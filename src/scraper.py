"""
Movie review and public sentiment scraper.

Sources
-------
Critic reviews
  1. OMDB API       — RT + Metacritic + IMDB aggregate scores (free key required)
  2. Metacritic     — individual critic review list (HTML scraping)
  3. Google News    — recent review articles, filterable by publication (RSS, no key)

Public sentiment
  4. Reddit         — posts + comments via anonymous JSON API (no key required)
  5. X / Twitter    — recent tweets via API v2 (free developer bearer token required)

Not included
  • Facebook  — public post scraping requires Graph API + app-level user auth; not feasible
  • Quora     — ToS prohibits scraping; heavy JS rendering requires Selenium

Setup
-----
    pip install requests beautifulsoup4 lxml pandas

Usage
-----
    from src.scraper import MovieScraper

    s = MovieScraper(
        "Warfare",
        omdb_key="your_key",          # https://www.omdbapi.com/apikey.aspx (free)
        twitter_bearer="your_token",  # https://developer.x.com (free Basic tier)
    )
    results = s.scrape_all()
    dfs = s.to_dataframe(results)     # dict of DataFrames keyed by source
"""

import re
import time

import pandas as pd
import requests


_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

_RATE_LIMIT_SEC = 1.5   # minimum seconds between outgoing requests


class MovieScraper:
    """
    Scrape critic reviews and public sentiment for a given movie title.

    Parameters
    ----------
    movie_name : str
        Title of the movie (e.g. "Inception").
    data_path : str, optional
        Path to master_table.csv from the Kaggle dataset. When provided,
        the scraper automatically selects the top N publications by lowest
        combined RMSE + bias from your historical data instead of the
        hard-coded fallback list.
    n_publications : int
        How many top publications to target (default 50).
    min_reviews : int
        Minimum review count a publication must have in the dataset to
        be considered (default 100, for a stable RMSE estimate).
    omdb_key : str, optional
        Free key from https://www.omdbapi.com/apikey.aspx (1,000 req/day).
    twitter_bearer : str, optional
        Bearer token from https://developer.x.com (free Basic tier).
    """

    # Fallback used only when data_path is not provided
    _FALLBACK_PUBLICATIONS = [
        "Variety", "The Hollywood Reporter", "IndieWire", "RogerEbert.com",
        "The Guardian", "New York Times", "Chicago Sun-Times", "NPR",
        "Entertainment Weekly", "New York Post", "San Francisco Chronicle",
        "Time Out", "Slant Magazine", "Austin Chronicle", "The Wrap",
        "Los Angeles Times", "Washington Post", "Rolling Stone", "Filmsite", 
        "The Jam Report", "Impression Blend", "NPR", "Women's Voices for Change",
        "CineXpress Podcast", "Access Hollywood", "Austin Burke/Flick Fan Nation",
        "Cine Sin Fronteras", "Cinema Siren", "Newshub", "Dark Horizons", 
        "Detroit News", "iNews.co.uk", "Alternative Lens", "Kalamazoo Gazette", 
        "CNN Radio", "Suburban Journals of St. Louis", "Panorama", "MovieJuice!"
    ]

    def __init__(
        self,
        movie_name: str,
        data_path: str | None = None,
        n_publications: int = 50,
        min_reviews: int = 100,
        omdb_key: str | None = None,
        twitter_bearer: str | None = None,
    ):
        self.movie_name = movie_name
        self.omdb_key = omdb_key
        self.twitter_bearer = twitter_bearer
        self._last_request_at = 0.0

        if data_path:
            self.publications = self.top_publications(
                data_path, n=n_publications, min_reviews=min_reviews
            )
            print(f"[Publications] Loaded {len(self.publications)} publications from dataset.")
        else:
            self.publications = self._FALLBACK_PUBLICATIONS

    # ------------------------------------------------------------------
    # Derive top publications from historical data
    # ------------------------------------------------------------------

    @staticmethod
    def top_publications(
        data_path: str,
        n: int = 50,
        min_reviews: int = 100,
    ) -> list[str]:
        """
        Read master_table.csv and return the top N publication names ranked
        by a combined score of low RMSE and low absolute bias — i.e. the
        publications whose scores most closely track the final tomatoMeter.

        Parameters
        ----------
        data_path : str
            Path to master_table.csv.
        n : int
            Number of publications to return.
        min_reviews : int
            Minimum reviews a publication must have for a stable estimate.
        """
        master = pd.read_csv(data_path)
        master = master.dropna(subset=["tomatoMeter", "reviewId", "originalScore"])
        master = master.drop_duplicates(subset=["reviewId"])

        # reuse standardize_score to get numeric scores
        try:
            from src.cleaning import standardize_score
        except ImportError:
            from cleaning import standardize_score

        master["standardized_score"] = master["originalScore"].apply(standardize_score)
        master = master.dropna(subset=["standardized_score"])

        master["resid"]        = master["standardized_score"] - master["tomatoMeter"]
        master["squared_resid"] = master["resid"] ** 2

        pub_stats = (
            master.groupby("publicatioName")
            .agg(
                review_count=("reviewId",       "count"),
                rmse        =("squared_resid",  lambda x: x.mean() ** 0.5),
                mean_bias   =("resid",          "mean"),
            )
            .reset_index()
        )

        pub_stats = pub_stats[pub_stats["review_count"] >= min_reviews].copy()

        # normalize both metrics to [0, 1] then average — lower is better
        for col in ("rmse", "mean_bias"):
            lo = pub_stats[col].abs().min()
            hi = pub_stats[col].abs().max()
            pub_stats[f"{col}_norm"] = (pub_stats[col].abs() - lo) / (hi - lo + 1e-9)

        pub_stats["combined"] = (pub_stats["rmse_norm"] + pub_stats["mean_bias_norm"]) / 2
        top = pub_stats.nsmallest(n, "combined")

        print(f"[Publications] Top {len(top)} publications (RMSE range: "
              f"{top['rmse'].min():.1f}–{top['rmse'].max():.1f}, "
              f"bias range: {top['mean_bias'].min():.1f}–{top['mean_bias'].max():.1f})")

        return top["publicatioName"].tolist()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(
        self,
        url: str,
        params: dict | None = None,
        headers: dict | None = None,
        delay: float = _RATE_LIMIT_SEC,
    ) -> requests.Response:
        """Rate-limited GET. Raises on non-2xx status."""
        elapsed = time.time() - self._last_request_at
        if elapsed < delay:
            time.sleep(delay - elapsed)
        resp = requests.get(
            url,
            params=params,
            headers=headers if headers is not None else _HEADERS,
            timeout=15,
        )
        self._last_request_at = time.time()
        resp.raise_for_status()
        return resp

    def _slug(self, text: str) -> str:
        """Convert a movie title to a URL-friendly slug."""
        return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")

    # ------------------------------------------------------------------
    # 1. OMDB API — RT + Metacritic + IMDB scores in a single call
    # ------------------------------------------------------------------

    def get_omdb_scores(self) -> dict:
        """
        Fetch Rotten Tomatoes, Metacritic, and IMDB scores via OMDB.
        Returns an empty dict if no API key is set or the movie isn't found.
        """
        if not self.omdb_key:
            print("[OMDB] No API key — skipping. Get one free at https://www.omdbapi.com/apikey.aspx")
            return {}

        resp = self._get(
            "https://www.omdbapi.com/",
            params={"t": self.movie_name, "apikey": self.omdb_key},
            headers={},     # OMDB doesn't need browser headers
        )
        data = resp.json()

        if data.get("Response") == "False":
            print(f"[OMDB] Not found: {data.get('Error')}")
            return {}

        scores = {
            "title":   data.get("Title"),
            "year":    data.get("Year"),
            "imdb_id": data.get("imdbID"),
            "genre":   data.get("Genre"),
            "director": data.get("Director"),
        }
        for rating in data.get("Ratings", []):
            if rating["Source"] == "Rotten Tomatoes":
                scores["rt_score"] = rating["Value"]
            elif rating["Source"] == "Metacritic":
                scores["metacritic"] = rating["Value"]
            elif rating["Source"] == "Internet Movie Database":
                scores["imdb"] = rating["Value"]

        print(f"[OMDB] {scores}")
        return scores

    # ------------------------------------------------------------------
    # 2. Publication reviews — search Google News per publication,
    #    scrape full article text, extract score with standardize_score
    # ------------------------------------------------------------------

    def scrape_publication_reviews(
        self,
        publications: list[str] | None = None,
    ) -> list[dict]:
        """
        For each publication, search Google News for a review of the movie,
        scrape the full article text, and attempt to extract a numeric score.

        Uses trafilatura for article text extraction (more reliable than
        newspaper3k on Python 3.11+).  Install: pip install trafilatura gnews

        Parameters
        ----------
        publications : list of str, optional
            Publication names to search. Defaults to DEFAULT_PUBLICATIONS.
        """
        try:
            from gnews import GNews
        except ImportError:
            print("[Publications] Run: pip install gnews")
            return []

        try:
            import trafilatura
        except ImportError:
            print("[Publications] Run: pip install trafilatura")
            return []

        try:
            from src.cleaning import standardize_score
        except ImportError:
            from cleaning import standardize_score

        pubs = publications or self.publications
        reviews = []

        for pub in pubs:
            gn = GNews(language="en", country="US", max_results=5)
            query = f'"{self.movie_name}" review {pub}'

            try:
                candidates = gn.get_news(query)
            except Exception as e:
                print(f"[Publications] {pub}: search failed — {e}")
                continue

            # prefer articles whose publisher name matches
            matched = [
                r for r in candidates
                if pub.lower() in r.get("publisher", {}).get("title", "").lower()
            ]
            item = (matched or candidates or [None])[0]
            if not item:
                continue

            url = item.get("url")
            if not url:
                continue

            # scrape full article text + metadata
            article_date = None
            html = ""
            try:
                html = trafilatura.fetch_url(url) or ""
                extracted = trafilatura.bare_extraction(html) or {}
                text = extracted.get("text") or ""
                article_date = extracted.get("date")
            except Exception:
                text = ""

            # fall back to article title if scraping failed
            if not text:
                text = item.get("title", "")

            # skip articles not actually about this movie (e.g. related-article sidebars)
            if not _is_relevant_article(text, item.get("title", ""), self.movie_name):
                print(f"  [{pub}] Skipping — article does not appear to be about {self.movie_name!r}")
                continue

            pub_date = item.get("published date") or article_date

            # structured data (JSON-LD / schema.org) is most reliable; fall back to text
            score = (
                _extract_structured_score(html)
                or _extract_score(text)
                or _extract_score(item.get("title", ""))
            )

            reviews.append({
                "source":      "publication",
                "publication": pub,
                "url":         url,
                "title":       item.get("title"),
                "text":        text,
                "score":       score,
                "has_score":   score is not None,
                "pub_date":    pub_date,
            })
            print(f"  [{pub}] {'score=' + str(score) if score else 'no score'} — {len(text)} chars")

        print(f"[Publications] {len(reviews)}/{len(pubs)} publications found.")
        return reviews

    # ------------------------------------------------------------------
    # 3. Google News — via gnews package (handles URL decoding correctly)
    # ------------------------------------------------------------------

    def scrape_google_news(
        self,
        max_articles: int = 20,
        publications: list[str] | None = None,
    ) -> list[dict]:
        """
        Search Google News for recent articles about the movie.
        Uses the gnews package which correctly resolves article URLs.

        Install: pip install gnews

        Parameters
        ----------
        publications : list of str, optional
            Filter to articles from specific outlets
            (e.g. ["NPR", "The Guardian"]).
        """
        try:
            from gnews import GNews
        except ImportError:
            print("[Google News] gnews not installed. Run: pip install gnews")
            return []

        gn = GNews(language="en", country="US", max_results=max_articles * 3)
        query = f'"{self.movie_name}" movie review'

        try:
            raw_results = gn.get_news(query)
        except Exception as e:
            print(f"[Google News] {e}")
            return []

        articles = []
        for item in raw_results:
            pub = item.get("publisher", {}).get("title", "")

            if publications:
                pubs_lower = [p.lower() for p in publications]
                if not any(p in pub.lower() for p in pubs_lower):
                    continue

            articles.append({
                "source":      "google_news",
                "title":       item.get("title"),
                "link":        item.get("url"),
                "pub_date":    item.get("published date"),
                "publication": pub,
            })
            if len(articles) >= max_articles:
                break

        print(f"[Google News] {len(articles)} articles found.")
        return articles

    # ------------------------------------------------------------------
    # 4. Reddit — anonymous JSON API (no key required)
    # ------------------------------------------------------------------

    def scrape_reddit(
        self,
        subreddits: list[str] | None = None,
        limit: int = 25,
        sort: str = "top",
    ) -> list[dict]:
        """
        Search Reddit for posts mentioning the movie.

        Parameters
        ----------
        subreddits : list of str, optional
            Restrict search to specific subreddits.
            Defaults to ["movies", "flicks", "TrueFilm", "MovieSuggestions"].
        limit : int
            Max posts to return (Reddit caps at 100 per request).
        sort : str
            "top", "new", or "relevance".
        """
        subs = subreddits or ["movies", "flicks", "TrueFilm", "MovieSuggestions"]
        sr_filter = "+".join(subs)
        url = f"https://www.reddit.com/r/{sr_filter}/search.json"

        params = {
            "q": self.movie_name,
            "sort": sort,
            "limit": min(limit, 100),
            "restrict_sr": "true",
            "type": "link",
        }

        try:
            resp = self._get(url, params=params)
        except requests.HTTPError as e:
            print(f"[Reddit] {e}")
            return []

        posts = []
        for child in resp.json().get("data", {}).get("children", []):
            d = child["data"]
            posts.append({
                "source":       "reddit",
                "subreddit":    d.get("subreddit"),
                "title":        d.get("title"),
                "upvotes":      d.get("score"),
                "upvote_ratio": d.get("upvote_ratio"),
                "num_comments": d.get("num_comments"),
                "url":          d.get("url"),
                "selftext":     d.get("selftext", "")[:600],
                "created_utc":  d.get("created_utc"),
            })

        print(f"[Reddit] {len(posts)} posts found across r/{sr_filter}.")
        return posts

    def scrape_reddit_comments(
        self,
        post_url: str,
        limit: int = 50,
    ) -> list[dict]:
        """
        Fetch top-level comments from a single Reddit post URL.
        Useful for drilling into a high-upvote post from scrape_reddit().
        """
        json_url = post_url.rstrip("/") + ".json"
        try:
            resp = self._get(json_url)
        except requests.HTTPError as e:
            print(f"[Reddit comments] {e}")
            return []

        data = resp.json()
        comment_listing = data[1] if isinstance(data, list) and len(data) > 1 else {}
        comments = []

        for child in comment_listing.get("data", {}).get("children", [])[:limit]:
            if child.get("kind") != "t1":   # t1 = comment; skip "more" stubs
                continue
            d = child["data"]
            comments.append({
                "source":  "reddit_comment",
                "body":    d.get("body", "")[:600],
                "upvotes": d.get("score"),
                "author":  d.get("author"),
            })

        print(f"[Reddit comments] {len(comments)} comments fetched.")
        return comments

    # ------------------------------------------------------------------
    # 5. X / Twitter — API v2 (free developer bearer token)
    # ------------------------------------------------------------------

    def scrape_twitter(
        self,
        max_results: int = 20,
        lang: str = "en",
    ) -> list[dict]:
        """
        Search recent tweets via the X API v2.

        Requires a bearer token from https://developer.x.com/en/portal/dashboard
        Free Basic tier: ~500k tweet reads/month.

        Excludes retweets and replies to focus on original opinions.
        """
        if not self.twitter_bearer:
            print("[Twitter/X] No bearer token — skipping.")
            print("  Get a free token at: https://developer.x.com/en/portal/dashboard")
            return []

        query = f'"{self.movie_name}" movie lang:{lang} -is:retweet -is:reply'
        try:
            resp = self._get(
                "https://api.twitter.com/2/tweets/search/recent",
                params={
                    "query":        query,
                    "max_results":  min(max(max_results, 10), 100),  # API minimum is 10
                    "tweet.fields": "text,created_at,public_metrics",
                },
                headers={"Authorization": f"Bearer {self.twitter_bearer}"},
            )
        except requests.HTTPError as e:
            print(f"[Twitter/X] {e}")
            return []

        tweets = []
        for t in resp.json().get("data", []):
            m = t.get("public_metrics", {})
            tweets.append({
                "source":     "twitter",
                "text":       t.get("text"),
                "created_at": t.get("created_at"),
                "likes":      m.get("like_count"),
                "retweets":   m.get("retweet_count"),
                "replies":    m.get("reply_count"),
            })

        print(f"[Twitter/X] {len(tweets)} tweets found.")
        return tweets

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------

    def scrape_all(
        self,
        publications: list[str] | None = None,
        reddit_subreddits: list[str] | None = None,
        reddit_limit: int = 25,
        twitter_limit: int = 20,
        news_articles: int = 20,
        news_publications: list[str] | None = None,
    ) -> dict:
        """
        Run all scrapers and return a combined results dict.

        Keys: "omdb", "publication_reviews", "reddit_posts", "twitter", "google_news"
        """
        print(f"\n{'='*54}")
        print(f"  Scraping: {self.movie_name}")
        print(f"{'='*54}\n")

        results = {
            "omdb":                self.get_omdb_scores(),
            "publication_reviews": self.scrape_publication_reviews(publications),
            "reddit_posts":        self.scrape_reddit(
                                       subreddits=reddit_subreddits,
                                       limit=reddit_limit,
                                   ),
            "twitter":             self.scrape_twitter(twitter_limit),
            "google_news":         self.scrape_google_news(news_articles, news_publications),
        }

        print(f"\nSummary for '{self.movie_name}':")
        print(f"  OMDB scores found:      {bool(results['omdb'])}")
        print(f"  Publication reviews:    {len(results['publication_reviews'])}")
        print(f"  Reddit posts:           {len(results['reddit_posts'])}")
        print(f"  Tweets:                 {len(results['twitter'])}")
        print(f"  News articles:          {len(results['google_news'])}")
        return results

    def to_dataframe(self, results: dict) -> dict[str, pd.DataFrame]:
        """
        Convert scrape_all() output to a dict of DataFrames, one per source.

        Keys: "critic_reviews", "reddit", "twitter", "news"
        """
        dfs = {}
        if results.get("publication_reviews"):
            dfs["critic_reviews"] = pd.DataFrame(results["publication_reviews"])
        if results.get("reddit_posts"):
            dfs["reddit"] = pd.DataFrame(results["reddit_posts"])
        if results.get("twitter"):
            dfs["twitter"] = pd.DataFrame(results["twitter"])
        if results.get("google_news"):
            dfs["news"] = pd.DataFrame(results["google_news"])
        return dfs

    def to_sentiment_df(self, results: dict) -> pd.DataFrame:
        """
        Flatten scrape_all() results into a single DataFrame ready for
        sentiment analysis.

        Columns
        -------
        source       : where the text came from
        text         : the text to run sentiment analysis on
        score        : numeric score 0-100 where available, else NaN
                       - Metacritic: critic score (already 0-100)
                       - Reddit:     upvote_ratio * 100 (proxy for crowd approval)
                       - Twitter:    NaN (no score)
                       - News:       NaN (title only, no score)
        has_score    : True if a real numeric score exists (Metacritic only)
        """
        rows = []

        # Publication reviews — full article text + extracted score
        for r in results.get("publication_reviews", []):
            score = r.get("score")
            rows.append({
                "source":    "publication",
                "text":      (r.get("text") or r.get("title") or "").strip(),
                "score":     float(score) if score is not None else float("nan"),
                "has_score": score is not None,
                "meta":      r.get("publication", ""),
            })

        # Reddit — combine post title + body text; no score
        for r in results.get("reddit_posts", []):
            title = r.get("title") or ""
            body  = r.get("selftext") or ""
            text  = (title + " " + body).strip()
            rows.append({
                "source":    "reddit",
                "text":      text,
                "score":     float("nan"),
                "has_score": False,
                "meta":      "r/" + (r.get("subreddit") or ""),
            })

        # Twitter — tweet text only, no score
        for r in results.get("twitter", []):
            rows.append({
                "source":    "twitter",
                "text":      (r.get("text") or "").strip(),
                "score":     float("nan"),
                "has_score": False,
                "meta":      "",
            })

        # Google News — article title as text proxy (full text needs article scraping)
        for r in results.get("google_news", []):
            rows.append({
                "source":    "google_news",
                "text":      (r.get("title") or "").strip(),
                "score":     float("nan"),
                "has_score": False,
                "meta":      r.get("publication") or "",
            })

        df = pd.DataFrame(rows)
        df = df[df["text"].str.len() > 0].reset_index(drop=True)   # drop blank rows
        return df


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _is_relevant_article(text: str, title: str, movie_name: str) -> bool:
    """
    Return True only if the article is actually about this movie.
    A page where the movie appears once in a sidebar will fail this check.
    """
    name = movie_name.lower()
    if name in (title or "").lower():
        return True
    return (text or "").lower().count(name) >= 3


def _extract_structured_score(html: str) -> float | None:
    """
    Extract a critic rating from structured data embedded in the page HTML:
      - JSON-LD schema.org  (reviewRating / aggregateRating)
      - itemprop="ratingValue" + itemprop="bestRating"
      - <meta name="rating" content="...">
    This is checked before any text-scraping because it is unambiguous.
    """
    if not html:
        return None
    try:
        import json
        from bs4 import BeautifulSoup
        try:
            from src.cleaning import standardize_score
        except ImportError:
            from cleaning import standardize_score

        soup = BeautifulSoup(html, "html.parser")

        # 1. JSON-LD schema.org
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(script.string or "")
                if isinstance(data, list):
                    data = data[0] if data else {}
                for key in ("reviewRating", "aggregateRating"):
                    rating = data.get(key, {})
                    if not isinstance(rating, dict):
                        continue
                    value = rating.get("ratingValue")
                    best  = rating.get("bestRating", 10)
                    if value:
                        return round(float(value) / float(best) * 100, 1)
            except Exception:
                continue

        # 2. itemprop microdata
        rv_el = soup.find(attrs={"itemprop": "ratingValue"})
        br_el = soup.find(attrs={"itemprop": "bestRating"})
        if rv_el:
            value = rv_el.get("content") or rv_el.get_text(strip=True)
            best  = (br_el.get("content") or br_el.get_text(strip=True)) if br_el else "10"
            try:
                return round(float(value) / float(best) * 100, 1)
            except Exception:
                pass

        # 3. <meta name="rating" content="...">
        meta = soup.find("meta", attrs={"name": re.compile(r"rating", re.I)})
        if meta:
            score = standardize_score(meta.get("content", ""))
            if score is not None:
                return score

        return None
    except Exception:
        return None


def _extract_score(text: str) -> float | None:
    """
    Scan article text for a critic score and normalize it to 0-100.

    Strategy (in priority order):
    1. Explicit rating labels anywhere in the text ("Rating: 1.5/5", "Score: B+")
       — catches dedicated rating blocks like the one on rendyreviews.com
    2. Score patterns in the first + last 600 chars (lede and sign-off)
    3. Score patterns anywhere in the full text (middle of review)
    """
    if not text:
        return None
    try:
        from src.cleaning import standardize_score
    except ImportError:
        from cleaning import standardize_score

    # Written-out star ratings: "four stars out of five", "three and a half stars out of four"
    _word_num = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
    }
    word_stars = re.search(
        r'\b(zero|one|two|three|four|five|six|seven|eight|nine|ten)'
        r'(?:\s+and\s+(?:a\s+)?half)?'
        r'\s*[-\s]?stars?\s+(?:out\s+of|/)\s*'
        r'(zero|one|two|three|four|five|six|seven|eight|nine|ten)\b',
        text, re.IGNORECASE
    )
    if word_stars:
        num = _word_num.get(word_stars.group(1).lower(), 0)
        # check for "and a half" between the number word and "stars"
        between = text[word_stars.start(1):word_stars.start(2)]
        if re.search(r'and\s+(?:a\s+)?half', between, re.IGNORECASE):
            num += 0.5
        den = _word_num.get(word_stars.group(2).lower(), 0)
        if den:
            score = standardize_score(f"{num}/{den}")
            if score is not None:
                return score

    # Pre-normalize "1.5 stars / 5 stars" → "1.5/5" so standardize_score can handle it
    stars_pattern = re.compile(
        r'(\d+\.?\d*)\s*(?:stars?\s*/|/)\s*(\d+\.?\d*)\s*stars?'
        r'|(\d+\.?\d*)\s*(?:stars?\s+)?out\s+of\s+(\d+\.?\d*)\s*stars?',
        re.IGNORECASE
    )
    sm = stars_pattern.search(text)
    if sm:
        num, den = (sm.group(1), sm.group(2)) if sm.group(1) else (sm.group(3), sm.group(4))
        score = standardize_score(f"{num}/{den}")
        if score is not None:
            return score

    score_patterns = [
        r'\d+\.?\d*\s*/\s*\d+',           # 1.5/5, 8/10
        r'\d+\.?\d*\s*out\s*of\s*\d+',    # 3 out of 5
        r'\d+\s*%',                         # 80%
        r'[A-Fa-f][+-]?(?=\s|$)',          # B+, C-
        r'[★✩]{1,5}',                      # star symbols
    ]

    # 1. Look for explicit label + score anywhere in the text
    label_pattern = re.compile(
        r'(?:rating|score|grade|verdict|stars?|our\s+rating|critic\s+score|review\s+score)'
        r'\s*[:\-–]\s*([^\n]{1,40})',
        re.IGNORECASE
    )
    for m in label_pattern.finditer(text):
        candidate = m.group(1).strip()
        for sp in score_patterns:
            sm = re.search(sp, candidate, re.IGNORECASE)
            if sm:
                score = standardize_score(sm.group(0))
                if score is not None:
                    return score

    # 2. Search first + last 600 chars
    search_zone = text[:600] + " " + text[-600:]
    for sp in score_patterns:
        m = re.search(sp, search_zone, re.IGNORECASE)
        if m:
            score = standardize_score(m.group(0))
            if score is not None:
                return score

    # 3. Full text scan as last resort
    for sp in score_patterns:
        m = re.search(sp, text, re.IGNORECASE)
        if m:
            score = standardize_score(m.group(0))
            if score is not None:
                return score

    return None
