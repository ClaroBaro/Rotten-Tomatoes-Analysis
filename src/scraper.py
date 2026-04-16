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
from urllib.parse import quote_plus

import pandas as pd
import requests
from bs4 import BeautifulSoup


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
    omdb_key : str, optional
        Free key from https://www.omdbapi.com/apikey.aspx (1,000 req/day).
    twitter_bearer : str, optional
        Bearer token from https://developer.x.com (free Basic tier).
    """

    def __init__(
        self,
        movie_name: str,
        omdb_key: str = None,
        twitter_bearer: str = None,
    ):
        self.movie_name = movie_name
        self.omdb_key = omdb_key
        self.twitter_bearer = twitter_bearer
        self._last_request_at = 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(
        self,
        url: str,
        params: dict = None,
        headers: dict = None,
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
    # 2. Metacritic — individual critic review list
    # ------------------------------------------------------------------

    def scrape_metacritic(self, max_reviews: int = 30) -> list[dict]:
        """
        Scrape individual critic reviews from Metacritic.

        Returns a list of dicts with keys:
            critic, publication, score (0-100), snippet

        Note: Metacritic may return a 403 if they detect automated requests.
        If that happens try adding a 'Cookie' header from a real browser session.
        """
        url = f"https://www.metacritic.com/movie/{self._slug(self.movie_name)}/critic-reviews/"
        try:
            resp = self._get(url)
        except requests.HTTPError as e:
            print(f"[Metacritic] HTTP {e.response.status_code} — the movie slug may be wrong, "
                  f"or Metacritic is blocking the request.")
            print(f"  Tried URL: {url}")
            return []

        soup = BeautifulSoup(resp.text, "html.parser")
        reviews = []

        for card in soup.select(".c-siteReview")[:max_reviews]:
            reviews.append({
                "source":      "metacritic",
                "critic":      _text(card, ".c-siteReview_criticName"),
                "publication": _text(card, ".c-siteReview_publicationName"),
                "score":       _text(card, ".c-siteReviewScore"),
                "snippet":     _text(card, ".c-siteReview_quote"),
            })

        print(f"[Metacritic] {len(reviews)} reviews scraped.")
        return reviews

    # ------------------------------------------------------------------
    # 3. Google News — via gnews package (handles URL decoding correctly)
    # ------------------------------------------------------------------

    def scrape_google_news(
        self,
        max_articles: int = 20,
        publications: list[str] = None,
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
        subreddits: list[str] = None,
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
        reddit_subreddits: list[str] = None,
        reddit_limit: int = 25,
        metacritic_reviews: int = 30,
        twitter_limit: int = 20,
        news_articles: int = 20,
        news_publications: list[str] = None,
    ) -> dict:
        """
        Run all scrapers and return a combined results dict.

        Keys: "omdb", "metacritic_reviews", "reddit_posts", "twitter", "google_news"
        """
        print(f"\n{'='*54}")
        print(f"  Scraping: {self.movie_name}")
        print(f"{'='*54}\n")

        results = {
            "omdb": self.get_omdb_scores(),
            "metacritic_reviews": self.scrape_metacritic(metacritic_reviews),
            "reddit_posts": self.scrape_reddit(
                subreddits=reddit_subreddits,
                limit=reddit_limit,
            ),
            "twitter": self.scrape_twitter(twitter_limit),
            "google_news": self.scrape_google_news(news_articles, news_publications),
        }

        print(f"\nSummary for '{self.movie_name}':")
        print(f"  OMDB scores found:    {bool(results['omdb'])}")
        print(f"  Metacritic reviews:   {len(results['metacritic_reviews'])}")
        print(f"  Reddit posts:         {len(results['reddit_posts'])}")
        print(f"  Tweets:               {len(results['twitter'])}")
        print(f"  News articles:        {len(results['google_news'])}")
        return results

    def to_dataframe(self, results: dict) -> dict[str, pd.DataFrame]:
        """
        Convert scrape_all() output to a dict of DataFrames, one per source.

        Keys: "critic_reviews", "reddit", "twitter", "news"
        """
        dfs = {}
        if results.get("metacritic_reviews"):
            dfs["critic_reviews"] = pd.DataFrame(results["metacritic_reviews"])
        if results.get("reddit_posts"):
            dfs["reddit"] = pd.DataFrame(results["reddit_posts"])
        if results.get("twitter"):
            dfs["twitter"] = pd.DataFrame(results["twitter"])
        if results.get("google_news"):
            dfs["news"] = pd.DataFrame(results["google_news"])
        return dfs


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _text(tag, selector: str) -> str | None:
    """Extract stripped text from a CSS selector within a BeautifulSoup tag."""
    el = tag.select_one(selector)
    return el.get_text(strip=True) if el else None


def _text_tag(item, tag_name: str) -> str | None:
    """Extract stripped text from a named XML/HTML tag."""
    el = item.find(tag_name)
    return el.get_text(strip=True) if el else None


def _resolve_google_news_url(url: str) -> str | None:
    """
    Decode the real article URL from a Google News RSS link.

    Google News RSS links (CBMi...) are base64url-encoded protobufs.
    Following the redirect fails because Google requires browser cookies.
    Instead, decode the protobuf locally and extract the embedded URL.
    """
    import base64
    if not url:
        return None
    try:
        match = re.search(r"/articles/([^?]+)", url)
        if not match:
            return url
        encoded = match.group(1)
        # Restore base64 padding
        encoded += "=" * (4 - len(encoded) % 4)
        decoded = base64.urlsafe_b64decode(encoded)
        # The article URL is embedded as a UTF-8 string inside the protobuf binary
        url_match = re.search(rb"https?://[^\x00-\x20\x7f-\xff]+", decoded)
        if url_match:
            return url_match.group().decode("utf-8", errors="ignore")
        return url
    except Exception:
        return url
