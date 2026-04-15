# Rotten-Tomatoes-Analysis
## Inquiry: 
We have all experienced the excitement of being recommended an interesting movie, only to google it and find its incredibly low ratings on Rotten Tomatoes. This project will explore the breakdown of these ratings, their validity, and what specific bias might exist using a wide critic rating system.

## Data Sources:
Historical Rotten Tomatoes Data: [https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset](url)

## EDA Inquiries & Findings
#### 1. [Audience Ratings] Does Rotten Tomatoes' Popcorn Score (public audience score) have a positive association with Tomato Score? 
<img width="681" height="545" alt="image" src="https://github.com/user-attachments/assets/2104ec84-78cb-4d12-9803-37ef2c3c65e5" />

The $R^2$ of this relationship is only 0.343. Audience scores differed from critic scores in the following numbers: 


- Movies where critics scored higher: 14786 (55%)
  
- Movies where audience scored higher: 11411 (42%)
  
- Movies were scores were equal: 562 (2%)

  
To our surprise, only 2% of movies had equal critic and audience scores, indicating that we must we cautious in using public audience sentiment to predict a movie's tomato score. 


Additionally, to answer if audience scores are biased towards underating or overating, the proportions across 25000+ movies tell us they are only slightly more likely to underate.

#### 2. [Critic Ratings] Who are the most 'consistent' critics? Do they have the 'topCritic' designation? 
Top 10 critics with highest fresh/rotten accuracy & 100+ reviews:

<img width="50%" height="584" alt="image" src="https://github.com/user-attachments/assets/35716f2b-5133-4ccd-a76d-4ca24c164fd3" />

Top 10 critics with highest fresh/rotten accuracy & 1000+ reviews:

<img width="50%" height="582" alt="image" src="https://github.com/user-attachments/assets/e5c34524-3190-44ad-b207-c9ef3b1d3556" />

It appears that the majority of the most accurate critics are not top critics, so being a top critic on Rotten Tomatoes does not necessarily mean your ratings is more consistent with the movie's actual fresh/rotten status.

#### 3. [Critic Ratings] How many critic reviews do we need until we can compute a good approximate of final tomato score? 

<img width="70%" height="964" alt="image" src="https://github.com/user-attachments/assets/f6328a1d-2b7c-4298-9d30-9974131325cc" />

It appears at least 10 reviews are required to drastically reduce RMSE, with 20 reviews being preferred (cuts RMSE in half). 


#### 4. [Metadata] Is a movie's metadata (ie. Genre, Director, Writer, Maturity Rating) good features for predicting tomato score? 
Converted preceding features to categorical variables (one-hot encoded). For example, genre is represented by 'Fantasy' ~ 0/1, 'Sci-fi' ~ 0/1, and so on. Using the four metadata features above in a ridge regression model, we achieve an $R^2$ of 0.334. Thus, a movie's metadata alone isn't a great constructor of a tomato score prior. 

#### 5a. [Critic Media Sources] What are the most common media sources for reviews? Are they biased toward overating or underating movies? 
Two metrics are computed to evaluate a media source's bias: 
- Mean Residual ~ positive value indicates media source scores higher than concensus on average, negative value indicates media source scores lower than concensus on average.
- RMSE ~ how consistently aligned the media source is with with the final score, with lower values indicating more accurate/predictive.

<img width="50%" height="961" alt="image" src="https://github.com/user-attachments/assets/1b5f5a2d-bd2b-4d50-bb86-7012573077ab" />

#### 5b. [Critic Media Sources] Distribution of the bias of all the media sources provided in the dataset. 
<img width="1488" height="489" alt="image" src="https://github.com/user-attachments/assets/f1f888d4-20b4-41cf-adb8-ec82d3f39811" />

Bias is normally distributed with mean -1.7, slightly underating on average.

#### 6. [Review Sentiment] Are the contents of critic reviews a good predictor of a movie's tomato score? 
Vader model performs very poorly with our review data. roBERTa model performs decently well, achieving an $R^2$ of 0.491 when tomato score is regressed on roBERTa's sentiment score. Correlation is 0.687. Sentiment model is not accurate enough to benefit our model currently; retry after more review text data is obtained via webscraping. 






