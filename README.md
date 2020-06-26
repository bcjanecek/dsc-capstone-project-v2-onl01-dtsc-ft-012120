# Emerald City Travel Recommendations

## Table of Contents
* [General Info](#General-Info)
* [Technologies](#Technologies)
* [Review Usefulness Prediction](#Review-Usefulness-Prediction)
* [Content Based Recommendation Engine](#Content-Based-Recommendation-Engine)
* [Potential Improvements](#Potential-Improvements)

## General Info
The scope of this project is to create a local business recommendation system for visitors to the Seattle area. The Seattle area sees over forty million visitors per year, who spend over $7.8 billion, and growth is expected to continue. Many flock here for culture and reputation of Seattle being a fertile ground for artists, musicians, and innovators. Studies show that younger generations are more concientious about supporting local businesses as well as being more inclined to seek unique travel experiences. Thus this project seeks to provide visitors with quality local business recommendations based on favorite local businesses in their hometown. 

In the final jupyter notebook, "Emerald City Travel Recommendations Application", users may select a destination of interest including:

1. Coffeeshops - Find a relaxing venue to work from or enjoy a local musician while sampling coffee from some of the country's finest roasters.
2. Boutique Stores - Shop for unique and consignment items in some of the quirkiest corners of Seattle.
3. Adult Entertainment - Sip a glass of the Columbia Valley's finest at a local winery or enjoy one of the city's hippest cocktail bars, speakeasies, and breweries. 

Once the destination of interest is selected the user will simply input the URL of the Yelp page of a favorite, related business in their hometown and notebook will return five recommendations. 

## Technologies
This project was created using the following languages and libraries. An environment with the correct versions of the following libraries will allow re-production and improvement on this project. 

* Python version: 3.6.9
* Matplotlib version: 3.0.3
* Seaborn version: 0.9.0
* Sklearn version: 0.20.3
* NLTK version: X.X.X
* TextBlob version: X.X.X
* XGBoost version: X.X.X

## Review Usefulness Prediction
In order to optimize the review-based recommendation engine it would be ideal to only use the most "useful" reviews. With respect to using content to generate recommendations it would be wise to only utilize nuetral to positive reviews as well as reviews which users recognize as useful. With this in mind, the "Predicting Useful Reviews" sought to use sentence structure features to predict whether a Yelp review has been classified as useful or not. Using the Yelp Academic Dataset we established a binary classfication problem with any review having one or more "useful" votes being classified as a useful review. 

Feature extraction yielded features such as part-of-speech tag counts, total word counts, average sentence length, average sentence subjectivity, and amount of "positive emotion" words used. odels trained, tuned, and evaluated included logistic regression, random forest, and XGBoost. Unfortunately these models using current features are unable to detect review usefulness with an accuracy beyond 65% which is hardly better than a random guess. This could be due to a number of reasons including insufficient features as well as the "usefulness" vote on Yelp not be an accurate indicator of whether or not a review is truly useful. For example, what is actually a useful review could be hidden on the fourth page and never receive votes. 

Fortunately, the exploratory data analysis and embedded feature selection conducted in this phase of the project yielded interesting insights regarding what makes a review useful so the work was not in vain. In general, it is apparent that:

1. Reviews classified as useful are wordier - this is evident accross nearly all POS tag counts. The difference is more apparent in certain tags (i.e. subordinating conjuctions and adjectives) compared to others (i.e. comparative adverbs). 
2. Reviews with a higher "positive emotion" word count are generally viewed as more useful. 
3. Useful reviews typically contain slightly more objective tone overall. 


## Content Based Recommendation Engine
By scraping review data of hundreds of coffeeshops, thrift stores, breweries and other points-of-interest across the Seattle area we've obtained a gold mine of information relating to the products, qualities, and atmosphere available at these businesses. This data has been cleaned, tokenized, and grouped by business to create three dataframes for each of our point-of-interest categories. Upon request of the user the notebook will scrape the review data of their inputted URL, clean the data in the same manner as the other reviews, and append the aggregated review data to the appropriate existing point-of-interest dataframe. The review data is then vectorized into a TF-IDF matrix and cosine similarity is used to find the most similar businesses in the matrix. 

An example below shows the recommendations for a traveler from Austin, TX seeking boutique store recommendations. 

## Potential Improvements
As with all data science projects there is always more work to do. Most notably I've been considering the following. 

### Exploring Word Embeddings

### Implementing n-grams Into TF-IDF Matrix
