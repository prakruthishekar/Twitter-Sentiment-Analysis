# Twitter-Sentiment-Analysis

This is a Python program that uses the Tweepy and TextBlob libraries to perform sentiment analysis on tweets fetched from Twitter. The program is designed to retrieve tweets containing a specific keyword or phrase and then analyze the sentiment of each tweet using TextBlob's sentiment analysis algorithm.

## Getting Started
Prerequisites
To run this program, you'll need:

Python 3.x
Tweepy library
TextBlob library


## Installing
Install Python 3.x on your computer. You can download it from the official website: https://www.python.org/downloads/

Install Tweepy by running the following command in your terminal:
pip install tweepy

Install TextBlob by running the following command in your terminal:
pip install textblob

## Setting up Twitter API
To use Tweepy to fetch tweets, you need to create a Twitter Developer account and create an application. Follow the instructions here to create a Twitter Developer account: https://developer.twitter.com/en/docs/twitter-api/getting-started/about-twitter-api

Once you have created your application, you will have access to your Consumer API keys and Access token & secret. You need to add these keys and tokens to the twitter_credentials.py file in the following format:

consumer_key = 'your_consumer_key'
consumer_secret = 'your_consumer_secret'
access_token = 'your_access_token'
access_token_secret = 'your_access_token_secret'


Running the Program
Clone this repository to your local machine
Update the twitter_credentials.py file with your Twitter API keys and tokens.

The program will retrieve the latest tweets containing your keyword or phrase and perform sentiment analysis on each tweet.
The program will output the overall sentiment analysis score as well as the sentiment analysis of each individual tweet.
