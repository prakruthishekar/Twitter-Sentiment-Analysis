# About the alogorithm of TextBlob: TextBlob's sentiment analysis algorithm is based on a simple pattern analysis approach that uses a pre-defined set of rules to identify and score the sentiment of words and phrases in a sentence. The algorithm uses a sentiment lexicon, which is a dictionary of words and their associated sentiment scores (e.g., positive or negative), to determine the overall sentiment of a sentence.

# The sentiment lexicon used by TextBlob is based on the MPQA (Multi-Perspective Question Answering) lexicon, which was developed by researchers at the University of Pittsburgh. The MPQA lexicon contains over 8,000 words and phrases, each of which is assigned a polarity score (ranging from -1 for very negative to +1 for very positive) and a subjectivity score (ranging from 0 for very objective to +1 for very subjective).

# When TextBlob analyzes the sentiment of a sentence, it first tokenizes the text into individual words and phrases, and then looks up each word or phrase in the MPQA lexicon to determine its polarity and subjectivity scores. The algorithm then combines these scores to calculate an overall polarity score for the sentence, which ranges from -1 for very negative to +1 for very positive. The subjectivity score, which indicates how subjective or objective the sentence is, is also calculated and returned as part of the sentiment analysis result.

# TextBlob's part-of-speech tagging algorithm is based on NLTK's pos_tag function, which uses a Hidden Markov Model (HMM) to assign part-of-speech tags to words in a sentence. The HMM is trained on a large corpus of text to learn the probability of each word being a particular part of speech based on its context in the sentence.

# In TextBlob's sentiment analysis algorithm, the polarity score ranges from -1 to +1, where -1 indicates very negative sentiment, +1 indicates very positive sentiment, and 0 indicates neutral sentiment.


# TextBlob is a Python library for processing textual data. It provides a simple and intuitive API for performing common natural language processing (NLP) tasks such as part-of-speech tagging, noun phrase extraction, sentiment analysis, classification, translation, and more.


# To create a TextBlob object, you can simply pass a string of text to the constructor:


# text = "This is a sample sentence."
# blob = TextBlob(text)

# Once you have a TextBlob object, you can perform various NLP tasks on it. For example, you can get the part-of-speech tags for each word in the text by calling the tags property:

# tags = blob.tags
# print(tags)
# You can also extract noun phrases from the text using the noun_phrases property:

# noun_phrases = blob.noun_phrases
# print(noun_phrases)

# TextBlob also provides a built-in sentiment analyzer that you can use to analyze the sentiment of a piece of text:
# sentiment = blob.sentiment

# print(sentiment.polarity)  # polarity ranges from -1 to 1
# print(sentiment.subjectivity)  # subjectivity ranges from 0 to 1


from tweepy import API 
from tweepy import Cursor
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream

from textblob import TextBlob 
 
import twitter_credentials

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re


# # # # TWITTER CLIENT # # # #
class TwitterClient():
    def __init__(self, twitter_user=None):
        self.auth = TwitterAuthenticator().authenticate_twitter_app()
        self.twitter_client = API(self.auth)

        self.twitter_user = twitter_user

    def get_twitter_client_api(self):
        return self.twitter_client

    def get_user_timeline_tweets(self, num_tweets):
        tweets = []
        for tweet in Cursor(self.twitter_client.user_timeline, id=self.twitter_user).items(num_tweets):
            tweets.append(tweet)
        return tweets

    def get_friend_list(self, num_friends):
        friend_list = []
        for friend in Cursor(self.twitter_client.friends, id=self.twitter_user).items(num_friends):
            friend_list.append(friend)
        return friend_list

    def get_home_timeline_tweets(self, num_tweets):
        home_timeline_tweets = []
        for tweet in Cursor(self.twitter_client.home_timeline, id=self.twitter_user).items(num_tweets):
            home_timeline_tweets.append(tweet)
        return home_timeline_tweets


# # # # TWITTER AUTHENTICATER # # # #
class TwitterAuthenticator():

    def authenticate_twitter_app(self):
        auth = OAuthHandler(twitter_credentials.CONSUMER_KEY, twitter_credentials.CONSUMER_SECRET)
        auth.set_access_token(twitter_credentials.ACCESS_TOKEN, twitter_credentials.ACCESS_TOKEN_SECRET)
        return auth

# # # # TWITTER STREAMER # # # #
class TwitterStreamer():
    """
    Class for streaming and processing live tweets.
    """
    def __init__(self):
        self.twitter_autenticator = TwitterAuthenticator()    

    def stream_tweets(self, fetched_tweets_filename, hash_tag_list):
        # This handles Twitter authetification and the connection to Twitter Streaming API
        listener = TwitterListener(fetched_tweets_filename)
        auth = self.twitter_autenticator.authenticate_twitter_app() 
        stream = Stream(auth, listener)

        # This line filter Twitter Streams to capture data by the keywords: 
        stream.filter(track=hash_tag_list)


# # # # TWITTER STREAM LISTENER # # # #
class TwitterListener(StreamListener):
    """
    This is a basic listener that just prints received tweets to stdout.
    """
    def __init__(self, fetched_tweets_filename):
        self.fetched_tweets_filename = fetched_tweets_filename

    def on_data(self, data):
        try:
            print(data)
            with open(self.fetched_tweets_filename, 'a') as tf:
                tf.write(data)
            return True
        except BaseException as e:
            print("Error on_data %s" % str(e))
        return True
          
    def on_error(self, status):
        if status == 420:
            # Returning False on_data method in case rate limit occurs.
            return False
        print(status)


class TweetAnalyzer():
    """
    Functionality for analyzing and categorizing content from tweets.
    """

    def clean_tweet(self, tweet):
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

    def analyze_sentiment(self, tweet):
        analysis = TextBlob(self.clean_tweet(tweet))
        
        if analysis.sentiment.polarity > 0:
            return 1
        elif analysis.sentiment.polarity == 0:
            return 0
        else:
            return -1

    def tweets_to_data_frame(self, tweets):
        df = pd.DataFrame(data=[tweet.text for tweet in tweets], columns=['tweets'])

        df['id'] = np.array([tweet.id for tweet in tweets])
        df['len'] = np.array([len(tweet.text) for tweet in tweets])
        df['date'] = np.array([tweet.created_at for tweet in tweets])
        df['source'] = np.array([tweet.source for tweet in tweets])
        df['likes'] = np.array([tweet.favorite_count for tweet in tweets])
        df['retweets'] = np.array([tweet.retweet_count for tweet in tweets])

        return df

 
if __name__ == '__main__':

    twitter_client = TwitterClient()
    tweet_analyzer = TweetAnalyzer()

    api = twitter_client.get_twitter_client_api()

    tweets = api.user_timeline(screen_name="realDonaldTrump", count=200)

    df = tweet_analyzer.tweets_to_data_frame(tweets)
    df['sentiment'] = np.array([tweet_analyzer.analyze_sentiment(tweet) for tweet in df['tweets']])

    print(df.head(10))

