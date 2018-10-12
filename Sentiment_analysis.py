import matplotlib.pyplot as plt
import nltk.classify.util
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import string
import tweepy
from random import shuffle
from sklearn.externals import joblib
from IPython.display import display
import pickle
from credentials import *
from nltk import NaiveBayesClassifier
from nltk import classify
from nltk.corpus import stopwords
from nltk.corpus import twitter_samples
from nltk.stem import PorterStemmer  # this module removes words that mean the same but have different tenses
from nltk.tokenize import TweetTokenizer
from tweepy import Stream, OAuthHandler
from tweepy.streaming import StreamListener

emoticons_happy = {':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}', ':^)', ':-D', ':D', '8-D',
                   '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D', '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P',
                   ':-P', ':P', 'X-P', 'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)', '<3'}

emoticons_sad = {':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<', ':-[', ':-<', '=\\', '=/',
                 '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c', ':c', ':{', '>:\\', ';('}

emoticons = emoticons_happy.union(emoticons_sad)
stopwords_english = stopwords.words('english')


# Tweepy
def twitter_setup():
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)
    api = tweepy.API(auth)
    return api


def extraction(name):
    extractor = twitter_setup()
    tweets = extractor.user_timeline(screen_name=name, count=200)
    data = pd.DataFrame(data=[tweet.text for tweet in tweets], columns=['Tweets'])
    data['len'] = np.array([len(tweet.text) for tweet in tweets])
    data['ID'] = np.array([tweet.id for tweet in tweets])
    data['Date'] = np.array([tweet.created_at for tweet in tweets])
    data['Likes'] = np.array([tweet.favorite_count for tweet in tweets])
    data['RTs'] = np.array([tweet.retweet_count for tweet in tweets])
    data['sentiment'] = np.array(["" for tweet in tweets])
    # display(data.head(10))
    data.to_csv('Data_collected.csv', sep=',')


def clean_tweet(tweet):
    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)

    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)

    # remove hyperlinks
    tweet = re.sub(r"https?://.*[\r\n]*", '', tweet)

    # remove hashtags
    tweet = re.sub(r'#', '', tweet)

    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tokens = tokenizer.tokenize(tweet)
    stemmer = PorterStemmer()
    cleaned_tweets = []
    for w in tokens:
        if w not in stopwords_english and w not in emoticons and w not in string.punctuation:
            stem_word = stemmer.stem(w)
            cleaned_tweets.append(stem_word)
    return cleaned_tweets


def get_words_and_clean(s):
    s = s.lower()
    s = re.sub("[^a-zA-Z]", " ", s)
    stops = set(stopwords.words("english"))
    final = ""
    for a in s.split():
        if not a in stops:
            final += str(a + " ")
    return final


def bag_of_words(tweet):
    words = clean_tweet(tweet)
    words_dictionary = dict([word, True] for word in words)
    return words_dictionary


def ml():
    pos_tweets = twitter_samples.strings('positive_tweets.json')
    neg_tweets = twitter_samples.strings('negative_tweets.json')
    pos_tweets_set = []
    for tweet in pos_tweets:
        pos_tweets_set.append((bag_of_words(tweet), 'pos'))

    neg_tweets_set = []
    for tweet in neg_tweets:
        neg_tweets_set.append((bag_of_words(tweet), 'neg'))

    test_set = pos_tweets_set[:1000] + neg_tweets_set[:1000]
    train_set = pos_tweets_set[1000:] + neg_tweets_set[1000:]
    classifier = NaiveBayesClassifier.train(train_set)
    accuracy = classify.accuracy(classifier, test_set)
    # print(accuracy)
    joblib.dump(classifier, 'ml_model.pkl')


def stats(no_of_pos, no_of_neg):
    df = pd.DataFrame.from_csv('Data_collected.csv', sep=',')
    total_likes,rts = 0,0
    for row in range(df.shape[0]):
        total_likes+=df.loc[row, 'Likes']
        rts +=df.loc[row,'RTs']

    print('Total tweets considered: ',no_of_neg+no_of_pos)
    print('No of positive tweets: ', no_of_pos)
    print('No of negative tweets: ', no_of_neg)
    print('Total likes: ',total_likes)
    print('Total rates',rts)
    if (no_of_pos > no_of_neg):
        print('The overall attitude recently has been positive.')
    else:
        print('The overall attitude recently has been negative.')

    print('Check generated csv file for individual predictions ')

def sentiment_analysis():
    ml()
    no_of_pos, no_of_neg = 0, 0
    df = pd.DataFrame.from_csv('Data_collected.csv', sep=',')
    clf = joblib.load('ml_model.pkl')
    # print(clf.show_most_informative_features(5))
    for row in range(df.shape[0]):
        prediction = clf.classify(bag_of_words(str(df['Tweets'][row])))
        df.loc[row, 'sentiment'] = prediction
        if prediction == 'pos':
            no_of_pos += 1
        else:
            no_of_neg += 1
    df.to_csv('Data_found.csv', sep=',')
    stats(no_of_pos, no_of_neg)


name = input("Enter twitter id: ")
extraction(name)
sentiment_analysis()

