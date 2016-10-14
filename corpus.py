from nltk.corpus import movie_reviews
from nltk.corpus import CategorizedPlaintextCorpusReader
from nltk.corpus import twitter_samples


def load_movies():
    return movie_reviews, 'pos', 'neg'


def load_tweets():
    tweets = CategorizedPlaintextCorpusReader('data', r'(positive|negative|neutral)/.*', cat_pattern=r'(\w+)\/.*')
    return tweets, 'positive', 'negative'

def load_tweets_nltk():
    return
