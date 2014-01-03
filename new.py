import os
import string
from django.conf import settings
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import WordPunctTokenizer
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy
from nltk.classify import *
from nltk.probability import DictionaryProbDist
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.linear_model import LogisticRegression
from nltk.corpus import wordnet
#from sklearn.pipeline import Pipeline
import sys
import urllib2
from nltk.corpus import brown
import random
import numpy as np
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.datasets import fetch_20newsgroups
from sklearn.multiclass import OneVsRestClassifier
#from sklearn.feature_extraction.text import CountVectorizer
import cPickle

''' Function that returns the posted_at parameter to sort the targetted_post list '''
def rearrange(post):
    return post.posted_at

def extract_words(text):

    '''
   here we are extracting features to use in our classifier. We want to pull all the words in our input
   porterstem them and grab the most significant bigrams to add to the mix as well.
   '''

    stemmer = PorterStemmer()

    tokenizer = WordPunctTokenizer()
    tokens = tokenizer.tokenize(text)
    bigram_finder = BigramCollocationFinder.from_words(tokens)
    bigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 500)

    for bigram_tuple in bigrams:
        x = "%s %s" % bigram_tuple
        tokens.append(x)

    result =  [stemmer.stem(x.lower()) for x in tokens if x not in stopwords.words('english') and len(x) > 1]
    return result

def get_feature(word):
    return dict([(word, True)])

def bag_of_words(words):
    return dict([(word, True) for word in words])

def create_training_dict(text, sense):
    ''' returns a dict ready for a classifier's test method '''
    tokens = extract_words(text)
    return [(bag_of_words(tokens), sense)]

line = raw_input("enter the data to be classified")
newline = line.translate(None,string.punctuation)
print newline
tokens = bag_of_words(extract_words(line))
with open('my_dataset.pkl', 'r') as f1:
  classifier=cPickle.load(f1)
decision = classifier.classify(tokens)
labels  = classifier.prob_classify(tokens)
print "misc     :%s" % labels.prob('Miscellaneous')
print "coding   : %s" %labels.prob('Coding')
print "startups    :%s " %labels.prob('Startups')
print "food    :%s " %labels.prob('Food')
print "music    :%s " %labels.prob('Music')
print "technology   :%s " %labels.prob('Technology')
print decision
