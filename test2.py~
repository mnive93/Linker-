from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import WordPunctTokenizer
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy
from nltk.classify import *
from nltk.probability import DictionaryProbDist
from nltk.corpus import wordnet
#from sklearn.svm.sparse import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import sys
import urllib2
from nltk.corpus import brown
import random
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
#from sklearn.feature_extraction.text import CountVectorizer
import cPickle
from goose import Goose

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
   # print result
    return result

def get_feature(word):
    return dict([(word, True)])


def bag_of_words(words):
    return dict([(word, True) for word in words])


def create_training_dict(text, sense):
    ''' returns a dict ready for a classifier's test method '''
    tokens = extract_words(text)
    return [(bag_of_words(tokens), sense)]




def parse_url(url):
 print url
 g = Goose()
 article = g.extract(url=url)
 print article.title
 print article.cleaned_text
 if article.title:
    if(len(article.cleaned_text)>200):
        return article.cleaned_text




def training_set():
   # create our dict of training data
    topics = {}
    topics['Finance'] = 'finance.txt'
    topics['Food'] = 'food.txt'
    topics['Literature'] = 'literature.txt'
    topics['Education'] = 'education.txt'
    topics['Family and Relationships'] = 'family.txt'
    topics['Computer Science'] = 'computerscience.txt'
    topics['Philosophy and Religion'] = 'philosophy.txt'
    topics['Health and Nutrition'] = 'health.txt'
    topics['Medicine'] = 'medicine.txt'
    topics['Science'] = 'science.txt'
    topics['Sports and Games'] = 'sports.txt'
    topics['Fashion'] = 'fashion.txt'
    topics['Movies'] = 'movies.txt'
    topics['Music'] = 'music.txt'
    topics['Startups'] = 'startups.txt'
    topics['Nature'] = 'nature.txt'
    topics['Technology and Innovations'] = 'technology.txt'
    topics['Politics'] = 'politics.txt'
    topics['Travel and Tourism'] ='travel.txt'
    topics['Art'] = 'art.txt'
    topics['Photography'] = 'photography.txt'
    topics['Law and Order'] = 'law.txt'
    topics['Gadgets and Electronics'] = 'gadget.txt'
    topics['Beauty'] = 'beauty.txt'
    topics['Dance'] = 'dance.txt'
    topics['Economics'] = 'economic.txt'
    topics['Social Networking'] = 'social network.txt'
    topics['Pets and Animals'] ='pets.txt'
    topics['Automobiles'] = 'cars&bikes.txt'





     #holds a dict of features for training our classifier
    train_set = []
   #train_test2 = []
  # loop through each item, grab the text, tokenize it and create a training feature with it
    for sense, file in topics.iteritems():
       print "training %s " % sense
       f=open(file, 'r')
       text = f.read()
       # print text
       features = extract_words(text)
       train_set = train_set + [(get_feature(word), sense) for word in features]
#    classifier = NaiveBayesClassifier.train(train_set)
   # pipeline = Pipeline([('tfidf', TfidfTransformer()),
     #                  ('chi2', SelectKBest(chi2, k=1000)),
    #                     ('nb', LogisticRegression())])
    classifier=SklearnClassifier(OneVsRestClassifier(LogisticRegression())).train(train_set)

    with open('my_dataset.pkl','wb') as fid:
       cPickle.dump(classifier,fid)
#   # uncomment out this line to see the most informative words the classifier will use
#classifier.show_most_informative_features(50)



   # uncomment out this line to see how well our accuracy is using some hand curated tweets
 #  run_classifier_tests(classifier)


#training_set()
url = raw_input("Enter the URL to be parsed")
line = parse_url(url)
if(line):
 tokens = bag_of_words(extract_words(line))
 f1= open('my_dataset.pkl')
 classifier=cPickle.load(f1)
 f1.close()
 #print tokens
 decision = classifier.classify(tokens)
 labels  = classifier.prob_classify(tokens)
  #batch = classifier.batch_classify(tokens)
  #for b in batch:
   # print b
 #print labels.samples()
 '''
 print "Economics    : %s" %labels.prob('Economics')
 print "Finance %s" %labels.prob("Finance")
 print "Family and Relationships   : %s" %labels.prob('Family and Relationships')
 print "Food :%s " %labels.prob('Food')
 print "Literature    :%s " %labels.prob('Literature')
 print "Science       : %s"%labels.prob('Science')
 print "Computer Science        :%s" %labels.prob('Computer Science')
 print "Education      :%s"%labels.prob('Education')
 print "Health and Nutrition   :%s"%labels.prob('Health and Nutrition')
 print "Movies       :%s"%labels.prob('Movies')
 print "Medicine       :%s"%labels.prob('Medicine')
 print "Music       :%s"%labels.prob('Music')
 print "Art      :%s"%labels.prob('Art')
 print "Photography       :%s"%labels.prob('Photography')
 print "Politics         :%s"%labels.prob('Politics')
 print "Startups    :%s"%labels.prob('Startups')
 print "Nature     :%s" %labels.prob('Nature')
 print "Travel and Tourism %s" %labels.prob("Travel and Tourism")
 print "Technology and Innovations %s" %labels.prob("Technology and Innovations")
 print "Law and Order %s" %labels.prob("Law and Order")
 print "Sports and Games %s" %labels.prob("Sports and Games")
 print "Gadgets and Electronics %s" %labels.prob("Gadgets and Electronics")
 print "Fashion %s" %labels.prob("Fashion")
 print "Dance %s" %labels.prob("Dance")
 '''
 print "******* CATEGORY OF THE WEB DOCUMENT *******"
 print decision
 print"*********************************************"
else:
    print "The site has no content or article to classify"

