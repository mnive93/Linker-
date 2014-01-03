
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
from sklearn.pipeline import Pipeline
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
#    print result
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

 g = Goose({'target_language':'en'})
 article = g.extract(url=url)
 if article.title:
     print article.title
     if(len(article.cleaned_text)>200):
        print article.cleaned_text
        return article.cleaned_text
     else:
         return "The site has no content it is invalid"



def training_set():
   # create our dict of training data
    texts = {}
    texts['Music'] = 'music.txt'
    texts['Programming']='coding.txt'
    texts['Politics']='politics.txt'
    texts['Startups']='startups.txt'
    texts['Miscellaneous']='misc.txt'
    texts['Food']='food.txt'
    texts['Business']='business.txt'
    texts['Books']='books.txt'
     #holds a dict of features for training our classifier
    train_set = []
   #train_test2 = []
  # loop through each item, grab the text, tokenize it and create a training feature with it
    for sense, file in texts.iteritems():
       print "training %s " % sense
       f=open(file, 'r')
       text = f.read()
       # print text
       features = extract_words(text)
       train_set = train_set + [(get_feature(word), sense) for word in features]
    classifier = NaiveBayesClassifier.train(train_set)
   # pipeline = Pipeline([('tfidf', TfidfTransformer()),
     #                  ('chi2', SelectKBest(chi2, k=1000)),
    #                     ('nb', LogisticRegression())])
    #classifier=SklearnClassifier(OneVsRestClassifier(LogisticRegression())).train(train_set)

    with open('my_dataset.pkl','wb') as fid:
       cPickle.dump(classifier,fid)
#   # uncomment out this line to see the most informative words the classifier will use
#classifier.show_most_informative_features(50)



   # uncomment out this line to see how well our accuracy is using some hand curated tweets
 #  run_classifier_tests(classifier)


#training_set()
print "Performance of Naive Bayes Classifier"
url = raw_input("Enter the url to be parsed")
line = parse_url(url)
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
print labels.samples()
print "music    : %s" %labels.prob('Music')
print "coding   : %s" %labels.prob('Programming')
print "politics  :%s " %labels.prob('Politics')
print "startups    :%s " %labels.prob('Startups')
print "misc        : %s"%labels.prob('Miscellaneous')
print "food         :%s" %labels.prob('Food')
print "business     :%s"%labels.prob('Business')
print "books        :%s"%labels.prob('Books')

print decision
#label_names = ['food', 'books','technology','music','travel','sports','company','coding','hobbies','careers','religion','education']
#predictions = [label_names[pred] for pred in classifier.predict(new_samples)]
result = "%s - %s" % (decision,line)
choice = raw_input("is the decision correct?")
if choice == "no":
 filename = raw_input("enter the name of the file to which u want to enter")
 f = open(filename,"a")
 f.write("\n %s" %str(line))
 training_set()