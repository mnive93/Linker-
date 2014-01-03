import sklearn.datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import numpy as np
data_train = sklearn.datasets.load_files('/home/nivedita/nltkprac', description=None, categories=None, load_content=True, shuffle=True, random_state=0)
data_test = sklearn.datasets.load_files('/home/nivedita/test', description=None, categories=None, load_content=True, shuffle=True, random_state=0)
categories = data_train.target_names
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,stop_words='english')
X_train = vectorizer.fit_transform(data_train.data)
y_train = data_train.target
X_test = vectorizer.transform(data_test.data)

ch2 = SelectKBest(chi2, k=1000)
X_train = ch2.fit_transform(X_train, y_train)
X_test = ch2.transform(X_test)
clf = MultinomialNB(alpha=.01)
clf.fit(X_train, y_train)
predicted = clf.predict(X_test)
for x in predicted:
    print data_train.target_names[x]


