for sense, file in topics.iteritems():
       print "training %s " % sense
       f=open(file, 'r')
       text = f.read()
       # print text
       features = extract_words(text)
       train_set = train_set + [(get_feature(word), sense) for word in features]

    classifier=SklearnClassifier(OneVsRestClassifier(LogisticRegression())).train(train_set)

    with open('my_dataset.pkl','wb') as fid:
       cPickle.dump(classifier,fid)

tokens = bag_of_words(extract_words(line))
 f1= open('my_dataset.pkl')
 classifier=cPickle.load(f1)
 f1.close()
 decision = classifier.classify(tokens)
 labels  = classifier.prob_classify(tokens)
