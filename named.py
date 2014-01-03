import nltk
import os

book= raw_input("enter")
for sentences in nltk.sent_tokenize(book):
        print sentences
        for chunked in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sentences))):
            if hasattr(chunked,'node'):
                if chunked.node == 'PERSON':
                    name=' '.join(leaf[0] for leaf in chunked.leaves())
                    print "name : %s"  % name
