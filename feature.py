def extract_words(text):
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
