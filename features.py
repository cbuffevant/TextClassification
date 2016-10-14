import itertools
import collections
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.corpus import stopwords


stopset = set(stopwords.words('english'))


def word_feats(words):
    return dict([(w, True) for w in words])


def non_stop_word_feats(words):
    return dict([(w, True) for w in words if w not in stopset])


def bigram_word_feats(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    return dict([(ngram, True) for ngram in itertools.chain(words, bigrams)])


def build_features(corpus, cat, ids, feats_ext, mapper):
    return [(feats_ext(corpus.words(fileids=[f])), mapper[cat]) for f in ids]


def get_features(corpus, neg_cat, pos_cat, feature_extractor, mapper=None):
    negids = corpus.fileids(neg_cat)
    posids = corpus.fileids(pos_cat)
    # default mapper
    if mapper is None:
        mapper = collections.defaultdict()
        mapper[pos_cat] = pos_cat
        mapper[neg_cat] = neg_cat

    negfeats = build_features(corpus, neg_cat, negids, feature_extractor, mapper)
    posfeats = build_features(corpus, pos_cat, posids, feature_extractor, mapper)
    return negfeats + posfeats


def split_dataset(corpus, neg_cat, pos_cat, feature_extractor, cutoff,  mapper=None):
    negids = corpus.fileids(neg_cat)
    posids = corpus.fileids(pos_cat)

    # default mapper
    if mapper is None:
        mapper = collections.defaultdict()
        mapper[pos_cat] = pos_cat
        mapper[neg_cat] = neg_cat

    negfeats = build_features(corpus, neg_cat, negids, feature_extractor, mapper)
    posfeats = build_features(corpus, pos_cat, posids, feature_extractor, mapper)

    negcutoff = int(len(negfeats)*cutoff)
    poscutoff = int(len(posfeats)*cutoff)

    trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
    testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]
    return trainfeats, testfeats
