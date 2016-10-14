import collections
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from sklearn.svm import LinearSVC
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.metrics import precision, recall
from sklearn.metrics import matthews_corrcoef


def trainNB(trainset):
    return NaiveBayesClassifier.train(trainset)


def trainSVM(trainset):
    classifier = SklearnClassifier(LinearSVC())
    return classifier.train(trainset)


def get_metrics(classifier, testfeats):
    results = collections.defaultdict(float)
    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)
    posrefset = []
    postestset = []
    negrefset = []
    negtestset = []
    for i, (feats, label) in enumerate(testfeats):
        observed = classifier.classify(feats)
        posrefset.append(1 if label == 'positive' else -1)
        negrefset.append(1 if label == 'negative' else -1)
        postestset.append(1 if observed == 'positive' and label == 'positive' else -1)
        negtestset.append(1 if observed == 'negative' and label == 'negative' else -1)
        refsets[label].add(i)
        testsets[observed].add(i)

    results['accuracy'] = nltk.classify.util.accuracy(classifier, testfeats)
    results['precision pos'] = precision(refsets['positive'], testsets['positive'])
    results['recall pos'] = recall(refsets['positive'], testsets['positive'])
    results['mmc pos'] = matthews_corrcoef(posrefset, postestset)
    results['precision neg'] = precision(refsets['negative'], testsets['negative'])
    results['recall neg'] = recall(refsets['negative'], testsets['negative'])
    results['mmc neg'] = matthews_corrcoef(negrefset, negtestset)
    print 'accuracy:', results['accuracy']
    print 'pos precision:', results['precision pos']
    print 'pos recall:', results['recall pos']
    print 'pos mcc:', results['mmc pos']
    print 'neg precision:', results['precision neg']
    print 'neg recall:', results['recall neg']
    print 'neg mcc:', results['mmc neg']
    # classifier.show_most_informative_features()
    print '\n\n'
    return results
