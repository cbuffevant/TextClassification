import corpus as c
import features as f
import sentiment as s
import collections
import matplotlib.pyplot as plt
import numpy as np
plt.rcdefaults()


def test_movie_tweets(feature_extractor, classify):
    training_corpus, training_positive_cat, training_negative_cat = c.load_movies()
    testing_corpus, testing_positive_cat, testing_negative_cat = c.load_tweets()

    category_mapper = collections.defaultdict()
    category_mapper[training_positive_cat] = testing_positive_cat
    category_mapper[training_negative_cat] = testing_negative_cat

    trainfeats = f.get_features(
        training_corpus,
        training_negative_cat,
        training_positive_cat,
        feature_extractor,
        category_mapper)

    testfeats = f.get_features(
        testing_corpus,
        testing_negative_cat,
        testing_positive_cat,
        feature_extractor)

    print 'train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats))
    classifier = classify(trainfeats)
    return s.get_metrics(classifier, testfeats)


def test_tweets(feature_extractor, classify):
    corpus, positive_cat, negative_cat = c.load_tweets()

    trainfeats, testfeats = f.split_dataset(corpus, negative_cat, positive_cat, feature_extractor, 3.0/4.0)
    print 'train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats))

    classifier = classify(trainfeats)
    return s.get_metrics(classifier, testfeats)

test_cases = []
accuracy = []
precision = ([], [])
recall = ([], [])
mmc = ([], [])

test_cases.append('NB - movie_revies - BoW')
print test_cases[len(test_cases)-1]
results = test_movie_tweets(f.word_feats, s.trainNB)
accuracy.append(results['accuracy'])
precision[0].append(results['precision pos'])
precision[1].append(results['precision neg'])
recall[0].append(results['recall pos'])
recall[1].append(results['recall neg'])
mmc[0].append(results['mmc pos'])
mmc[1].append(results['mmc neg'])

test_cases.append('NB - movie_revies - BoW stopwords')
print test_cases[len(test_cases)-1]
results = test_movie_tweets(f.non_stop_word_feats, s.trainNB)
accuracy.append(results['accuracy'])
precision[0].append(results['precision pos'])
precision[1].append(results['precision neg'])
recall[0].append(results['recall pos'])
recall[1].append(results['recall neg'])
mmc[0].append(results['mmc pos'])
mmc[1].append(results['mmc neg'])

test_cases.append('NB - movie_revies - bigrams')
print test_cases[len(test_cases)-1]
results = test_movie_tweets(f.bigram_word_feats, s.trainNB)
accuracy.append(results['accuracy'])
precision[0].append(results['precision pos'])
precision[1].append(results['precision neg'])
recall[0].append(results['recall pos'])
recall[1].append(results['recall neg'])
mmc[0].append(results['mmc pos'])
mmc[1].append(results['mmc neg'])

test_cases.append('NB - tweets - BoW')
print test_cases[len(test_cases)-1]
results = test_tweets(f.word_feats, s.trainNB)
accuracy.append(results['accuracy'])
precision[0].append(results['precision pos'])
precision[1].append(results['precision neg'])
recall[0].append(results['recall pos'])
recall[1].append(results['recall neg'])
mmc[0].append(results['mmc pos'])
mmc[1].append(results['mmc neg'])

test_cases.append('NB - tweets - BoW stopwords')
print test_cases[len(test_cases)-1]
results = test_tweets(f.non_stop_word_feats, s.trainNB)
accuracy.append(results['accuracy'])
precision[0].append(results['precision pos'])
precision[1].append(results['precision neg'])
recall[0].append(results['recall pos'])
recall[1].append(results['recall neg'])
mmc[0].append(results['mmc pos'])
mmc[1].append(results['mmc neg'])

test_cases.append('NB - tweets - bigrams')
print test_cases[len(test_cases)-1]
results = test_tweets(f.bigram_word_feats, s.trainNB)
accuracy.append(results['accuracy'])
precision[0].append(results['precision pos'])
precision[1].append(results['precision neg'])
recall[0].append(results['recall pos'])
recall[1].append(results['recall neg'])
mmc[0].append(results['mmc pos'])
mmc[1].append(results['mmc neg'])

test_cases.append('SVM - tweets - bigrams')
print test_cases[len(test_cases)-1]
results = test_tweets(f.bigram_word_feats, s.trainSVM)
accuracy.append(results['accuracy'])
precision[0].append(results['precision pos'])
precision[1].append(results['precision neg'])
recall[0].append(results['recall pos'])
recall[1].append(results['recall neg'])
mmc[0].append(results['mmc pos'])
mmc[1].append(results['mmc neg'])


def add_plot(index, data, label, cases):
    y_pos = np.arange(len(cases))
    height = 0.35
    fig = plt.figure(index)
    ax = fig.add_subplot(111)
    if isinstance(data, tuple):
        rects1 = ax.barh(y_pos, data[0], height, color='b')
        rects2 = ax.barh(y_pos + height, data[1], height, color='r')
        ax.legend((rects1[0], rects2[0]), ('Pos', 'Neg'))
    else:
        ax.barh(y_pos, data, height)
    ax.set_xlabel(label)
    ax.set_yticks(y_pos + height)
    ax.set_yticklabels(cases)

add_plot(1, precision, 'Precision', test_cases)
add_plot(2, recall, 'Recall', test_cases)
add_plot(3, mmc, 'MMC', test_cases)
add_plot(4, accuracy, 'Accuracy', test_cases)

plt.show()
