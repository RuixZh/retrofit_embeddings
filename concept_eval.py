import sys
from collections import defaultdict
import re
import gensim
import pandas as pd
import numpy as np
import scipy.stats
from scipy.spatial.distance import cosine
from sklearn import metrics


def read_word_vecs(filename):
    wordVectors = {}

    if filename.endswith('.bin'):
        model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)
        for w in model.wv.vocab:
            wordVectors[w] = model[w]
    else:
        with open(filename) as fileObject:
            lines = fileObject.readlines()
        for l in lines:
            line = l.strip()
            split = line.split()
            word = split[0]
            wordVectors[word] = np.array([float(split[i]) for i in range(1, len(split))], dtype='float64')
    sum = np.zeros(100, dtype=float)
    i = 0
    for wv in wordVectors.values():
        try:
            sum += wv
        except:
            i += 1
            continue
    mean_vector = sum / (len(wordVectors) - i)
    #mean_vector = np.mean(np.array([wv for wv in wordVectors.values()]), axis=0)
    sys.stderr.write("Vectors read from: " + filename + " \n")
    return wordVectors, mean_vector


def read_word_vecs1(filename):
    wordVectors = {}
    wordList = []
    senseList = defaultdict(list)

    if filename.endswith('.bin'):
        model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)
        for w in model.wv.vocab:
            wordVectors[w] = model[w]
    else:
        with open(filename, 'r') as fileObject:
            for l in fileObject:
                line = l.strip()
                if len(line.split()[1:]) != 100: continue
                word = line.split()[0]
                wordList.append(word.split('#')[0])
                wordVectors[word] = np.zeros(len(line.split()) - 1, dtype=float)
                for index, vecVal in enumerate(line.split()[1:]):
                    wordVectors[word][index] = float(vecVal)

    sys.stderr.write("Vectors read from: " + filename + " \n")
    s = []
    i = 0
    for wv in wordVectors.values():
        try:
            s.append(wv)
        except:
            i += 1
            continue
    mean_vector = sum(s) / (len(wordVectors)-i)
    #mean_vector = np.mean(np.array([wv for wv in wordVectors.values()]), axis=0)
    for sense in wordVectors:
        senseList[sense.split('#')[0]].append(sense)

    return wordVectors, wordList, mean_vector, senseList


def read_lexicon(file):
    w_dict = defaultdict()
    group =[]
    with open(file, 'r') as lexicon:
        lines = lexicon.readlines()
        for line in lines:
            split = line.split('\t')
            w1, w2, f, g = split[0].lower(), split[1].lower(), split[2].lower(), int(split[3])
            w_pair = (w1, w2)
            w_dict[w_pair] = f
            group.append(g)
    return w_dict, group


def score_cosine(w1, w2, w3, wordVecs, mean_vector):

    missing = 0

    if w1 not in set(wordVecs.keys()):
        vecs0 = mean_vector
        missing += 1
    else:
        vecs0 = wordVecs[w1]

    if w2 not in set(wordVecs.keys()):
        vecs1 = mean_vector
        missing += 1
    else:
        vecs1 = wordVecs[w2]

    if w3 not in set(wordVecs.keys()):
        vecs2 = mean_vector
        missing += 1
    else:
        vecs2 = wordVecs[w3]

    if (1 - cosine(vecs0, vecs2)) > (1 - cosine(vecs1, vecs2)):
        return 1
    else:
        return 0


def score_cosinus1(wordVecs, wordList, mean_vector, senseList):
    missing = 0
    vecs0 = []
    vecs1 = []
    vecs2 = []
    if w1 not in set(wordList):
        vecs0.append(mean_vector)
        missing += 1
    else:
        for sense in senseList[w1]:
            vecs0.append(wordVecs[sense])

    if w2 not in set(wordList):
        vecs1.append(mean_vector)
        missing += 1
    else:
        for sense in senseList[w2]:
            vecs1.append(wordVecs[sense])

    if w3 not in set(wordList):
        vecs2.append(mean_vector)
        missing += 1
    else:
        for sense in senseList[w3]:
            vecs2.append(wordVecs[sense])

    max_value13 = np.max([1 - cosine(a, b) for a in vecs0 for b in vecs2])
    avg_value13 = np.mean([1 - cosine(a, b) for a in vecs0 for b in vecs2])
    max_value23 = np.max([1 - cosine(a, b) for a in vecs1 for b in vecs2])
    avg_value23 = np.mean([1 - cosine(a, b) for a in vecs1 for b in vecs2])
    if max_value13 > max_value23:
        mg = 1
    else:
        mg = 0
    if avg_value13 > avg_value23:
        ag = 1
    else:
        ag = 0
    return mg, ag



if __name__ == '__main__':
    wV, mv = read_word_vecs("word_vec/glove.twitter.27B.100d.txt")
    wordVectors, wordList, mean_vector, senseList = read_word_vecs1("testvec/Genvec27B.txt")
    wordVectors1, wordList1, mean_vector1, senseList1 = read_word_vecs1("revec/resg/twitter.txt")
    f = "eval_data/concept.txt"
    w_dict, group = read_lexicon(f)
    glo = []
    G_avg = []
    G_max = []
    R_avg = []
    R_max = []
    for (w1, w2), w3 in w_dict.items():
        glo.append(score_cosine(w1, w2, w3, wV, mv))
        mg, ag = score_cosinus1(wordVectors, wordList, mean_vector, senseList)
        G_max.append(mg)
        G_avg.append(ag)
        mg, ag = score_cosinus1(wordVectors1, wordList1, mean_vector1, senseList1)
        R_max.append(mg)
        R_avg.append(ag)
    print("glove:\naccuracy: %.2f\tprecision: %.2f\trecall: %.2f\tF1: %.2f\n"
          %(metrics.accuracy_score(group, glo), metrics.precision_score(group, glo),
            metrics.recall_score(group, glo), metrics.f1_score(group, glo)))
    print("GenSense_Max:\naccuracy: %.2f\tprecision: %.2f\trecall: %.2f\tF1: %.2f\n"
          % (metrics.accuracy_score(group, G_max), metrics.precision_score(group, G_max),
             metrics.recall_score(group, G_max), metrics.f1_score(group, G_max)))
    print("GenSense_Avg:\naccuracy: %.2f\tprecision: %.2f\trecall: %.2f\tF1: %.2f\n"
          % (metrics.accuracy_score(group, G_avg), metrics.precision_score(group, G_avg),
             metrics.recall_score(group, G_avg), metrics.f1_score(group, G_avg)))
    print("NV-sv_Max:\naccuracy: %.2f\tprecision: %.2f\trecall: %.2f\tF1: %.2f\n"
          % (metrics.accuracy_score(group, R_max), metrics.precision_score(group, R_max),
             metrics.recall_score(group, R_max), metrics.f1_score(group, R_max)))
    print("NV-sv_Avg:\naccuracy: %.2f\tprecision: %.2f\trecall: %.2f\tF1: %.2f\n"
          % (metrics.accuracy_score(group, R_avg), metrics.precision_score(group, R_avg),
             metrics.recall_score(group, R_avg), metrics.f1_score(group, R_avg)))
