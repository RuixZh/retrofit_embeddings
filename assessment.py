import sys
from collections import defaultdict

import gensim
import pandas as pd
import numpy as np
import scipy.stats
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import seaborn as sns


#用weight计算出的spearman correlation 0.37
#用sentiWordNet计算出的 0.39

def read_lexicon(f1):
    dict1 = defaultdict(float)
    #dict2 = defaultdict(float)

    with open(f1, 'r') as lexicon:
        lines = lexicon.readlines()
        for line in lines:
            split = line.split()
            w1, w2, score = split[0].lower(), split[1].lower(), float(split[3])
            #w1, w2, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, score = line.strip().split(",")
            w_pair = (w1, w2)
            dict1[w_pair] = float(score)

    sys.stderr.write("Dict: " + f1 + "is ready! \n")
    return dict1#, dict2


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


def score_cosinus(wordVecs, mean_vector, lexicon):
    score_cos = {}
    missing = 0
    scores = []
    #mean_vector = np.mean(wordVecs.values(), axis=0)
    for (w1, w2) in lexicon.keys():

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

        score_cos[(w1, w2)] = (1 - cosine(vecs0, vecs1))
    return score_cos


def Spearman(lexicon, score):
    X, Y = list(), list()
    for w_pair in lexicon.keys():
        if w_pair in score.keys():
            X.append(lexicon[w_pair])
            Y.append(score[w_pair])
    X, Y = np.asarray(X, dtype=float), np.asarray(Y, dtype=float)
    if len(X) != len(Y):
        raise ValueError("X and Y are not of equal size!")
    r, p_value = scipy.stats.spearmanr(X, Y)
    print("r = " + str(r) + " p = " + str(p_value) + "\n")
    #plot_scatter_spearman(X, Y, filename)


def plot_scatter_spearman(X, Y, filename):
    sub_data = pd.DataFrame(
        {
            'lexicon': X,
            'wordVector': Y,
        })
    #fig = plt.figure(figsize=(10, 5))
    sns.regplot(x="lexicon", y="wordVector", fit_reg=True, data=sub_data)
    plt.xlabel('Score EN-SimLex-999')
    plt.ylabel('Score W2V retrofitted')
    plt.title(
        'Scatterplot for the Spearman correlation between ws353 and w2v retrofitted')
    plt.savefig(filename)


if __name__ == '__main__':
    wordVecs, mean_vector = read_word_vecs("/home/ruizhang/Desktop/PycharmProjects/refinewv/retrofit/word_vec/glove.twitter.27B.100d.txt")
    #wv1 = read_word_vecs("w2v.txt.syn1")
    #for w in wordVecs.keys():
        #wordVecs[w] += wv1[w]
    #wordVecs = read_word_vecs("/home/ruizhang/Desktop/demo/word2vecpy-master/cbow_hs_100.txt")
    l1= read_lexicon("retrofit/eval_data/EN-SimLex-999.txt")#, "EN-SimLex-999.txt")
    ns1 = score_cosinus(wordVecs, mean_vector, l1)
    #ns2 = score_cosinus(wordVecs, mean_vector, l2)
    #ns = score_cosinus(w2v, lexicon)

    Spearman(l1, ns1)
   # Spearman(l2, ns2)
    #Spearman(lexicon, ns, "eval/oldEN-WS353.png")
