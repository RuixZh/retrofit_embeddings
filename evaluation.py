import sys
from collections import defaultdict

import gensim
import pandas as pd
import numpy as np
import scipy.stats
from scipy.spatial.distance import cosine



def read_lexicon(filename):
    dict = defaultdict(float)

    with open(filename, 'r') as lexicon:
        lines = lexicon.readlines()
        for line in lines:
            split = line.split()
            w1, w2, score = split[0].lower(), split[1].lower(), float(split[3])
            w_pair = (w1, w2)
            dict[w_pair] = float(score)

    sys.stderr.write("Dict: " + filename + " is ready! \n")
    return dict


def read_word_vecs(filename):
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
    for sense in wordVectors.keys():
        senseList[sense.split('#')[0]].append(sense)

    return wordVectors, wordList, mean_vector, senseList


def score_cosinus(wordVecs, wordList, mean_vector, senseList, lexicon):
    missing = 0
    cos_max = []
    cos_avg = []
    scores = []
    for (w1, w2) in lexicon.keys():
        vecs0 = []
        vecs1 = []
        scores.append(lexicon[(w1, w2)])
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

        max_value = np.max([1 - cosine(a, b) for a in vecs0 for b in vecs1])
        avg_value = np.mean([1 - cosine(a, b) for a in vecs0 for b in vecs1])
        cos_max.append(max_value)
        cos_avg.append(avg_value)
    spear_max, p_max = scipy.stats.spearmanr(scores, cos_max)
    #pearson_max = scipy.stats.pearsonr(scores, cos_max)
    spear_avg, p_avg = scipy.stats.spearmanr(scores, cos_avg)
    #pearson_avg = scipy.stats.pearsonr(scores, cos_avg)
    print("r_max/r_avg = " + str(spear_max) + ' / ' + str(spear_avg) + "\n")
    print(" p_max/p_avg = " + str(p_max) + ' / ' + str(p_avg) + "\n")
    return spear_max, spear_avg

    #plot_scatter_spearman(scores, cos_max, "Max")
    #plot_scatter_spearman(scores, cos_avg, "Average")


def plot_scatter_spearman(X, Y, annotation):
    sub_data = pd.DataFrame(
        {
            'lexicon': X,
            'wordVector': Y,        })
    #fig = plt.figure(figsize=(10, 5))
    ax = sns.regplot(x="lexicon", y="wordVector", fit_reg=True, data=sub_data, line_kws={'label': annotation})
    ax.legend()
    plt.xlabel('Score Lexicon')
    plt.ylabel('Score Retrofitting Vectors')
    plt.title(
        'Scatterplot between EN-SimLex-999 and re-glove_42B')
    plt.savefig('glove42B_SL.png')


def word_vecs(wordVecs, dim):
    wordList = []
    senseList = defaultdict(list)
    mean_vector = np.zeros(shape=(dim))
    for word, vec in wordVecs.items():
        wordList.append(word.split('#')[0])
        senseList[word.split('#')[0]].append(word)
        mean_vector += vec
    mean_vector /= len(wordVecs)

    return wordList, mean_vector, senseList


def main(wordVecs, eval_data, dim):
    #lexicon= read_lexicon("eval_data/EN-SimLex-999.txt")
    #lexicon = read_lexicon("eval_data/SimVerb-3500.txt")
    #wordVecs, wordList, mean_vector, senseList = read_word_vecs(path)
    wordList, mean_vector, senseList = word_vecs(wordVecs, dim)
    spear_max, spear_mavg = score_cosinus(wordVecs, wordList, mean_vector, senseList, eval_data)
    print('Finish Evaluation!')
    return spear_max, spear_mavg

if __name__ == '__main__':

    lexicon= read_lexicon("eval_data/EN-SimLex-999.txt")
    #lexicon = read_lexicon("eval_data/SimVerb-3500.txt")
    wordVecs, wordList, mean_vector, senseList = read_word_vecs("ft_crawl_100it.txt")
    score_cosinus(wordVecs, wordList, mean_vector, senseList, lexicon)
    print('Finish Evaluation!')

