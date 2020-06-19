import sys
from collections import defaultdict
import re
import gensim
import pandas as pd
import numpy as np
import scipy.stats
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords


stopWords = set(stopwords.words('english'))

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
                if len(line.split()[1:]) != 100: continue
                word = line.split()[0]
                wordList.append(word.split('#')[0])
                wordVectors[word] = np.zeros(len(line.split()) - 1, dtype=float)
                for index, vecVal in enumerate(line.split()[1:]):
                    wordVectors[word][index] = float(vecVal)

    sys.stderr.write("Vectors read from: " + filename + " \n")
    sumvec = np.zeros(100, dtype=float)
    i = 0
    for wv in wordVectors.values():
        try:
            sumvec += wv
        except:
            i += 1
            continue
    mean_vector = sumvec / (len(wordVectors)-i)
    #mean_vector = np.mean(np.array([wv for wv in wordVectors.values()]), axis=0)
    for sense in wordVectors:
        senseList[sense.split('#')[0]].append(sense)

    return wordVectors, wordList, mean_vector, senseList


def read_cont_lexicon(file):
    w_dict = defaultdict()
    s_dict = defaultdict(list)

    with open(file, 'r') as lexicon:
        lines = lexicon.readlines()
        for line in lines:
            split = line.split('\t')
            w1, w2, s1, s2, score = split[1].lower(), split[3].lower(), split[5].lower(), split[6].lower(), float(
                split[7])
            w_pair = (w1, w2)
            w_dict[w_pair] = float(score)
            s_dict[w_pair].extend([s1, s2])
    return w_dict, s_dict


def contexts(word, sen, wordVecs, wordList, mean_vector, senseList, window=5):
    prior_s = sen.split('<b>')[0]
    after_s = sen.split('</b>')[1]
    prior_s = re.sub('[^a-zA-Z]+\s', '', prior_s)
    after_s = re.sub('[^a-zA-Z]+\s', '', after_s)
    prior = [w for w in prior_s.split()[-window:] if w not in stopWords]
    after = [w for w in after_s.split()[:window] if w not in stopWords]
    prior_cont = context_vec(prior, wordVecs, wordList, mean_vector, senseList)
    after_cont = context_vec(after, wordVecs, wordList, mean_vector, senseList)
    contVector = (prior_cont + after_cont) # / (2 * window)
    vecs0 = []
    if word not in set(wordList):
        vecs0.append(mean_vector)
    else:
        for sense in senseList[word]:
            vecs0.append(wordVecs[sense])
    # contextual_sim(contVector, vecs0)
    return contVector, vecs0


def contexts_avg(word, sen, wordVecs, wordList, mean_vector, senseList, window=5):
    prior_s = sen.split('<b>')[0]
    after_s = sen.split('</b>')[1]
    prior_s = re.sub('[^a-zA-Z]+\s', '', prior_s)
    after_s = re.sub('[^a-zA-Z]+\s', '', after_s)
    prior = [w for w in prior_s.split()[-window:] if w not in stopWords]
    after = [w for w in after_s.split()[:window] if w not in stopWords]
    prior_cont = context_vec(prior, wordVecs, wordList, mean_vector, senseList)
    after_cont = context_vec(after, wordVecs, wordList, mean_vector, senseList)
    contVector = (prior_cont + after_cont) / (2 * window)
    vecs0 = []
    if word not in set(wordList):
        vecs0.append(mean_vector)
    else:
        for sense in senseList[word]:
            vecs0.append(wordVecs[sense])
    # contextual_sim(contVector, vecs0)
    return contVector, vecs0


def context_vec(sen, wordVecs, wordList, mean_vector, senseList):
    missing = 0
    contVector = []

    for w2 in sen:
        vecs1 = []
        if w2 not in set(wordList):
            vecs1.append(mean_vector)
            missing += 1
        else:
            for sense in senseList[w2]:
                vecs1.append(wordVecs[sense])
        contVector.append(np.mean(vecs1, axis=0))

    return np.sum(contVector, axis=0)


def contextual_sim(contVector1, vecs1, contVector2, vecs2):
    avg_value = 0
    for i in range(len(vecs1)):
        for j in range(len(vecs2)):
            # max_value += w1_max[i] * w2_max[j] * w1_w2_max
            avg_value += (1 - cosine(contVector1, vecs1[i])) * \
                         (1 - cosine(contVector2, vecs2[j])) * \
                         (1 - cosine(vecs1[i], vecs2[j]))
    s1 = np.argmax([1 - cosine(contVector1, b) for b in vecs1])
    s2 = np.argmax([1 - cosine(contVector2, b) for b in vecs2])
    max_value = 1 - cosine(vecs1[s1], vecs2[s2])

    return avg_value/(len(vecs1) * len(vecs2)), max_value


def word_vecs(wordVecs,dim):
    wordList = []
    senseList = defaultdict(list)
    for word, vec in wordVecs.items():
        wordList.append(word.split('#')[0])
        senseList[word.split('#')[0]].append(word)
    mean_vector = np.mean(np.array([wv for wv in wordVecs.values()if len(wv) == dim]), axis=0)

    return wordList, mean_vector, senseList


def main(wordVec, dim):
    wordList, mean_vector, senseList = word_vecs(wordVec, dim)
    f = "ratings.txt"
    w_dict, s_dict = read_cont_lexicon(f)
    G_avg_dict = []
    G_max_dict = []

    G_avg_dict_avg = []
    G_max_dict_avg = []

    for (w1, w2) in w_dict.keys():
        s1 = s_dict[(w1, w2)][0]
        s2 = s_dict[(w1, w2)][1]
        GcV1, Gvecs1 = contexts(w1, s1, wordVec, wordList, mean_vector, senseList)
        GcV2, Gvecs2 = contexts(w2, s2, wordVec, wordList, mean_vector, senseList)
        GcV_avg, Gvecs_avg = contexts_avg(w1, s1, wordVec, wordList, mean_vector, senseList)
        GcV2_avg, Gvecs2_avg = contexts_avg(w2, s2, wordVec, wordList, mean_vector, senseList)

        avg, mx = contextual_sim(GcV1, Gvecs1, GcV2, Gvecs2)
        G_avg_dict.append(avg)
        G_max_dict.append(mx)
        avg, mx = contextual_sim(GcV_avg, Gvecs_avg, GcV2_avg, Gvecs2_avg)
        G_avg_dict_avg.append(avg)
        G_max_dict_avg.append(mx)

    print('context_sum:\n')

    print("NS-sv:\n")
    spear_avg, p_avg = scipy.stats.spearmanr(list(w_dict.values()), G_avg_dict)
    print("AvgSimC_r / AvgSimC_p = " + str(spear_avg) + ' / ' + str(p_avg) + "\n")
    spear_max, p_max = scipy.stats.spearmanr(list(w_dict.values()), G_max_dict)
    print("MaxSimC_r / MaxSimC_p = " + str(spear_max) + ' / ' + str(p_max) + "\n")

    print('context_avg:\n')

    spear_avg, p_avg = scipy.stats.spearmanr(list(w_dict.values()), G_avg_dict_avg)
    print("AvgSimC_r / AvgSimC_p = " + str(spear_avg) + ' / ' + str(p_avg) + "\n")
    spear_max, p_max = scipy.stats.spearmanr(list(w_dict.values()), G_max_dict_avg)
    print("MaxSimC_r / MaxSimC_p = " + str(spear_max) + ' / ' + str(p_max) + "\n")



if __name__ == '__main__':
    # wV, mv = read_word_vecs("word_vec/glove.twitter.27B.100d.txt")
    # wordVectors1, wordList1, mean_vector1, senseList1 = read_word_vecs1("re_glove.twitter.27B.100d.txt")
    wordVec, wordList, mean_vector, senseList = read_word_vecs("re_sum_glove.twitter.27B.100d.txt")
    f = "ratings.txt"
    w_dict, s_dict = read_cont_lexicon(f)
    G_avg_dict = []
    G_max_dict = []

    G_avg_dict_avg = []
    G_max_dict_avg = []

    for (w1, w2) in w_dict.keys():
        s1 = s_dict[(w1, w2)][0]
        s2 = s_dict[(w1, w2)][1]
        GcV1, Gvecs1 = contexts(w1, s1, wordVec, wordList, mean_vector, senseList)
        GcV2, Gvecs2 = contexts(w2, s2, wordVec, wordList, mean_vector, senseList)
        GcV_avg, Gvecs_avg = contexts_avg(w1, s1, wordVec, wordList, mean_vector, senseList)
        GcV2_avg, Gvecs2_avg = contexts_avg(w2, s2, wordVec, wordList, mean_vector, senseList)

        avg, mx = contextual_sim(GcV1, Gvecs1, GcV2, Gvecs2)
        G_avg_dict.append(avg)
        G_max_dict.append(mx)
        avg, mx = contextual_sim(GcV_avg, Gvecs_avg, GcV2_avg, Gvecs2_avg)
        G_avg_dict_avg.append(avg)
        G_max_dict_avg.append(mx)

    print('context_sum:\n')

    print("NS-sv:\n")
    spear_avg, p_avg = scipy.stats.spearmanr(list(w_dict.values()), G_avg_dict)
    print("AvgSimC_r / AvgSimC_p = " + str(spear_avg) + ' / ' + str(p_avg) + "\n")
    spear_max, p_max = scipy.stats.spearmanr(list(w_dict.values()), G_max_dict)
    print("MaxSimC_r / MaxSimC_p = " + str(spear_max) + ' / ' + str(p_max) + "\n")

    print('context_avg:\n')

    spear_avg, p_avg = scipy.stats.spearmanr(list(w_dict.values()), G_avg_dict_avg)
    print("AvgSimC_r / AvgSimC_p = " + str(spear_avg) + ' / ' + str(p_avg) + "\n")
    spear_max, p_max = scipy.stats.spearmanr(list(w_dict.values()), G_max_dict_avg)
    print("MaxSimC_r / MaxSimC_p = " + str(spear_max) + ' / ' + str(p_max) + "\n")
