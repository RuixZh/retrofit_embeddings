import argparse
import math
import time
import gensim
import numpy as np
import re
import sys
import evaluation
from collections import defaultdict


isNumber = re.compile(r'\d+.*')
#f_sl = open('SLiteration.txt', 'w')
#f_sv = open('SViteration.txt', 'w')


def norm_word(word):
    if isNumber.search(word):
        return '---num---'
    elif re.sub(r'\W+', '', word) == '':
        return '---punc---'
    else:
        return word


def read_word_vecs(filename):
    wordVectors = {}
    newWordVecs = {}
    #open the format of w2v
    if filename.endswith('.bin.gz'):
        model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)
        wordVectors = dict(zip(model.index2word, model.vectors))
        newWordVecs = dict(zip(model.index2word, model.vectors))
        dim = model.vector_size
    else:
        with open(filename, 'r') as fileObject:
            for line in fileObject:
                line = line.strip()
                word = line.split()[0]
                #wordVectors is the input vectors, while newWordVecs is output vectors
                #here we set they have the same initialization vectors
                wordVectors[word] = np.zeros(len(line.split()) - 1, dtype=np.float64)
                newWordVecs[word] = np.zeros(len(line.split()) - 1, dtype=np.float64)
                for index, vecVal in enumerate(line.split()[1:]):
                    wordVectors[word][index] = float(vecVal)
                    newWordVecs[word][index] = float(vecVal)
                ''' normalize weight vector '''
            dim = len(line.split()) - 1

    sys.stderr.write("Vectors read from: " + filename + " \n")
    return wordVectors, newWordVecs, dim


def read_lexicon(filename):
    lexicon = {}
    no = defaultdict(lambda:0)
    with open(filename, 'r') as f:
        lines = f.readlines()
        for j, line in enumerate(lines):
            words = line.strip().split()
            lexicon[norm_word(words[0]) + '#' + str(no[words[0]])] = [norm_word(word) for word in words[1:]]
            no[words[0]] += 1
    sys.stderr.write("Lexicons read from: " + filename + " \n")
    return lexicon


def read_evaluation_data(filename):
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


# eval_data = read_evaluation_data('eval_data/EN-SimLex-999.txt')
# eval_verb = read_evaluation_data("eval_data/SimVerb-3500.txt")


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# correct solution:
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x)
    return e_x / e_x.sum(axis=0)


def retrofit(wordVec, newWordVecs, dim, syn_lexicon, ant_lexicon, numIters=5, starting_alpha=1.0, lamda=0.5):
    wvVocab = set(wordVec.keys())
    dim = int(dim)
    loop_synVocab = set()
    loop_antVocab = set()
    for w in syn_lexicon.keys():
        if w.split('#')[0] in wvVocab:
            loop_synVocab.add(w)
    for w in ant_lexicon.keys():
        if w.split('#')[0] in wvVocab:
            loop_antVocab.add(w)

    word_count = len(loop_synVocab)

    for it in range(numIters):
        count = 0
        for word in loop_synVocab:
            count += 1
            wordNeighbours = set(syn_lexicon[word])
            numNeighbours = len(wordNeighbours)
            # no neighbours, pass - use data estimate
            if numNeighbours == 0:
                continue
            # Calculate learning rate
            alpha = starting_alpha * (1 - float(it / numIters)) / numNeighbours
            if alpha < starting_alpha * 0.00001:
                alpha = starting_alpha * 0.00001

            #negative sampling
            classifiers = [(word.split('#')[0], 1)]
            if word in loop_antVocab:
                classifiers += [(target, 0) for target in set(ant_lexicon[word]).intersection(wvVocab)]

            neu1 = np.zeros(dim)
            for nbWord in wordNeighbours:
                #estimate the vector which word not exist in the pretrained vectors
                if nbWord not in wvVocab:
                    wordVec[nbWord] = np.random.uniform(low=-0.5 / dim, high=0.5 / dim, size=(dim))
                    # wordVec[nbWord] = np.random.normal(scale=0.5/dim,size=(dim))
                    #adjust the random vector by synonyms already appeared
                    wordVec[nbWord] = lamda * wordVec[nbWord] + (1 - lamda) * wordVec[word.split('#')[0]]
                    newWordVecs[nbWord] = wordVec[nbWord]

                neu1e = np.zeros(dim)
                for target, label in classifiers:
                    z = np.dot(wordVec[nbWord], newWordVecs[target])
                    p = sigmoid(z) #relu??
                    g = alpha * (label - p)
                    neu1e += g * newWordVecs[target]
                    newWordVecs[target] += g * wordVec[nbWord]  # Update syn1

                # accumulate the contributions from neighbours
                neu1 += wordVec[nbWord]
                #update synonyms
                wordVec[nbWord] += neu1e
            #update the word vector which with the definite sense
            wordVec[word] = (wordVec[word.split('#')[0]] + neu1) / (1 + numNeighbours)

            sys.stdout.write("\rIteration: No. %d Alpha: %f Progress: %d of %d (%.2f%%)" %
                                 ((it+1), alpha, count, word_count, float(count / word_count) * 100))
            sys.stdout.flush()
        '''
        if it == 0 or (it+1) % 10 == 0:
            print('num of iteration:\n', str(it+1))
            print('simlex_999:\n')
            sl_max, sl_avg = evaluation.main(wordVec, eval_data, dim)
            f_sl.write(str(it+1) + '\t' + str(round(sl_max + 0.00001, 3)) + '\t' + 'NS-svMax' + '\n')
            f_sl.write(str(it+1) + '\t' + str(round(sl_avg + 0.00001, 3)) + '\t' + 'NS-svMean' + '\n')
            print('simver_3500:\n')
            sv_max, sv_avg = evaluation.main(wordVec, eval_verb, dim)
            f_sv.write(str(it+1) + '\t' + str(round(sv_max + 0.00001, 3)) + '\t' + 'NS-svMax' + '\n')
            f_sv.write(str(it+1) + '\t' + str(round(sv_avg + 0.00001, 3)) + '\t' + 'NS-svMean' + '\n')
        '''
        # sl_max, sl_avg = evaluation.main(wordVec_sv, eval_data, dim)
        # if sl_max > 0.7644 or sl_avg >0.7644:
        #     return wordVec
    return wordVec


def retrofit_sum(wordVec, newWordVecs, dim, syn_lexicon, ant_lexicon,  numIters=5, starting_alpha=1.0, lamda=0.5):

    wvVocab = set(wordVec.keys())
    dim = int(dim)
    loop_synVocab = set()
    loop_antVocab = set()
    for w in syn_lexicon.keys():
        if w.split('#')[0] in wvVocab:
            loop_synVocab.add(w)
    for w in ant_lexicon.keys():
        if w.split('#')[0] in wvVocab:
            loop_antVocab.add(w)

    word_count = len(loop_synVocab)

    for it in range(numIters):
        count = 0
        for word in loop_synVocab:
            count += 1
            wordNeighbours = set(syn_lexicon[word])
            numNeighbours = len(wordNeighbours)
            # no neighbours, pass - use data estimate
            if numNeighbours == 0:
                continue
            # Calculate learning rate
            alpha = starting_alpha * (1 - float(it / numIters)) / numNeighbours
            if alpha < starting_alpha * 0.00001:
                alpha = starting_alpha * 0.00001

            #negative sampling
            classifiers = [(word.split('#')[0], 1)]
            if word in loop_antVocab:
                classifiers += [(target, 0) for target in set(ant_lexicon[word]).intersection(wvVocab)]

            neu1 = np.zeros(dim)
            #neu = np.zeros(dim)
            for nbWord in wordNeighbours:
                #estimate the vector which word not exist in the pretrained vectors
                if nbWord not in wvVocab:
                    wordVec[nbWord] = np.random.uniform(low=-0.5 / dim, high=0.5 / dim, size=(dim))
                    #adjust the random vector by synonyms already appeared
                    wordVec[nbWord] = lamda * wordVec[nbWord] + (1 - lamda) * wordVec[word.split('#')[0]]
                    newWordVecs[nbWord] = wordVec[nbWord]
                # accumulate the contributions from neighbours
                neu1 += wordVec[nbWord]
            neu1e = np.zeros(dim)
            for target, label in classifiers:
                z = np.dot(neu1, newWordVecs[target])
                p = sigmoid(z)
                g = alpha * (label - p)
                neu1e += g * newWordVecs[target]
                newWordVecs[target] += g * neu1  # Update syn1

            for nbWord in wordNeighbours:
                #update synonyms
                wordVec[nbWord] += neu1e
            #update the word vector which with the definite sense
            wordVec[word] = (wordVec[word.split('#')[0]] + neu1) / (1 + numNeighbours)

            sys.stdout.write("\rIteration: No. %d Alpha: %f Progress: %d of %d (%.2f%%)" %
                                 ((it+1), alpha, count, word_count, float(count / word_count) * 100))
            sys.stdout.flush()

        '''
        if it == 0 or (it+1) % 10 == 0:
            print('num of iteration:\n', str(it+1))
            print('simlex_999:\n')
            sl_max, sl_avg = evaluation.main(wordVec, eval_data, dim)
            f_sl.write(str(it+1) + '\t' + str(round(sl_max + 0.00001, 3)) + '\t' + 'NS-sv-sumMax' + '\n')
            f_sl.write(str(it+1) + '\t' + str(round(sl_avg + 0.00001, 3)) + '\t' + 'NS-sv-sumMean' + '\n')
            print('simver_3500:\n')
            sv_max, sv_avg = evaluation.main(wordVec, eval_verb, dim)
            f_sv.write(str(it+1) + '\t' + str(round(sv_max + 0.00001, 3)) + '\t' + 'NS-sv-sumMax' + '\n')
            f_sv.write(str(it+1) + '\t' + str(round(sv_avg + 0.00001, 3)) + '\t' + 'NS-sv-sumMean' + '\n')
        '''


    return wordVec


def print_word_vecs(wordVec, outFileName):
    sys.stderr.write('\nWriting down the vectors in ' + outFileName + '\n')
    outFile = open(outFileName, 'w')
    for word, values in wordVec.items():
        outFile.write(word + ' ')
        for val in wordVec[word].tolist():
            outFile.write('%.5f' % (val) + ' ')
        outFile.write('\n')
    outFile.close()
    sys.stderr.write('Finish writing!' + '\n')


if __name__ == '__main__':

    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default=None, help="Input word vectors")
    parser.add_argument("-s", "--syn_lexicon", type=str, default=None, help="Synonyms lexicon file name")
    parser.add_argument("-a", "--ant_lexicon", type=str, default=None, help="Antonyms lexicon file name")
    parser.add_argument("-o", "--output", type=str, help="Output word vecs")
    parser.add_argument("-n", "--numiter", type=int, default=4, help="NO. iterations")
    parser.add_argument("-lr", "--alpha", type=float, default=1.0, help="Learning rate")
    parser.add_argument("-w", "--lamda", type=float, default=0.5, help="Weight for unknown neighbor vectors")
    args = parser.parse_args()
    '''
    syn_lexicon = read_lexicon('lexicon/wiki-syns.txt')
    ant_lexicon = read_lexicon('lexicon/wiki-ants.txt')

    # input_files = ['word_vec/glove.6B.300d.txt']
    # input_files = ['word_vec/ft-crawl-300d-2M.vec']
    wordVecs, newWordVecs, dim = read_word_vecs('word_vec/glove.6B.300d.txt')
    t0 = time.time()
    wordVec_sv = retrofit(wordVecs, newWordVecs, dim, syn_lexicon, ant_lexicon, numIters=10)
    t1 = time.time()
    print('\nCompleted retrofitting. Retrofitting took', (t1 - t0), 'seconds')
    '''
    for input_file in input_files:
        # iterations = [100]

        outFileName = 're_' + input_file
        # outFileName_sum = 're_sum_' + input_file
        # for it in iterations:

        while 1:

            print('NS-sv:\n')
            wordVecs, newWordVecs, dim = read_word_vecs(input_file)
            #t0 = time.time()
            wordVec_sv = retrofit(wordVecs, newWordVecs, dim, syn_lexicon, ant_lexicon, numIters=100)
            #t1 = time.time()
            #print('\nCompleted retrofitting. Retrofitting took', (t1 - t0), 'seconds')
            sl_max, sl_avg = evaluation.main(wordVec_sv, eval_data, dim)
            if sl_max > 0.7644 or sl_avg > 0.7644:
                print_word_vecs(wordVec_sv, outFileName)
                break
            #t0 = time.time()
            #print('NS-sv-sum:\n')
            #wordVecs, newWordVecs, dim = read_word_vecs(input_file)
            #wordVec_sum = retrofit_sum(wordVecs, newWordVecs, dim, syn_lexicon, ant_lexicon, numIters=it)
            #t1 = time.time()
            #print('\nCompleted retrofitting. Retrofitting took', (t1 - t0) / 60, 'minutes')

        #f_sl.close()
        #f_sv.close()

'''

#python retrofit.py -i 'word_vec/ft-crawl-300d-2M.vec' -s 'lexicon/wiki-syns.txt'
# -a  -o 'ft_crawl_50it.txt' -n 50
