from thesaurus import Word


if __name__ == '__main__':

    #lexicon = {}
    fo = open('synonyms.txt','w')
    fou = open('antonyms.txt','w')
    for line in open('word.txt', 'r'):
        word = line.split()
        w = Word(word[0])
        for arr in w.synonyms('all', relevance=[3]):
            fo.write(word[0]+' '+' '.join([syn for syn in arr if ' ' not in syn])+'\n')
        for arr in w.antonyms('all', relevance=[3]):
            fou.write(word[0]+' '+' '.join([ant for ant in arr if ' ' not in ant])+'\n')
    fo.close()
    fou.close()
