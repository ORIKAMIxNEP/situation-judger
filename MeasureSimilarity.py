import gensim

file = "lexvec.enwiki+newscrawl.300d.W.pos.vectors.gz"
wm_en = gensim.models.KeyedVectors.load_word2vec_format(file)


def MeasureSimilarity(nouns, similarityList):
    vectors = {noun: {"type": "", "max": -1} for noun in nouns}
    for noun in nouns:
        for i in range(len(similarityList)):
            for label, similarity in similarityList[i].items():
                for key in similarity:
                    similarityList[i][label][key] = wm_en.similarity(key, noun)
        print(noun + "：" + str(similarityList))
        maxValue = -1
        for i in range(len(similarityList)):
            for label, similarity in similarityList[i].items():
                if max(similarity.values()) > maxValue:
                    vectors[noun]["type"] = label
                    vectors[noun]["max"] = maxValue = max(similarity.values())
    print("\n" + str(vectors))
    return vectors
