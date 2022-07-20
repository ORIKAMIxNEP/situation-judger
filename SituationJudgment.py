import nltk
import gensim
import collections

nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
file = "~/lexvec.enwiki+newscrawl.300d.W.pos.vectors.gz"
wm_en = gensim.models.KeyedVectors.load_word2vec_format(file)


def SituationJudgment(string):
    print("\n\n画像キャプション生成："+string)

    words = nltk.word_tokenize(string)
    nouns = []
    for word in nltk.pos_tag(words):
        if word[1] == "NN" or word[1] == "NNS" or word[1] == "NNP" or word[1] == "NNS":
            for n in nouns:
                if word[0] == n:
                    break
            else:
                nouns.append(word[0])
    print("\n抽出した名詞："+str(nouns))

    similarityList = [{"食事中": {"food": 0, "dish": 0, "meal": 0, "snack": 0}}, {"観光中(建物)": {"building": 0, "structure": 0}}, {
        "観光中(風景)": {"landscape": 0, "scene": 0}}, {"動物と触れ合い中": {"animal": 0}}, {"自撮り中、又は他人を撮影中": {"human": 0, "person": 0, "people": 0}}]
    vectors = {noun: {"type": "", "max": -1} for noun in nouns}

    for noun in nouns:
        for i in range(len(similarityList)):
            for label, similarity in similarityList[i].items():
                for key in similarity:
                    similarityList[i][label][key] = wm_en.similarity(key, noun)
        print(noun+"："+str(similarityList))
        maxValue = -1
        for i in range(len(similarityList)):
            for label, similarity in similarityList[i].items():
                if max(similarity.values()) > maxValue:
                    vectors[noun]["type"] = label
                    vectors[noun]["max"] = maxValue = max(similarity.values())
    print("\n"+str(vectors))

    counts = {}
    sums = {}
    for similarity in similarityList:
        for label in similarity:
            counts.update([(label, 0)])
            sums.update([(label, 0)])
    for vector in vectors.values():
        for count in counts:
            if vector["type"] == count:
                counts[count] += 1
                sums[count] += vector["max"]
    print("\n"+str(counts))
    print("\n"+str(sums))

    print(max(counts.values()))
    print(collections.Counter(counts.values()))
    print(collections.Counter(counts.values())[max(counts.values())])
    if collections.Counter(counts.values())[max(counts.values())] == 1:
        CaptionData = dict(caption=max(counts, key=counts.get))
    else:
        CaptionData = dict(caption=max(sums, key=sums.get))

    print(CaptionData["caption"])

    return CaptionData
