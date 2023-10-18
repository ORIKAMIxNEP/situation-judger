import nltk

nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")


def ExtractNouns(string):
    words = nltk.word_tokenize(string)
    nouns = []
    for word in nltk.pos_tag(words):
        if word[1] == "NN" or word[1] == "NNS" or word[1] == "NNP" or word[1] == "NNS":
            for n in nouns:
                if word[0] == n:
                    break
            else:
                nouns.append(word[0])
    print("\n抽出した名詞：" + str(nouns) + "\n")
    return nouns
