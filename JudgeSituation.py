import collections

from ExtractNouns import ExtractNouns
from MeasureSimilarity import MeasureSimilarity


def JudgeSituation(caption):
    SituationData = dict(caption=caption)

    nouns = ExtractNouns(caption)
    SituationData.update(dict(nouns=str(nouns)))

    similarityList = [
        {"食事中": {"food": 0, "dish": 0, "meal": 0, "snack": 0}},
        {"観光中(建物)": {"building": 0, "structure": 0}},
        {"観光中(風景)": {"landscape": 0, "scene": 0, "spot": 0}},
        {"動物に癒され中": {"animal": 0}},
        {"人と交流中": {"human": 0, "person": 0, "people": 0}},
    ]

    vectors = MeasureSimilarity(nouns, similarityList)

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
    print("\n" + str(counts))
    print("\n" + str(sums))

    if collections.Counter(counts.values())[max(counts.values())] == 1:
        SituationData.update(dict(situation=max(counts, key=counts.get)))
    else:
        SituationData.update(dict(situation=max(sums, key=sums.get)))
    print("\n状況説明：" + SituationData["situation"] + "\n")
    return SituationData
