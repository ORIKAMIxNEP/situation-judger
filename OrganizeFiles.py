from operator import itemgetter
import os


def OrganizeFiles():
    if len(os.listdir("../images/")) > 10:
        files = []
        for file in os.listdir("../images/"):
            files.append([file, os.path.getctime(file)])
        files.sort(key=itemgetter(1), reverse=True)
        for i, file in enumerate(files):
            if i > 9:
                os.remove("../images/"+file[i])
