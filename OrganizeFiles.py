import os
from operator import itemgetter


def OrganizeFiles():
    if len(os.listdir("static/images")) > 10:
        files = []
        for file in os.listdir("static/images"):
            files.append([file, os.path.getctime("static/images/"+file)])
        files.sort(key=itemgetter(1), reverse=True)
        for i, file in enumerate(files):
            if i > 9:
                os.remove("static/images/"+file[0])
