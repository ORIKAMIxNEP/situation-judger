import os
from operator import itemgetter


def OrganizeFiles():
    # 画像ファイルの枚数を10に制限する関数
    if len(os.listdir("../images/")) > 10:
        files = []
        for file in os.listdir("../images/"):
            files.append([file, os.path.getctime("../images/"+file)])
        files.sort(key=itemgetter(1), reverse=True)
        for i, file in enumerate(files):
            if i > 9:
                os.remove("../images/"+file[0])
