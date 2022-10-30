import datetime

import requests
from flask import Flask, jsonify, render_template, request

from ImageCaption import ImageCaption
from JudgeSituation import JudgeSituation
from OrganizeFiles import OrganizeFiles

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


@app.route("/", methods=["GET", "POST"])
def API():
    if request.method == "POST":
        fs = request.files["image"]
        imagePath = "../images/" + datetime.datetime.now().strftime("%Y%m%d%H%M%S%f") + ".jpg"
        fs.save(imagePath)
        OrganizeFiles()
    return jsonify(JudgeSituation(ImageCaption()))


@app.route("/web_test", methods=["GET", "POST"])
def WebTest():
    if request.method == "GET":
        SituationData = {"caption": None, "nouns": None, "situation": None}
    elif request.method == "POST":
        fs = request.files["image"]
        imagePath = "../images/" + datetime.datetime.now().strftime("%Y%m%d%H%M%S%f") + ".jpg"
        fs.save(imagePath)
        SituationData = requests.get(
            "http://172.31.50.221:8081/").json()
    return render_template("test.html", caption=SituationData["caption"], nouns=SituationData["nouns"], situation=SituationData["situation"])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8081, debug=True)
