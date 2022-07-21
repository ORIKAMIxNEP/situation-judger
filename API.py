from flask import Flask, render_template, request, jsonify
import datetime
import requests
from ImageCaption import ImageCaption
from SituationJudgment import SituationJudgment

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024


@app.route("/", methods=["GET", "POST"])
def API(imagePath):
    if request.method == "POST":
        fs = request.files["image"]
        imagePath = "../images/" + datetime.datetime.now().strftime("%Y%m%d%H%M%S%f") + ".jpg"
        fs.save(imagePath)
    return jsonify(SituationJudgment(ImageCaption(imagePath)))


@app.route("/web_test", methods=["GET", "POST"])
def WebTest():
    if request.method == "GET":
        caption = None
    elif request.method == "POST":
        fs = request.files["image"]
        imagePath = "../images/" + datetime.datetime.now().strftime("%Y%m%d%H%M%S%f") + ".jpg"
        fs.save(imagePath)
        caption = API(imagePath).json()["caption"]
    return render_template("test.html", caption=caption)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=20221, debug=True)
