from flask import Flask, request
import datetime

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def API():
    if request.method == "POST":
        fs = request.files["image"]
        imagePath = "../images/" + datetime.datetime.now().strftime("%Y%m%d%H%M%S%f") + ".jpg"
        fs.save(imagePath)
        print("ok")
    return "test"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8081, debug=True)
