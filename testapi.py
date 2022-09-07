from flask import Flask

app = Flask(__name__)


@app.route("/", methods=["GET"])
def testapi():
    return "test"


if __name__ == "__main__":
    app.run(port=8082, debug=True)
