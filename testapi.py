from flask import Flask, jsonify

app = Flask(__name__)


@app.route("/", methods=["GET"])
def testapi():
    data = {"data": "test"}
    return jsonify(data)


if __name__ == "__main__":
    app.run(port=8082, debug=True)
