from flask import Flask, request, jsonify
from prediction import predict


app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict_api():
data = request.json
result = predict(data)
return jsonify(result)




if __name__ == "__main__":
app.run(debug=True)
