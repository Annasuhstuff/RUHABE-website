# from eve import Eve

# app = Eve()

# if __name__ == '__main__':
#     app.run(debug=True, host="0.0.0.0")

from flask import Flask, jsonify, request
from joblib import load

trained = load("pretrained_models/pipe.joblib")

app = Flask(__name__)

def predict(text):
    prediction = trained.predict([text])
    return prediction[0]

@app.route("/predict")
def ruhabe_index():
    text = request.args.get("text", "DEFAULT")
    return jsonify({'group': predict(text)})

@app.route("/hello")
def hello():
    return "<p>hello!</p>"

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8000)
