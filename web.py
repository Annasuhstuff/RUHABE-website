from flask import Flask, render_template, request
from joblib import load

trained = load("pretrained_models/pipe.joblib")

def predict(text):

    prediction = trained.predict([text])
    result = prediction[0]

    return f'Группа хейта в тексте: {result}'


app = Flask(__name__)


@app.route("/index")
def ruhabe_index():
    return render_template("index.html.j2")

@app.route("/about")
def ruhabe_about():
    return render_template("about.html.j2")

@app.route("/download")
def ruhabe_download():
    return render_template("download.html.j2")

@app.route("/model_pred", methods =["GET", "POST"])
def ruhabe_predict():
    if len(request.args) > 1:
        text = request.args['text']
        model = request.args['selected_model']
        predicted_text = predict(text)
        return render_template("model_predict_result.html.j2", predicted_text=predicted_text)

    return render_template("model_predict.html.j2")

@app.route("/stats")
def ruhabe_stats():
    return render_template("stats.html.j2")

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8080)


# http://localhost:8080/model_pred
