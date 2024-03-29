from flask import Flask, render_template, request
import os
import requests
# from joblib import load

# trained = load("pretrained_models/pipe.joblib")

# def predict(text):
#     prediction = trained.predict([text])
#     result = prediction[0]
#     return f'Группа хейта в тексте: {result}'


app = Flask(__name__)

REST_URL = os.environ.get('REST_URL', 'localhost:8000')


@app.route("/index")
def ruhabe_index():
    return render_template("index.html.j2")

@app.route("/about")
def ruhabe_about():
    return render_template("about.html.j2")

@app.route("/team")
def ruhabe_team():
    return render_template("team.html.j2")

@app.route("/download")
def ruhabe_download():
    return render_template("download.html.j2")

@app.route("/simple_pred", methods = ["GET", "POST"])
def ruhabe_simple_pred():
    print(request.args)
    if len(request.args) >= 1:
        text = request.args['text']
        if len(text) == 0:
            return render_template("simple_pred.html.j2", empty_text=True)
        response = requests.get(f'{REST_URL}/predict', params={'text': text}).json()
        return render_template("simple_pred_result.html.j2", predicted_text=response['group'], input_text=text)
    return render_template("simple_pred.html.j2")

@app.route("/model_pred", methods = ["GET", "POST"])
def ruhabe_predict():
    print(request.args)
    if len(request.args) > 1:
        text = request.args['text']
        model = request.args['selected_model']
        if len(text) == 0:
            return render_template("model_predict.html.j2", empty_text=True)
        response = requests.get(f'{REST_URL}/predict', params={'text': text, 'model': model}).json()
        return render_template("model_predict_result.html.j2", predicted_text=response['group'], input_text=text, input_model=model)
    return render_template("model_predict.html.j2")

@app.route("/stats")
def ruhabe_stats():
    return render_template("stats.html.j2")

if __name__ == '__main__':
    print('Hello!')
    app.run(debug=True, host="0.0.0.0", port=5000)


# http://localhost:8080/model_pred
