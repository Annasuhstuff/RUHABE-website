from flask import Flask, render_template

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

@app.route("/model_pred")
def ruhabe_predict():
    return render_template("model_predict.html.j2")

@app.route("/stats")
def ruhabe_stats():
    return render_template("stats.html.j2")

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
