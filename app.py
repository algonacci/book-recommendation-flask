from flask import Flask, request, render_template
import module as md

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/title", methods=["GET", "POST"])
def title():
    if request.method == "POST":
        judul = request.form["judul"]
        recommendation = md.rec_pvdbow(Title=judul)
        return render_template("title.html", recommendation=recommendation, judul=judul)
    else:
        return render_template("index.html")


@app.route("/keyword", methods=["GET", "POST"])
def keyword():
    if request.method == "POST":
        keyword = request.form["keyword"]
        recommendation = md.recommend_books(keyword=keyword)
        return render_template("keyword.html", recommendation=recommendation, keyword=keyword)
    else:
        return render_template("index.html")


if __name__ == "__main__":
    app.run()
