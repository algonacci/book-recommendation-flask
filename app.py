from flask import Flask, request, render_template
import pandas as pd
import module as md

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/recommendation")
def recommendation():
    return render_template("recommendation.html")


@app.route("/title", methods=["GET", "POST"])
def title():
    if request.method == "POST":
        judul = request.form["judul"]
        try:
            recommendation = md.rec_pvdbow(Title=judul)
            return render_template("title.html", recommendation=recommendation, judul=judul)
        except KeyError:
            return render_template("recommendation.html", error="Judul novel yang dicari tidak ada"), 500
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


@app.errorhandler(500)
def internal_server_error(error):
    return render_template("recommendation.html", error="Judul novel yang dicari tidak ada"), 500


@app.route("/daftar_novel")
def daftar_novel():
    df = pd.read_csv('Novel.csv')

    # Add image column to DataFrame
    df['Sampul'] = df['Sampul'].apply(
        lambda x: f'<img src="{x}" width="100">' if isinstance(x, str) else '')

    # Convert the DataFrame to an HTML table with Bootstrap classes
    table = df.to_html(
        index=False, classes='table table-striped', escape=False)

    # Render the HTML template with the Bootstrap table
    return render_template('daftar_novel.html', table=table)


if __name__ == "__main__":
    app.run()
