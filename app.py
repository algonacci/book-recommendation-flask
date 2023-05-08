from flask import Flask, request, render_template
import module as md

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        judul = request.form["judul"]
        recommendation = md.rec_pvdbow(Title=judul)
        return render_template("index.html", recommendation=recommendation)
    else:
        return render_template("index.html")


if __name__ == "__main__":
    app.run()
