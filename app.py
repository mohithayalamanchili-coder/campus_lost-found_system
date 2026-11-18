
import os
import uuid
from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Upload Folder
app.config["UPLOAD_FOLDER"] = os.path.join(
    app.root_path, "static", "uploads"
)

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Database Configuration
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///campus.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)


# Database Model
class Item(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    item_name = db.Column(db.String(100), nullable=False)
    category = db.Column(db.String(50))
    date = db.Column(db.String(20))
    location = db.Column(db.String(100))
    description = db.Column(db.String(300))
    item_type = db.Column(db.String(20))
    email = db.Column(db.String(120))
    image = db.Column(db.String(200))


# Save Uploaded Image
def save_image(file):

    if file and file.filename:
        filename = secure_filename(file.filename)

        extension = os.path.splitext(filename)[1].lower()

        if extension not in [".jpg", ".jpeg", ".png", ".gif", ".webp"]:
            return None

        new_filename = str(uuid.uuid4()) + extension

        file.save(
            os.path.join(
                app.config["UPLOAD_FOLDER"],
                new_filename
            )
        )

        return new_filename

    return None


# Home Page
@app.route("/")
def home():
    return render_template("index.html")


# Report Lost Item
@app.route("/lost", methods=["GET", "POST"])
def lost():

    if request.method == "POST":

        image_file = request.files.get("image")
        image_name = save_image(image_file)

        item = Item(
            item_name=request.form.get("item_name"),
            category=request.form.get("category"),
            date=request.form.get("date"),
            location=request.form.get("location"),
            description=request.form.get("description"),
            item_type="Lost",
            email=request.form.get("email"),
            image=image_name
        )

        db.session.add(item)
        db.session.commit()

        return "Lost Item Report Submitted Successfully!"

    return render_template("lost.html")


# Report Found Item
@app.route("/found", methods=["GET", "POST"])
def found():

    if request.method == "POST":

        image_file = request.files.get("image")
        image_name = save_image(image_file)

        item = Item(
            item_name=request.form.get("item_name"),
            category=request.form.get("category"),
            date=request.form.get("date"),
            location=request.form.get("location"),
            description=request.form.get("description"),
            item_type="Found",
            email=request.form.get("email"),
            image=image_name
        )

        db.session.add(item)
        db.session.commit()

        return "Found Item Report Submitted Successfully!"

    return render_template("found.html")


# View All Reports
@app.route("/items")
def items():
    all_items = Item.query.all()
    return render_template("items.html", items=all_items)


# AI Matching Feature
@app.route("/matches")
def matches():

    lost_items = Item.query.filter_by(item_type="Lost").all()
    found_items = Item.query.filter_by(item_type="Found").all()

    match_results = []

    if lost_items and found_items:

        all_items = lost_items + found_items
        item_texts = []

        for item in all_items:
            text = (
                str(item.item_name) + " " +
                str(item.category) + " " +
                str(item.description)
            )
            item_texts.append(text)

        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(item_texts)

        similarity_matrix = cosine_similarity(
            vectors[:len(lost_items)],
            vectors[len(lost_items):]
        )

        for i in range(len(lost_items)):
            for j in range(len(found_items)):

                score = similarity_matrix[i][j]

                if score >= 0.15:
                    match_results.append({
                        "lost": lost_items[i],
                        "found": found_items[j],
                        "score": round(score * 100, 2)
                    })

        match_results.sort(
            key=lambda x: x["score"],
            reverse=True
        )

    return render_template(
        "matches.html",
        matches=match_results
    )


# Create Database and Update Existing Table
with app.app_context():

    db.create_all()

    with db.engine.begin() as connection:

        columns = connection.exec_driver_sql(
            "PRAGMA table_info(item)"
        ).fetchall()

        column_names = [column[1] for column in columns]

        if "email" not in column_names:
            connection.exec_driver_sql(
                "ALTER TABLE item ADD COLUMN email VARCHAR(120)"
            )

        if "image" not in column_names:
            connection.exec_driver_sql(
                "ALTER TABLE item ADD COLUMN image VARCHAR(200)"
            )


if __name__ == "__main__":
    app.run(debug=True)
