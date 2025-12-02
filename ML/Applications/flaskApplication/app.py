# from flask import Flask, request , jsonify, render_template
# app = Flask(__name__)
# from urllib.parse import quote
# password = "Admin@123"
# encoded_password = quote(password)
# password = "Admin@123"
# app.config['SQLALCHEMY_DATABASE_URI'] = f'mysql+mysqlconnector://root:{encoded_password}@localhost/datascience'
# from models import db,User

# with app.app_context():
#     db.create_all()
# @app.route("/<name>")
# def hello(name): 
#     return render_template("index.html",name=name)

# @app.route("/predict",methods = ["GET","POST"])
# def predict():
#     if request.method == "POST":
#         name = request.form.get("name")
#         return "hello " + name


# if __name__ == '__main__':
#     app.run(debug=True)

# # Add a new user
# new_user = User(username='testuser', email='test@example.com')
# db.session.add(new_user)
# db.session.commit()


# users = User.query.all()
# for u in users:
#     print(users)
from flask import Flask, request , jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from urllib.parse import quote

app = Flask(__name__)

password = "Admin@123"
encoded_password = quote(password)

app.config['SQLALCHEMY_DATABASE_URI'] = f'mysql+mysqlconnector://root:{encoded_password}@localhost/datascience'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)

    def __repr__(self):
        return f"<User {self.username}>"

@app.route("/<name>")
def hello(name):
    return render_template("index.html",name=name)

@app.route("/predict",methods = ["GET","POST"])
def predict():
    if request.method == "POST":
        name = request.form.get("name")
        return "hello " + name


if __name__ == '__main__':
    with app.app_context():
        db.create_all()

        # Check if user already exists
        if not User.query.filter_by(username='testuser').first():
            new_user = User(username='testuser', email='test@example.com')
            db.session.add(new_user)
            db.session.commit()
            print("User created")
        else:
            print("User already exists")

        # Print all users
        all_users = User.query.all()
        print("Users in DB:", all_users)

    app.run(debug=True)
