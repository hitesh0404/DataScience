from flask import Flask,render_template,request
app = Flask(__name__)

@app.route("/")
def greet():
    return render_template("index.html")

@app.route("/about",methods = ["GET"])
def about():
    return render_template("about.html")

@app.route("/prediction",methods= ["GET","POST"])
def prediction():
    if request.method == "GET":
        return render_template("predict.html")
    elif request.method == "POST":
        data = request.form.get("age")
        return render_template("result.html",result=f"this is result {data}")