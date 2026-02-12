from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("diabetes_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = [
        int(request.form["Pregnancies"]),
        int(request.form["Glucose"]),
        int(request.form["BloodPressure"]),
        int(request.form["SkinThickness"]),
        int(request.form["Insulin"]),
        float(request.form["BMI"]),
        float(request.form["DiabetesPedigreeFunction"]),
        int(request.form["Age"])
    ]

    prediction = model.predict([data])

    result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"

    return render_template("result.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
