from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load trained pipeline model
with open("house_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return render_template("form.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get data from form
    data = {
        "area": float(request.form["area"]),
        "bedrooms": int(request.form["bedrooms"]),
        "bathrooms": int(request.form["bathrooms"]),
        "stories": int(request.form["stories"]),
        "mainroad": request.form["mainroad"],
        "guestroom": request.form["guestroom"],
        "basement": request.form["basement"],
        "hotwaterheating": request.form["hotwaterheating"],
        "airconditioning": request.form["airconditioning"],
        "parking": int(request.form["parking"]),
        "prefarea": request.form["prefarea"],
        "furnishingstatus": request.form["furnishingstatus"]
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([data])

    # Predict
    prediction = model.predict(input_df)[0]

    return render_template("form.html", prediction=round(prediction, 2))

if __name__ == "__main__":
    app.run(debug=True)
