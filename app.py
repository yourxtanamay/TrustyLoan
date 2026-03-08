from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load your trained pipeline/model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        person_age = int(request.form["person_age"])
        person_gender = int(request.form["person_gender"])
        person_education = int(request.form["person_education"])
        person_annual_income = float(request.form["person_annual_income"])
        person_emp_exp = float(request.form["person_emp_exp"])
        person_home_ownership = int(request.form["person_home_ownership"])
        loan_amnt = float(request.form["loan_amnt"])
        loan_intent = int(request.form["loan_intent"])
        loan_int_rate = float(request.form["loan_int_rate"])
        person_cred_hist_length = float(request.form["person_cred_hist_length"])
        credit_score = float(request.form["credit_score"])
        previous_defaults = int(request.form["previous_loan_defaults_on_file"])

        input_data = pd.DataFrame([[ 
            person_age,
            person_gender,
            person_education,
            person_annual_income,
            person_emp_exp,
            person_home_ownership,
            loan_amnt,
            loan_intent,
            loan_int_rate,
            person_cred_hist_length,
            credit_score,
            previous_defaults
        ]],columns=[
            "person_age",
            "person_gender",
            "person_education",
            "person_annual_income",
            "person_emp_exp",
            "person_home_ownership",
            "loan_amnt",
            "loan_intent",
            "loan_int_rate",
            "person_cred_hist_length",
            "credit_score",
            "previous_loan_defaults_on_file"
        ])


        prediction = model.predict(input_data)

        if prediction[0] == 0:
            result = " Risky ⚠️ "
        else:
            result = " Safe ✅"

        return render_template(
                "index.html",
                prediction_text=result,
                form_data=request.form
)


    except Exception as e:
        return render_template(
            "index.html",
            prediction_text="Error occurred",
            probability_text=str(e),
            form_data=request.form)   


if __name__ == "__main__":
    app.run(debug=True)
