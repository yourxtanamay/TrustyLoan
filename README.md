# Loan Prediction Web Application

## 📌 Project Description

The **Loan Prediction Web Application** is a Machine Learning project that predicts whether a loan application is **Safe** or **Risky** based on user financial information.

This project combines **Machine Learning and Web Development**. A trained model is used to analyze user inputs and provide a prediction through a simple web interface.

The application is built using **Python, Flask, and HTML**.

---

# 🚀 Features

* Web-based interface for user input
* Machine Learning model for loan prediction
* Predicts loan status:

  * ✅ Safe (Low risk)
  * ⚠️ Risky (High risk)
* Easy to run locally using Flask

---

# 🛠 Technologies Used

* Python
* Flask
* Pandas
* NumPy
* Scikit-learn
* HTML
* Pickle (for saving the trained model)

---

# 📂 Project Structure

```
Loan-Prediction-Project
│
├── app.py
├── model.pkl
├── templates
│   └── index.html
│
└── README.md
```

---

# 📊 Input Parameters

The model uses the following input features:

* Person Age
* Person Gender
* Person Education
* Annual Income
* Employment Experience
* Home Ownership
* Loan Amount
* Loan Intent
* Interest Rate
* Credit History Length
* Credit Score
* Previous Loan Defaults

---

# ⚙️ Installation & Setup

## 1️⃣ Install Required Libraries

Run the following command:

```bash
pip install flask pandas numpy scikit-learn
```

## 2️⃣ Run the Application

```bash
python app.py
```

## 3️⃣ Open in Browser

```
http://127.0.0.1:5000/
```

---

# 📈 How It Works

1. The user enters loan details in the web form.
2. The Flask application collects the input data.
3. The data is converted into a Pandas DataFrame.
4. The trained Machine Learning model (`model.pkl`) predicts the result.
5. The result is displayed on the web page as **Safe** or **Risky**.

---

# 🎯 Future Improvements

* Improve the UI design
* Add probability score for prediction
* Deploy the application online
* Use advanced ML models for better accuracy

---

# 👨‍💻 Author

**Tanmay Paul**
Diploma in Computer Science Engineering Student
Interested in **Data Science, Machine Learning, and AI**




