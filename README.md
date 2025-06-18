# 🏋️‍♂️ Nexus-Gym: ML-Based Personalized Fitness Recommendation App

Nexus-Gym is a machine learning-powered Flask web application that provides **personalized gym recommendations** based on a user’s health profile and fitness goals. It recommends suitable exercises, equipment, diet plans, and general fitness advice.

---

## 🚀 Features

- 🔍 Accepts user details: age, sex, height, weight, hypertension, diabetes, BMI level, and fitness goals
- 🤖 Uses a trained ML model to predict the best fitness strategy
- 🏃 Suggests personalized:
  - Fitness Type (e.g., Cardio, Muscular)
  - Exercises
  - Equipment
  - Diet Plans
  - General Recommendations
- 💡 Simple, user-friendly web interface built using Flask

---

## 🧠 Tech Stack

- Python 3
- Flask (Web Framework)
- Scikit-learn (Machine Learning)
- Pandas (Data Handling)
- HTML/CSS (Frontend)

---

## 📁 Project Structure

NEXUS-GYM/
├── app.py # Main Flask application
├── train_model.py # Script to train and save the ML model
├── model.pkl # Trained ML model
├── label_encoders.pkl # Encoders for input labels
├── target_encoder.pkl # Encoder for target variable
├── dataset_info.pkl # Metadata used in the app
├── gym recommendation.csv # Original dataset
├── requirements.txt # Python dependencies
├── templates/
│ ├── index.html # Input form
│ └── result.html # Output recommendation page
└── static/
└── style.css # Styling for the web pages


---

## ⚙️ How to Run Locally

1. **Clone this repo**

git clone https://github.com/yourusername/nexus-gym-ml-app.git
cd nexus-gym-ml-app

2. Install dependencies

bash
Copy
Edit

3. Run the app

python app.py
![image](https://github.com/user-attachments/assets/ff70f7f9-7c91-4c00-98c6-9f82df5c5e39)

Results page:
![image](https://github.com/user-attachments/assets/ce2161ab-0ceb-4c8f-86a0-9adf059ea23c)


