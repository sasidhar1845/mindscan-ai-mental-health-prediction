# 🧠 MindScan AI

MindScan AI is an AI-powered mental health prediction system that analyzes lifestyle and behavioral factors to estimate potential mental health conditions. The project uses machine learning models to provide insights into mental health patterns and demonstrate how artificial intelligence can support mental health awareness.

## 📌 Project Overview

Mental health is an important part of overall well-being. This project uses Artificial Intelligence and Machine Learning to analyze various lifestyle indicators such as sleep patterns, stress levels, physical activity, and social interaction to predict possible mental health conditions.

The system predicts the mental health state based on user inputs and provides insights into factors affecting mental health.

## 🎯 Objectives

- Build a machine learning model to predict mental health conditions
- Analyze behavioral and lifestyle factors affecting mental health
- Compare multiple machine learning models
- Provide an interactive prediction interface

## 🛠️ Technologies Used

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Streamlit  
- Matplotlib  
- Seaborn  

## 📂 Project Structure

mindscan-ai  
│  
├── app.py  
├── data_generation.py  
├── model_training.py  
│  
├── mental_health_data.csv  
├── model_comparison.csv  
│  
├── decision_tree_model.pkl  
├── gender_encoder.pkl  
├── scaler.pkl  
├── target_encoder.pkl  
│  
├── requirements.txt  
└── README.md  

## 📊 Dataset

The dataset contains multiple features that influence mental health such as:

- Age  
- Gender  
- Sleep Hours  
- Physical Activity  
- Stress Level  
- Social Interaction  
- Work Pressure  
- Diet Quality  

Target Variable:

Depression_State

Classes:

- No Depression  
- Mild  
- Moderate  
- Severe  

## 🤖 Machine Learning Models Used

The project compares several machine learning models:

- Logistic Regression  
- Random Forest  
- K-Nearest Neighbors  
- Decision Tree  

The **Decision Tree model** was selected as the final model and saved as:

decision_tree_model.pkl

## 📈 Model Evaluation

Model comparison results are stored in:

model_comparison.csv

The best performing model is used for predictions in the application.

## ▶️ How to Run the Project

### 1. Clone the repository

git clone git clone https://github.com/sasidhar1845/mindscan-ai.git

### 2. Install dependencies

pip install -r requirements.txt

### 3. Train the model (optional)

python model_training.py

### 4. Run the Streamlit application

streamlit run app.py

## 📊 Features

- Mental health prediction using machine learning
- Interactive user interface using Streamlit
- Model comparison analysis
- Data visualization
- Lifestyle factor analysis

## ⚠️ Disclaimer

This project is created for **educational purposes only**.  
It is not intended to replace professional medical advice or diagnosis.  
If you are experiencing mental health issues, please consult a qualified healthcare professional.

## 👨‍💻 Author

Sasidhar M  
MSc Data Science Student  
Chennai, India

## ⭐ Future Improvements

- Improve prediction accuracy using advanced models  
- Deploy the application on cloud platforms  
- Integrate real-time mental health monitoring data  
- Improve the user interface and visualization
