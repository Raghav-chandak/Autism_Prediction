#  Autism Spectrum Disorder (ASD) Prediction Using Machine Learning
Check out the site by  <a href="https://autismprediction27.streamlit.app/"> clicking here </a> <br/>
This project uses **machine learning and Python libraries** to predict the likelihood of Autism Spectrum Disorder (ASD) based on behavioral and demographic data. A user-friendly **Streamlit web app** allows users to interactively input features and receive a prediction.

---

##  Problem Statement

Early detection of Autism can lead to timely intervention and improved outcomes. This project builds a predictive model based on the **Autism Screening Adult Dataset**, using behavioral scores and personal information to assess ASD risk.

---

##  Tech Stack & Python Libraries

| Category        | Libraries Used                                                                 |
|-----------------|--------------------------------------------------------------------------------|
| Data Handling   | `pandas`, `numpy`                                                              |
| ML Algorithms   | `scikit-learn`, `xgboost`, `lightgbm` (optional)                               |
| Model Saving    | `pickle`                                                                       |
| Data Encoding   | `LabelEncoder` from `sklearn.preprocessing`                                    |
| UI / Deployment | `streamlit`                                                                    |
| Visualization   | `matplotlib`, `seaborn`                                                        |
| Imbalanced Data | `imblearn` (e.g., SMOTE, if applied)                                           |

---

##  Project Structure

├── app.py # Streamlit app script
├── Autism_Prediction_using_ML.ipynb # Model training and preprocessing notebook
├── best_model.pkl # Trained machine learning model
├── encoders.pkl # Saved label encoders
├── train.csv # Training dataset
├── requirements.txt # Python dependencies
└── README.md # Project documentation (this file)

---


---

##  How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/autism-prediction-ml.git
cd autism-prediction-ml
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Streamlit App
```bash
streamlit run app.py
```

---

##  Dataset Info

- **Source**: Autism Screening Adult Dataset  
- **Features**:  
  - Behavioral Scores: A1–A10  
  - Age  
  - Gender  
  - Ethnicity  
  - Jaundice History  
  - Autism Diagnosis History  
  - Country of Residence  
  - Screening Score Result  
  - Relation to Respondent  
- **Target**: `Class/ASD`  
  - `1` = ASD  
  - `0` = No ASD  

---

##  Model Building

- **Preprocessing**:  
  - Label Encoding for all categorical variables using `LabelEncoder` from `sklearn.preprocessing`
- **Model**:  
  - Trained using `RandomForestClassifier` from `sklearn.ensemble`
- **Evaluation**:  
  - Accuracy, Precision, Recall, F1-Score (metrics available in the notebook)
- **Persistence**:  
  - Model saved using `pickle` as `best_model.pkl`  
  - Label encoders saved as `encoders.pkl`

---

##  Features of the Streamlit App

-  Interactive UI for behavioral score and demographic input  
-  Model loaded using `pickle`  
-  Real-time prediction based on user inputs  
- Clear visual output with risk indicators:  
  - 🟢 Low Risk  
  - 🔴 High Risk  
-  Consistent data encoding via saved `LabelEncoders`  
-  Lightweight and fast – runs locally using `streamlit`

---

##  Key Skills Demonstrated

- Python for Data Science  
- Data preprocessing & feature engineering  
- Model training and evaluation using `scikit-learn`  
- Serialization using `pickle`  
- Streamlit for ML app deployment  
- Real-world dataset handling  
- Working with imbalanced data (optional use of `imblearn`)

---

##  Sample Prediction Workflow

1. User selects answers to 10 behavioral questions (A1–A10)  
2. Provides age, gender, and other demographics  
3. Inputs are encoded using stored `LabelEncoders`  
4. Pre-trained model predicts ASD likelihood  
5. Result is shown to user with visual feedback:  
   - 🟢 Low Risk  
   - 🔴 High Risk  

---

##  Future Improvements

- Add explainability with SHAP or LIME  
- Integrate additional ML models like SVM, XGBoost for comparison  
- Display confidence probability alongside prediction  
- Deploy the app on Streamlit Cloud or Hugging Face Spaces  
- Improve UI/UX with responsive mobile design  

---

##  License

This project is intended strictly for **educational and research purposes only**.  
Not suitable for clinical or diagnostic use.





