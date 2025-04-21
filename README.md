# 🍷 Wine Quality Prediction App

This Streamlit app predicts the quality of red wine based on its physicochemical properties using machine learning models.

## 🚀 Features

- Predicts whether a wine sample is of **good** or **bad quality**
- Offers **model selection** (Random Forest, Logistic Regression, SVM)
- Displays **model accuracy, precision, recall, confusion matrix**
- Supports **batch prediction via CSV, XLSX, or TXT upload**
- Shows **training vs testing accuracy chart**
- Optionally downloads **predictions and a sample template**
- Clean and interactive **UI built with Streamlit**

## 📂 Files Included

- `wine-quality-pred.py`: Main Streamlit app
- `requirements.txt`: Python dependencies
- `winequality-red.xlsx`: Sample dataset from UCI ML Repository
- `wine.png`: Image displayed on the app homepage
- `README.md`: You’re reading it!

## 🧠 Models Used

- Random Forest
- Logistic Regression
- Support Vector Machine (SVM)

## 🧪 Features Used

- Fixed Acidity
- Volatile Acidity
- Citric Acid
- Residual Sugar
- Chlorides
- Free Sulfur Dioxide
- Total Sulfur Dioxide
- Density
- pH
- Sulphates
- Alcohol

## 🛠 How to Run Locally

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
python -m venv venv
source venv/bin/activate  # or venv\\Scripts\\activate on Windows
pip install -r requirements.txt
streamlit run wine-quality-pred.py

