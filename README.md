# 🏥 Heart Disease Prediction using AI/ML Techniques

## 🚀 Overview
This project focuses on predicting heart disease using AI/ML techniques. It involves data preprocessing, exploratory data analysis, feature engineering, and model implementation using **Multi-class Logistic Regression, SVM, and Deep Learning models with TensorFlow/Keras**. The objective is to build a robust predictive model that can assist in early diagnosis and risk assessment for Coronary heart disease.

## 🔥 Features
- ✅ **Data Consolidation & Preprocessing** – Cleaning, handling missing values, and feature engineering
- ✅ **Exploratory Data Analysis (EDA)** – Visualizations and statistical insights
- ✅ **Machine Learning Models** – Multi-Class Logistic Regression, SVM, and deep learning models
- ✅ **Performance Evaluation** – F1-Score, Recall, Confusion Matrix
- ✅ **Real-World Application** – AI-driven approach for healthcare diagnostics

## 📂 Project Structure
```
📦 heart-disease-prediction
├── 📂 data                # Raw and processed datasets
├── 📂 notebooks           # Jupyter notebooks with EDA and model training
├── 📂 models              # Trained model architectures and saved weights
├── 📂 src                 # Python scripts for training and inference
├── 📂 results             # Output logs, metrics, and model visualizations
├── 📄 requirements.txt    # Dependencies
└── 📄 README.md           # Project documentation
```

## 📊 Dataset
- **Source:** Processed Cleveland Heart Disease Dataset
- **Preprocessing Steps:**
  - Handled missing values
  - Standardized numerical features
  - One-hot encoded categorical variables

## 📈 Exploratory Data Analysis (EDA)
- Distribution analysis of heart disease presence
- Correlation heatmaps and feature importance
- Outlier detection and data balancing

## 🏋️ Model Training
```bash
python train.py --model logistic_regression --epochs 50 --batch_size 32 --lr 0.001
```
### **Implemented Models:**
- **Logistic Regression** – Baseline model
- **Support Vector Machine (SVM)** – Improved classification
- **Deep Learning Model (TensorFlow/Keras)** – Neural network for enhanced accuracy

## 🎯 Results & Performance
| Model                 | Recall | F1-score |
|-----------------------|--------|----------|
| Logistic Regression   | 63%    | 53%      |
| SVM                   | 60%    | 46%      |
| Deep Learning Model   | 54%    | 40%      |

## 📌 Future Improvements
- 🔹 Hyperparameter tuning for better optimization
- 🔹 Incorporating more advanced deep learning architectures
- 🔹 Testing on additional real-world datasets

## 🤝 Contributing
Feel free to submit issues or pull requests. Contributions are always welcome!

## Conclusion 
- In this project, we implemented Multiclass Logistic Regression, Neural Network Classifier, and Support Vector Machine methods to predict coronary heart disease. Though these models showed great promise in handling the classification task, they could not perform to an optimum predictive level due to limitations of the dataset.
- The dataset used for this project was insufficient in several ways:
     **Sample size** : Limited samples in the dataset made models generalize less efficiently to
 the new coming data.
    **Diversity in features** : Despite there being a fair number of different features included
 in the dataset, they were less representative of further, detailed features of health status that
 may have improved how well those models learned from coronary heart disease risk factors.

