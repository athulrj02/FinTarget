# **FinTarget - Smarter Predictions for Smarter Banking!**  

## **Overview**  
**FinTarget** is a machine learning project aimed at predicting whether a bank customer will subscribe to a term deposit based on demographic, financial, and campaign-related features. This model helps optimize marketing strategies by focusing efforts on high-potential customers, enhancing efficiency and performance.  

## **Problem Statement**  
The goal is to build a classification model that predicts term deposit subscriptions based on customer data. Accurate predictions can help banks improve their marketing campaigns by targeting the right customers.  

## **Dataset**  
The dataset used is the **Bank Marketing Dataset**, consisting of 11,162 rows and 17 columns with the following key attributes:  

- **Demographics:** Age, job, marital status, education level.  
- **Financial Information:** Balance, housing loan status, personal loan status.  
- **Campaign Details:** Last contact duration, number of contacts, and previous outcomes.  
- **Target Variable:** Subscription to term deposit ('yes' or 'no').  

**Source:**  
Dataset obtained from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing).  

## **Workflow**  
1. **Exploratory Data Analysis (EDA):**  
   - Visualized numerical and categorical data relationships.  
2. **Data Preprocessing:**  
   - Missing values handled using median and mode.  
   - Encoded categorical variables and scaled numerical features.  
3. **Feature Selection:**  
   - Analyzed correlations and extracted top features using Random Forest.  
4. **Model Development:**  
   - Trained and tested Logistic Regression, Decision Tree, and Random Forest classifiers.  
5. **Hyperparameter Optimization:**  
   - Tuned Random Forest using Grid Search CV for the best performance.  
6. **Evaluation Metrics:**  
   - Compared models using accuracy, precision, recall, and F1-score.  

## **Results**  

| Model                         | Accuracy | Precision | Recall | F1-Score |
|-------------------------------|----------|-----------|--------|----------|
| Logistic Regression           | 75.43%   | 73.46%    | 81.97% | 77.61%   |
| Decision Tree                 | 78.26%   | 78.95%    | 80.13% | 79.64%   |
| Random Forest                 | 84.20%   | 87.23%    | 81.57% | 84.30%   |
| Optimized Random Forest (Grid Search) | **84.52%** | **87.85%** | **81.80%** | **84.59%** |

## **Key Insights**  
- Features like **duration**, **balance**, and **age** played a critical role in predictions.  
- Optimized Random Forest delivered the best results, highlighting the importance of ensemble methods.  
- Feature selection improved performance without sacrificing accuracy.  

## **Technologies Used**  
- **Programming Language:** Python  
- **Libraries:**  
  - Pandas, NumPy - Data manipulation  
  - Matplotlib, Seaborn - Data visualization  
  - Scikit-learn - Machine learning modeling and evaluation  

## **File Structure**  
```
|-- FinTarget/
|   |-- data/
|       |-- bank.csv
|   |-- notebooks/
|       |-- Bank_marketing_deposit.ipynb
|   |-- reports/
|       |-- SUPERVISED_MACHINE_LEARNING_report.pdf
|   |-- README.md
|-- requirements.txt
```

## **How to Run**  
1. Clone this repository:  
   ```
   git clone https://github.com/athulrj02/FinTarget.git
   ```  
2. Install dependencies:  
   ```
   pip install -r requirements.txt
   ```  
3. Open the Jupyter Notebook:  
   ```
   jupyter notebook
   ```  
4. Execute the cells in **Bank_marketing_deposit.ipynb** step-by-step.  

## **Future Enhancements**  
- Incorporate additional customer behavior data for better predictions.  
- Test deep learning models to boost performance.  
- Deploy the model using Flask or Streamlit for real-time predictions.  

