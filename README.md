# Customer Churn Prediction

Machine learning project for predicting whether a bank customer is likely to churn. The project includes exploratory data analysis, feature engineering, model comparison, evaluation, and a serialized scikit-learn pipeline for reuse.

## Project Overview

The analysis uses the **Bank Customer Churn Prediction** dataset containing 10,000 customer records and 12 original columns. The target variable is `churn`:

- `0` — customer retained
- `1` — customer churned

The notebook compares several classification algorithms using stratified 5-fold cross-validation and selects the model with the highest mean ROC AUC.

## Results

Gradient Boosting performed best during cross-validation and was selected as the final model.

| Model | Mean CV ROC AUC |
| --- | ---: |
| Logistic Regression | 0.7878 |
| Random Forest | 0.8497 |
| **Gradient Boosting** | **0.8631** |
| AdaBoost | 0.8464 |
| SVC | 0.8348 |

Performance on the held-out test set:

| Metric | Score |
| --- | ---: |
| Accuracy | 0.8650 |
| Precision | 0.7708 |
| Recall | 0.4791 |
| F1 score | 0.5909 |
| ROC AUC | 0.8695 |

Because the classes are imbalanced, ROC AUC, recall, and the churn-class F1 score should be considered alongside accuracy.

## Feature Engineering

The workflow adds the following derived features:

- `balance_per_product`
- `salary_balance_ratio`
- `age_group`
- `tenure_bucket`
- `high_balance`

Numeric features are median-imputed and standardized. Categorical features are imputed with their most frequent value and one-hot encoded. `customer_id` is excluded from model training.

## Repository Contents

| File | Description |
| --- | --- |
| `Analysis.ipynb` | Complete analysis, visualization, training, evaluation, and prediction workflow |
| `Bank Customer Churn Prediction.csv` | Source dataset |
| `best_churn_pipeline.pkl` | Serialized trained preprocessing and Gradient Boosting pipeline |
| `README.md` | Project documentation |

## Getting Started

### Requirements

- Python 3.9 or newer
- Jupyter Notebook or JupyterLab
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- joblib

Install the dependencies with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib jupyter
```

### Run the analysis

1. Clone the repository:

   ```bash
   git clone https://github.com/dhanushram24/Customer-Churn-Prediction.git
   cd Customer-Churn-Prediction
   ```

2. Start Jupyter:

   ```bash
   jupyter notebook
   ```

3. Open `Analysis.ipynb` and run the cells from top to bottom.

The notebook reads `Bank Customer Churn Prediction.csv`, trains and evaluates the models, and saves the final pipeline as `best_churn_pipeline.pkl`.

## Using the Trained Pipeline

The saved pipeline contains both preprocessing and the trained classifier. Load it with `joblib`:

```python
import joblib

pipeline = joblib.load("best_churn_pipeline.pkl")
```

For a new customer, apply the same feature engineering shown in `Analysis.ipynb`, remove `customer_id`, and pass a one-row pandas DataFrame to the pipeline:

```python
prediction = pipeline.predict(customer_df)[0]
churn_probability = pipeline.predict_proba(customer_df)[0, 1]

print("Predicted churn:", prediction)
print(f"Churn probability: {churn_probability:.3f}")
```

## Notes

- The reported metrics are based on the current notebook run and a fixed `random_state=42`.
- Model performance may change if the data, feature engineering, or train/test split changes.
- The trained pipeline is intended for experimentation and learning; production use would require monitoring, validation, and a suitable decision threshold for the business objective.
