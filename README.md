# House Prices — Feature Engineering & Regression

An end-to-end machine learning workflow for predicting residential property prices using the Kaggle House Prices dataset.

The project focuses on **feature engineering, leakage-safe preprocessing, semantic treatment of categorical variables, and model evaluation** rather than simply fitting a model to the raw dataset.

## Why this project?

Real-world tabular data rarely arrives in a form that can be passed directly into a machine learning model.

A house-price dataset contains:

* Numerical measurements such as living area and lot size
* Categorical features such as neighborhood and foundation type
* Ordinal features such as quality and condition ratings
* Missing values that can mean either **"not present"** or **"unknown"**
* Variables whose stored data type does not necessarily represent their true meaning

This project explores how these differences affect the modeling pipeline and how feature engineering can be designed around the **meaning of the data**, not just its datatype.

## Project Goals

* Build a reproducible house-price regression pipeline
* Handle missing values according to their semantic meaning
* Correct misleading variable types before modeling
* Preserve ordering in ordinal categorical variables
* Encode nominal categorical variables appropriately
* Prevent preprocessing leakage between training and validation data
* Compare a linear baseline with a nonlinear ensemble model
* Retrain the selected model on the complete labelled dataset
* Generate a Kaggle-ready prediction file

## Machine Learning Workflow

```text
Raw Data
   ↓
Data Inspection
   ↓
Missing-Value Analysis
   ↓
Semantic Feature Engineering
   ↓
Train / Validation Split
   ↓
Leakage-Safe Preprocessing
   ├── Numeric Imputation
   ├── Ordinal Encoding
   └── One-Hot Encoding
   ↓
Model Training
   ├── Linear Regression
   └── Random Forest
   ↓
Validation
   ├── RMSE
   ├── R²
   └── MAE
   ↓
Model Selection
   ↓
Retraining on Full Training Data
   ↓
Test Prediction
   ↓
submission.csv
```

## Key Engineering Decisions

### 1. Missing Values Are Interpreted Semantically

A missing value does not always mean the same thing.

For features related to optional house components such as garages, basements, pools, and fireplaces, missing values can indicate that the component **does not exist**.

These are therefore treated differently from values where the underlying feature exists but its measurement is unknown.

For remaining missing values, appropriate imputation strategies are applied within the preprocessing pipeline.

### 2. Feature Meaning Determines Encoding

`MSSubClass` is stored as an integer, but its values represent building-class categories rather than a continuous measurement.

It is therefore treated as a categorical variable rather than allowing the model to interpret the class codes as numerical quantities.

For categorical variables:

* **Ordinal encoding** is used where categories have a meaningful order
* **One-hot encoding** is used for nominal categories without an inherent ranking

### 3. Leakage-Safe Preprocessing

The dataset is split before preprocessing.

Imputation and encoding are fitted using the training data and subsequently applied to validation and test data.

This prevents information from the validation or test sets from influencing the preprocessing stage.

## Target Transformation

`SalePrice` is strongly right-skewed, so the model is trained using:

```python
log1p(SalePrice)
```

Predictions are transformed back to the original price scale before generating the final submission.

This also aligns the modeling target with the competition's logarithmic evaluation approach.

## Models

Two models are compared using the same validation set.

### Linear Regression

Used as an interpretable baseline for understanding how far a relatively simple model can perform after appropriate feature engineering.

### Random Forest

Used as a nonlinear model capable of capturing more complex relationships between property characteristics and price.

## Evaluation

Models are evaluated using:

| Metric           | Purpose                                       |
| ---------------- | --------------------------------------------- |
| RMSE (log price) | Primary validation metric                     |
| R²               | Measures explained variance                   |
| MAE              | Interpretable prediction error in price units |

The model with the lowest validation RMSE is selected for the final training stage.

## Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Jupyter Notebook

## Dataset

The project uses the **Kaggle House Prices: Advanced Regression Techniques** dataset.

The dataset contains residential property characteristics and corresponding sale prices.

## Project Structure

```text
house-prices-feature-engineering/
│
├── Features.ipynb
├── train.csv
├── test.csv
├── submission.csv
└── README.md
```

## Reproducibility

Clone the repository, install the required dependencies, place the Kaggle dataset files in the project directory, and run `Features.ipynb` from top to bottom.

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook Features.ipynb
```

The notebook performs the complete workflow from data inspection through final test-set prediction.

## Key Takeaway

The main focus of this project is not simply predicting house prices.

It demonstrates how **understanding the meaning of features can inform preprocessing and model design**, while maintaining a leakage-safe and reproducible machine learning workflow.
