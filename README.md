# Customer Churn Prediction Dashboard

A Streamlit research prototype for comparing customer churn models and predicting churn with a class-weighted logistic regression pipeline.

## Required files

- `app.py`
- `requirements.txt`
- `churn_pipeline.joblib` (exported from the Colab notebook)
- `cleaned_telco.csv`
- `model_comparison.csv`
- `balanced_model_comparison.csv`
- `cross_validation_results.csv`

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment

Push these files to a GitHub repository and deploy `app.py` via Streamlit Community Cloud. Use the exact scikit-learn version from the notebook when exporting the model; adjust `requirements.txt` accordingly.

This is a research prototype and is not validated for production decisions.
