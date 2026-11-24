import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title='Customer Churn Prediction', page_icon='📊', layout='wide')
st.title('Customer Churn Prediction Dashboard')
st.caption('IBM Telco Customer Churn | Honours research prototype')

@st.cache_resource
def load_model():
    return joblib.load('churn_pipeline.joblib')

@st.cache_data
def load_data():
    return pd.read_csv('cleaned_telco.csv')

try:
    model = load_model()
    df = load_data()
except Exception as exc:
    st.error(f'Could not load project files: {exc}')
    st.stop()

page = st.sidebar.radio('Navigate', ['Customer analytics', 'Model comparison', 'Predict churn'])
if page == 'Customer analytics':
    st.subheader('Customer analytics')
    c1, c2, c3 = st.columns(3)
    c1.metric('Customers', f'{len(df):,}')
    c2.metric('Churn rate', f"{df['Churn'].eq('Yes').mean():.1%}")
    c3.metric('Average tenure (months)', f"{df['tenure'].mean():.1f}")
    st.write('Churn distribution')
    st.bar_chart(df['Churn'].value_counts())
    st.write('Churn rate by contract')
    st.bar_chart(df.groupby('Contract')['Churn'].apply(lambda x: x.eq('Yes').mean()))
elif page == 'Model comparison':
    st.subheader('Model comparison')
    files = [('Baseline test results', 'model_comparison.csv'), ('Class-weighted test results', 'balanced_model_comparison.csv'), ('Cross-validation results', 'cross_validation_results.csv')]
    for title, filename in files:
        st.markdown(f'**{title}**')
        if Path(filename).exists():
            table = pd.read_csv(filename)
            st.dataframe(table, use_container_width=True, hide_index=True)
            if 'F1_Mean' in table.columns:
                plot = table.copy()
                plot['Configuration'] = plot['Model'] + ' — ' + plot['Experiment']
                st.bar_chart(plot.set_index('Configuration')['F1_Mean'])
        else:
            st.info(f'Add {filename} to the repository to display this table.')
else:
    st.subheader('Predict customer churn')
    st.caption('Research demonstration only; predictions are not validated for real-world retention decisions.')
    X = df.drop(columns=['customerID', 'Churn'])
    with st.form('prediction_form'):
        cols = st.columns(2)
        values = {}
        for idx, field in enumerate(X.columns):
            with cols[idx % 2]:
                if pd.api.types.is_numeric_dtype(X[field]):
                    series = pd.to_numeric(X[field], errors='coerce').dropna()
                    is_int = pd.api.types.is_integer_dtype(X[field])
                    default = float(series.median()) if len(series) else 0.0
                    values[field] = st.number_input(field, value=int(default) if is_int else round(default, 2), step=1 if is_int else 0.5)
                else:
                    options = sorted(X[field].dropna().astype(str).unique().tolist())
                    values[field] = st.selectbox(field, options)
        submitted = st.form_submit_button('Predict churn', type='primary')
    if submitted:
        row = pd.DataFrame([values], columns=X.columns)
        prediction = int(model.predict(row)[0])
        st.metric('Predicted outcome', 'Likely to churn' if prediction else 'Unlikely to churn')
        if hasattr(model, 'predict_proba'):
            probability = float(model.predict_proba(row)[0, 1])
            st.metric('Estimated churn probability', f'{probability:.1%}')
            st.caption('This is a model score, not a calibrated or guaranteed probability.')
