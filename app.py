import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans

st.set_page_config(page_title="Health Dashboard", layout="wide")

# =========================================================
# helper functions
# =========================================================
def get_bmi(weight, height_cm):
    h = height_cm / 100
    if h <= 0:
        return 0
    return round(weight / (h * h), 2)

def bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    return "Obese"

def cluster_meaning(cluster_id):
    mapping = {
        0: "Lower-risk profile group",
        1: "Moderate profile group",
        2: "Higher-risk profile group"
    }
    return mapping.get(cluster_id, f"Cluster {cluster_id}")

# =========================================================
# train diabetes model
# =========================================================
diabetes_df = pd.read_csv("diabetes.csv")

zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in zero_cols:
    diabetes_df[col] = diabetes_df[col].replace(0, np.nan)
    diabetes_df[col] = diabetes_df[col].fillna(diabetes_df[col].median())

X_diabetes = diabetes_df.drop("Outcome", axis=1)
y_diabetes = diabetes_df["Outcome"]

X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
    X_diabetes, y_diabetes, test_size=0.2, random_state=42, stratify=y_diabetes
)

diabetes_scaler = StandardScaler()
X_train_d_scaled = diabetes_scaler.fit_transform(X_train_d)
X_test_d_scaled = diabetes_scaler.transform(X_test_d)

diabetes_model = LogisticRegression(max_iter=5000)
diabetes_model.fit(X_train_d_scaled, y_train_d)
diabetes_accuracy = round(accuracy_score(y_test_d, diabetes_model.predict(X_test_d_scaled)) * 100, 2)

# =========================================================
# train heart model
# =========================================================
heart_df = pd.read_csv("heart_disease.csv")
heart_df = heart_df.replace("?", np.nan)

for col in heart_df.columns:
    heart_df[col] = pd.to_numeric(heart_df[col], errors="coerce")

heart_df["target"] = heart_df["num"].apply(lambda x: 0 if x == 0 else 1)

drop_cols = [col for col in ["id", "num", "dataset"] if col in heart_df.columns]
heart_df = heart_df.drop(columns=drop_cols)

X_heart = heart_df.drop("target", axis=1)
y_heart = heart_df["target"]

for col in X_heart.columns:
    X_heart[col] = pd.to_numeric(X_heart[col], errors="coerce")

heart_num_cols = X_heart.columns.tolist()

heart_preprocessor = ColumnTransformer(transformers=[
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), heart_num_cols)
])

heart_model = Pipeline(steps=[
    ("preprocessor", heart_preprocessor),
    ("classifier", LogisticRegression(max_iter=5000))
])

X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
    X_heart, y_heart, test_size=0.2, random_state=42, stratify=y_heart
)

heart_model.fit(X_train_h, y_train_h)
heart_accuracy = round(accuracy_score(y_test_h, heart_model.predict(X_test_h)) * 100, 2)

# =========================================================
# clustering model
# =========================================================
cluster_df = pd.read_csv("height_weight.csv")
cluster_df = cluster_df[["age", "sex", "bmi", "children", "smoker", "region"]].copy()

cluster_df["sex"] = cluster_df["sex"].map({"male": 1, "female": 0})
cluster_df["smoker"] = cluster_df["smoker"].map({"yes": 1, "no": 0})
cluster_df["region"] = cluster_df["region"].map({
    "northwest": 0,
    "northeast": 1,
    "southwest": 2,
    "southeast": 3
})

cluster_df = cluster_df.fillna(cluster_df.median(numeric_only=True))

cluster_scaler = StandardScaler()
X_cluster_scaled = cluster_scaler.fit_transform(cluster_df)

kmeans_model = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_model.fit(X_cluster_scaled)

# =========================================================
# UI style
# =========================================================
st.markdown("""
<style>
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
}
[data-testid="stMetric"] {
    background: #f7faff;
    border: 1px solid #dbe6f5;
    padding: 12px;
    border-radius: 14px;
}
.demo-box {
    background: #f8fbff;
    border: 1px solid #d9e5f4;
    padding: 14px;
    border-radius: 14px;
    margin-bottom: 14px;
}
.title-box {
    background: linear-gradient(135deg, #eef4ff, #f8fbff);
    border: 1px solid #dbe6f5;
    padding: 18px;
    border-radius: 18px;
    margin-bottom: 18px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="title-box">
    <h1 style="margin-bottom:8px;">Smart Health Dashboard</h1>
    <p style="margin:0; color:#55606f;">
        Separate tabs for BMI, Diabetes, Heart Disease and Clustering.
    </p>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs(["BMI", "Diabetes", "Heart", "Clustering"])

# =========================================================
# BMI TAB
# =========================================================
with tabs[0]:
    st.subheader("BMI Calculator")

    st.markdown("""
    <div class="demo-box">
        <b>Demo values:</b> Height = 170 cm, Weight = 70 kg
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        bmi_height = st.number_input("Height (cm)", min_value=1.0, value=170.0, key="bmi_height")
    with c2:
        bmi_weight = st.number_input("Weight (kg)", min_value=1.0, value=70.0, key="bmi_weight")

    if st.button("Calculate BMI", use_container_width=True):
        bmi = get_bmi(bmi_weight, bmi_height)
        category = bmi_category(bmi)

        x, y = st.columns(2)
        x.metric("BMI", bmi)
        y.metric("Category", category)

# =========================================================
# DIABETES TAB
# =========================================================
with tabs[1]:
    st.subheader("Diabetes Prediction")

    st.markdown("""
    <div class="demo-box">
        <b>Demo values:</b> Pregnancies = 0, Glucose = 120, Blood Pressure = 80, Skin Thickness = 20, Insulin = 85, Height = 170, Weight = 70, DPF = 0.5, Age = 22
    </div>
    """, unsafe_allow_html=True)

    d1, d2 = st.columns(2)

    with d1:
        pregnancies = st.number_input("Pregnancies", value=0, step=1, key="d_preg")
        glucose = st.number_input("Glucose", value=120.0, key="d_glucose")
        bloodpressure = st.number_input("Blood Pressure", value=80.0, key="d_bp")
        skinthickness = st.number_input("Skin Thickness", value=20.0, key="d_skin")
        insulin = st.number_input("Insulin", value=85.0, key="d_insulin")

    with d2:
        d_height = st.number_input("Height (cm)", value=170.0, key="d_height")
        d_weight = st.number_input("Weight (kg)", value=70.0, key="d_weight")
        dpf = st.number_input("DPF", value=0.5, key="d_dpf")
        age_d = st.number_input("Age", value=22, step=1, key="d_age")

    if st.button("Predict Diabetes", use_container_width=True):
        bmi = get_bmi(d_weight, d_height)

        diabetes_input = pd.DataFrame([{
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": bloodpressure,
            "SkinThickness": skinthickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": dpf,
            "Age": age_d
        }])

        diabetes_scaled = diabetes_scaler.transform(diabetes_input)
        diabetes_pred = int(diabetes_model.predict(diabetes_scaled)[0])
        diabetes_prob = float(diabetes_model.predict_proba(diabetes_scaled)[0][1])

        result = "Diabetes Risk" if diabetes_pred == 1 else "No Diabetes Risk"

        x, y, z = st.columns(3)
        x.metric("Result", result)
        y.metric("Confidence", f"{round(diabetes_prob * 100, 2)}%")
        z.metric("Model Accuracy", f"{diabetes_accuracy}%")

# =========================================================
# HEART TAB
# =========================================================
with tabs[2]:
    st.subheader("Heart Disease Prediction")

    st.markdown("""
    <div class="demo-box">
        <b>Demo values:</b> Age = 22, Gender = male, CP = 1, Trestbps = 120, Chol = 200, FBS = 0, Restecg = 1, Thalch = 150, Exang = 0, Oldpeak = 1.2, Slope = 2, CA = 0, Thal = 0
    </div>
    """, unsafe_allow_html=True)

    h1, h2, h3 = st.columns(3)

    with h1:
        age_h = st.number_input("Age", value=22, step=1, key="h_age")
        sex_h = st.selectbox("Gender", ["male", "female"], key="h_sex")
        cp = st.number_input("CP", value=1.0, key="h_cp")
        trestbps = st.number_input("Trestbps", value=120.0, key="h_trest")

    with h2:
        chol = st.number_input("Chol", value=200.0, key="h_chol")
        fbs = st.number_input("FBS", value=0.0, key="h_fbs")
        restecg = st.number_input("Restecg", value=1.0, key="h_restecg")
        thalch = st.number_input("Thalch", value=150.0, key="h_thalch")

    with h3:
        exang = st.number_input("Exang", value=0.0, key="h_exang")
        oldpeak = st.number_input("Oldpeak", value=1.2, key="h_oldpeak")
        slope = st.number_input("Slope", value=2.0, key="h_slope")
        ca = st.number_input("CA", value=0.0, key="h_ca")
        thal = st.number_input("Thal", value=0.0, key="h_thal")

    if st.button("Predict Heart Disease", use_container_width=True):
        heart_input = pd.DataFrame([{
            "age": age_h,
            "sex": 1 if sex_h == "male" else 0,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalch": thalch,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal
        }])

        for col in heart_input.columns:
            heart_input[col] = pd.to_numeric(heart_input[col], errors="coerce")

        heart_pred = int(heart_model.predict(heart_input)[0])
        heart_prob = float(heart_model.predict_proba(heart_input)[0][1])

        result = "Heart Disease Risk" if heart_pred == 1 else "No Heart Disease Risk"

        x, y, z = st.columns(3)
        x.metric("Result", result)
        y.metric("Confidence", f"{round(heart_prob * 100, 2)}%")
        z.metric("Model Accuracy", f"{heart_accuracy}%")

# =========================================================
# CLUSTERING TAB
# =========================================================
with tabs[3]:
    st.subheader("Health Profile Clustering")

    st.markdown("""
    <div class="demo-box">
        <b>Demo values:</b> Age = 22, Gender = male, Height = 170, Weight = 70, Children = 0, Smoker = no, Region = northwest
    </div>
    """, unsafe_allow_html=True)

    cl1, cl2 = st.columns(2)

    with cl1:
        age_c = st.number_input("Age", value=22, step=1, key="c_age")
        sex_c = st.selectbox("Gender", ["male", "female"], key="c_sex")
        c_height = st.number_input("Height (cm)", value=170.0, key="c_height")
        c_weight = st.number_input("Weight (kg)", value=70.0, key="c_weight")

    with cl2:
        children_c = st.number_input("Children", value=0, step=1, key="c_children")
        smoker_c = st.selectbox("Smoker", ["no", "yes"], key="c_smoker")
        region_c = st.selectbox("Region", ["northwest", "northeast", "southwest", "southeast"], key="c_region")

    if st.button("Generate Cluster", use_container_width=True):
        bmi_c = get_bmi(c_weight, c_height)

        cluster_input = pd.DataFrame([{
            "age": age_c,
            "sex": 1 if sex_c == "male" else 0,
            "bmi": bmi_c,
            "children": children_c,
            "smoker": 1 if smoker_c == "yes" else 0,
            "region": {"northwest": 0, "northeast": 1, "southwest": 2, "southeast": 3}[region_c]
        }])

        cluster_scaled = cluster_scaler.transform(cluster_input)
        cluster_id = int(kmeans_model.predict(cluster_scaled)[0])

        x, y, z = st.columns(3)
        x.metric("BMI Used", bmi_c)
        y.metric("Cluster ID", cluster_id)
        z.metric("Cluster Type", cluster_meaning(cluster_id))