import streamlit as st
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier

st.set_page_config(page_title="Student Churn Predictor - Manual Entry", layout="wide")
st.title("🎓 Predict Student Churn (Form-Based)")

st.markdown("Fill out the student information below:")

with st.form("student_form"):
   STDNT_AGE = st.number_input("Age", min_value=18, max_value=60)
   STDNT_GENDER = st.selectbox("Gender", ["M", "F"])
   STDNT_BACKGROUND = st.selectbox("Background", ["BGD 1", "BGD 2", "BGD 3", "BGD 4", "BGD 5", "BGD 6", "BGD 7", "BGD 8"])
   IN_STATE_FLAG = st.selectbox("In State Flag", ["Y", "N"])
   INTERNATIONAL_STS = st.selectbox("International Student", ["Y", "N"])
   STDNT_MAJOR = st.selectbox("Major Subject", ["Accounting", "Applied Computer Science", "Art", "Art Education", "Biology", "Biology and Secondary Ed", "Chemistry", "Chemistry and Secondary Ed", "Communication", "Computer Science - Games", "Computer Science - Systems", "Criminal Justice", "Early Admission - Accel", "Early Childhood Education", "Earth and Space Science", "Engineering Studies", "English and Secondary Ed", "English Language/Literature", "Exercise Science", "Finance", "French", "French with Teacher Cert", "General Business", "General Studies/AS", "Geology", "Health and Physical Education", "Health Science", "History", "History and Secondary Ed", "Information Technology", "Joint Enrollment - Accel", "Liberal Arts", "Management", "Management Information Systems", "Marketing", "Mathematics", "Mathematics and Secondary Ed", "Middle Grades Education", "Music", "Music Education", "Music Performance", "Nursing", "Political Science",  "Pre-Business", "Pre-Engineering/RETP", "Pre-Nursing", "Psychology", "Sociology", "Spanish", "Spanish with Teacher Cert", "Spec Ed: Gen. Curr. - Reading", "Theatre Arts", "Theatre Education", "Undeclared"])
   STDNT_MINOR = st.selectbox("Minor Subject", ["Accounting", "African Studies", "Anthropology", "Art", "Art History", "Asian Studies", "Biology", "Chemistry", "Communication", "Computer Info. Management", "Computer Science - Systems", "Creative Writing", "Criminal Justice", "Early Childhood Education", "Economics", "English Language/Literature", "Exercise Science", "Finance", "Foundations of Business", "French", "Health Science", "History", "International Business", "Management", "Marketing", "Mathematics", "Military Sci & Adv Leadership", "Music", "N", "Philosophy", "Political Science", "Professional Writing", "Psychology", "Sociology", "Spanish", "Theatre Arts", "Women's Studies"])
   STDNT_TEST_ENTRANCE1 = st.number_input("Entrance 1 Score", min_value=0, max_value=32)
   STDNT_TEST_ENTRANCE2 = st.number_input("Entrance 2 Score", min_value=0, max_value=1500)
   FIRST_TERM = st.selectbox("First Term Joining Year", ["200508", "200608", "200708", "200808", "200908", "201008"])
   HOUSING_STS = st.selectbox("Housing Status", ["On Campus", "Off Campus"])
   DISTANCE_FROM_HOME = st.number_input("Distance From Home", min_value=0, max_value=6000)
   HIGH_SCHL_GPA = st.number_input("High School GPA", min_value=0, max_value=4)
   FATHER_HI_EDU_CD = st.selectbox("Father Education Code", ["1", "2", "3", "4"])
   MOTHER_HI_EDU_CD = st.selectbox("Mother Education Code", ["1", "2", "3", "4"])
   DEGREE_GROUP_DESC = st.selectbox("Degree Group Description", ["Associate", "Bachelor", "Career Associate"])
   COST_OF_ATTEND = st.number_input("Cost Of Attending", min_value=0, max_value=2200000)
   UNMET_NEED = st.number_input("Unmet Need", value=0, min_value=-1500000, max_value=1700000, step=1000)
   FIRST_YR_GPA = st.number_input("First Year GPA", min_value=0, max_value=4)
   FIRST_YR_EFFICIENCY = st.number_input("First Year Efficiency", min_value=0, max_value=2)
   SUBMITTED = st.form_submit_button("Predict")

if SUBMITTED:
    input_data = pd.DataFrame([{
        'STDNT_AGE': STDNT_AGE,
        'STDNT_GENDER': STDNT_GENDER,
        'STDNT_BACKGROUND': STDNT_BACKGROUND,
        'IN_STATE_FLAG': IN_STATE_FLAG,
        'INTERNATIONAL_STS': INTERNATIONAL_STS,
        'STDNT_MAJOR': STDNT_MAJOR,
        'STDNT_MINOR': STDNT_MINOR,
        'STDNT_TEST_ENTRANCE1': STDNT_TEST_ENTRANCE1,
        'STDNT_TEST_ENTRANCE2': STDNT_TEST_ENTRANCE2,
        'FIRST_TERM': FIRST_TERM,
        'HOUSING_STS': HOUSING_STS,
        'DISTANCE_FROM_HOME': DISTANCE_FROM_HOME,
        'HIGH_SCHL_GPA': HIGH_SCHL_GPA,
        'FATHER_HI_EDU_CD': FATHER_HI_EDU_CD,
        'MOTHER_HI_EDU_CD': MOTHER_HI_EDU_CD,
        'DEGREE_GROUP_DESC': DEGREE_GROUP_DESC,
        'COST_OF_ATTEND': COST_OF_ATTEND, 
        'UNMET_NEED': UNMET_NEED,
        'FIRST_YR_GPA': FIRST_YR_GPA,
        'FIRST_YR_EFFICIENCY': FIRST_YR_EFFICIENCY,
    }])

    df = pd.read_csv("student_application_and_performance_data.csv")
    df = df.drop(columns='RETURNED_2ND_YR', errors='ignore')
    full_data = pd.concat([df.iloc[:50], input_data], ignore_index=True)

    X = full_data
    input_row = X.tail(1)
    X = X.iloc[:-1]

    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())]), num_cols),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
    ])

    X_trans = preprocessor.fit_transform(X)
    input_trans = preprocessor.transform(input_row)

    y_dummy = pd.read_csv("student_application_and_performance_data.csv")['RETURNED_2ND_YR'][:50]
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, solver='liblinear', random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    }
    stacking_model = StackingClassifier(estimators=[(n, m) for n, m in models.items()], final_estimator=LogisticRegression())
    models["Stacking Classifier"] = stacking_model

    st.subheader("🔮 Prediction Results")
    for name, model in models.items():
        model.fit(X_trans, y_dummy)
        pred = model.predict(input_trans)[0]
        st.write(f"**{name}** predicts: {'✅ This Student will Return' if pred == 1 else '❌ This Student will not Return'}")
