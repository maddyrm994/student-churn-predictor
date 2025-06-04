import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt

# MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="Student Churn Prediction App", layout="wide")

# Sidebar for mode selection
st.sidebar.title("Prediction Modes")
mode = st.sidebar.radio("Select Prediction Mode", ["Form-Based Prediction", "Bulk Upload Prediction"])

# ------------------- FORM-BASED PREDICTION ------------------------
if mode == "Form-Based Prediction":
    st.title("🎓 Predict Student Churn (Form-Based)")

    st.markdown("📋 Fill out the student information below:")
    with st.form("student_form"):
        STDNT_AGE = st.number_input("Age", min_value=16, max_value=26)
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
        HIGH_SCHL_GPA = st.number_input("High School GPA", min_value=0.0, max_value=4.0)
        FATHER_HI_EDU_CD = st.selectbox("Father Education Code", ["1", "2", "3", "4"])
        MOTHER_HI_EDU_CD = st.selectbox("Mother Education Code", ["1", "2", "3", "4"])
        DEGREE_GROUP_DESC = st.selectbox("Degree Group Description", ["Associate", "Bachelor", "Career Associate"])
        COST_OF_ATTEND = st.number_input("Cost Of Attending", min_value=0, max_value=2200000)
        UNMET_NEED = st.number_input("Unmet Need", value=0, min_value=-1500000, max_value=1700000, step=1000)
        FIRST_YR_GPA = st.number_input("First Year GPA", min_value=0.0, max_value=4.0)
        FIRST_YR_EFFICIENCY = st.number_input("First Year Efficiency", min_value=0.0, max_value=2.0)
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

        preprocessor = ColumnTransformer([
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
        #stacking_model = StackingClassifier(estimators=[(n, m) for n, m in models.items()], final_estimator=LogisticRegression())
        #models["Stacking Classifier"] = stacking_model

        st.subheader("🔮 Prediction Results")
        cols = st.columns(2)
        ohe = preprocessor.named_transformers_['cat']['encoder']
        encoded_cat_cols = ohe.get_feature_names_out(cat_cols)
        feature_names = num_cols + list(encoded_cat_cols)

        for idx, (name, model) in enumerate(models.items()):
            with cols[idx % 2]:
                model.fit(X_trans, y_dummy)
                pred = model.predict(input_trans)[0]
                proba = model.predict_proba(input_trans)[0][1]

                msg = f"**{name}**\n\n🔢 Confidence: `{proba:.2f}`\n\n"
                if pred == 1:
                    msg += "✅ This Student will Return"
                    st.success(msg)
                else:
                    msg += "❌ This Student will Not Return\n"
                    reason = "N/A"
                    try:
                        if hasattr(model, "coef_"):
                            weights = model.coef_[0]
                            contribs = input_trans[0] * weights
                            top_idx = np.argsort(contribs)[0]
                            reason = f"{feature_names[top_idx]} (contribution={contribs[top_idx]:.2f})"
                        elif hasattr(model, "feature_importances_"):
                            importances = model.feature_importances_
                            input_values = input_trans[0]
                            top_idx = np.argmax(importances * np.abs(input_values))
                            reason = f"{feature_names[top_idx]} (importance={importances[top_idx]:.2f})"
                    except Exception as e:
                        reason = f"Could not determine (error: {str(e)})"
                    msg += f"\n\n📉 **Reason for not returning:** `{reason}`"
                    st.error(msg)

# ------------------- BULK UPLOAD PREDICTION ------------------------
else:
    st.title("🎓 Predict Student Churn (Bulk Upload)")
    uploaded_file = st.file_uploader("📂 Upload your dataset (CSV file)", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        drop_cols = ['STDNT_ID', 'CORE_COURSE_GRADE_X_F']
        df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')
        df = df.dropna(subset=['RETURNED_2ND_YR'])

        X = df.drop(columns=['RETURNED_2ND_YR'])
        y = df['RETURNED_2ND_YR']
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

        preprocessor = ColumnTransformer([
            ('num', Pipeline([('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())]), numerical_cols),
            ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), categorical_cols)
        ])

        X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train = preprocessor.fit_transform(X_train_raw)
        X_test = preprocessor.transform(X_test_raw)

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "Gradient Boosting": GradientBoostingClassifier()
        }
        #stacking_model = StackingClassifier(estimators=[(n, m) for n, m in models.items()], final_estimator=LogisticRegression())
        #models["Stacking Classifier"] = stacking_model

        results = []
        predictions = {}

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            predictions[name] = y_pred
            results.append({
                "Model": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred),
                "Recall": recall_score(y_test, y_pred),
                "ROC AUC": roc_auc_score(y_test, y_pred)
            })

        result_df = pd.DataFrame(results).set_index("Model")

        st.subheader("📊 Model Performance Metrics")
        st.dataframe(result_df.style.format("{:.2f}"))

        st.subheader("📈 Model Comparison")
        fig, ax = plt.subplots(figsize=(10, 4))
        result_df.plot(kind='bar', ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

        best_model = result_df["Accuracy"].idxmax()
        final_preds = predictions[best_model]
        returned = sum(final_preds)
        not_returned = len(final_preds) - returned

        st.success(f"✅ Using **{best_model}**, prediction summary on test data:")
        st.write(f"- 🎓 Students predicted to **return**: **{returned}**")
        st.write(f"- ❌ Students predicted **not to return**: **{not_returned}**")
