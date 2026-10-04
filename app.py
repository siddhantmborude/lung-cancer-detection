import streamlit as st
import pandas as pd
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Lung Cancer Risk Prediction",
    page_icon="🫁",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 30px;
}

.section-title {
    font-size: 24px;
    font-weight: bold;
    margin-top: 20px;
}

.result-box {
    padding: 25px;
    border-radius: 12px;
    text-align: center;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(
    "models/lung_cancer_model.pkl"
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🫁 Lung Cancer Risk Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning Based Healthcare Prediction System'
    '</div>',
    unsafe_allow_html=True
)

st.warning(
    "⚠️ This system is developed for academic and educational "
    "purposes only. It does not provide a medical diagnosis."
)


# ============================================================
# INPUT SECTIONS
# ============================================================

st.markdown(
    '<div class="section-title">👤 Patient Information</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

with col2:

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=50
    )

with col3:

    smoking = st.selectbox(
        "Smoking",
        ["No", "Yes"]
    )


# ============================================================
# LIFESTYLE FACTORS
# ============================================================

st.markdown(
    '<div class="section-title">🏃 Lifestyle & Medical Factors</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

with col1:

    yellow_fingers = st.selectbox(
        "Yellow Fingers",
        ["No", "Yes"]
    )

with col2:

    anxiety = st.selectbox(
        "Anxiety",
        ["No", "Yes"]
    )

with col3:

    peer_pressure = st.selectbox(
        "Peer Pressure",
        ["No", "Yes"]
    )

with col4:

    chronic_disease = st.selectbox(
        "Chronic Disease",
        ["No", "Yes"]
    )


col1, col2, col3, col4 = st.columns(4)

with col1:

    fatigue = st.selectbox(
        "Fatigue",
        ["No", "Yes"]
    )

with col2:

    allergy = st.selectbox(
        "Allergy",
        ["No", "Yes"]
    )

with col3:

    alcohol = st.selectbox(
        "Alcohol Consuming",
        ["No", "Yes"]
    )

with col4:

    wheezing = st.selectbox(
        "Wheezing",
        ["No", "Yes"]
    )


# ============================================================
# SYMPTOMS
# ============================================================

st.markdown(
    '<div class="section-title">🩺 Symptoms</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:

    coughing = st.selectbox(
        "Coughing",
        ["No", "Yes"]
    )

with col2:

    shortness_breath = st.selectbox(
        "Shortness of Breath",
        ["No", "Yes"]
    )

with col3:

    swallowing = st.selectbox(
        "Swallowing Difficulty",
        ["No", "Yes"]
    )


col1, col2, col3 = st.columns(3)

with col1:

    chest_pain = st.selectbox(
        "Chest Pain",
        ["No", "Yes"]
    )


# ============================================================
# ENCODING
# ============================================================

def yes_no(value):

    if value == "Yes":
        return 2

    return 1


gender_value = 1 if gender == "Male" else 0


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()

predict_button = st.button(
    "🔍  PREDICT LUNG CANCER RISK",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    input_data = pd.DataFrame({

        "GENDER": [gender_value],

        "AGE": [age],

        "SMOKING": [yes_no(smoking)],

        "YELLOW_FINGERS": [yes_no(yellow_fingers)],

        "ANXIETY": [yes_no(anxiety)],

        "PEER_PRESSURE": [yes_no(peer_pressure)],

        "CHRONIC DISEASE": [yes_no(chronic_disease)],

        "FATIGUE": [yes_no(fatigue)],

        "ALLERGY": [yes_no(allergy)],

        "WHEEZING": [yes_no(wheezing)],

        "ALCOHOL CONSUMING": [yes_no(alcohol)],

        "COUGHING": [yes_no(coughing)],

        "SHORTNESS OF BREATH": [
            yes_no(shortness_breath)
        ],

        "SWALLOWING DIFFICULTY": [
            yes_no(swallowing)
        ],

        "CHEST PAIN": [
            yes_no(chest_pain)
        ]
    })


    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    prediction = model.predict(
        input_data
    )[0]


    probabilities = model.predict_proba(
        input_data
    )[0]

    probability = probabilities[1]


    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    st.subheader("📊 Prediction Result")


    if prediction == 1:

        st.error(
            "⚠️ HIGH RISK CLASSIFICATION"
        )

        st.write(
            "The model classified this input as "
            "**Lung Cancer = YES**."
        )

    else:

        st.success(
            "✅ LOW RISK CLASSIFICATION"
        )

        st.write(
            "The model classified this input as "
            "**Lung Cancer = NO**."
        )


    # --------------------------------------------------------
    # PROBABILITY
    # --------------------------------------------------------

    st.metric(
        "Predicted Probability of Positive Classification",
        f"{probability * 100:.2f}%"
    )

    st.progress(
        float(probability)
    )


    # --------------------------------------------------------
    # INPUT SUMMARY
    # --------------------------------------------------------

    st.subheader("📋 Patient Input Summary")

    summary = pd.DataFrame({

        "Feature": [
            "Gender",
            "Age",
            "Smoking",
            "Yellow Fingers",
            "Anxiety",
            "Peer Pressure",
            "Chronic Disease",
            "Fatigue",
            "Allergy",
            "Wheezing",
            "Alcohol",
            "Coughing",
            "Shortness of Breath",
            "Swallowing Difficulty",
            "Chest Pain"
        ],

        "Value": [
            gender,
            age,
            smoking,
            yellow_fingers,
            anxiety,
            peer_pressure,
            chronic_disease,
            fatigue,
            allergy,
            wheezing,
            alcohol,
            coughing,
            shortness_breath,
            swallowing,
            chest_pain
        ]
    })

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # MODEL INFORMATION
    # --------------------------------------------------------

    st.subheader("🤖 Model Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Algorithm",
            "Logistic Regression"
        )

    with col2:

        st.metric(
            "Test Accuracy",
            "90.32%"
        )

    with col3:

        st.metric(
            "F1 Score",
            "94.55%"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Lung Cancer Risk Prediction | Machine Learning Mini Project"
)

st.caption(
    "For educational purposes only — not a substitute for professional medical advice."
)