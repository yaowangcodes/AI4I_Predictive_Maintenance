
import os
import streamlit as st
import requests


# ==========================================
# 1. API Configuration
# ==========================================

API_URL = os.getenv(
    "API_URL",
    "http://localhost:8000"
)


# ==========================================
# 2. Page Configuration
# ==========================================

st.set_page_config(
    page_title="AI4I Predictive Maintenance",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ AI4I Predictive Maintenance")

st.caption(
    "Machine Learning-Based Industrial Failure Prediction"
)

st.divider()

st.write(
    "Enter the machine operating conditions below "
    "to predict potential machine failure."
)


# ==========================================
# 3. User Input Form
# ==========================================

with st.form("prediction_form"):

    st.subheader("Machine Operating Conditions")

    col1, col2 = st.columns(2)

    with col1:

        product_type = st.selectbox(
            "Product Type",
            options=["L", "M", "H"],
            index=1
        )

        air_temperature = st.number_input(
            "Air Temperature (K)",
            min_value=0.01,
            value=298.1,
            step=0.1
        )

        process_temperature = st.number_input(
            "Process Temperature (K)",
            min_value=0.01,
            value=308.6,
            step=0.1
        )

    with col2:

        rotational_speed = st.number_input(
            "Rotational Speed (rpm)",
            min_value=0,
            value=1551,
            step=1
        )

        torque = st.number_input(
            "Torque (Nm)",
            min_value=0.0,
            value=42.8,
            step=0.1
        )

        tool_wear = st.number_input(
            "Tool Wear (min)",
            min_value=0,
            value=120,
            step=1
        )

    submitted = st.form_submit_button(
        "Predict Machine Failure",
        use_container_width=True,
        type="primary"
    )


# ==========================================
# 4. Connect to FastAPI
# ==========================================

if submitted:

    payload = {
    "Type": product_type,
    "air_temperature": air_temperature,
    "process_temperature": process_temperature,
    "rotational_speed": rotational_speed,
    "torque": torque,
    "tool_wear": tool_wear
}

    try:

        with st.spinner("Analyzing machine conditions..."):

            response = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=15
            )

        response.raise_for_status()

        result = response.json()


        # ==================================
        # 5. Display Prediction Results
        # ==================================

        st.divider()
        st.subheader("Prediction Results")

        prediction = result.get("prediction")

        probability = result.get(
            "failure_probability",
            result.get("probability")
        )

        model_version = result.get(
            "model_version",
            "Not provided"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            if prediction == 0:

                st.metric(
                    "Prediction",
                    "Normal"
                )

            elif prediction == 1:

                st.metric(
                    "Prediction",
                    "Failure"
                )

            else:

                st.metric(
                    "Prediction",
                    "Unknown"
                )

        with col2:

            if probability is not None:

                st.metric(
                    "Failure Probability",
                    f"{float(probability):.2%}"
                )

            else:

                st.metric(
                    "Failure Probability",
                    "N/A"
                )

        with col3:

            st.metric(
                "Model Version",
                str(model_version)
            )


        # ==================================
        # 6. Display Risk Status
        # ==================================

        if prediction == 1:

            st.error(
                "⚠️ Potential machine failure detected."
            )

            st.warning(
                "Equipment inspection is recommended."
            )

        elif prediction == 0:

            st.success(
                "✅ Normal operation predicted."
            )

        else:

            st.warning(
                "The API returned an unrecognized prediction."
            )

        if probability is not None:

            probability = float(probability)

            if 0 <= probability <= 1:

                st.write("Estimated Failure Risk")

                st.progress(probability)


        # ==================================
        # 7. Display Technical Details
        # ==================================

        with st.expander("API Request Details"):

            st.json(payload)

        with st.expander("API Response Details"):

            st.json(result)


    # ======================================
    # 8. Error Handling
    # ======================================

    except requests.exceptions.ConnectionError:

        st.error(
            "Cannot connect to FastAPI. "
            "Please check whether the API is running."
        )

    except requests.exceptions.Timeout:

        st.error(
            "The prediction request timed out."
        )

    except requests.exceptions.HTTPError:

        st.error(
            f"API Error: {response.status_code}"
        )

        st.write(response.text)

    except Exception as e:

        st.error(f"Unexpected error: {e}")


# ==========================================
# 9. Sidebar
# ==========================================

with st.sidebar:

    st.header("System Information")

    st.write("**Project:** AI4I Predictive Maintenance")

    st.write("**Course:** INSY 684")

    st.write("**Model Serving:** FastAPI")

    st.write("**Model Registry:** MLflow")

    st.write("**Frontend:** Streamlit")

    st.write("**Database:** PostgreSQL")

    st.divider()

    st.caption("API Endpoint")

    st.code(f"{API_URL}/predict")

    if st.button("Check API Connection"):

        try:

            response = requests.get(
                f"{API_URL}/health",
                timeout=5
            )

            if response.status_code == 200:

                st.success("API Connected")

            else:

                st.warning(
                    f"API returned {response.status_code}"
                )

        except requests.exceptions.RequestException:

            st.error("API Disconnected")


st.divider()

st.caption(
    "INSY 684 | End-to-End MLOps System | AI4I Dataset"
)
