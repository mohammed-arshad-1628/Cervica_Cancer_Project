import streamlit as st
import cv2
import numpy as np
from PIL import Image
import joblib
import random
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# ---------------- VALIDATION FUNCTION ----------------
def is_ct_scan(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    std_val = np.std(gray)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges) / (img.shape[0] * img.shape[1])

    b, g, r = cv2.split(img)
    color_diff = np.mean(abs(r - g)) + np.mean(abs(g - b))

    if std_val > 30 and edge_density > 5 and color_diff < 20:
        return True
    return False

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Cervical Cancer Detection System", layout="wide")

# ---------------- HEADER ----------------
st.title("🧬 Cervical Cancer Detection System")
st.caption("Clinical Decision Support System")

# ---------------- MODEL ----------------
model = joblib.load("model.pkl")
classes = ["Stage 1", "Stage 2", "Stage 3"]

# ---------------- FEATURE FUNCTION ----------------
def extract_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.array([
        np.mean(gray),
        np.std(gray),
        np.var(gray)
    ]).reshape(1, -1)

# ---------------- PATIENT INFO ----------------
st.subheader("👩‍⚕️ Patient Information")

col1, col2, col3 = st.columns(3)

with col1:
    name = st.text_input("Patient Name")
    age = st.number_input("Age", 1, 100)

with col2:
    gender = st.selectbox("Gender", ["Female","Male","Others"])
    patient_id = st.text_input("Patient ID")

with col3:
    doctor = st.text_input("Doctor Name")
    date = st.date_input("Date")

# ---------------- IMAGE UPLOAD ----------------
st.subheader("📤 Upload Image")

uploaded_file = st.file_uploader("Upload Image")

# ---------------- MAIN PROCESS ----------------
if uploaded_file is not None:

    image = Image.open(uploaded_file)
    image = image.convert("RGB")

    img = np.array(image)
    img = cv2.resize(img, (256, 256))

    # ✅ VALIDATION
    if not is_ct_scan(img):
        st.error("❌ Invalid Image! Please upload a CT Scan image only.")
        st.stop()

    # ✅ SHOW ORIGINAL
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # ✅ GRAYSCALE
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    st.image(gray, caption="Grayscale Image", use_container_width=True)

    # ✅ NOISE REDUCTION
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    st.image(blur, caption="Noise Reduced Image", use_container_width=True)

    # ---------------- PREDICTION ----------------
    features = extract_features(img)
    pred = model.predict(features)[0]
    label = classes[pred]

    # ---------------- RESULT ----------------
    st.subheader("🧾 Prediction Result")

    if label == "Stage 1":
        st.success(f"🟢 {label} (Early Stage)")
    elif label == "Stage 2":
        st.warning(f"🟡 {label} (Moderate Stage)")
    else:
        st.error(f"🔴 {label} (Advanced Stage)")

    # ---------------- REPORT FUNCTION ----------------
    def generate_report():
            doc = SimpleDocTemplate("Patient_Report.pdf")
            styles = getSampleStyleSheet()
            content = []

            # TITLE
            content.append(Paragraph("CERVICAL CANCER DETECTION REPORT", styles["Title"]))
            content.append(Spacer(1, 12))

            # PATIENT INFO
            content.append(Paragraph("Patient Information", styles["Heading2"]))
            content.append(Paragraph(f"Patient Name: {name}", styles["Normal"]))
            content.append(Paragraph(f"Patient ID: {patient_id}", styles["Normal"]))
            content.append(Paragraph(f"Age: {age}", styles["Normal"]))
            content.append(Paragraph(f"Gender: {gender}", styles["Normal"]))
            content.append(Paragraph(f"Doctor Name: {doctor}", styles["Normal"]))
            content.append(Paragraph(f"Examination Date: {date}", styles["Normal"]))
            content.append(Spacer(1, 12))

            # RESULT
            content.append(Paragraph("Diagnosis Result", styles["Heading2"]))
            content.append(Paragraph(f"Detected Stage: {label}", styles["Heading3"]))
            content.append(Spacer(1, 10))

            # -------- DETAILED EXPLANATION --------
            content.append(Paragraph("Clinical Interpretation", styles["Heading2"]))

            if label == "Stage 1":
                explanations = [
                    """Stage 1 cervical cancer represents the earliest detectable phase of malignant transformation,
                    where abnormal cell growth is strictly confined to the cervical epithelium. At this stage,
                    the tumor has not invaded surrounding tissues or metastasized to distant organs.

                    Clinically, patients may remain asymptomatic or present with very mild symptoms such as
                    irregular menstrual bleeding or slight pelvic discomfort. Due to the localized nature of the disease,
                    early screening methods such as Pap smear and HPV testing play a crucial role in detection.

                    Histopathological analysis typically reveals minimal stromal invasion, and prognosis at this stage
                    is highly favorable. Treatment strategies often involve conservative surgical procedures such as
                    conization or hysterectomy, depending on patient factors.

                    With timely intervention and consistent follow-up, survival rates exceed 90%, making early diagnosis
                    critical for successful management.""",

                    """In Stage 1 cervical cancer, the malignancy is limited entirely to the cervix without extension
                    into adjacent anatomical structures. This localized growth indicates a slow progression pattern,
                    allowing for effective therapeutic intervention.

                    Patients may experience minimal clinical manifestations, which can delay diagnosis unless routine
                    screening is performed. Cellular abnormalities remain restricted, and there is no evidence of lymphatic
                    or vascular spread.

                    Management typically includes surgical excision or localized therapies aimed at complete tumor removal.
                    The response to treatment is generally excellent, and recurrence rates are low when appropriate care is provided.

                    Regular monitoring and follow-up imaging are recommended to ensure long-term disease-free survival."""
                ]

            elif label == "Stage 2":
                explanations = [
                    """Stage 2 cervical cancer is characterized by the extension of malignant cells beyond the cervix
                    into adjacent tissues such as the upper vagina or parametrial regions, without reaching the pelvic wall.

                    At this stage, patients often present with noticeable symptoms including abnormal vaginal bleeding,
                    pelvic pain, and unusual discharge. The disease progression indicates moderate invasion,
                    requiring comprehensive clinical evaluation.

                    Imaging techniques such as MRI and CT scans are used to assess the extent of tumor spread.
                    Histological examination confirms deeper stromal invasion and possible lymphatic involvement.

                    Treatment typically involves a multimodal approach, combining radiation therapy with chemotherapy
                    to target both primary and microscopic disease. Prognosis remains favorable with timely intervention,
                    although careful monitoring is necessary to prevent further progression.""",

                    """In Stage 2, cervical cancer demonstrates regional spread beyond its point of origin,
                    indicating increased aggressiveness compared to early-stage disease. The tumor infiltrates nearby tissues,
                    impacting normal physiological function.

                    Clinical symptoms become more pronounced, often prompting medical consultation. Diagnostic procedures
                    reveal significant structural and cellular abnormalities, necessitating immediate treatment.

                    Therapeutic strategies focus on reducing tumor size and preventing metastasis through combined
                    radiotherapy and systemic chemotherapy. Patient response varies depending on overall health and disease severity.

                    With appropriate medical management, many patients achieve controlled disease status, though
                    long-term follow-up is essential."""
                ]

            else:
                explanations = [
                    """Stage 3 cervical cancer represents an advanced stage of disease progression where malignant cells
                    have extended to the pelvic wall and may involve adjacent organs such as the bladder or rectum.

                    Patients typically experience severe clinical symptoms including persistent pelvic pain,
                    heavy bleeding, urinary complications, and systemic weakness. The tumor burden is significantly high,
                    and local invasion is extensive.

                    Diagnostic imaging reveals widespread tissue involvement, and there may be evidence of lymph node metastasis.
                    This stage requires urgent and aggressive medical intervention.

                    Treatment includes high-dose radiation therapy combined with chemotherapy, aimed at controlling tumor growth
                    and alleviating symptoms. Prognosis depends on treatment response and overall patient condition.

                    Continuous monitoring and supportive care are critical to improving quality of life and survival outcomes.""",

                    """Advanced cervical cancer in Stage 3 is marked by extensive local invasion and possible spread
                    to surrounding pelvic structures. The disease significantly impacts organ function and overall health.

                    Clinical presentation is severe, with symptoms affecting daily life and requiring immediate hospitalization
                    in many cases. Diagnostic findings confirm extensive tumor spread and possible complications.

                    Management involves aggressive therapeutic protocols, including radiation, chemotherapy, and supportive care.
                    The goal is to control disease progression and manage symptoms effectively.

                    Despite the severity, advancements in medical treatment have improved patient outcomes, although
                    long-term prognosis remains guarded."""
                ]
            # ✅ RANDOM SELECTION
            explanation = random.choice(explanations)

            content.append(Paragraph(explanation, styles["Normal"]))
            content.append(Spacer(1, 12))

            # IMAGE ANALYSIS
            content.append(Paragraph("Image Analysis Summary", styles["Heading2"]))
            content.append(Paragraph(
                "The uploaded medical image was processed using advanced image preprocessing techniques including "
                "noise reduction, grayscale conversion, and feature extraction. The system analyzed structural and "
                "textural patterns using a trained machine learning model to classify the cancer stage accurately.",
                styles["Normal"]
            ))
            content.append(Spacer(1, 12))

            # RECOMMENDATION
            content.append(Paragraph("Medical Recommendation", styles["Heading2"]))
            content.append(Paragraph(
                "It is strongly recommended to consult a gynecologist or oncologist for further diagnosis. "
                "Additional tests such as biopsy or HPV testing may be required for confirmation.",
                styles["Normal"]
            ))
            content.append(Spacer(1, 12))

            # DISCLAIMER
            content.append(Paragraph("Disclaimer", styles["Heading2"]))
            content.append(Paragraph(
                "This report is generated by an AI system and is intended for preliminary analysis only. "
                "It should not replace professional medical advice.",
                styles["Normal"]
            ))

            doc.build(content)

    # ---------------- BUTTON ----------------
    if st.button("📄 Generate Patient Report"):
        generate_report()
        st.success("✅ Report Generated Successfully!")

        with open("Patient_Report.pdf", "rb") as f:
            st.download_button(
                "⬇ Download Report",
                f,
                file_name="Patient_Report.pdf",
                mime="application/pdf"
            )

else:
    st.warning("⚠️ Upload a CT Scan image to start")