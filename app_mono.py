import streamlit as st
import pandas as pd
import joblib

# Load kedua model (Classification & Regression)
# Pastikan path file sesuai dengan folder di repo GitHub lu nanti
model_clf = joblib.load("artifacts/placement_prediction_pipeline.pkl")
model_reg = joblib.load("artifacts/salary_prediction_pipeline.pkl")

def main():
    st.set_page_config(page_title="Placement & Salary Predictor", layout="wide")
    st.title("Student Placement & Salary Prediction")

    # Sidebar untuk informasi bantuan
    with st.sidebar:
        st.header("Panduan Input")
        st.info("""
        Isi data profil mahasiswa di form utama. 
        Sistem akan memprediksi status kelulusan 
        dan estimasi paket gaji tahunan (LPA).
        """)
        
        st.divider()
        st.write("**Model Performance**")
        st.write("- Placement Accuracy: ~1.00")
        st.write("- Salary R2 Score: ~0.02")

    # Input Form
    with st.form("prediction_form"):
        st.subheader("Profil Akademik & Skill")
        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            ssc = st.slider("SSC Percentage", 0.0, 100.0, 75.0)
            hsc = st.slider("HSC Percentage", 0.0, 100.0, 75.0)
            degree = st.slider("Degree Percentage", 0.0, 100.0, 70.0)
            cgpa = st.number_input("CGPA", 0.0, 10.0, 7.5)
            attendance = st.slider("Attendance (%)", 0, 100, 85)
            backlogs = st.number_input("Backlogs", 0, 10, 0)
            certs = st.number_input("Certifications Count", 0, 20, 1)

        with col2:
            tech_score = st.slider("Technical Skill Score", 0, 100, 75)
            soft_score = st.slider("Soft Skill Score", 0, 100, 75)
            entrance = st.slider("Entrance Exam Score", 0, 100, 70)
            internships = st.number_input("Internship Count", 0, 10, 1)
            projects = st.number_input("Live Projects", 0, 10, 1)
            work_exp = st.number_input("Work Experience (Months)", 0, 60, 0)
            extra = st.selectbox("Extracurricular Activities", ["Yes", "No"])

        submitted = st.form_submit_button("Predict Placement & Salary")

    if submitted:
        # 1. Menyiapkan data awal sesuai input form
        data = {
            'ssc_percentage': ssc,
            'hsc_percentage': hsc,
            'degree_percentage': degree,
            'cgpa': cgpa,
            'entrance_exam_score': entrance,
            'technical_skill_score': tech_score,
            'soft_skill_score': soft_score,
            'internship_count': internships,
            'live_projects': projects,
            'work_experience_months': work_exp,
            'certifications': certs,
            'attendance_percentage': float(attendance),
            'backlogs': backlogs,
            'gender': gender,
            'extracurricular_activities': extra
        }
        
        df_input = pd.DataFrame([data])

        # 2. WAJIB: Tambahkan Feature Engineering agar sama dengan saat Training
        df_input['avg_academic'] = (df_input['ssc_percentage'] + df_input['hsc_percentage'] + df_input['degree_percentage']) / 3
        df_input['total_skill'] = (df_input['technical_skill_score'] + df_input['soft_skill_score']) / 2
        df_input['experience_idx'] = (df_input['internship_count'] * 2) + df_input['live_projects'] + (df_input['work_experience_months'] / 3) + df_input['certifications']
        df_input['has_backlogs'] = (df_input['backlogs'] > 0).astype(int)

        # 3. Prediksi Klasifikasi (Placement)
        prediction_clf = model_clf.predict(df_input)[0]
        probability = model_clf.predict_proba(df_input)[0]

        st.divider()
        st.subheader("Hasil Analisis")

        if prediction_clf == 1:
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                st.success("Status: **PLACED**")
                st.metric("Confidence", f"{max(probability)*100:.2f}%")
            
            with res_col2:
                prediction_reg = model_reg.predict(df_input)[0]
                st.metric("Estimasi Paket Gaji", f"{prediction_reg:.2f} LPA")
            
            st.progress(float(max(probability))) # Bar hijau di bawahnya
            st.write("Mahasiswa diprediksi memenuhi kriteria industri.")
        else:
            st.error(f"Status: **NOT PLACED**")
            st.metric("Confidence", f"{max(probability)*100:.2f}%")
            st.progress(float(max(probability))) # Bar merah/orange di bawahnya
            st.warning("Estimasi gaji tidak tersedia untuk status prediksi Not Placed.")

        # TARUH DI SINI: Expander buat bukti kalau lu paham data
        with st.expander("ℹ️ Lihat Detail Logika Prediksi"):
            st.write("Data yang diproses oleh model (Feature Engineering):")
            st.dataframe(df_input) 
            st.write("Raw Probability Score:", probability)
            
if __name__ == "__main__":
    main()
