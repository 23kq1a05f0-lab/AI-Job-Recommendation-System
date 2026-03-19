import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Job Recommender", page_icon="💼", layout="centered")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #1f4037, #99f2c8);
}

.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: white;
}

.subtitle {
    text-align: center;
    color: #f0f0f0;
    margin-bottom: 30px;
}

label {
    color: white !important;
    font-weight: bold;
}

textarea, input {
    background-color: white !important;
    color: black !important;
    border-radius: 10px !important;
    padding: 10px !important;
}

div.stButton > button {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    border-radius: 10px;
    height: 45px;
    font-size: 16px;
    width: 100%;
}

.section {
    color: white;
    font-size: 24px;
    margin-top: 20px;
}

.job-card {
    background: white;
    color: black;
    padding: 20px;
    border-radius: 15px;
    margin-top: 15px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.3);
}

.skill {
    background: #0072ff;
    color: white;
    padding: 5px 10px;
    border-radius: 8px;
    margin: 3px;
    display: inline-block;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
jobs = pd.read_csv("jobs_dataset.csv")

# ---------------- HEADER ----------------
st.markdown('<div class="title">💼 AI Job Recommendation System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Find the best job based on your skills</div>', unsafe_allow_html=True)

# ---------------- INPUT ----------------
skills = st.text_area("Enter your skills (comma separated)")
exp = st.number_input("Years of experience", min_value=0, max_value=20)

# ---------------- BUTTON ----------------
if st.button("🔍 Recommend Jobs"):

    # Reset index (fix index error)
    jobs_filtered = jobs[jobs["Min_Exp"] <= exp].copy().reset_index(drop=True)

    if jobs_filtered.empty:
        st.warning("No jobs match your experience level")

    else:
        # ---------------- CLEAN INPUT ----------------
        user_skills = [s.strip().lower() for s in skills.split(",") if s.strip() != ""]

        jobs_filtered["Skills"] = jobs_filtered["Skills"].str.lower()

        # ---------------- TF-IDF ----------------
        text_data = jobs_filtered["Skills"].tolist()
        text_data.append(" ".join(user_skills))

        tfidf = TfidfVectorizer()
        matrix = tfidf.fit_transform(text_data)

        similarity = cosine_similarity(matrix[-1], matrix[:-1])[0]

        # ---------------- COMBINED SCORE ----------------
        final_scores = []

        for idx, row in jobs_filtered.iterrows():

            job_skills = [s.strip() for s in row["Skills"].split(",")]

            # manual matching
            match_count = len(set(user_skills) & set(job_skills))
            manual_score = match_count / len(job_skills)

            # combine TF-IDF + manual
            final_score = (0.7 * similarity[idx]) + (0.3 * manual_score)

            final_scores.append(final_score)

        jobs_filtered["Score"] = final_scores

        # ---------------- SORT ----------------
        top = jobs_filtered.sort_values(by="Score", ascending=False).head(3)

        # ---------------- OUTPUT ----------------
        st.markdown('<div class="section">🎯 Recommended Jobs</div>', unsafe_allow_html=True)

        for _, row in top.iterrows():

            score = round(row["Score"] * 100, 2)

            st.markdown(f"""
            <div class="job-card">
                <h3>💼 {row['Job_Title']}</h3>
                <p><b>💰 Salary:</b> {row['Salary']}</p>
                <p><b>📊 Match Score:</b> {score}%</p>
            </div>
            """, unsafe_allow_html=True)

            st.progress(score / 100)

            # Skills display
            st.write("🧠 Required Skills:")

            skills_list = row["Skills"].split(",")

            skill_html = ""
            for s in skills_list:
                skill_html += f'<span class="skill">{s.strip().upper()}</span>'

            st.markdown(skill_html, unsafe_allow_html=True)

            st.divider()
