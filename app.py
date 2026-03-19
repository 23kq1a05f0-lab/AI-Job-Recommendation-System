import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Job Recommender", page_icon="💼", layout="centered")



# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #141e30, #243b55);
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: white;
}

.subtitle {
    text-align: center;
    color: #d1d1d1;
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
}

div.stButton > button {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    border-radius: 10px;
    height: 45px;
    width: 100%;
}

.job-card {
    background: white;
    color: black;
    padding: 20px;
    border-radius: 15px;
    margin-top: 15px;
    box-shadow: 0 5px 15px rgba(0,0,0,0.3);
}

.skill {
    background: #0072ff;
    color: white;
    padding: 5px 10px;
    border-radius: 8px;
    margin: 3px;
    display: inline-block;
}

.badge {
    background: gold;
    color: black;
    padding: 5px 10px;
    border-radius: 8px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
jobs = pd.read_csv("jobs_dataset.csv")

# ---------------- HEADER ----------------
st.markdown('<div class="title">💼 AI Job Recommendation System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Smart job suggestions based on your skills</div>', unsafe_allow_html=True)

# ---------------- INPUT ----------------
skills = st.text_area("Enter your skills (comma or space separated)")
exp = st.number_input("Years of experience", min_value=0, max_value=20)

# ---------------- BUTTON ----------------
if st.button("🔍 Recommend Jobs"):

    jobs_filtered = jobs[jobs["Min_Exp"] <= exp].copy().reset_index(drop=True)

    if jobs_filtered.empty:
        st.warning("No jobs match your experience level")

    else:
        # -------- FLEXIBLE INPUT CLEANING --------
        user_skills = re.split(r"[,\s]+", skills.lower())
        user_skills = [s.strip() for s in user_skills if s.strip()]

        jobs_filtered["Skills"] = jobs_filtered["Skills"].str.lower()

        text_data = jobs_filtered["Skills"].tolist()
        text_data.append(" ".join(user_skills))

        tfidf = TfidfVectorizer()
        matrix = tfidf.fit_transform(text_data)

        similarity = cosine_similarity(matrix[-1], matrix[:-1])[0]

        final_scores = []
        missing_skills_list = []

        for idx, row in jobs_filtered.iterrows():

            # -------- FLEXIBLE JOB SKILLS --------
            job_skills = re.split(r"[,\s]+", row["Skills"])
            job_skills = [s.strip() for s in job_skills if s.strip()]

            match = set(user_skills) & set(job_skills)
            missing = set(job_skills) - set(user_skills)

            manual_score = len(match) / len(job_skills)
            final_score = (0.7 * similarity[idx]) + (0.3 * manual_score)

            final_scores.append(final_score)
            missing_skills_list.append(list(missing))

        jobs_filtered["Score"] = final_scores
        jobs_filtered["Missing"] = missing_skills_list

        top = jobs_filtered.sort_values(by="Score", ascending=False).head(3)

        st.subheader("🎯 Recommended Jobs")

        for i, row in top.iterrows():

            score = round(row["Score"] * 100, 2)

            # Color logic
            if score > 70:
                color = "green"
            elif score > 40:
                color = "orange"
            else:
                color = "red"

            badge = ""
            if i == top.index[0]:
                badge = '<span class="badge">🏆 Top Match</span>'

            st.markdown(f"""
            <div class="job-card">
                <h3>💼 {row['Job_Title']} {badge}</h3>
                <p><b>💰 Salary:</b> {row['Salary']}</p>
                <p><b style="color:{color};">📊 Match Score: {score}%</b></p>
            </div>
            """, unsafe_allow_html=True)

            st.progress(score / 100)

            # Skills
            st.write("🧠 Required Skills:")
            skills_html = ""
            for s in row["Skills"].split(","):
                skills_html += f'<span class="skill">{s.strip().upper()}</span>'
            st.markdown(skills_html, unsafe_allow_html=True)

            # Missing Skills
            if row["Missing"]:
                st.write("💡 Improve by learning:")
                miss_html = ""
                for m in row["Missing"]:
                    miss_html += f'<span class="skill" style="background:red;">{m.upper()}</span>'
                st.markdown(miss_html, unsafe_allow_html=True)

            st.divider()
