import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
jobs = pd.read_csv("jobs_dataset.csv")

st.title("🤖 AI Job Recommendation System")

skills = st.text_area("Enter your skills")
exp = st.number_input("Years of experience", min_value=0, max_value=20, step=1)

if st.button("Recommend Jobs"):

    # Filter by experience
    jobs_filtered = jobs[jobs["Min_Exp"] <= exp]

    if jobs_filtered.empty:
        st.warning("No jobs match your experience level")
    else:

        text_data = jobs_filtered["Skills"].tolist()
        text_data.append(skills)

        tfidf = TfidfVectorizer()
        matrix = tfidf.fit_transform(text_data)

        similarity = cosine_similarity(matrix[-1], matrix[:-1])
        scores = similarity[0]

        jobs_filtered["Score"] = scores

        top = jobs_filtered.sort_values(by="Score", ascending=False).head(3)

        st.subheader("🎯 Recommended Jobs")

        for _,row in top.iterrows():
            st.markdown(f"""
            ### 💼 {row['Job_Title']}
            **Salary:** {row['Salary']}  
            **Match Score:** {round(row['Score'],2)}
            """)
