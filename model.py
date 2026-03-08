import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

jobs = pd.read_csv("jobs_dataset.csv")

def recommend_jobs(user_skills, user_exp):

    # Filter by experience
    jobs_filtered = jobs[jobs["Min_Exp"] <= user_exp]

    if jobs_filtered.empty:
        return ["No job matches your experience"]

    # Combine text
    text_data = jobs_filtered["Skills"].tolist()
    text_data.append(user_skills)

    # TF-IDF
    tfidf = TfidfVectorizer()
    matrix = tfidf.fit_transform(text_data)

    similarity = cosine_similarity(matrix[-1], matrix[:-1])

    scores = similarity[0]
    jobs_filtered["Score"] = scores

    top = jobs_filtered.sort_values(by="Score", ascending=False).head(3)

    return top[["Job_Title","Salary","Score"]]

