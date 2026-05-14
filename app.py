import pandas as pd
import gradio as gr

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Loading Dataset

df= pd.read_csv(r"C:\Users\Siddharth\Desktop\pyp\ML Project\netflix_titles.csv")
df

# Handeling Missing Values

df['director'] = df['director'].fillna('Unknown')

df['cast'] = df['cast'].fillna('Unknown')

df['country'] = df['country'].fillna('Unknown')

# Createing Combined Features

df['combined_features'] = (

    df['listed_in'] + ' ' +

    df['description'] + ' ' +

    df['type'] + ' ' +

    df['director'] + ' ' +

    df['cast']
)

# TF-IDF Vectorization

tfidf = TfidfVectorizer(stop_words='english')

tfidf_matrix = tfidf.fit_transform(
    df['combined_features']
)

# Cosine Similarity

cosine_sim = cosine_similarity(tfidf_matrix)

# Creating Index Mapping

indices = pd.Series(
    df.index,
    index=df['title']
).drop_duplicates()

# Recommendation Function

def recommend(title):

    try:

        # Find movie index
        idx = indices[title]

        # Get similarity scores
        sim_scores = list(
            enumerate(cosine_sim[idx])
        )

        # Sort similarities
        sim_scores = sorted(
            sim_scores,
            key=lambda x: x[1],
            reverse=True
        )

        # Top 5 similar titles
        sim_scores = sim_scores[1:6]

        # Get indices
        movie_indices = [
            i[0] for i in sim_scores
        ]

        # Return titles
        return "\n".join(
            df['title'].iloc[movie_indices]
        )

    except:

        return "Title not found"

# Gradio Interface

iface = gr.Interface(

    fn=recommend,

    inputs=gr.Textbox(
        lines=1,
        placeholder="Enter a Netflix title..."
    ),

    outputs="text",

    title="🎬 Netflix Recommendation System",

    description="Get Netflix recommendations using NLP and Machine Learning"
)

# Launch App

iface.launch()
