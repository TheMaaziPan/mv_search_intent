import streamlit as st
import requests
import numpy as np
import collections
import plotly.express as px
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

st.title("Search Intent Explorer")

# User input for keyword, country, and language
seed_keyword = st.text_input("Enter a seed keyword:")
country = st.selectbox("Select a country:", ["us", "ca", "gb", "in", "au", "de", "fr", "es", "it", "nl"])
language = st.selectbox("Select a language:", ["en", "fr", "es", "de", "it", "nl"])  

# Function to get Google Auto Suggest results
def get_google_suggestions(keyword, country, language):
    url = f"http://suggestqueries.google.com/complete/search?client=firefox&q={keyword}&hl={language}&gl={country}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()[1]
    return []

def expand_suggestions(seed_keyword, country, language):
    suggestions = set()
    level_1 = get_google_suggestions(seed_keyword, country, language)[:10]
    suggestions.update(level_1)
    
    level_2 = []
    for kw in level_1:
        level_2.extend(get_google_suggestions(kw, country, language)[:10])
    suggestions.update(level_2)
    
    level_3 = []
    for kw in level_2:
        level_3.extend(get_google_suggestions(kw, country, language)[:10])
    suggestions.update(level_3)
    
    return list(suggestions)

# Load pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

def cluster_suggestions(suggestions):
    embeddings = model.encode(suggestions)
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=1.0).fit(embeddings)
    
    clusters = {}
    for i, label in enumerate(clustering.labels_):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(suggestions[i])
    
    return clusters, embeddings

def generate_cluster_labels(clusters, embeddings, suggestions):
    labels = {}
    for cluster, terms in clusters.items():
        term_embeddings = [embeddings[suggestions.index(term)] for term in terms]
        centroid = np.mean(term_embeddings, axis=0).reshape(1, -1)
        similarities = cosine_similarity(centroid, term_embeddings)[0]
        best_match_index = np.argmax(similarities)
        labels[cluster] = terms[best_match_index]
    return labels

def visualize_clusters(clusters, cluster_labels):
    data = []
    for cluster, items in clusters.items():
        for item in items:
            data.append({"Cluster": cluster_labels[cluster], "Keyword": item})
    df = px.data.tips()
    fig = px.treemap(data, path=["Cluster", "Keyword"], title="Keyword Clusters")
    st.plotly_chart(fig)

if st.button("Get Suggestions"):
    if seed_keyword:
        suggestions = expand_suggestions(seed_keyword, country, language)
        if suggestions:
            clusters, embeddings = cluster_suggestions(suggestions)
            cluster_labels = generate_cluster_labels(clusters, embeddings, suggestions)
            st.write("### Clustered Google Auto Suggest Results:")
            visualize_clusters(clusters, cluster_labels)
        else:
            st.write("No suggestions found.")
    else:
        st.warning("Please enter a seed keyword.")
