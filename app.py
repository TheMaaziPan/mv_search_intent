import streamlit as st
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import collections

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

def optimal_kmeans(X):
    best_k = 2
    best_score = -1
    for k in range(2, min(10, len(X.toarray()))):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        if score > best_score:
            best_k = k
            best_score = score
    return best_k

def generate_cluster_labels(clusters):
    labels = {}
    for cluster, terms in clusters.items():
        common_words = collections.Counter(" ".join(terms).split()).most_common(2)
        cluster_label = " ".join([word[0] for word in common_words])
        labels[cluster] = cluster_label
    return labels

def cluster_suggestions(suggestions):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(suggestions)
    
    optimal_clusters = optimal_kmeans(X)
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
    kmeans.fit(X)
    
    clusters = {}
    for i, label in enumerate(kmeans.labels_):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(suggestions[i])
    
    cluster_labels = generate_cluster_labels(clusters)
    return clusters, cluster_labels

if st.button("Get Suggestions"):
    if seed_keyword:
        suggestions = expand_suggestions(seed_keyword, country, language)
        if suggestions:
            clusters, cluster_labels = cluster_suggestions(suggestions)
            st.write("### Clustered Google Auto Suggest Results:")
            for cluster, items in clusters.items():
                st.write(f"#### {cluster_labels[cluster].title()}")
                for item in items:
                    st.write(f"- {item}")
        else:
            st.write("No suggestions found.")
    else:
        st.warning("Please enter a seed keyword.")