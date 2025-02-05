import streamlit as st
import requests
import numpy as np
import collections
import plotly.express as px
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO

st.set_page_config(layout="wide")

st.title("Search Intent Explorer")

seed_keyword = st.text_input("Enter a seed keyword:")
country = st.selectbox("Select a country:", ["us", "ca", "gb", "in", "au", "de", "fr", "es", "it", "nl"])
language = st.selectbox("Select a language:", ["en", "fr", "es", "de", "it", "nl"])  

def get_google_suggestions(keyword, country, language):
    url = f"http://suggestqueries.google.com/complete/search?client=firefox&q={keyword}&hl={language}&gl={country}"
    response = requests.get(url)
    if response.status_code == 200:
        print(response.json()[1])
        return response.json()[1]
    return []

def get_modifier_suggestions(keyword):
    alphabet_modifiers = [f"{keyword} {chr(i)}" for i in range(97, 123)]
    number_modifiers = [f"{keyword} {i}" for i in range(1, 21)]
    question_modifiers = [
        f"how {keyword}", f"what {keyword}", f"why {keyword}",
        f"can {keyword}", f"where {keyword}", f"when {keyword}",
        f"which {keyword}", f"who {keyword}", f"will {keyword}",
        f"should {keyword}", f"whose {keyword}", f"whom {keyword}",
        f"is {keyword}", f"are {keyword}", f"does {keyword}",
        f"do {keyword}", f"was {keyword}", f"were {keyword}",
        f"has {keyword}", f"have {keyword}", f"had {keyword}",
        f"may {keyword}", f"might {keyword}", f"must {keyword}",
        f"shall {keyword}", f"could {keyword}", f"would {keyword}",
        f"did {keyword}", f"am {keyword}", f"been {keyword}"
    ]
    return alphabet_modifiers + number_modifiers + question_modifiers

def expand_suggestions(seed_keyword, country, language):
    suggestions = set()
    
    level_1 = get_google_suggestions(seed_keyword, country, language)
    suggestions.update(level_1)
    
    level_2 = []
    for kw in level_1:
        level_2.extend(get_google_suggestions(kw, country, language))
    suggestions.update(level_2)
    
    level_3 = []
    for kw in level_2:
        level_3.extend(get_google_suggestions(kw, country, language))
    suggestions.update(level_3)
    
    modifier_suggestions = get_modifier_suggestions(seed_keyword)
    for modifier in modifier_suggestions:
        suggestions.update(get_google_suggestions(modifier, country, language))

    print(len(suggestions))
    
    return list(suggestions)

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
    return data

def export_to_excel(data):
    df = pd.DataFrame(data)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Clustered Keywords', index=False)
    return output.getvalue()

if st.button("Get Suggestions"):
    if seed_keyword:
        with st.spinner('Fetching and analyzing suggestions... Please wait.'):
            suggestions = expand_suggestions(seed_keyword, country, language)
            if suggestions:
                clusters, embeddings = cluster_suggestions(suggestions)
                cluster_labels = generate_cluster_labels(clusters, embeddings, suggestions)
                st.write("### Clustered Google Auto Suggest Results:")
                data = visualize_clusters(clusters, cluster_labels)
                
                excel_data = export_to_excel(data)
                st.download_button(
                    label="Download Excel File",
                    data=excel_data,
                    file_name="clustered_keywords.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.write("No suggestions found.")
    else:
        st.warning("Please enter a seed keyword.")