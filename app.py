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

# Add a new section explaining what the Search Intent Explorer is
with st.expander("What is Search Intent Explorer?", expanded=True):
    st.write("""
    The Search Intent Explorer is a powerful tool designed to help you understand user search behavior. 
    By analyzing search queries, it provides insights into what users are looking for, enabling you to create 
    content that aligns with their intent. This tool can help you identify relevant keywords, develop 
    content strategies, and improve your website's SEO performance.
    """)

# Add a new section on how to use the tool
with st.expander("How to Use It", expanded=True):
    st.write("""
    Follow these steps to get started:
    1. **Enter a Seed Keyword**: Type a keyword that you want to explore.
    2. **Select a Country and Language**: Choose the relevant country and language for your search.
    3. **Get Suggestions**: Click the "Get Suggestions" button to fetch related search queries.
    4. **View Clustered Results**: The tool will display clustered keywords based on user intent.
    5. **Download Data**: You can download the results in an Excel file for further analysis.
    """)

# Create a distinct section for input forms
st.subheader("Search Intent Research Parameters")
st.markdown("""
Enter your target keyword and market parameters below to discover high-value search intent patterns 
and keyword opportunities for your SEO strategy:
""")

# Input fields for user interaction
seed_keyword = st.text_input("Enter a seed keyword:", placeholder="e.g., digital marketing")
country = st.selectbox("Select a country:", ["us", "ca", "gb", "in", "au", "de", "fr", "es", "it", "nl"])
language = st.selectbox("Select a language:", ["en", "fr", "es", "de", "it", "nl"])  

def get_google_suggestions(keyword, country, language, cache={}):
    # Check if the result is already cached
    cache_key = (keyword, country, language)
    if cache_key in cache:
        return cache[cache_key]
    
    url = f"http://suggestqueries.google.com/complete/search?client=firefox&q={keyword}&hl={language}&gl={country}"
    response = requests.get(url)
    if response.status_code == 200:
        suggestions = response.json()[1]
        # Cache the result
        cache[cache_key] = suggestions
        return suggestions
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
    
    # Create a placeholder for real-time updates
    status_placeholder = st.empty()
    
    # Level 1
    status_placeholder.write("🔍 Fetching first level suggestions...")
    level_1 = get_google_suggestions(seed_keyword, country, language)
    suggestions.update(level_1)
    status_placeholder.write(f"📊 Found {len(level_1)} first level suggestions")
    
    # Level 2
    status_placeholder.write("🔍 Fetching second level suggestions...")
    level_2 = []
    progress_bar = st.progress(0)
    for i, kw in enumerate(level_1):
        level_2.extend(get_google_suggestions(kw, country, language))
        progress = (i + 1) / len(level_1)
        progress_bar.progress(progress)
        status_placeholder.write(f"Processing: {kw}")
    suggestions.update(level_2)
    status_placeholder.write(f"📊 Found {len(level_2)} second level suggestions")
    
    # Level 3
    status_placeholder.write("🔍 Fetching third level suggestions...")
    level_3 = []
    progress_bar = st.progress(0)
    for i, kw in enumerate(level_2):
        level_3.extend(get_google_suggestions(kw, country, language))
        progress = (i + 1) / len(level_2)
        progress_bar.progress(progress)
        status_placeholder.write(f"Processing: {kw}")
    suggestions.update(level_3)
    status_placeholder.write(f"📊 Found {len(level_3)} third level suggestions")
    
    # Modifier suggestions
    status_placeholder.write("🔍 Fetching modifier suggestions...")
    modifier_suggestions = get_modifier_suggestions(seed_keyword)
    progress_bar = st.progress(0)
    for i, modifier in enumerate(modifier_suggestions):
        suggestions.update(get_google_suggestions(modifier, country, language))
        progress = (i + 1) / len(modifier_suggestions)
        progress_bar.progress(progress)
        status_placeholder.write(f"Processing: {modifier}")
    
    total_suggestions = len(suggestions)
    status_placeholder.write(f"✅ Completed! Total unique suggestions found: {total_suggestions}")
    
    return list(suggestions)

model = SentenceTransformer('all-MiniLM-L6-v2')

def cluster_suggestions(suggestions):
    embeddings = model.encode(suggestions)
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.5).fit(embeddings)
    
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
        if len(items) > 1:  # Only include clusters with more than one keyword
            for item in items:
                data.append({"Cluster": cluster_labels[cluster], "Keyword": item})
    # Increase the height of the treemap
    fig = px.treemap(data, path=["Cluster", "Keyword"], title="Keyword Clusters", height=900)  # Adjust height as needed
    st.plotly_chart(fig)
    return data

def export_to_excel(data):
    df = pd.DataFrame(data)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Clustered Keywords', index=False)
    return output.getvalue()

# Button to get suggestions
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
