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
clustering_threshold = st.slider(
    "Clustering Tightness (distance threshold)", 
    min_value=0.1, 
    max_value=1.5, 
    value=0.5, 
    step=0.1,
    help="Lower values create tighter clusters (more specific groups), higher values create looser clusters (broader groups)"
)

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
    number_modifiers = [f"{keyword} {i}" for i in range(0, 11)]
    question_modifiers = [
        # How variations
        f"how to {keyword}", f"how does {keyword}", f"how do {keyword}", 
        f"how is {keyword}", f"how are {keyword}", f"how much {keyword}",
        f"how many {keyword}", f"how long {keyword}", f"how often {keyword}",
        
        # What variations
        f"what is {keyword}", f"what are {keyword}", f"what does {keyword}",
        f"what type of {keyword}", f"what kind of {keyword}", f"what makes {keyword}",
        f"what if {keyword}", f"what about {keyword}", f"what happened to {keyword}",
        
        # Why variations
        f"why is {keyword}", f"why are {keyword}", f"why does {keyword}",
        f"why do {keyword}", f"why not {keyword}", f"why use {keyword}",
        f"why choose {keyword}", f"why buy {keyword}",
        
        # Where variations
        f"where to {keyword}", f"where is {keyword}", f"where are {keyword}",
        f"where can i {keyword}", f"where does {keyword}", f"where do {keyword}",
        f"where buy {keyword}", f"where find {keyword}",
        
        # When variations
        f"when to {keyword}", f"when is {keyword}", f"when are {keyword}",
        f"when does {keyword}", f"when do {keyword}", f"when will {keyword}",
        f"when can {keyword}", f"when should {keyword}",
        
        # Which variations
        f"which {keyword}", f"which is best {keyword}", f"which are best {keyword}",
        f"which type of {keyword}", f"which kind of {keyword}",
        
        # Who variations
        f"who is {keyword}", f"who are {keyword}", f"who does {keyword}",
        f"who can {keyword}", f"who should {keyword}", f"who needs {keyword}",
        
        # Can variations
        f"can {keyword}", f"can i {keyword}", f"can you {keyword}",
        f"can we {keyword}", f"can it {keyword}", f"can they {keyword}",
        
        # Other question starters
        f"should i {keyword}", f"should you {keyword}", f"should we {keyword}",
        f"will {keyword}", f"does {keyword}", f"do {keyword}",
        f"is {keyword}", f"are {keyword}", f"was {keyword}",
        f"were {keyword}", f"has {keyword}", f"have {keyword}",
        f"had {keyword}", f"vs {keyword}", f"versus {keyword}",
        f"or {keyword}", f"and {keyword}", f"without {keyword}",
        
        # Action-oriented
        f"compare {keyword}", f"buy {keyword}", f"sell {keyword}",
        f"find {keyword}", f"download {keyword}", f"install {keyword}",
        f"fix {keyword}", f"repair {keyword}", f"solve {keyword}",
        
        # Best/Top variations
        f"best {keyword}", f"top {keyword}", f"cheapest {keyword}",
        f"fastest {keyword}", f"easiest {keyword}", f"free {keyword}",
        f"premium {keyword}", f"professional {keyword}", f"alternative {keyword}"
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

discord_webhook = st.secrets["discord_webhook"]
print(discord_webhook)

def send_discord_notification(message):
    
    webhook_url = discord_webhook
    
    if not webhook_url:
        print("Discord webhook URL not found in environment variables")
        return
        
    try:
        payload = {
            "content": message,
            "username": "Search Intent Explorer Bot",
            "avatar_url": "https://cdn-icons-png.flaticon.com/512/1998/1998342.png"  # Optional: bot avatar
        }

        response = requests.post(webhook_url, json=payload)
        
        response.raise_for_status()
        
        return True
    except Exception as e:
        print(f"Failed to send Discord notification: {str(e)}")
        return False

model = SentenceTransformer('all-MiniLM-L6-v2')

def cluster_suggestions(suggestions, distance_threshold):
    embeddings = model.encode(suggestions)
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold).fit(embeddings)
    
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
    
    # Create a clean and minimal treemap
    fig = px.treemap(
        data, 
        path=["Cluster", "Keyword"],
        title="Keyword Clusters",
        height=900,
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )

    
    # Display the plot
    st.plotly_chart(fig, use_container_width=True)
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
                clusters, embeddings = cluster_suggestions(suggestions, clustering_threshold)
                cluster_labels = generate_cluster_labels(clusters, embeddings, suggestions)
                st.write("### Clustered Google Auto Suggest Results:")
                data = visualize_clusters(clusters, cluster_labels)
                
                # Send success notification with more detailed data
                cluster_details = "\n".join([f"- {label}: {len(terms)} terms" for label, terms in clusters.items()])
                top_keywords = "\n".join([f"- {label}" for label in cluster_labels.values()][:5])  # Show top 5 cluster labels
                
                notification_message = f"New search analysis completed\nKeyword: {seed_keyword}\nCountry: {country}\nLanguage: {language}\nClusters found: {len(clusters)}"

                send_discord_notification(notification_message)

                
                
                excel_data = export_to_excel(data)
                st.download_button(
                    label="Download Excel File",
                    data=excel_data,
                    file_name="clustered_keywords.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                # Send empty result notification
                send_discord_notification(f"Search analysis failed - No suggestions found for keyword: {seed_keyword}")
                st.write("No suggestions found.")
    else:
        st.warning("Please enter a seed keyword.")
