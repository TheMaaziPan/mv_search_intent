import streamlit as st
import requests

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

def cluster_suggestions(suggestions, num_clusters=5):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(suggestions)
    
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    kmeans.fit(X)
    
    clusters = {}
    for i, label in enumerate(kmeans.labels_):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(suggestions[i])
    
    return clusters

if st.button("Get Suggestions"):
    if seed_keyword:
        suggestions = expand_suggestions(seed_keyword, country, language)
        if suggestions:
            clusters = cluster_suggestions(suggestions)
            st.write("### Clustered Google Auto Suggest Results:")
            for cluster, items in clusters.items():
                st.write(f"#### Cluster {cluster+1}:")
                for item in items:
                    st.write(f"- {item}")
        else:
            st.write("No suggestions found.")
    else:
        st.warning("Please enter a seed keyword.")
