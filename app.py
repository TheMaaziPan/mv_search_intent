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
country = st.selectbox(
    label="Select a country:",
    options=[
        ("us", "United States"), ("ca", "Canada"), ("gb", "United Kingdom"), ("in", "India"), 
        ("au", "Australia"), ("de", "Germany"), ("fr", "France"), ("es", "Spain"), ("it", "Italy"), 
        ("nl", "Netherlands"), ("br", "Brazil"), ("mx", "Mexico"), ("ru", "Russia"), ("jp", "Japan"), 
        ("kr", "South Korea"), ("cn", "China"), ("hk", "Hong Kong"), ("tw", "Taiwan"), 
        ("sg", "Singapore"), ("my", "Malaysia"), ("th", "Thailand"), ("id", "Indonesia"), 
        ("ph", "Philippines"), ("vn", "Vietnam"), ("za", "South Africa"), ("ng", "Nigeria"), 
        ("eg", "Egypt"), ("sa", "Saudi Arabia"), ("ae", "United Arab Emirates"), ("tr", "Turkey"), 
        ("pl", "Poland"), ("se", "Sweden"), ("no", "Norway"), ("dk", "Denmark"), ("fi", "Finland"), 
        ("be", "Belgium"), ("ch", "Switzerland"), ("at", "Austria"), ("ie", "Ireland"), 
        ("pt", "Portugal"), ("gr", "Greece"), ("cz", "Czech Republic"), ("hu", "Hungary"), 
        ("ro", "Romania"), ("bg", "Bulgaria"), ("hr", "Croatia"), ("sk", "Slovakia"), 
        ("si", "Slovenia"), ("lt", "Lithuania"), ("lv", "Latvia"), ("ee", "Estonia"), 
        ("is", "Iceland"), ("mt", "Malta"), ("cy", "Cyprus"), ("lu", "Luxembourg")
    ],
    format_func=lambda x: x[1],
    placeholder="Choose a country"
)

language = st.selectbox(
    label="Select a language:",
    options=[
        ("en", "English"), ("fr", "French"), ("es", "Spanish"), ("de", "German"), ("it", "Italian"), 
        ("nl", "Dutch"), ("pt", "Portuguese"), ("ru", "Russian"), ("ja", "Japanese"), ("ko", "Korean"), 
        ("zh", "Chinese"), ("ar", "Arabic"), ("tr", "Turkish"), ("pl", "Polish"), ("sv", "Swedish"), 
        ("no", "Norwegian"), ("da", "Danish"), ("fi", "Finnish"), ("el", "Greek"), ("cs", "Czech"), 
        ("hu", "Hungarian"), ("ro", "Romanian"), ("bg", "Bulgarian"), ("hr", "Croatian"), 
        ("sk", "Slovak"), ("sl", "Slovenian"), ("lt", "Lithuanian"), ("lv", "Latvian"), 
        ("et", "Estonian"), ("is", "Icelandic"), ("mt", "Maltese"), ("cy", "Cypriot"), ("ga", "Irish")
    ],
    format_func=lambda x: x[1],
    placeholder="Choose a language"
)

clustering_threshold = st.slider(
    "Set Clustering Precision: Adjust the distance threshold to control how closely related search queries are grouped together. Lower values create more clusters with tighter groupings, while higher values result in fewer, broader clusters.", 
    min_value=0.1, 
    max_value=1.5, 
    value=0.5, 
    step=0.1
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
        f"how {keyword}", f"how can {keyword}", f"how does {keyword}", 
        f"how should {keyword}", f"how would {keyword}", f"how did {keyword}", 
        f"how might {keyword}", f"how could {keyword}", f"how will {keyword}", 
        f"how is {keyword}", f"how has {keyword}", f"how would {keyword}", 
        f"how could {keyword}", f"how should {keyword}", f"how often {keyword}",
        f"how long {keyword}", f"how much {keyword}", f"how many {keyword}",
        f"how far {keyword}", f"how come {keyword}", f"how soon {keyword}",
        
        # What variations
        f"what {keyword}", f"what is {keyword}", f"what does {keyword}",
        f"what can {keyword}", f"what are the {keyword}", f"what was the {keyword}",
        f"what will be {keyword}", f"what should {keyword}", f"what could {keyword}",
        f"what might {keyword}", f"what has been {keyword}", f"what would be {keyword}",
        f"what could be {keyword}", f"what if {keyword}", f"what about {keyword}",
        f"what makes {keyword}", f"what causes {keyword}", f"what happens if {keyword}",
        
        # Why variations
        f"Why {keyword}", f"Why did {keyword}", f"Why can't {keyword}",
        f"Why should {keyword}", f"Why would {keyword}", f"Why does {keyword}",
        f"Why will {keyword}", f"Why might {keyword}", f"Why could {keyword}",
        f"Why is {keyword}", f"Why hasn't {keyword}", f"Why wouldn't {keyword}",
        f"Why couldn't {keyword}", f"Why should've {keyword}", f"Why do {keyword}",
        f"Why are {keyword}", f"Why were {keyword}", f"Why was {keyword}",
        
        # Where variations
        f"where {keyword}", f"where is {keyword}", f"where can {keyword}",
        f"where do {keyword}", f"where are the {keyword}", f"where was {keyword}",
        f"where will {keyword}", f"where should {keyword}", f"where could {keyword}",
        f"where might {keyword}", f"where has {keyword}", f"where would {keyword}",
        f"where could have {keyword}", f"where should have {keyword}", f"where did {keyword}",
        f"where does {keyword}", f"where have {keyword}", f"where had {keyword}",
        
        # When variations
        f"when {keyword}", f"when did {keyword}", f"when can {keyword}",
        f"when will {keyword}", f"when was the {keyword}", f"when should {keyword}",
        f"when could {keyword}", f"when might {keyword}", f"when does {keyword}",
        f"when is {keyword}", f"when has {keyword}", f"when would {keyword}",
        f"when could {keyword}", f"when should {keyword}", f"when was {keyword}",
        f"when were {keyword}", f"when have {keyword}", f"when had {keyword}",
        
        # Which variations
        f"which {keyword}", f"which is {keyword}", f"which are the {keyword}",
        f"which was the {keyword}", f"which can {keyword}", f"which should {keyword}",
        f"which would {keyword}", f"which could {keyword}", f"which might {keyword}",
        f"which does {keyword}", f"which has been {keyword}", f"which would be {keyword}",
        f"which could be {keyword}", f"which should be {keyword}", f"which one {keyword}",
        f"which type {keyword}", f"which kind {keyword}", f"which category {keyword}",
        
        # Who variations
        f"who {keyword}", f"who was {keyword}", f"who can {keyword}",
        f"who will {keyword}", f"who has {keyword}", f"who does {keyword}",
        f"who might {keyword}", f"who should {keyword}", f"who would {keyword}",
        f"who could {keyword}", f"who has been {keyword}", f"who would have {keyword}",
        f"who could have {keyword}", f"who is {keyword}", f"who are {keyword}",
        f"who did {keyword}", f"who had {keyword}", f"who have {keyword}",
        
        # Can variations
        f"Can {keyword}", f"Can you {keyword}", f"Can I {keyword}",
        f"Can we {keyword}", f"Can they {keyword}", f"Can he {keyword}",
        f"Can she {keyword}", f"Can it {keyword}", f"Can there {keyword}",
        f"Can this {keyword}", f"Can there be {keyword}", f"Can this happen {keyword}",
        f"Can anyone {keyword}", f"Can anything {keyword}", f"Can someone {keyword}",
        
        # Could variations
        f"Could {keyword}", f"Could you {keyword}", f"Could we {keyword}",
        f"Could I {keyword}", f"Could they {keyword}", f"Could he {keyword}",
        f"Could she {keyword}", f"Could it {keyword}", f"Could there {keyword}",
        f"Could this {keyword}", f"Could there be {keyword}", f"Could this happen {keyword}",
        f"Could anyone {keyword}", f"Could anything {keyword}", f"Could someone {keyword}",
        
        # Should variations
        f"Should {keyword}", f"Should you {keyword}", f"Should we {keyword}",
        f"Should I {keyword}", f"Should they {keyword}", f"Should he {keyword}",
        f"Should she {keyword}", f"Should it {keyword}", f"Should there {keyword}",
        f"Should this {keyword}", f"Should you ever {keyword}", f"Should it ever {keyword}",
        f"Should there ever {keyword}", f"Should anyone {keyword}", f"Should anything {keyword}",
        f"Should someone {keyword}",
        
        # Would variations
        f"Would {keyword}", f"Would you {keyword}", f"Would we {keyword}",
        f"Would I {keyword}", f"Would they {keyword}", f"Would he {keyword}",
        f"Would she {keyword}", f"Would it {keyword}", f"Would there {keyword}",
        f"Would this {keyword}", f"Would you ever {keyword}", f"Would it ever {keyword}",
        f"Would there ever {keyword}", f"Would anyone {keyword}", f"Would anything {keyword}",
        f"Would someone {keyword}",
        
        # Do variations
        f"Do {keyword}", f"Do you {keyword}", f"Do we {keyword}",
        f"Do I {keyword}", f"Do they {keyword}", f"Do they have {keyword}",
        f"Do you know {keyword}", f"Do we need {keyword}", f"Do I need {keyword}",
        
        # Does variations
        f"Does {keyword}", f"Does it {keyword}", f"Does this {keyword}",
        f"Does that {keyword}", f"Does he {keyword}", f"Does she {keyword}",
        f"Does there {keyword}", f"Does anyone {keyword}", f"Does everybody {keyword}",
        f"Does someone {keyword}", f"Does it ever {keyword}", f"Does this mean {keyword}",
        f"Does that happen {keyword}", f"Does it work {keyword}", f"Does it matter {keyword}",
        
        # Did variations
        f"Did {keyword}", f"Did you {keyword}", f"Did we {keyword}",
        f"Did I {keyword}", f"Did they {keyword}", f"Did he {keyword}",
        f"Did it {keyword}", f"Did there {keyword}", f"Did this {keyword}",
        f"Did anyone {keyword}", f"Did anything {keyword}", f"Did someone {keyword}",
        
        # Are variations
        f"Are {keyword}", f"Are you {keyword}", f"Are we {keyword}",
        f"Are they {keyword}", f"Are there {keyword}", f"Are there any {keyword}",
        f"Are you sure {keyword}", f"Are we ready {keyword}", f"Are they available {keyword}",
        
        # Is variations
        f"Is {keyword}", f"Is it {keyword}", f"Is this {keyword}",
        f"Is there {keyword}", f"Is that {keyword}", f"Is he {keyword}",
        f"Is there a {keyword}", f"Is there any {keyword}", f"Is it possible {keyword}",
        f"Is this true {keyword}", f"Is that correct {keyword}", f"Is he/she available {keyword}",
        f"Is there a reason {keyword}", f"Is it necessary {keyword}", f"Is it worth {keyword}",
        
        # Was variations
        f"Was {keyword}", f"Was it {keyword}", f"Was this {keyword}",
        f"Was there {keyword}", f"Was that {keyword}", f"Was he {keyword}",
        f"Was it possible {keyword}", f"Was there any {keyword}", f"Was it necessary {keyword}",
        f"Was this known {keyword}", f"Was that the case {keyword}", f"Was he/she there {keyword}",
        f"Was there a problem {keyword}", f"Was it worth {keyword}", f"Was it true {keyword}",
        
        # Have variations
        f"Have {keyword}", f"Have there been {keyword}", f"Have there been changes {keyword}",
        f"Have you seen {keyword}", f"Have we done {keyword}", f"Have they tried {keyword}",
        
        # Has variations
        f"Has {keyword}", f"Has it {keyword}", f"Has this {keyword}",
        f"Has that {keyword}", f"Has he {keyword}", f"Has she {keyword}",
        f"Has there been {keyword}", f"Has it ever {keyword}", f"Has there ever been {keyword}",
        f"Has anyone ever {keyword}", f"Has it changed {keyword}", f"Has this happened {keyword}",
        f"Has that been {keyword}", f"Has it worked {keyword}", f"Has it mattered {keyword}",
        
        # Will variations
        f"Will {keyword}", f"Will you {keyword}", f"Will we {keyword}",
        f"Will they {keyword}", f"Will it {keyword}", f"Will he {keyword}",
        f"Will she {keyword}", f"Will there {keyword}", f"Will there be {keyword}",
        f"Will it work {keyword}", f"Will it matter {keyword}", f"Will it change {keyword}",
        
        # Other question starters
        f"vs {keyword}", f"versus {keyword}", f"or {keyword}", f"and {keyword}", f"without {keyword}",
        
        # Action-oriented
        f"compare {keyword}", f"buy {keyword}", f"sell {keyword}",
        f"find {keyword}", f"download {keyword}", f"install {keyword}",
        f"fix {keyword}", f"repair {keyword}", f"solve {keyword}",
        
        # Best/Top variations
        f"best {keyword}", f"top {keyword}", f"cheapest {keyword}",
        f"fastest {keyword}", f"easiest {keyword}", f"free {keyword}",
        f"premium {keyword}", f"professional {keyword}", f"alternative {keyword}",
        
        # New variations
        # Time-related
        f"today {keyword}", f"tomorrow {keyword}", f"yesterday {keyword}",
        f"next week {keyword}", f"last year {keyword}", f"this month {keyword}",
        
        # Location-based
        f"near me {keyword}", f"in my area {keyword}", f"local {keyword}",
        f"nearby {keyword}", f"closest {keyword}", f"around here {keyword}",
        
        # Comparison
        f"better than {keyword}", f"worse than {keyword}", f"similar to {keyword}",
        f"different from {keyword}", f"compared to {keyword}", f"as good as {keyword}"
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
    
    # Display the data in an interactive table with the specified columns
    st.write("### Interactive Table of Clusters")
    cluster_data = [
        {
            "Cluster Name": cluster_labels[cluster],
            "Number of Search Queries in a Cluster": len(items),
            "List of Search Queries in a Cluster": ", ".join(items)
        }
        for cluster, items in clusters.items()
    ]
    df = pd.DataFrame(cluster_data)
    df_sorted = df.sort_values(by="Number of Search Queries in a Cluster", ascending=False)
    st.dataframe(df_sorted)
    
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
                # Add progress bar for clustering
                st.write("🔄 Clustering suggestions...")
                progress_bar = st.progress(0)
                clusters, embeddings = cluster_suggestions(suggestions, clustering_threshold)
                progress_bar.progress(1.0)  # Complete the progress bar
                cluster_labels = generate_cluster_labels(clusters, embeddings, suggestions)
                st.write("### Clustered Google Auto Suggest Results:")
                data = visualize_clusters(clusters, cluster_labels)
                
                # Send success notification with more detailed data
                cluster_details = "\n".join([f"- {label}: {len(terms)} terms" for label, terms in clusters.items()])
                top_keywords = "\n".join([f"- {label}" for label in cluster_labels.values()][:5])  # Show top 5 cluster labels
                
                # Add more details to the notification message
                notification_message = f"""
                **New search analysis completed**
                \nKeyword: {seed_keyword}
                \nCountry: {country}
                \nLanguage: {language}
                \nClustering Threshold: {clustering_threshold}
                \nClusters found: {len(clusters)}
                \nUnique Suggestions found: {len(suggestions)}
                \nTop Keywords: {top_keywords}  # Add top keywords to the notification message
                """

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