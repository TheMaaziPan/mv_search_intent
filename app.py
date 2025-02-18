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
import random
import time
import umap.umap_ as umap 


import pymongo
from datetime import datetime
from bson import ObjectId


mongodb_url = st.secrets["mongodb_uri"]
client = pymongo.MongoClient(mongodb_url)

# Initialize database connection
db = client["search-intent-explorer"]

print(db)

# Define collections
searches_collection = db["searches"]

def save_search_results(seed_keyword, country, language, suggestions, clusters, cluster_labels):
    """Save search results to MongoDB"""

    # Prepare cluster data array
    clusters_data = []
    for cluster_id, queries in clusters.items():
        clusters_data.append({
            "cluster_id": str(cluster_id),
            "cluster_label": cluster_labels.get(cluster_id),
            "queries": queries,
            "size": len(queries)
        })


    # Create single document with all data
    search_data = {
        "timestamp": datetime.utcnow(),
        "seed_keyword": seed_keyword,
        "country": country[1],
        "language": language[1],
        "total_suggestions": len(suggestions),
        "total_clusters": len(clusters),
        "suggestions": suggestions,
        "clusters": {str(k): v for k, v in clusters.items()},
        "cluster_labels": {str(k): v for k, v in cluster_labels.items()},
        "clusters_data": clusters_data
    }
    
    # Save to searches collection
    search_result = searches_collection.insert_one(search_data)
    
    return search_result



st.set_page_config(
    layout="wide",
    page_title="🔍 Search Intent Explorer"
)


# Add informative sections to the sidebar
with st.sidebar:

    st.markdown("<h2 style='font-size:20px;'>🔗 Credits</h2>", unsafe_allow_html=True)
    st.markdown("""
    <p style='font-size:14px;'>Explore more about my work and insights:</p>
    <ul style='font-size:14px;'>
    <li><a href='https://www.mihirnaik.com' target='_blank'>Visit my personal website</a></li>
    <li><a href='https://mihirnaik.substack.com/' target='_blank'>Subscribe to my newsletter: SEO Workflow Automation</a></li>
    </ul>
    """, unsafe_allow_html=True)

    st.markdown("<h2 style='font-size:20px;'>🚀 Welcome to the Search Intent Explorer!</h2>", unsafe_allow_html=True)
    st.markdown("""
    <p style='font-size:14px;'>Enhance your SEO and content marketing with the Search Intent Explorer! 
    Discover what your audience is truly searching for and create content that connects and ranks. 
    Ideal for SEO managers and content marketers looking to increase online visibility and attract targeted traffic. 📈</p>
    """, unsafe_allow_html=True)

    st.markdown("<h2 style='font-size:20px;'>🛠️ How to Use the Tool</h2>", unsafe_allow_html=True)
    st.markdown("""
    <p style='font-size:14px;'>Get started in just a few steps:</p>
    <ol style='font-size:14px;'>
    <li><strong>🔍 Enter a Seed Search Query</strong>: Input a search query you wish to explore.</li>
    <li><strong>🌍 Select a Country and Language</strong>: Tailor your search to specific regions and languages.</li>
    <li><strong>💡 Get Suggestions</strong>: Hit the "Get Suggestions" button to generate related search queries.</li>
    <li><strong>🔗 View Clustered Results</strong>: See how search queries are grouped by user intent for strategic insights.</li>
    <li><strong>📥 Download Data</strong>: Export your findings in an Excel file for detailed analysis and reporting.</li>
    </ol>
    """, unsafe_allow_html=True)

st.title("🔍 Search Intent Explorer")

st.warning("""
⏳ **Please Note**: To ensure reliable data collection and comply with API rate limits, reports now take around 5 minutes to complete.

This slight delay allows us to gather more comprehensive and accurate search intent data while avoiding API throttling. Feel free to grab a coffee while we prepare your detailed search analysis! ☕
""")


st.markdown("""
Enter your target search query and market parameters below to discover high-value search intent patterns 
and search query opportunities for your SEO strategy:
""")

# Input fields for user interaction in a single line
col1, col2, col3 = st.columns(3)

with col1:
    seed_keyword = st.text_input("🔑 Enter a seed search query:", placeholder="e.g., digital marketing")

with col2:
    country = st.selectbox(
        label="🌍 Select a country:",
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

with col3:
    language = st.selectbox(
        label="🗣️ Select a language:",
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
    """🔧 Optimize Clustering Precision:
    \n Fine-tune the distance threshold to define the granularity of search query groupings. 
    \n Lower values yield more specific clusters, ideal for targeting niche search intents, while higher values create broader clusters for general insights.""", 
    min_value=0.1, 
    max_value=1.5, 
    value=0.5, 
    step=0.1
)

def get_google_suggestions(keyword, country, language, cache={}):
    # Check if the result is already cached
    cache_key = (keyword, country[0], language[0])  # Extract country/language codes
    if cache_key in cache:
        return cache[cache_key]
    
    # List of different user agents to rotate through
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.864.59'
    ]
    
    headers = {
        'User-Agent': random.choice(user_agents),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
    
    url = f"https://google.com/complete/search?client=chrome&hl={language[0]}&gl={country[0]}&q={keyword}"
    print(url)

    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            # Add a small random delay between requests
            time.sleep(random.uniform(0.5, 1.5))
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                suggestions = response.json()[1]
                print(suggestions)
                cache[cache_key] = suggestions
                return suggestions
            elif response.status_code == 403:
                if attempt < max_retries - 1:  # If not the last attempt
                    time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                    continue
                else:
                    st.warning(f"Rate limit reached while searching for '{keyword}'. Try again in a few minutes.")
                    return []
            else:
                st.error(f"Unexpected status code: {response.status_code}")
                return []
                
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
                continue
            else:
                st.error(f"Error accessing Google Suggestions API: {str(e)}")
                return []
    
    return []

def get_modifier_suggestions(keyword):
    alphabet_modifiers = [f"{keyword} {chr(i)}" for i in range(97, 123)]
    number_modifiers = [f"{keyword} {i}" for i in range(0, 11)]
    question_modifiers = [
        # Basic question modifiers
        f"who {keyword}",
        f"what {keyword}",
        f"when {keyword}",
        f"where {keyword}",
        f"why {keyword}",
        f"how {keyword}",
        f"which {keyword}",
        f"can {keyword}",
        f"could {keyword}",
        f"should {keyword}",
        f"would {keyword}",
        f"do {keyword}",
        f"does {keyword}",
        f"did {keyword}",
        f"are {keyword}",
        f"is {keyword}",
        f"was {keyword}",
        f"have {keyword}",
        f"has {keyword}",
        f"will {keyword}",
        
        # Other question starters
        #f"vs {keyword}", f"versus {keyword}", f"or {keyword}", f"and {keyword}", f"without {keyword}",
        
        # Action-oriented
        #f"compare {keyword}", f"buy {keyword}", f"sell {keyword}",
        #f"find {keyword}", f"download {keyword}", f"install {keyword}",
        #f"fix {keyword}", f"repair {keyword}", f"solve {keyword}",
        
        # Best/Top variations
        #f"best {keyword}", f"top {keyword}", f"cheapest {keyword}",
        #f"fastest {keyword}", f"easiest {keyword}", f"free {keyword}",
        #f"premium {keyword}", f"professional {keyword}", f"alternative {keyword}",
        
        # New variations
        # Time-related
        #f"today {keyword}", f"tomorrow {keyword}", f"yesterday {keyword}",
        #f"next week {keyword}", f"last year {keyword}", f"this month {keyword}",
        
        # Location-based
        #f"near me {keyword}", f"in my area {keyword}", f"local {keyword}",
        #f"nearby {keyword}", f"closest {keyword}", f"around here {keyword}",
        
        # Comparison
        #f"better than {keyword}", f"worse than {keyword}", f"similar to {keyword}",
        #f"different from {keyword}", f"compared to {keyword}", f"as good as {keyword}"
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
#print(discord_webhook)

def send_discord_notification(message):
    webhook_url = discord_webhook
    
    if not webhook_url:
        print("Discord webhook URL not found in environment variables")
        return False
        
    try:
        # Clean up the message by removing extra indentation
        cleaned_message = "\n".join(line.strip() for line in message.split("\n"))
        
        payload = {
            "content": cleaned_message,
            "username": "Search Intent Explorer Bot",
            "avatar_url": "https://cdn-icons-png.flaticon.com/512/1998/1998342.png"
        }

        response = requests.post(webhook_url, json=payload)
        
        if response.status_code == 404:
            print("Discord webhook not found - URL may be invalid")
            return False
        elif response.status_code == 429:
            print("Rate limited by Discord")
            return False
            
        response.raise_for_status()
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"Failed to send Discord notification: {str(e)}")
        print(f"Status code: {getattr(e.response, 'status_code', 'N/A')}")
        print(f"Response text: {getattr(e.response, 'text', 'N/A')}")
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
    # Create data for visualization
    data = []
    for cluster, items in clusters.items():
        if len(items) > 1:  # Only include clusters with more than one search query
            for item in items:
                data.append({
                    "Cluster": cluster_labels[cluster], 
                    "Search Query": item,
                    "Cluster_ID": cluster  # Add this for color mapping
                })
    
    # Create scatter plot data
    df = pd.DataFrame(data)
    
    # Get top 10 clusters by size
    cluster_sizes = df['Cluster'].value_counts()
    top_10_clusters = cluster_sizes.head(10).index.tolist()
    
    # Filter dataframe for top 10 clusters
    df_top_10 = df[df['Cluster'].isin(top_10_clusters)]
    
    # Get embeddings for filtered queries
    top_10_queries = df_top_10["Search Query"].tolist()
    embeddings_2d = umap.UMAP(n_components=2, random_state=42).fit_transform(
        model.encode(top_10_queries)
    )
    
    # Add 2D coordinates to filtered dataframe
    df_top_10["x"] = embeddings_2d[:, 0]
    df_top_10["y"] = embeddings_2d[:, 1]
    
    # Create scatter plot with filtered data
    scatter_fig = px.scatter(
        df_top_10,
        x="x",
        y="y",
        color="Cluster",
        hover_data=["Search Query"],
        title="Top 10 Largest Search Query Clusters Visualization",
        labels={"x": "", "y": ""},
        #color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    # Update scatter plot layout
    scatter_fig.update_traces(marker=dict(size=8))
    scatter_fig.update_layout(
        plot_bgcolor="white",
        showlegend=True,
        legend_title_text="Clusters"
    )
    
    # Create and display visualizations
    st.write("### Cluster Visualizations")
    
    # Create treemap
    treemap_fig = px.treemap(
        df, 
        path=["Cluster", "Search Query"],
        title="Search Query Clusters Treemap",
        height=600,
        color_discrete_sequence=px.colors.qualitative.Set3,
    )
    
    # Display treemap
    st.plotly_chart(treemap_fig, use_container_width=True)

    # Display scatter plot
    st.plotly_chart(scatter_fig, use_container_width=True)
    
    # Display the data in an interactive table
    st.write("### Interactive Table of Clusters")
    cluster_data = [
        {
            "Cluster Name": cluster_labels[cluster],
            "Number of Search Queries in a Cluster": len(items),
            "List of Search Queries in a Cluster": ", ".join(items)
        }
        for cluster, items in clusters.items()
    ]
    df_table = pd.DataFrame(cluster_data)
    df_sorted = df_table.sort_values(by="Number of Search Queries in a Cluster", ascending=False)
    st.dataframe(df_sorted)

def export_to_excel(data):
    df = pd.DataFrame(data)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Clustered Search Queries', index=False)
    return output.getvalue()

# Button to get suggestions
if st.button("Get Suggestions"):
    if seed_keyword:
        # Track time
        # Track start time
        start_time = time.time()
        
        # Send initial notification when job starts
        start_notification = f"""
        
        **New search analysis started**
        \nSearch Query: {seed_keyword}
        \nCountry: {country[1]}  
        \nLanguage: {language[1]}
        \nClustering Threshold: {clustering_threshold}
        """
        send_discord_notification(start_notification)

        with st.spinner('Fetching and analyzing suggestions... Please wait.'):
            suggestions = expand_suggestions(seed_keyword, country, language)
            if suggestions:
                # Add progress bar for clustering
                st.write("🔄 Clustering suggestions...")
                progress_bar = st.progress(0)
                clusters, embeddings = cluster_suggestions(suggestions, clustering_threshold)
                progress_bar.progress(1.0)  # Complete the progress bar
                cluster_labels = generate_cluster_labels(clusters, embeddings, suggestions)

                # Save results to MongoDB
                search_id = save_search_results(
                    seed_keyword, 
                    country, 
                    language, 
                    suggestions, 
                    clusters, 
                    cluster_labels
                )


                st.write("### Clustered Search Query Results:")
                data = visualize_clusters(clusters, cluster_labels)
                
                # Send success notification with more detailed data
                cluster_details = "\n".join([f"- {label}: {len(terms)} terms" for label, terms in clusters.items()])
                top_keywords = "\n".join([f"- {label}" for label in cluster_labels.values()][:5])  # Show top 5 cluster labels
                top_queries = "\n".join([f"- {query}" for query in suggestions[:5]])  # Show top 5 suggestions
                
                # Add more details to the notification message
                # Calculate duration
                duration = time.time() - start_time
                duration_mins = int(duration // 60)
                duration_secs = int(duration % 60)


                # Create a more concise notification message
                notification_message = f"""**Search Analysis Complete**
                • Query: {seed_keyword}
                • Region: {country[1]} ({language[1]})
                • Duration: {duration_mins}m {duration_secs}s
                • Results: {len(suggestions)} suggestions in {len(clusters)} clusters
                • Threshold: {clustering_threshold}

                Top 3 Clusters:
                {chr(10).join(f"• {label}" for label in list(cluster_labels.values())[:3])}"""

                notification_message = str(notification_message)

                # Add debug logging
                print("Attempting to send Discord notification...")
                notification_sent = send_discord_notification(notification_message)
                if notification_sent:
                    print("Discord notification sent successfully")
                else:
                    print("Failed to send Discord notification")
                
                excel_data = export_to_excel(data)
                
                st.download_button(
                    label="Download Excel File",
                    data=excel_data,
                    file_name="clustered_search_queries.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                # Send empty result notification
                send_discord_notification(f"Search analysis failed - No suggestions found for search query: {seed_keyword}")
                st.write("No suggestions found.")
    else:
        st.warning("Please enter a seed search query.")


recent_searches = searches_collection.find().sort("timestamp", -1).limit(10)

for search in recent_searches:
    with st.expander(f"🔍 {search['seed_keyword']} - {search['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}"):
        st.write(f"Country: {search['country']}")
        st.write(f"Language: {search['language']}")
        st.write(f"Total Suggestions: {search['total_suggestions']}")
        st.write(f"Total Clusters: {search['total_clusters']}")
