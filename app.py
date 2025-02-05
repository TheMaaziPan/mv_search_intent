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
        suggestions = response.json()[1]
        return suggestions
    return []

if st.button("Get Suggestions"):
    if seed_keyword:
        suggestions = get_google_suggestions(seed_keyword, country, language)
        if suggestions:
            st.write("### Google Auto Suggest Results:")
            for suggestion in suggestions:
                st.write(f"- {suggestion}")
        else:
            st.write("No suggestions found.")
    else:
        st.warning("Please enter a seed keyword.")
