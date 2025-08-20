import streamlit as st
import pandas as pd
from io import BytesIO
import re
import numpy as np

# Imports for the TF-IDF model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==============================================================================
# --- 1. Model Training Setup ---
# This part sets up the data and trains the vectorizer your function needs.
# In a real app, you might load a pre-trained model, but for a single script,
# this is the most straightforward way.
# ==============================================================================

# Define the training data for intents
intent_data = {
    "TOTAL_SALES": ["total sales", "what are the total sales", "sum of sales"],
    "TREND_MONTH": ["monthly sales trend", "sales over months", "show sales by month for 2024", "sales trend for 2023"],
    "FILTER_REGION": ["show sales for east region", "data for west", "what about north region"],
    "AVG_PROFIT_YEAR": ["average profit per year", "show me the average profit for 2024", "yearly average profit"],
    "AGGREGATE_REGION": ["sales by region", "summarize sales per region", "region-wise sales"],
    "TOP_PRODUCTS": ["top 5 products", "which are the top 10 selling products", "top products by sales"],
}

# Prepare training texts and labels
train_texts = []
train_labels = []
for intent, examples in intent_data.items():
    train_texts.extend(examples)
    train_labels.extend([intent] * len(examples))

# Create and train the TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
train_emb = vectorizer.fit_transform(train_texts)


# ==============================================================================
# --- 2. Your Custom Functions ---
# These are the two functions you provided, pasted directly here.
# ==============================================================================

def predict_intent_tfidf(user_query: str, threshold: float = 0.2):
    q_emb = vectorizer.transform([user_query])
    sims = cosine_similarity(train_emb, q_emb).ravel()
    idx = np.argmax(sims)
    best_sim = sims[idx]
    best_intent = train_labels[idx]
    best_example = train_texts[idx]
    
    if best_sim < threshold:
        return None, best_sim, best_example
    return best_intent, best_sim, best_example

def run_intent_on_dataframe(intent, df, user_query=None):
    if intent == "TOTAL_SALES":
        return pd.DataFrame({"Total Sales": [df["Sales"].sum()]})
    
    if intent == "TREND_MONTH":
        tmp = df.copy()
        tmp["Month"] = pd.to_datetime(df["Date"]).dt.to_period("M").astype(str)
        
        # Detect year from query
        if user_query:
            year_match = re.search(r"\b(20\d{2})\b", user_query)
            if year_match:
                tmp = tmp[pd.to_datetime(tmp["Date"]).dt.year == int(year_match.group(1))]
        return tmp.groupby("Month")["Sales"].sum().sort_index()
    
    if intent == "FILTER_REGION":
        regions = ["North", "South", "East", "West"]
        region = "East" # Default region
        if user_query:
            # Find the first region in the list that is mentioned in the query
            found_region = next((r for r in regions if r.lower() in user_query.lower()), None)
            if found_region:
                region = found_region
        return df[df["Region"] == region][["Region","Sales","Profit"]]
    
    if intent == "AVG_PROFIT_YEAR":
        tmp = df.copy()
        tmp["Year"] = pd.to_datetime(df["Date"]).dt.year
        if user_query:
            year_match = re.search(r"\b(20\d{2})\b", user_query)
            if year_match:
                tmp = tmp[tmp["Year"] == int(year_match.group(1))]
        return tmp.groupby("Year")["Profit"].mean()
    
    if intent == "AGGREGATE_REGION":
        return df.groupby("Region")["Sales"].sum().sort_values(ascending=False)
    
    if intent == "TOP_PRODUCTS":
        tmp = df.groupby("Product")["Sales"].sum().sort_values(ascending=False)
        n = 5 # Default to top 5
        if user_query:
            n_match = re.search(r"\b(?:top|show|give me) (\d+)\b", user_query.lower())
            if n_match:
                n = int(n_match.group(1))
        return tmp.head(n)
    
    return pd.DataFrame({"Message": [f"Intent '{intent}' not mapped yet."]})

# ==============================================================================
# --- 3. Streamlit Application ---
# This is the main application interface code.
# ==============================================================================

st.set_page_config(page_title="Excel NLP Assistant", layout="wide")

st.title("📊 Excel NLP Assistant")
st.write("Upload any Excel file and ask queries in natural language!")

uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.success("File loaded successfully!")
    st.dataframe(df.head())

    user_query = st.text_input("Enter your query (e.g., 'Top 5 products in East region')")

    if st.button("Run Query") and user_query.strip() != "":
        intent, sim, match = predict_intent_tfidf(user_query, threshold=0.2)
        
        if intent is None:
            st.warning("I didn’t understand your query, please rephrase.")
        else:
            st.info(f"Intent: {intent} (Similarity: {sim:.2f}, Matched Example: '{match}')")
            
            try:
                result = run_intent_on_dataframe(intent, df, user_query)
                
                if isinstance(result, (int, float, np.number)):
                    # Handle single numeric results
                    st.metric(label=intent.replace("_", " ").title(), value=f"{result:,.2f}")
                    result = pd.DataFrame({intent: [result]}) # Convert to df for download
                else:
                    if isinstance(result, pd.Series):
                        result = result.to_frame()
                    
                    st.subheader("Result:")
                    st.dataframe(result)

                    if not result.empty and len(result) > 1:
                        st.subheader("Chart:")
                        try:
                            st.bar_chart(result)
                        except Exception as e:
                            st.warning(f"Could not generate a chart for this result.")

                # Download Button Logic
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    result.to_excel(writer, index=True)
                data_to_download = output.getvalue()

                st.download_button(
                    label="📥 Download Result as Excel",
                    data=data_to_download,
                    file_name="query_result.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            except Exception as e:
                st.error(f"An error occurred while processing the query: {e}")