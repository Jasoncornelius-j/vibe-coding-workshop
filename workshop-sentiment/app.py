"""2-Hour Workshop: Sentiment Analysis MVP
Fast, working demo using pre-trained model"""
import streamlit as st
import pandas as pd
from transformers import pipeline
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import re

# ==================== SETUP ====================
# Page config
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="📊",
    layout="wide"
)

# Load pre-trained model (cached so it only loads once)
@st.cache_resource
def load_model():
    """Load Hugging Face sentiment analysis pipeline"""
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

sentiment_analyzer = load_model()

# ==================== HELPER FUNCTIONS ====================
def clean_text_simple(text):
    """Basic text cleaning - keep it simple for speed"""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s!?]', '', text)   # Keep letters and basic punctuation
    text = re.sub(r'\s+', ' ', text).strip()    # Remove extra spaces
    return text

def analyze_sentiment(text):
    """
    Analyze sentiment using pre-trained model
    Returns: dict with label, score, emoji
    """
    if not text or len(text.strip()) < 3:
        return None

    cleaned = clean_text_simple(text)
    result = sentiment_analyzer(cleaned)[0]

    # Format result
    label = result['label']
    score = result['score']

    # Convert to positive/negative/neutral
    if label == 'POSITIVE':
        sentiment = 'Positive'
        emoji = '😊'
        color = 'green'
    else:
        sentiment = 'Negative'
        emoji = '😞'
        color = 'red'

    return {
        'sentiment': sentiment,
        'confidence': round(score * 100, 1),
        'emoji': emoji,
        'color': color
    }

def create_sentiment_badge(result):
    """Create a nice-looking sentiment badge"""
    st.markdown(f"""
        <div style='
            background-color: {"#d4edda" if result["color"] == "green" else "#f8d7da"};
            border: 2px solid {"#28a745" if result["color"] == "green" else "#dc3545"};
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            margin: 20px 0;
        '>
            <h1 style='margin: 0; color: {"#28a745" if result["color"] == "green" else "#dc3545"};'>
                {result['emoji']} {result['sentiment']}
            </h1>
            <h3 style='margin: 10px 0; color: #666;'>
                Confidence: {result['confidence']}%
            </h3>
        </div>
    """, unsafe_allow_html=True)

# ==================== UI STYLING ====================
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    h1 {
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-size: 18px;
        padding: 0.5rem;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background-color: #1557a0;
    }
    </style>""", unsafe_allow_html=True)

# ==================== MAIN APP ====================
st.title("📊 Customer Review Sentiment Analyzer")
st.markdown("*Powered by AI | Workshop Demo Version*")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("About")
    st.info("""
    This app analyzes customer reviews to determine if they're positive or negative.

    **Features:**
    - Single review analysis
    - Bulk CSV analysis
    - Visual insights

    **Model:** DistilBERT (90%+ accuracy)
    """)

    st.header("Sample Reviews")
    st.text("Positive:\n'Great product! Very satisfied.'")
    st.text("Negative:\n'Poor quality. Not worth it.'")

# ==================== TAB 1: SINGLE REVIEW ====================
tab1, tab2 = st.tabs(["📝 Single Review", "📊 Bulk Analysis"])

with tab1:
    st.header("Analyze a Single Review")

    # Input
    review_text = st.text_area(
        "Paste or type a customer review:",
        height=150,
        placeholder="Example: This product exceeded my expectations! Great quality and fast delivery."
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button("🔍 Analyze Sentiment", use_container_width=True)

    if analyze_button and review_text:
        with st.spinner("Analyzing..."):
            result = analyze_sentiment(review_text)

            if result:
                # Display result
                create_sentiment_badge(result)

                # Show cleaned text
                with st.expander("📋 Processed Text"):
                    st.write(clean_text_simple(review_text))

                # Confidence meter
                st.subheader("Confidence Level")
                st.progress(result['confidence'] / 100)
            else:
                st.error("⚠️ Please enter a valid review (at least 3 characters)")

# ==================== TAB 2: BULK CSV ====================
with tab2:
    st.header("Analyze Multiple Reviews from CSV")

    # Instructions
    st.info("""
    📤 **Upload a CSV file** with a column named 'review' or 'text'

    Example format:
    ```
    review
    "Great product, very satisfied"
    "Poor quality, disappointed"
    "Average, nothing special"
    ```
    """)

    # File upload
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="CSV must have a 'review' or 'text' column"
    )

    if uploaded_file:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)

            # Find review column
            review_col = None
            for col in ['review', 'text', 'Review', 'Text', 'reviews', 'comment']:
                if col in df.columns:
                    review_col = col
                    break

            if not review_col:
                st.error("❌ CSV must have a column named 'review' or 'text'")
            else:
                st.success(f"✅ Found {len(df)} reviews in '{review_col}' column")

                # Show sample
                with st.expander("👀 Preview Data"):
                    st.dataframe(df.head())

                # Analyze button
                if st.button("🚀 Analyze All Reviews", use_container_width=True):

                    # Progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    results = []

                    # Process each review
                    for idx, row in df.iterrows():
                        review = str(row[review_col])
                        result = analyze_sentiment(review)

                        if result:
                            results.append({
                                'original_review': review,
                                'sentiment': result['sentiment'],
                                'confidence': result['confidence']
                            })

                        # Update progress
                        progress = (idx + 1) / len(df)
                        progress_bar.progress(progress)
                        status_text.text(f"Processing: {idx + 1}/{len(df)} reviews")

                    progress_bar.empty()
                    status_text.empty()

                    # Create results dataframe
                    results_df = pd.DataFrame(results)

                    # ==================== VISUALIZATIONS ====================

                    st.success("✅ Analysis Complete!")
                    st.markdown("---")

                    # Summary metrics
                    col1, col2, col3 = st.columns(3)

                    positive_count = len(results_df[results_df['sentiment'] == 'Positive'])
                    negative_count = len(results_df[results_df['sentiment'] == 'Negative'])
                    avg_confidence = results_df['confidence'].mean()

                    col1.metric("😊 Positive Reviews", f"{positive_count}",
                                f"{positive_count/len(results_df)*100:.1f}%")
                    col2.metric("😞 Negative Reviews", f"{negative_count}",
                                f"{negative_count/len(results_df)*100:.1f}%")
                    col3.metric("🎯 Avg Confidence", f"{avg_confidence:.1f}%")

                    st.markdown("---")

                    # Charts
                    col1, col2 = st.columns(2)

                    with col1:
                        # Pie chart
                        st.subheader("Sentiment Distribution")
                        sentiment_counts = results_df['sentiment'].value_counts()
                        fig = px.pie(
                            values=sentiment_counts.values,
                            names=sentiment_counts.index,
                            color=sentiment_counts.index,
                            color_discrete_map={'Positive': '#28a745', 'Negative': '#dc3545'}
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    with col2:
                        # Confidence distribution
                        st.subheader("Confidence Scores")
                        fig = px.histogram(
                            results_df,
                            x='confidence',
                            nbins=20,
                            color='sentiment',
                            color_discrete_map={'Positive': '#28a745', 'Negative': '#dc3545'}
                        )
                        fig.update_layout(xaxis_title="Confidence (%)", yaxis_title="Count")
                        st.plotly_chart(fig, use_container_width=True)

                    # Sample results
                    st.subheader("📋 Sample Results")
                    st.dataframe(results_df.head(10), use_container_width=True)

                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Full Results",
                        data=csv,
                        file_name="sentiment_analysis_results.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>Built for ML Workshop | Using DistilBERT Pre-trained Model</p>
        <p><small>Expand this project: Add custom training, aspect analysis, deployment</small></p>
    </div>""", unsafe_allow_html=True)
