"""
═══════════════════════════════════════════════════════════════════════════
    SENTIMENT ANALYSIS DASHBOARD
    A beginner-friendly interactive tool for analyzing text sentiment
═══════════════════════════════════════════════════════════════════════════

"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sentiment_analyzer import SentimentAnalyzer
import matplotlib.pyplot as plt
from io import StringIO

# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="💭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# # Custom CSS for better styling
# st.markdown("""
#     <style>
#     .main {
#         padding-top: 1rem;
#     }
#     .metric-card {
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         padding: 20px;
#         border-radius: 10px;
#         color: white;
#         text-align: center;
#     }
#     </style>
#     """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# INITIALIZE SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════

# Session state is used to maintain data across page reruns
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = SentimentAnalyzer()

if 'results_df' not in st.session_state:
    st.session_state.results_df = None

if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

# ═══════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════

st.title("💭 Sentiment Analysis Dashboard")
st.markdown("""
    Analyze the emotional tone of text data using Machine Learning.
    Upload files, paste text, or use sample datasets to get started!
""")

# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR - INPUT OPTIONS
# ═══════════════════════════════════════════════════════════════════════════

st.sidebar.header("📥 Data Input")

# Input method selection
input_method = st.sidebar.radio(
    "Choose input method:",
    ["Paste Text", "Upload CSV", "Sample Dataset"]
)

texts = []

# ───────────────────────────────────────────────────────────────────────────
# METHOD 1: PASTE TEXT
# ───────────────────────────────────────────────────────────────────────────
if input_method == "Paste Text":
    st.sidebar.subheader("Enter Text")
    user_text = st.sidebar.text_area(
        "Paste your text here (one sentence per line):",
        height=150,
        placeholder="Example:\nI love this product!\nThis is terrible.\nIt's okay, nothing special."
    )

    if st.sidebar.button("Analyze Text", key="analyze_text"):
        if user_text.strip():
            texts = [line.strip() for line in user_text.split('\n') if line.strip()]
        else:
            st.sidebar.warning("⚠️ Please enter some text!")

# ───────────────────────────────────────────────────────────────────────────
# METHOD 2: UPLOAD CSV
# ───────────────────────────────────────────────────────────────────────────
elif input_method == "Upload CSV":
    st.sidebar.subheader("Upload CSV File")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="CSV should have a column with text data"
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success("✅ File uploaded successfully!")

            # Let user select which column contains the text
            column = st.sidebar.selectbox(
                "Select the text column:",
                df.columns,
                help="Choose the column that contains the text to analyze"
            )

            if st.sidebar.button("Analyze CSV", key="analyze_csv"):
                texts = df[column].astype(str).tolist()
        except Exception as e:
            st.sidebar.error(f"❌ Error reading file: {str(e)}")

# ───────────────────────────────────────────────────────────────────────────
# METHOD 3: SAMPLE DATASET
# ───────────────────────────────────────────────────────────────────────────
else:
    st.sidebar.subheader("Select Sample Dataset")

    sample_choice = st.sidebar.selectbox(
        "Choose a dataset:",
        ["Movie Reviews", "Product Reviews", "Twitter-like Posts"]
    )

    # Sample datasets
    samples = {
        "Movie Reviews": [
            "This movie was absolutely fantastic! Best film I've seen all year!",
            "Terrible waste of time. The plot made no sense.",
            "It was okay. Some good parts but could be better.",
            "Masterpiece! The acting, cinematography, everything was perfect!",
            "Don't watch this. One of the worst movies ever made.",
            "Pretty good entertainment. Worth watching on a lazy Sunday.",
            "Absolutely horrible! Couldn't even finish it.",
            "Not bad, not great. Average movie for a casual watch.",
        ],
        "Product Reviews": [
            "Love this product! Works exactly as described. Highly recommend!",
            "Complete garbage. Broke after one day of use.",
            "It's fine. Does what it's supposed to do.",
            "Amazing quality and fast shipping! Very satisfied.",
            "Worst purchase ever. Total waste of money.",
            "Good value for the price. No complaints.",
            "Disappointed with the quality. Expected much better.",
            "Perfect! Exceeded my expectations in every way.",
        ],
        "Twitter-like Posts": [
            "Just had the best day ever! 😊 #blessed",
            "I hate Mondays. Everything is going wrong.",
            "It's a beautiful day outside. Feeling grateful!",
            "This is absolutely ridiculous! Can't believe this happened!",
            "Meh, another boring day at work.",
            "Just finished an amazing book! Highly recommend it!",
            "Why is everything so complicated? Frustrated!",
            "Coffee and sunshine - that's all I need today!",
        ]
    }

    if st.sidebar.button("Load Sample Dataset", key="load_sample"):
        texts = samples[sample_choice]
        st.sidebar.success(f"✅ Loaded {len(texts)} {sample_choice.lower()}!")

# ═══════════════════════════════════════════════════════════════════════════
# MAIN ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

if texts:
    # Perform sentiment analysis on all texts
    with st.spinner("🔄 Analyzing sentiment..."):
        results = st.session_state.analyzer.analyze_batch(texts)
        st.session_state.results_df = pd.DataFrame(results)

    st.success("✅ Analysis complete!")

    # ═════════════════════════════════════════════════════════════════════
    # SECTION 1: SUMMARY METRICS
    # ═════════════════════════════════════════════════════════════════════

    st.markdown("---")
    st.header("📊 Summary Metrics")

    df = st.session_state.results_df
    sentiment_counts = df['sentiment'].value_counts()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Texts Analyzed", len(df), delta="samples")

    with col2:
        positive_count = sentiment_counts.get('positive', 0)
        st.metric("😊 Positive", positive_count, delta=f"{positive_count / len(df) * 100:}%")

    with col3:
        negative_count = sentiment_counts.get('negative', 0)
        st.metric("😞 Negative", negative_count, delta=f"{negative_count / len(df) * 100:}%")

    with col4:
        neutral_count = sentiment_counts.get('neutral', 0)
        st.metric("😐 Neutral", neutral_count, delta=f"{neutral_count / len(df) * 100:}%")

    # ═════════════════════════════════════════════════════════════════════
    # SECTION 2: VISUALIZATIONS
    # ═════════════════════════════════════════════════════════════════════

    st.markdown("---")
    st.header("📈 Visualizations")

    viz_col1, viz_col2 = st.columns(2)

    # Sentiment Distribution Pie Chart
    with viz_col1:
        st.subheader("Sentiment Distribution")

        # Create pie chart using Plotly
        fig_pie = go.Figure(data=[go.Pie(
            labels=sentiment_counts.index,
            values=sentiment_counts.values,
            marker=dict(colors=['#2ecc71', '#e74c3c', '#95a5a6']),
            hole=0.7,  # Donut chart
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        )])

        fig_pie.update_layout(
            height=400,
            showlegend=True,
            font=dict(size=12)
        )

        st.plotly_chart(fig_pie, use_container_width=True)

    # Sentiment Bar Chart
    with viz_col2:
        st.subheader("Sentiment Counts")

        # Create bar chart
        fig_bar = go.Figure(data=[go.Bar(
            x=sentiment_counts.index,
            y=sentiment_counts.values,
            marker=dict(color=['#2ecc71', '#e74c3c', '#95a5a6']),
            text=sentiment_counts.values,
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
        )])

        fig_bar.update_layout(
            height=400,
            xaxis_title="Sentiment",
            yaxis_title="Count",
            showlegend=False,
            hovermode='x unified'
        )

        st.plotly_chart(fig_bar, use_container_width=True)

    # Confidence Score Distribution
    st.subheader("Confidence Score Distribution")

    fig_confidence = go.Figure(data=[
        go.Histogram(
            x=df['confidence'],
            nbinsx=20,
            marker_color='#3498db',
            name='Confidence',
            hovertemplate='Confidence: %{x:.3f}<br>Count: %{y}<extra></extra>'
        )
    ])

    fig_confidence.update_layout(
        height=300,
        xaxis_title="Confidence Score",
        yaxis_title="Frequency",
        showlegend=False,
        hovermode='x unified'
    )

    st.plotly_chart(fig_confidence, use_container_width=True)

    # ═════════════════════════════════════════════════════════════════════
    # SECTION 4: DETAILED RESULTS TABLE
    # ═════════════════════════════════════════════════════════════════════

    st.markdown("---")
    st.header("📋 Detailed Results")

    # Add filter options
    filter_col1, filter_col2 = st.columns(2)

    with filter_col1:
        sentiment_filter = st.multiselect(
            "Filter by sentiment:",
            df['sentiment'].unique(),
            default=df['sentiment'].unique()
        )

    with filter_col2:
        confidence_min = st.slider(
            "Minimum confidence score:",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05
        )

    # Apply filters
    filtered_df = df[
        (df['sentiment'].isin(sentiment_filter)) &
        (df['confidence'] >= confidence_min)
        ].copy()

    # Add sentiment emoji
    filtered_df['Sentiment'] = filtered_df['sentiment'].map({
        'positive': '😊 Positive',
        'negative': '😞 Negative',
        'neutral': '😐 Neutral'
    })

    # Display as table
    display_df = filtered_df[['text', 'Sentiment', 'confidence']].copy()
    display_df.columns = ['Text', 'Sentiment', 'Confidence Score']
    display_df['Confidence Score'] = display_df['Confidence Score'].round(4)

    st.dataframe(display_df, use_container_width=True, height=400)

    # ═════════════════════════════════════════════════════════════════════
    # SECTION 5: EXPORT OPTIONS
    # ═════════════════════════════════════════════════════════════════════

    st.markdown("---")
    st.header("💾 Export Results")

    export_col1, export_col2 = st.columns(2)

    with export_col1:
        # Export to CSV
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name="sentiment_analysis_results.csv",
            mime="text/csv"
        )

    with export_col2:
        # Export to JSON
        json = df.to_json(orient='records', indent=2)
        st.download_button(
            label="📥 Download Results as JSON",
            data=json,
            file_name="sentiment_analysis_results.json",
            mime="application/json"
        )


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR: ABOUT SECTION
# ═══════════════════════════════════════════════════════════════════════════

st.sidebar.markdown("---")
st.sidebar.header("ℹ️ About")

with st.sidebar.expander("How it works"):
    st.markdown("""
    **Sentiment Analysis** is the process of determining the emotional tone 
    of text (positive, negative, or neutral).

    This dashboard uses **Machine Learning** specifically:
    - **TF-IDF** vectorization (converts text to numbers)
    - **Naive Bayes** classifier (learns sentiment patterns)

    **Confidence Score**: How confident the model is in its prediction
    (higher = more confident)
    """)

with st.sidebar.expander("Tech Stack"):
    st.markdown("""
    - **Streamlit**: User interface
    - **scikit-learn**: Machine learning
    - **Plotly**: Interactive charts
    - **Python**: Programming language
    """)

with st.sidebar.expander("Tips & Tricks"):
    st.markdown("""
    1. **Better Results**: Use longer, more descriptive texts
    2. **Performance**: Works best with 10-1000 texts at a time
    3. **Accuracy**: The model learns from patterns in the data
    4. **Experimentation**: Try different datasets to explore!
    """)

