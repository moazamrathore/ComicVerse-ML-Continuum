import time
import pandas as pd
import streamlit as st
import plotly.express as px
import yfinance as yf
from models import generate_sample_data

# Welcome page with hero information
def render_welcome_page():
    st.markdown("""
    <div class="welcome-container">
        <h1 class="centered-header">🦸‍♂️ ComicVerse ML Continuum</h1>
        <h3 class="centered-header">Where Financial Data meets Superhero Powers!</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    Welcome to the Superhero Financial ML Universe, where the analytical powers of your favorite superheroes are harnessed to tackle financial data! 
    Our heroes use their unique abilities to analyze market trends, predict outcomes, and cluster data points for maximum insight.
    
    Choose your superhero to explore their specialized machine learning powers, or learn more about our incredible roster of heroes!
    """)
    
    # Quick access to hero pages
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="hero-card batman-card">
            <h3>Batman</h3>
            <p>The Dark Knight's Linear Regression</p>
            <p><em>"Predicting values with detective precision"</em></p>
        </div>
        """, unsafe_allow_html=True)
        
    
    with col2:
        st.markdown("""
        <div class="hero-card spiderman-card">
            <h3>Spider-Man</h3>
            <p>The Web-Slinger's Logistic Regression</p>
            <p><em>"Classifying with spider-sense accuracy"</em></p>
        </div>
        """, unsafe_allow_html=True)
        
    
    with col3:
        st.markdown("""
        <div class="hero-card thanos-card">
            <h3>Thanos</h3>
            <p>The Mad Titan's K-Means Clustering</p>
            <p><em>"Perfectly balanced data grouping"</em></p>
        </div>
        """, unsafe_allow_html=True)
        
    
    # DC heroes section
    st.markdown("""
    <div class="hero-team">
        <h2 class="dc-header">⚡ DC Universe Heroes</h2>
    </div>
    """, unsafe_allow_html=True)
    
    dc_heroes = [
        {"name": "Batman", "power": "Linear Regression", "image": "https://static.thenounproject.com/png/157453-200.png"},
        {"name": "Superman", "power": "Random Forest", "image": "https://i.pinimg.com/736x/c5/33/e7/c533e7fffc2be890fbadcc85b729253c.jpg"},
        {"name": "Wonder Woman", "power": "Support Vector Machines", "image": "https://play-lh.googleusercontent.com/gle-JZB2RR1-Gky1KniUlQnZ5Nv1BfP1TpdVFDxCmVquK__JM4Ql0-gl4UFmuP63zek=w240-h480-rw"},
        {"name": "The Flash", "power": "Real-time Analysis", "image": "https://cdn0.iconfinder.com/data/icons/famous-character-vol-1-colored/48/JD-18-512.png"},
        {"name": "Aquaman", "power": "Time Series Analysis", "image": "https://i.pinimg.com/736x/7d/84/be/7d84be977fc9b7f1b3b3e6b53b137506.jpg"},
        {"name": "Green Lantern", "power": "Dimensionality Reduction", "image": "https://www.pngplay.com/wp-content/uploads/12/Green-Lantern-Transparent-Free-PNG.png"}
    ]
    
    # Display DC heroes
    dc_cols = st.columns(6)
    for i, hero in enumerate(dc_heroes):
        with dc_cols[i]:
            st.markdown(f"""
            <div class="hero-profile">
                <img src="{hero['image']}" width="80" height="80" style="border-radius: 50%; object-fit: cover;">
                <div class="hero-name">{hero['name']}</div>
                <div class="hero-power">{hero['power']}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Marvel heroes section
    st.markdown("""
    <div class="hero-team">
        <h2 class="marvel-header">🔴 Marvel Universe Heroes</h2>
    </div>
    """, unsafe_allow_html=True)
    
    marvel_heroes = [
        {"name": "Spider-Man", "power": "Logistic Regression", "image": "https://icons.veryicon.com/png/o/movie--tv/movie-hero-icon/spider-man-5.png"},
        {"name": "Iron Man", "power": "Neural Networks", "image": "https://icons.veryicon.com/png/o/movie--tv/movie-hero-icon/iron-man-6.png"},
        {"name": "Captain America", "power": "Anomaly Detection", "image": "https://cdn1.iconfinder.com/data/icons/user-pop-culture/512/Cap-512.png"},
        {"name": "Thor", "power": "Power Analysis", "image": "https://i.pinimg.com/736x/fd/9e/5c/fd9e5cd86b459f402c896e22e0e53f1b.jpg"},
        {"name": "Black Widow", "power": "Sequential Pattern Mining", "image": "https://i.pinimg.com/736x/e2/21/a4/e221a4aed0ede751e330a934268ae66b.jpg"},
        {"name": "Thanos", "power": "K-Means Clustering", "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQb6WqSw2oEGKsqfruV7EsHDvoNZoaVtSXnHg&s"}
    ]
    
    # Display Marvel heroes
    marvel_cols = st.columns(6)
    for i, hero in enumerate(marvel_heroes):
        with marvel_cols[i]:
            st.markdown(f"""
            <div class="hero-profile">
                <img src="{hero['image']}" width="80" height="80" style="border-radius: 50%; object-fit: cover;">
                <div class="hero-name">{hero['name']}</div>
                <div class="hero-power">{hero['power']}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Data source section
    st.markdown("---")
    st.subheader("📊 Select Your Data Source")
    
    data_source = st.radio(
        "",
        ["Upload your own files", "Yahoo Finance API", "Sample Datasets"],
        horizontal=True,
    )
    
    # Conditional display based on data source selection
    if data_source == "Upload your own files":
        st.markdown("""
        <div class="batman-panel">
            <h3>🦇 Batman's File Analyzer</h3>
            <p>"I'm good at puzzling through data others have missed."</p>
        </div>
        """, unsafe_allow_html=True)
        
        file_type = st.radio("Select file type:", ["CSV", "Excel"], horizontal=True)
        
        if file_type == "CSV":
            uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.session_state.data = df
                    st.success("File successfully uploaded to the Batcomputer!")
                    st.dataframe(df.head())
                except Exception as e:
                    st.error(f"Error reading file: {e}")
        else:
            uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"])
            if uploaded_file is not None:
                try:
                    df = pd.read_excel(uploaded_file)
                    st.session_state.data = df
                    st.success("File successfully uploaded to the Batcomputer!")
                    st.dataframe(df.head())
                except Exception as e:
                    st.error(f"Error reading file: {e}")
                
    elif data_source == "Yahoo Finance API":
        st.markdown("""
        <div class="spiderman-panel">
            <h3>🕸️ Spider-Man's Web Crawler</h3>
            <p>"Your friendly neighborhood data collector is on the job!"</p>
        </div>
        """, unsafe_allow_html=True)
        
        ticker_symbol = st.text_input("Enter Stock Ticker Symbol (e.g., AAPL, MSFT, GOOG)")
        period = st.select_slider("Select Time Period", 
                                options=["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"])
        
        if ticker_symbol:
            try:
                with st.spinner(f"Spider-Man is swinging through the web to fetch {ticker_symbol} data..."):
                    df = yf.download(ticker_symbol, period=period)
                    if not df.empty:
                        # Reset index to make Date a column
                        df = df.reset_index()
                        st.session_state.data = df
                        st.success(f"Thwip! Successfully fetched {ticker_symbol} data!")
                        st.dataframe(df.head())
                        
                        # Display a quick chart
                        fig = px.line(df, x='Date', y='Close', title=f'{ticker_symbol} Stock Price')
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("No data found. Is that ticker correct?")
            except Exception as e:
                st.error(f"My Spider-sense detected a problem: {e}")
                
    else:  # Sample Datasets
        st.markdown("""
        <div class="thanos-panel">
            <h3>💎 Thanos' Infinity Datasets</h3>
            <p>"I call that... mercy. Providing you with perfectly balanced datasets."</p>
        </div>
        """, unsafe_allow_html=True)
        
        dataset_choice = st.selectbox(
            "Choose a sample dataset:",
            ["S&P 500 Historical", "Bitcoin Price History", "Stock Market Crash 2008", 
             "Tech Stocks Performance", "Financial Classification Data", "Market Clustering Dataset"]
        )
        
        if st.button("Snap! Get the Dataset"):
            with st.spinner("Balancing the universe... and your dataset"):
                # Add slight delay for effect
                time.sleep(1)
                
                # Generate appropriate sample data based on selection
                if "Classification" in dataset_choice:
                    df = generate_sample_data("binary")
                elif "Clustering" in dataset_choice:
                    df = generate_sample_data("cluster")
                else:
                    df = generate_sample_data("stock")
                
                st.session_state.data = df
                st.success(f"With a snap, Thanos has balanced the {dataset_choice} for you!")
                st.dataframe(df.head())
