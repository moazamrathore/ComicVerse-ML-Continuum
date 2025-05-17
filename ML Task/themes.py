import base64
from io import BytesIO
import random
import pandas as pd
import streamlit as st

# Hero quotes for different contexts
batman_quotes = {
    "general": [
        "It's not who I am underneath, but what I predict that defines me.",
        "Criminals are a superstitious and cowardly lot. So are market downturns.",
        "I analyze this data because no one else can.",
        "Data preparation isn't something I do. It's something I am."
    ],
    "success": [
        "I'm Batman. I make predictions in a city that has none.",
        "Why do we predict? So we can learn to pick ourselves up when markets fall.",
        "It's not who I am inside, but what my models predict that defines me."
    ],
    "error": [
        "Sometimes the truth isn't good enough. Sometimes people deserve better data.",
        "The data is corrupt. The data is flawed.",
        "Our data needs an overhaul. A fresh start."
    ]
}

spiderman_quotes = {
    "general": [
        "With great probabilities comes great predictive power.",
        "Just your friendly neighborhood prediction system!",
        "My Spider-sense helps me detect patterns in financial data.",
        "I'm swinging through your data points as we speak!"
    ],
    "success": [
        "My Spider-sense is tingling! This prediction is looking good!",
        "When you help someone predict market trends, you help everyone.",
        "Whatever life holds in store for me, I will never forget these basic principles: financial responsibility comes from financial knowledge."
    ],
    "error": [
        "Uh-oh! My web fluid is jammed. We need to recalibrate this model!",
        "I guess one person really can make a difference... by cleaning this data.",
        "No matter what I do, no matter how hard I try, the outliers are always there."
    ]
}

thanos_quotes = {
    "general": [
        "The work of clustering begins. It is... inevitable.",
        "Today, I cluster data. With all six Infinity Stones, I could simply snap my fingers. The outliers would all cease to exist.",
        "The universe's financial patterns require correction.",
        "I call my data technique... mercy."
    ],
    "success": [
        "Perfectly balanced, as all financial portfolios should be.",
        "I finally rest, and watch the sunrise on a grateful dataset.",
        "The work is done. The clusters are complete."
    ],
    "error": [
        "The hardest choices require the strongest algorithms.",
        "Reality is often disappointing. Your model accuracy, even more so.",
        "You're not the only one cursed with inadequate data."
    ]
}


# Load CSS and theme assets
def load_css():
    # Base CSS for the entire app
    st.markdown("""
    <style>
    /* Common Styles */
    .superhero-welcome {text-align: center; margin-bottom: 2rem;}
    .quote-text {font-style: italic; font-size: 1.1rem; margin-bottom: 5px;}
    .quote-attribution {text-align: right; font-weight: bold;}
    .stButton>button {width: 100%;}
    .centered-header {text-align: center; margin-bottom: 20px;}
    .results-container {background-color: rgba(255, 255, 255, 0.1); padding: 20px; border-radius: 10px; margin-top: 20px;}
    
    /* Batman Theme */
    .batman-panel {background: linear-gradient(#1a1a1a, #2c3e50); padding: 20px; border-radius: 10px; color: #e6e6e6; margin-bottom: 15px;}
    .batman-quote {background-color: #1a1a1a; border-left: 5px solid #4169e1; padding: 15px; margin: 10px 0; color: #e6e6e6;}
    .batman-header {font-family: 'Gotham', sans-serif; color: #e6e6e6; text-shadow: 2px 2px 4px #000000;}
    .batman-button button {background-color: #4169e1; color: white;}
    .batman-metric-container {background-color: #1a1a1a; padding: 10px; border-radius: 5px; margin: 5px; text-align: center;}
    .batman-metric-label {color: #4169e1; font-weight: bold; font-size: 0.9rem;}
    .batman-metric-value {color: white; font-size: 1.5rem; font-weight: bold;}
    
    /* Spider-Man Theme */
    .spiderman-panel {background: linear-gradient(#b71c1c, #0d47a1); padding: 20px; border-radius: 10px; color: white; margin-bottom: 15px;}
    .spiderman-quote {background-color: #b71c1c; border-left: 5px solid #0d47a1; padding: 15px; margin: 10px 0; color: white;}
    .spiderman-header {font-family: 'Comic Sans MS', cursive; color: #ff0000; text-shadow: 2px 2px 4px #000000;}
    .spiderman-button button {background-color: #0d47a1; color: white;}
    .spiderman-metric-container {background-color: #0d47a1; padding: 10px; border-radius: 5px; margin: 5px; text-align: center;}
    .spiderman-metric-label {color: #ff0000; font-weight: bold; font-size: 0.9rem;}
    .spiderman-metric-value {color: white; font-size: 1.5rem; font-weight: bold;}
    
    /* Thanos Theme */
    .thanos-panel {background: linear-gradient(#4a148c, #7b1fa2); padding: 20px; border-radius: 10px; color: white; margin-bottom: 15px;}
    .thanos-quote {background-color: #4a148c; border-left: 5px solid #ffd700; padding: 15px; margin: 10px 0; color: white;}
    .thanos-header {font-family: 'Arial', sans-serif; color: #ffd700; text-shadow: 2px 2px 4px #4a148c;}
    .thanos-button button {background-color: #7b1fa2; color: white;}
    .thanos-metric-container {background-color: #4a148c; padding: 10px; border-radius: 5px; margin: 5px; text-align: center;}
    .thanos-metric-label {color: #ffd700; font-weight: bold; font-size: 0.9rem;}
    .thanos-metric-value {color: white; font-size: 1.5rem; font-weight: bold;}
    
    /* Download buttons */
    .download-button {
        display: inline-block;
        padding: 10px 15px;
        background-color: #2c3e50;
        color: white;
        text-decoration: none;
        border-radius: 5px;
        margin-top: 10px;
        font-weight: bold;
        text-align: center;
    }
    .batman-download {background-color: #4169e1;}
    .spiderman-download {background-color: #b71c1c;}
    .thanos-download {background-color: #7b1fa2;}
    
    /* Welcome page styling */
    .welcome-container {
        background: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)), url('https://i.ytimg.com/vi/n_wu370AShs/sddefault.jpg');
        background-size: cover;
        padding: 30px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
    }
    .hero-card {
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        transition: transform 0.3s;
        cursor: pointer;
    }
    .hero-card:hover {
        transform: scale(1.03);
    }
    .batman-card {background: linear-gradient(#1a1a1a, #2c3e50); color: white;}
    .spiderman-card {background: linear-gradient(#b71c1c, #0d47a1); color: white;}
    .thanos-card {background: linear-gradient(#4a148c, #7b1fa2); color: white;}
    .hero-team {margin-top: 30px; margin-bottom: 15px; text-align: center;}
    .dc-header {color: #0078f2; font-size: 28px; font-weight: bold; text-shadow: 1px 1px 3px black;}
    .marvel-header {color: #e62429; font-size: 28px; font-weight: bold; text-shadow: 1px 1px 3px black;}
    .hero-grid {display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 15px; margin-top: 15px;}
    .hero-profile {text-align: center; background-color: rgba(0,0,0,0.5); padding: 15px; border-radius: 10px; transition: transform 0.2s; height: 240px;}
    .hero-profile:hover {transform: scale(1.05);}
    .hero-name {font-weight: bold; margin-top: 8px;}
    .hero-power {font-style: italic; font-size: 0.9em; color: #cccccc;}
    
    /* App Header Styling */
    .show {display: block;}
    .app-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 10px 20px;
        background-color: #1a1a1a;
        margin-bottom: 20px;
        border-radius: 10px;
    }
    .app-title {
        font-size: 24px;
        font-weight: bold;
        color: white;
    }
    .header-controls {
        display: flex;
        align-items: center;
    }
                
    #MainMenu, header[data-testid="stHeader"], .stDeployButton {display: none !important;}

    .st-emotion-cache-1w723zb{
                padding: 2rem 1rem 10rem;}
    </style>
    
    """, unsafe_allow_html=True)

# Function to display themed quotes
def show_themed_quote(theme, category="general"):
    quotes = {
        "batman": batman_quotes,
        "spiderman": spiderman_quotes,
        "thanos": thanos_quotes
    }
    
    quote = random.choice(quotes.get(theme, {}).get(category, ["Data analysis is a hero's job."]))
    
    if theme == "batman":
        st.markdown(f"""
        <div class="batman-quote">
            <div class="quote-text">"{quote}"</div>
            <div class="quote-attribution">— Batman</div>
        </div>
        """, unsafe_allow_html=True)
    elif theme == "spiderman":
        st.markdown(f"""
        <div class="spiderman-quote">
            <div class="quote-text">"{quote}"</div>
            <div class="quote-attribution">— Spider-Man</div>
        </div>
        """, unsafe_allow_html=True)
    elif theme == "thanos":
        st.markdown(f"""
        <div class="thanos-quote">
            <div class="quote-text">"{quote}"</div>
            <div class="quote-attribution">— Thanos</div>
        </div>
        """, unsafe_allow_html=True)

# Function to create download buttons for exporting data
def create_download_buttons(df, model_results, figures=None, theme="batman"):
    st.markdown("---")
    
    if theme == "batman":
        st.markdown("""
        <div class="batman-panel">
            <h3>🦇 Batman's Evidence Locker</h3>
            <p>"Always keep the evidence. You never know when you'll need it."</p>
        </div>
        """, unsafe_allow_html=True)
        download_class = "batman-download"
        prefix = "batman_analysis"
        
    elif theme == "spiderman":
        st.markdown("""
        <div class="spiderman-panel">
            <h3>🕸️ Spider-Man's Web Archive</h3>
            <p>"Gotta save this for my homework... I mean, for future reference!"</p>
        </div>
        """, unsafe_allow_html=True)
        download_class = "spiderman-download"
        prefix = "spiderman_analysis"
        
    else:  # thanos
        st.markdown("""
        <div class="thanos-panel">
            <h3>💎 Thanos' Reality Stone Archive</h3>
            <p>"A small price to pay for saving your analysis."</p>
        </div>
        """, unsafe_allow_html=True)
        download_class = "thanos-download"
        prefix = "thanos_analysis"
    
    # Export format selection
    export_format = st.radio("Choose export format:", ["Excel", "CSV", "PDF"], horizontal=True)
    
    if export_format == "Excel":
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer) as writer:
            df.to_excel(writer, sheet_name="Raw Data", index=True)
            
            # Write model results to another sheet
            if model_results:
                results_df = pd.DataFrame([model_results])
                results_df.to_excel(writer, sheet_name="Analysis Results", index=False)
        
        excel_data = excel_buffer.getvalue()
        b64 = base64.b64encode(excel_data).decode()
        href = f'<a href="data:application/vnd.ms-excel;base64,{b64}" download="{prefix}_report.xlsx" class="download-button {download_class}">Download Excel File</a>'
        st.markdown(href, unsafe_allow_html=True)
        
    elif export_format == "CSV":
        csv_buffer = BytesIO()
        df.to_csv(csv_buffer, index=True)
        csv_data = csv_buffer.getvalue()
        b64 = base64.b64encode(csv_data).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{prefix}_data.csv" class="download-button {download_class}">Download CSV File</a>'
        st.markdown(href, unsafe_allow_html=True)
        
    else:  # PDF
        st.info("Generating PDF report with analysis details and visualizations...")
        st.success("PDF report generated!")
        
        # In a real app, you'd generate a proper PDF here
        # This is a placeholder for demonstration
        pdf_data = b"Sample PDF data"
        b64 = base64.b64encode(pdf_data).decode()
        href = f'<a href="data:application/pdf;base64,{b64}" download="{prefix}_report.pdf" class="download-button {download_class}">Download PDF Report</a>'
        st.markdown(href, unsafe_allow_html=True)

# Custom dropdown navigation
def create_header_navigation():
    col1, col2, col3, col4 = st.columns([2,3,3,3])
    with col1:
        if st.button("🏠 Mission Control"):
            st.query_params["page"] = "welcome"
            st.rerun()
    with col2:
        if st.button("🦇 Batman: Linear Regression"):
            st.query_params["page"] = "batman"
            st.rerun()
    with col3:
        if st.button("🕸️ Spider-Man: Logistic Regression"):
            st.query_params["page"] = "spiderman"
            st.rerun()
    with col4:
        if st.button("💎 Thanos: K-Means Clustering"):
            st.query_params["page"] = "thanos"
            st.rerun()
    st.markdown("---")
