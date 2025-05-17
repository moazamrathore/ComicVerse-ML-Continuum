import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
from models import run_linear_regression
from themes import show_themed_quote, create_download_buttons

# Batman Linear Regression page
def render():
    # Batman themed header
    st.markdown("""
    <div style="background: url('https://cdn.pixabay.com/photo/2024/06/22/16/55/ai-generated-8846672_640.jpg'); background-size: cover; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h1 class="batman-header">🦇 Batman's Detective Lab</h1>
        <h3 style="color: #e6e6e6;">Linear Regression Analysis</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Display Batman quote
    show_themed_quote("batman", "general")
    
    # Check if data is loaded
    if st.session_state.data is None:
        st.warning("No data loaded. Return to Mission Control to select your dataset.")
        if st.button("Return to Mission Control"):
            st.query_params["page"] = "welcome"
            st.rerun()
        return
    
    # Show data preprocessing options
    st.markdown("""
    <div class="batman-panel">
        <h3>Data Preprocessing & Feature Selection</h3>
        <p>"Preparation is everything in this line of work."</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get a clean copy of the data
    df = st.session_state.data.copy()
    
    # Display head of the dataset
    with st.expander("View Dataset Preview"):
        st.dataframe(df.head())
        
        # Display basic data info
        st.write("**Dataset Info:**")
        buffer = StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)
        
        # Display data statistics
        st.write("**Dataset Statistics:**")
        st.write(df.describe())
    
    # Preprocessing steps
    st.subheader("Data Preprocessing")
    
    # Handle missing values
    missing_values = df.isnull().sum().sum()
    if missing_values > 0:
        st.warning(f"Dataset contains {missing_values} missing values.")
        missing_strategy = st.radio(
            "Batman's Missing Value Strategy:",
            ["Drop rows with missing values", "Fill with mean/mode", "Fill with median"],
            horizontal=True
        )
        
        if missing_strategy == "Drop rows with missing values":
            df = df.dropna()
            st.success("Rows with missing values have been eliminated!")
        elif missing_strategy == "Fill with mean/mode":
            # Fill numerical with mean, categorical with mode
            for col in df.columns:
                if df[col].dtype.kind in 'ifc':  # numeric columns
                    df[col].fillna(df[col].mean(), inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0], inplace=True)
            st.success("Missing values filled with mean/mode!")
        else:
            # Fill with median
            for col in df.columns:
                if df[col].dtype.kind in 'ifc':  # numeric columns
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0], inplace=True)
            st.success("Missing values filled with median!")
    
    # Feature selection
    st.subheader("Feature Selection")
    
    # Only show numeric columns for linear regression
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if len(numeric_cols) < 2:
        st.error("Not enough numeric columns for linear regression!")
        return
    
    # Select target variable
    target_col = st.selectbox("Select Target Variable:", numeric_cols)
    
    # Select predictor variables
    available_features = [col for col in numeric_cols if col != target_col]
    if not available_features:
        st.error("Not enough features available after selecting target!")
        return
    
    selected_features = st.multiselect(
        "Select Features for Prediction:",
        available_features,
        default=available_features[:min(3, len(available_features))]
    )
    
    if not selected_features:
        st.warning("Please select at least one feature for prediction.")
        return
    
    # Model training section
    st.markdown("""
    <div class="batman-panel">
        <h3>Model Training</h3>
        <p>"Training is nothing. Will is everything."</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model parameters
    test_size = st.slider("Test Data Size (%)", 10, 50, 20) / 100
    
    # Train the model
    if st.button("Analyze with Batman's Precision", key="batman_analyze"):
        with st.spinner("Batman is analyzing the data..."):
            # Add slight delay for effect
            time.sleep(1)
            
            # Run linear regression
            model_results = run_linear_regression(df, selected_features, target_col, test_size)
            st.session_state.model_results = model_results
            
            # Show success quote
            show_themed_quote("batman", "success")
    
    # Display results if available
    if st.session_state.model_results:
        model_results = st.session_state.model_results
        
        st.markdown("""
        <div class="results-container">
            <h2 class="spiderman-header">Linear Regression Results</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Display model metrics
        st.subheader("Model Performance")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="batman-metric-container">
                <div class="batman-metric-label">R-squared (Test)</div>
                <div class="batman-metric-value">{model_results['test_r2']:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="batman-metric-container">
                <div class="batman-metric-label">MSE (Test)</div>
                <div class="batman-metric-value">{model_results['test_mse']:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="batman-metric-container">
                <div class="batman-metric-label">RMSE (Test)</div>
                <div class="batman-metric-value">{np.sqrt(model_results['test_mse']):.4f}</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="batman-metric-container">
                <div class="batman-metric-label">Intercept</div>
                <div class="batman-metric-value">{model_results['model'].intercept_:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Display feature importance
        st.subheader("Feature Importance")
        
        fig_importance = px.bar(
            model_results['coefficients'],
            x='Coefficient',
            y='Feature',
            orientation='h',
            color='Coefficient',
            color_continuous_scale=['#4169e1', '#1a1a1a'],
            title="Feature Coefficients"
        )
        fig_importance.update_layout(
            plot_bgcolor='rgba(0,0,0,0.1)',
            paper_bgcolor='rgba(0,0,0,0.1)',
            font=dict(color='white')
        )
        st.plotly_chart(fig_importance, use_container_width=True)
        
        # Actual vs Predicted plot
        st.subheader("Actual vs Predicted Values")
        
        fig_pred = px.scatter(
            x=model_results['y_test'],
            y=model_results['y_pred_test'],
            labels={'x': 'Actual Values', 'y': 'Predicted Values'},
            title="Batman's Prediction Accuracy"
        )
        
        # Add the ideal line
        min_val = min(model_results['y_test'].min(), model_results['y_pred_test'].min())
        max_val = max(model_results['y_test'].max(), model_results['y_pred_test'].max())
        fig_pred.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='Perfect Prediction'
            )
        )
        
        fig_pred.update_layout(
            plot_bgcolor='rgba(0,0,0,0.1)',
            paper_bgcolor='rgba(0,0,0,0.1)',
            font=dict(color='white')
        )
        st.plotly_chart(fig_pred, use_container_width=True)
        
        # Residuals plot
        st.subheader("Residuals Analysis")
        
        residuals = model_results['y_test'] - model_results['y_pred_test']
        fig_residuals = px.scatter(
            x=model_results['y_pred_test'],
            y=residuals,
            labels={'x': 'Predicted Values', 'y': 'Residuals'},
            title="Batman's Residual Analysis"
        )
        
        # Add a horizontal line at y=0
        fig_residuals.add_hline(
            y=0,
            line_dash="dash",
            line_color="red",
            annotation_text="Zero Residual",
            annotation_position="bottom right"
        )
        
        fig_residuals.update_layout(
            plot_bgcolor='rgba(0,0,0,0.1)',
            paper_bgcolor='rgba(0,0,0,0.1)',
            font=dict(color='white')
        )
        st.plotly_chart(fig_residuals, use_container_width=True)
        
        # Prediction section
        st.markdown("""
        <div class="batman-panel">
            <h3>Make New Predictions</h3>
            <p>"It's not who I am underneath, but what I predict that defines me."</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Get feature inputs
        feature_inputs = {}
        for feature in selected_features:
            min_val = float(df[feature].min())
            max_val = float(df[feature].max())
            default_val = float(df[feature].mean())
            feature_inputs[feature] = st.slider(
                f"{feature}:",
                min_val,
                max_val,
                default_val
            )
        
        # Make prediction
        if st.button("Make Prediction", key="batman_predict"):
            # Create input array
            X_new = np.array([feature_inputs[feature] for feature in selected_features]).reshape(1, -1)
            
            # Scale the input
            X_new_scaled = model_results['scaler'].transform(X_new)
            
            # Make prediction
            prediction = model_results['model'].predict(X_new_scaled)[0]
            
            # Display prediction result
            st.markdown(f"""
            <div class="batman-panel">
                <h3>Batman's Prediction</h3>
                <p>The predicted value of {target_col} is:</p>
                <h2 style="color: #4169e1; text-align: center;">{prediction:.4f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Create download buttons for exporting data
        create_download_buttons(df, model_results['model_summary'], None, "batman")
