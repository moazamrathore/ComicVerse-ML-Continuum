from io import StringIO
import time
import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from models import run_logistic_regression
from themes import show_themed_quote, create_download_buttons

def render():
    # Spider-Man themed header
    st.markdown("""
    <div style="background: url('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTN0qjBHdeGKVDSY5G-hCbKuF9muyFswDVidA&s'); background-size: cover; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h1 class="spiderman-header">🕸️ Spider-Man's Web of Decisions</h1>
        <h3 style="color: white;">Logistic Regression Classification</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Display Spider-Man quote
    show_themed_quote("spiderman", "general")
    
    # Check if data is loaded
    if st.session_state.data is None:
        st.warning("No data loaded. Return to Mission Control to select your dataset.")
        if st.button("Return to Mission Control"):
            st.experimental_set_query_params(page="welcome")
            st.experimental_rerun()
        return
    
    # Show data preprocessing options
    st.markdown("""
    <div class="spiderman-panel">
        <h3>Data Preprocessing & Feature Selection</h3>
        <p>"With great data comes great responsibility!"</p>
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
    
    # Check for target column
    if 'Target' not in df.columns:
        st.warning("No binary target column found. Spider-Man will create one for you!")
        
        # Create a target column
        target_options = st.radio(
            "Choose how to create a binary target:",
            ["Based on a threshold value", "Price movement direction (for stock data)", "Random classification (demo only)"],
            horizontal=True
        )
        
        if target_options == "Based on a threshold value":
            # Select column to threshold
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            threshold_col = st.selectbox("Select column to threshold:", numeric_cols)
            threshold_value = st.slider(
                "Threshold value:",
                float(df[threshold_col].min()),
                float(df[threshold_col].max()),
                float(df[threshold_col].median())
            )
            df['Target'] = (df[threshold_col] > threshold_value).astype(int)
            st.success(f"Target created: 1 if {threshold_col} > {threshold_value}, else 0")
            
        elif target_options == "Price movement direction (for stock data)":
            if 'Close' in df.columns:
                # For stock data, create target based on price movement
                df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
                df = df.dropna()  # Remove the last row where target is NA
                st.success("Target created: 1 if price goes up next period, 0 if down")
            else:
                st.error("No 'Close' column found in the data. Please use another method.")
                return
        else:
            # Random classification for demo
            df['Target'] = np.random.randint(0, 2, size=len(df))
            st.success("Random binary target created for demonstration")
    
    # Handle missing values
    missing_values = df.isnull().sum().sum()
    if missing_values > 0:
        st.warning(f"Dataset contains {missing_values} missing values.")
        missing_strategy = st.radio(
            "Spider-Man's Missing Value Strategy:",
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
    
    # Get all columns except Target
    all_cols = [col for col in df.columns if col != 'Target']
    
    # Separate numerical and categorical columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != 'Target']
    
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Select features
    selected_numeric_features = st.multiselect(
        "Select Numeric Features:",
        numeric_cols,
        default=numeric_cols[:min(3, len(numeric_cols))]
    )
    
    selected_categorical_features = st.multiselect(
        "Select Categorical Features:",
        categorical_cols,
        default=categorical_cols[:min(2, len(categorical_cols))]
    )
    
    selected_features = selected_numeric_features + selected_categorical_features
    
    if not selected_features:
        st.warning("Please select at least one feature for classification.")
        return
    
    # Convert categorical features
    if selected_categorical_features:
        st.info("Spider-Man will automatically convert categorical features using Label Encoding.")
        
        # Apply label encoding to categorical features
        for col in selected_categorical_features:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    
    # Model training section
    st.markdown("""
    <div class="spiderman-panel">
        <h3>Model Training</h3>
        <p>"Whatever comes our way, whatever battle we have raging inside us, we always have a choice."</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model parameters
    test_size = st.slider("Test Data Size (%)", 10, 50, 20) / 100
    
    # Train the model
    if st.button("Analyze with Spider-Sense", key="spiderman_analyze"):
        with st.spinner("Spider-Man is analyzing the data with his Spider-Sense..."):
            # Add slight delay for effect
            time.sleep(1)
            
            # Run logistic regression
            model_results = run_logistic_regression(df, selected_features, 'Target', test_size)
            st.session_state.model_results = model_results
            
            # Show success quote
            show_themed_quote("spiderman", "success")
    
    # Display results if available
    if st.session_state.model_results is not None:
        model_results = st.session_state.model_results
        
        st.markdown("""
        <div class="results-container">
            <h2 class="spiderman-header">Logistic Regression Results</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Display model metrics
        st.subheader("Model Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="spiderman-metric-container">
                <div class="spiderman-metric-label">Accuracy</div>
                <div class="spiderman-metric-value">{model_results['accuracy_test']:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="spiderman-metric-container">
                <div class="spiderman-metric-label">Precision</div>
                <div class="spiderman-metric-value">{model_results['precision']:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
            <div class="spiderman-metric-container">
                <div class="spiderman-metric-label">Recall</div>
                <div class="spiderman-metric-value">{model_results['recall']:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            st.markdown(f"""
            <div class="spiderman-metric-container">
                <div class="spiderman-metric-label">F1 Score</div>
                <div class="spiderman-metric-value">{model_results['f1']:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Display confusion matrix
        st.subheader("Confusion Matrix")
        
        # Calculate confusion matrix values
        tn, fp, fn, tp = model_results['conf_matrix'].ravel()
        
        # Create confusion matrix visualization
        conf_matrix_df = pd.DataFrame(
            [[tn, fp], [fn, tp]],
            columns=['Predicted Negative', 'Predicted Positive'],
            index=['Actual Negative', 'Actual Positive']
        )
        
        fig_conf = px.imshow(
            conf_matrix_df,
            text_auto=True,
            color_continuous_scale=['#b71c1c', '#0d47a1'],
            aspect="equal",
            title="Spider-Man's Confusion Matrix"
        )
        fig_conf.update_layout(
            plot_bgcolor='rgba(0,0,0,0.1)',
            paper_bgcolor='rgba(0,0,0,0.1)',
            font=dict(color='white')
        )
        st.plotly_chart(fig_conf, use_container_width=True)
        
        # Display feature importance
        st.subheader("Feature Importance")
        
        fig_importance = px.bar(
            model_results['coefficients'],
            x='Coefficient',
            y='Feature',
            orientation='h',
            color='Coefficient',
            color_continuous_scale=['#b71c1c', '#0d47a1'],
            title="Feature Coefficients"
        )
        fig_importance.update_layout(
            plot_bgcolor='rgba(0,0,0,0.1)',
            paper_bgcolor='rgba(0,0,0,0.1)',
            font=dict(color='white')
        )
        st.plotly_chart(fig_importance, use_container_width=True)
        
        # ROC Curve
        st.subheader("ROC Curve")
        
        # Calculate ROC curve
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, thresholds = roc_curve(model_results['y_test'], model_results['y_prob_test'])
        roc_auc = auc(fpr, tpr)
        
        fig_roc = px.area(
            x=fpr, y=tpr,
            title=f'Spider-Man\'s ROC Curve (AUC = {roc_auc:.4f})',
            labels=dict(x='False Positive Rate', y='True Positive Rate'),
            width=700, height=500
        )
        fig_roc.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1
        )
        fig_roc.update_layout(
            plot_bgcolor='rgba(0,0,0,0.1)',
            paper_bgcolor='rgba(0,0,0,0.1)',
            font=dict(color='white')
        )
        st.plotly_chart(fig_roc, use_container_width=True)
        
        # Prediction section
        st.markdown("""
        <div class="spiderman-panel">
            <h3>Make New Predictions</h3>
            <p>"My Spider-Sense is tingling! Let's predict something!"</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Get feature inputs
        feature_inputs = {}
        for feature in selected_features:
            if feature in numeric_cols:
                min_val = float(df[feature].min())
                max_val = float(df[feature].max())
                default_val = float(df[feature].mean())
                feature_inputs[feature] = st.slider(
                    f"{feature}:",
                    min_val,
                    max_val,
                    default_val,
                    key=f"spidey_{feature}"
                )
            else:
                # For categorical features (already encoded)
                unique_values = sorted(df[feature].unique())
                feature_inputs[feature] = st.selectbox(
                    f"{feature}:",
                    unique_values,
                    key=f"spidey_cat_{feature}"
                )
        
        # Make prediction
        if st.button("Make Prediction", key="spiderman_predict"):
            # Create input array
            X_new = np.array([feature_inputs[feature] for feature in selected_features]).reshape(1, -1)
            
            # Scale the input
            X_new_scaled = model_results['scaler'].transform(X_new)
            
            # Make prediction
            prediction_prob = model_results['model'].predict_proba(X_new_scaled)[0][1]
            prediction_class = 1 if prediction_prob > 0.5 else 0
            
            # Display prediction result
            st.markdown(f"""
            <div class="spiderman-panel">
                <h3>Spider-Man's Prediction</h3>
                <p>The predicted class is:</p>
                <h2 style="color: {'#ff0000' if prediction_class == 1 else '#0d47a1'}; text-align: center;">
                    {"Positive (1)" if prediction_class == 1 else "Negative (0)"}
                </h2>
                <p>Probability of positive class: {prediction_prob:.4f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Create a probability gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prediction_prob,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Spider-Sense Probability"},
                gauge={
                    'axis': {'range': [0, 1]},
                    'bar': {'color': "#ff0000"},
                    'steps': [
                        {'range': [0, 0.5], 'color': "#0d47a1"},
                        {'range': [0.5, 1], 'color': "#b71c1c"}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.5
                    }
                }
            ))
            fig_gauge.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0.1)',
                font=dict(color='white')
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Create download buttons for exporting data
        create_download_buttons(df, model_results['model_summary'], None, "spiderman")
