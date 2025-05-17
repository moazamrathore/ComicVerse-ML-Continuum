from io import StringIO
import streamlit as st
import time
import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
from models import run_kmeans_clustering
from themes import show_themed_quote, create_download_buttons

# Thanos K-Means Clustering page
def render():
    # Thanos themed header
    st.markdown("""
    <div style="background: url('https://i.pinimg.com/736x/46/19/99/461999fe4e05205ec4deff47a12ca57e.jpg'); background-size: cover; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h1 class="thanos-header">💎 Thanos' Infinity Clusters</h1>
        <h3 style="color: white;">K-Means Clustering Analysis</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Display Thanos quote
    show_themed_quote("thanos", "general")
    
    # Check if data is loaded
    if st.session_state.data is None:
        st.warning("No data loaded. Return to Mission Control to select your dataset.")
        if st.button("Return to Mission Control"):
            st.experimental_set_query_params(page="welcome")
            st.experimental_rerun()
        return
    
    # Show data preprocessing options
    st.markdown("""
    <div class="thanos-panel">
        <h3>Data Preprocessing & Feature Selection</h3>
        <p>"Perfectly balanced, as all data should be."</p>
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
            "Thanos' Missing Value Strategy:",
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
    
    # Only show numeric columns for clustering
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Exclude specific columns that might not be useful for clustering
    exclude_cols = ['Target', 'index', 'Index', 'id', 'ID']
    available_features = [col for col in numeric_cols if col not in exclude_cols]
    
    if len(available_features) < 2:
        st.error("Not enough numeric columns for clustering!")
        return
    
    # Select features for clustering
    selected_features = st.multiselect(
        "Select Features for Clustering:",
        available_features,
        default=available_features[:min(3, len(available_features))]
    )
    
    if len(selected_features) < 2:
        st.warning("Please select at least two features for clustering.")
        return
    
    # Model parameter - number of clusters
    st.subheader("Number of Clusters")
    
    n_clusters = st.slider("Select Number of Clusters (Infinity Stones):", 2, 8, 4)
    
    # Model training section
    st.markdown("""
    <div class="thanos-panel">
        <h3>Model Training</h3>
        <p>"A small price to pay for perfect clusters."</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Train the model with Thanos snap animation
    if st.button("Snap! Find the Clusters", key="thanos_analyze"):
        with st.spinner("Thanos is balancing the universe..."):
            # Infinity stone collection animation
            st.markdown("""
            <div style="text-align: center;">
                <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR2pM77y8WPQ_wDFA9cGkJkI6ShP104CeCKHQ&s" width="200">
                <p>Collecting Infinity Stones...</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Add slight delay for effect
            time.sleep(1.5)
            
            # Run K-Means clustering
            model_results = run_kmeans_clustering(df, selected_features, n_clusters)
            st.session_state.model_results = model_results
            
            # Snap animation
            st.markdown("""
            <div style="text-align: center;">
                <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRA_jGaodaXeWg9RhkO6UONecQJvJk1j8OyNQ&s" width="300">
                <p>Perfectly balanced...</p>
            </div>
            """, unsafe_allow_html=True)
            
            time.sleep(1)
            
            # Show success quote
            show_themed_quote("thanos", "success")
    
    # Display results if available
    if st.session_state.model_results:
        model_results = st.session_state.model_results
        
        st.markdown("""
        <div class="results-container">
            <h2 class="thanos-header">K-Means Clustering Results</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Display model metrics
        st.subheader("Clustering Performance")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="thanos-metric-container">
                <div class="thanos-metric-label">Number of Clusters</div>
                <div class="thanos-metric-value">{model_results['model_summary']['Number of Clusters']}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="thanos-metric-container">
                <div class="thanos-metric-label">Silhouette Score</div>
                <div class="thanos-metric-value">{model_results['silhouette']:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
            <div class="thanos-metric-container">
                <div class="thanos-metric-label">Inertia</div>
                <div class="thanos-metric-value">{model_results['inertia']:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Display cluster sizes
        st.subheader("Cluster Sizes")
        
        cluster_sizes = model_results['df_with_clusters']['Cluster'].value_counts().sort_index()
        
        fig_sizes = px.bar(
            x=cluster_sizes.index,
            y=cluster_sizes.values,
            labels={'x': 'Cluster', 'y': 'Number of Data Points'},
            title="Thanos' Perfectly Balanced Clusters",
            color=cluster_sizes.index,
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_sizes.update_layout(
            plot_bgcolor='rgba(0,0,0,0.1)',
            paper_bgcolor='rgba(0,0,0,0.1)',
            font=dict(color='white')
        )
        st.plotly_chart(fig_sizes, use_container_width=True)
        
        # Display cluster visualization for 2 features
        st.subheader("Cluster Visualization")
        
        if len(selected_features) >= 2:
            # Let user select which 2 features to visualize
            x_feature = st.selectbox("Select X-axis feature:", selected_features, index=0)
            remaining_features = [f for f in selected_features if f != x_feature]
            y_feature = st.selectbox("Select Y-axis feature:", remaining_features, index=0)
            
            # Create scatter plot
            fig_scatter = px.scatter(
                model_results['df_with_clusters'],
                x=x_feature,
                y=y_feature,
                color='Cluster',
                title=f"Thanos' Infinity Clusters ({x_feature} vs {y_feature})",
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            
            # Add cluster centers
            centers = model_results['cluster_centers']
            centers_df = pd.DataFrame(
                model_results['scaler'].inverse_transform(centers),
                columns=selected_features
            )
            
            # for i in range(model_results['model'].n_clusters):
            #     fig_scatter.add_trace(
            #         go.Scatter(
            #             x=[centers_df.loc[i, x_feature]],
            #             y=[centers_df.loc[i, y_feature]],
            #             mode='markers',
            #             marker=dict(
            #                 symbol='star',
            #                 size=15,
            #                 color='yellow',
            #                 line=dict(color='black', width=1)
            #             ),
            #             name=f'Infinity Stone {i+1}'
            #         )
            #     )
            
            fig_scatter.update_layout(
                plot_bgcolor='rgba(0,0,0,0.1)',
                paper_bgcolor='rgba(0,0,0,0.1)',
                font=dict(color='white')
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # 3D visualization if there are at least 3 features
        if len(selected_features) >= 3:
            st.subheader("3D Cluster Visualization")
            
            # Let user select which 3 features to visualize
            z_feature = st.selectbox("Select Z-axis feature:", 
                                    [f for f in selected_features if f != x_feature and f != y_feature], 
                                    index=0)
            
            # Create 3D scatter plot
            fig_3d = px.scatter_3d(
                model_results['df_with_clusters'],
                x=x_feature,
                y=y_feature,
                z=z_feature,
                color='Cluster',
                title=f"Thanos' 3D Infinity Clusters",
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            
            # Add cluster centers
            # for i in range(model_results['model'].n_clusters):
            #     fig_3d.add_trace(
            #         go.Scatter3d(
            #             x=[centers_df.loc[i, x_feature]],
            #             y=[centers_df.loc[i, y_feature]],
            #             z=[centers_df.loc[i, z_feature]],
            #             mode='markers',
            #             marker=dict(
            #                 symbol='diamond',
            #                 size=8,
            #                 color='yellow',
            #                 line=dict(color='black', width=1)
            #             ),
            #             name=f'Infinity Stone {i+1}'
            #         )
            #     )
            
            fig_3d.update_layout(
                scene=dict(
                    xaxis=dict(backgroundcolor='rgba(0,0,0,0.1)'),
                    yaxis=dict(backgroundcolor='rgba(0,0,0,0.1)'),
                    zaxis=dict(backgroundcolor='rgba(0,0,0,0.1)')
                ),
                paper_bgcolor='rgba(0,0,0,0.1)',
                font=dict(color='white')
            )
            st.plotly_chart(fig_3d, use_container_width=True)
        
        # Cluster characteristics
        st.subheader("Cluster Characteristics")
        
        # Radar chart for cluster profiles
        if len(selected_features) >= 3:
            # Calculate cluster means for each feature
            cluster_profiles = model_results['df_with_clusters'].groupby('Cluster')[selected_features].mean()
            
            # Normalize the profiles for radar chart
            from sklearn.preprocessing import MinMaxScaler
            scaler_radar = MinMaxScaler()
            cluster_profiles_normalized = pd.DataFrame(
                scaler_radar.fit_transform(cluster_profiles),
                columns=selected_features,
                index=cluster_profiles.index
            )
            
            # Create radar chart
            categories = selected_features
            fig_radar = go.Figure()
            
            for i in range(model_results['model'].n_clusters):
                values = cluster_profiles_normalized.loc[i].tolist()
                values.append(values[0])  # Close the loop
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories + [categories[0]],  # Close the loop
                    fill='toself',
                    name=f'Cluster {i}'
                ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                title="Thanos' Infinity Stones Power Profiles",
                plot_bgcolor='rgba(0,0,0,0.1)',
                paper_bgcolor='rgba(0,0,0,0.1)',
                font=dict(color='white')
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        
        # Display cluster statistics
        with st.expander("View Detailed Cluster Statistics"):
            st.dataframe(model_results['cluster_stats'])
        
        # Clustering prediction section
        st.markdown("""
        <div class="thanos-panel">
            <h3>Predict Cluster for New Data</h3>
            <p>"I am... inevitable. Your data will be classified."</p>
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
                default_val,
                key=f"thanos_{feature}"
            )
        
        # Make prediction
        if st.button("Classify New Data", key="thanos_predict"):
            # Create input array
            X_new = np.array([feature_inputs[feature] for feature in selected_features]).reshape(1, -1)
            
            # Scale the input
            X_new_scaled = model_results['scaler'].transform(X_new)
            
            # Make prediction
            cluster_prediction = model_results['model'].predict(X_new_scaled)[0]
            
            # Get the stone color based on cluster
            stone_colors = ["purple", "blue", "red", "orange", "green", "yellow"]
            stone_color = stone_colors[cluster_prediction % len(stone_colors)]
            
            # Calculate distance to cluster center
            cluster_center = model_results['model'].cluster_centers_[cluster_prediction]
            distance = np.sqrt(np.sum((X_new_scaled[0] - cluster_center) ** 2))
            
            # Display prediction result with infinity stone
            st.markdown(f"""
            <div class="thanos-panel" style="text-align: center;">
                <h3>Thanos' Classification</h3>
                <img src="https://i.pinimg.com/736x/46/19/99/461999fe4e05205ec4deff47a12ca57e.jpg" width="150">
                <p>The data point belongs to:</p>
                <h2 style="color: {stone_color}; text-align: center;">Cluster {cluster_prediction}</h2>
                <p>Distance to cluster center: {distance:.4f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Create download buttons for exporting data
        create_download_buttons(model_results['df_with_clusters'], model_results['model_summary'], None, "thanos")
