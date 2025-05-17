from sklearn.datasets import make_blobs
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, silhouette_score
import pandas as pd
import numpy as np

# Linear Regression implementation (Batman)
def run_linear_regression(df, features, target, test_size=0.2):
    """
    Run linear regression on the given data
    """
    # Prepare the data
    X = df[features]
    y = df[target]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    # Evaluate the model
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    # Create feature importance
    coefficients = pd.DataFrame({
        'Feature': features,
        'Coefficient': model.coef_
    })
    coefficients = coefficients.sort_values('Coefficient', ascending=False)
    
    # Prepare results
    results = {
        'model': model,
        'scaler': scaler,
        'y_pred_test': y_pred_test,
        'y_test': y_test,
        'X_test': X_test,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'coefficients': coefficients,
        'model_summary': {
            'R-squared (Test)': round(test_r2, 4),
            'MSE (Test)': round(test_mse, 4),
            'RMSE (Test)': round(np.sqrt(test_mse), 4),
            'Coefficients': {feature: round(coef, 4) for feature, coef in zip(features, model.coef_)},
            'Intercept': round(model.intercept_, 4)
        }
    }
    
    return results

# Logistic Regression implementation (Spider-Man)
def run_logistic_regression(df, features, target, test_size=0.2):
    """
    Run logistic regression on the given data
    """
    # Prepare the data
    X = df[features]
    y = df[target]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    y_prob_test = model.predict_proba(X_test_scaled)[:, 1]
    
    # Evaluate the model
    accuracy_train = accuracy_score(y_train, y_pred_train)
    accuracy_test = accuracy_score(y_test, y_pred_test)
    conf_matrix = confusion_matrix(y_test, y_pred_test)
    
    # Create feature importance
    coefficients = pd.DataFrame({
        'Feature': features,
        'Coefficient': model.coef_[0]
    })
    coefficients = coefficients.sort_values('Coefficient', ascending=False)
    
    # Calculate precision and recall
    tn, fp, fn, tp = conf_matrix.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Prepare results
    results = {
        'model': model,
        'scaler': scaler,
        'y_pred_test': y_pred_test,
        'y_prob_test': y_prob_test,
        'y_test': y_test,
        'X_test': X_test,
        'accuracy_train': accuracy_train,
        'accuracy_test': accuracy_test,
        'conf_matrix': conf_matrix,
        'coefficients': coefficients,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'model_summary': {
            'Accuracy (Test)': round(accuracy_test, 4),
            'Precision': round(precision, 4),
            'Recall': round(recall, 4),
            'F1 Score': round(f1, 4),
            'True Positives': int(tp),
            'False Positives': int(fp),
            'True Negatives': int(tn),
            'False Negatives': int(fn)
        }
    }
    
    return results

# K-Means Clustering implementation (Thanos)
def run_kmeans_clustering(df, features, n_clusters=4):
    """
    Run K-Means clustering on the given data
    """
    # Prepare the data
    X = df[features]
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train the model
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = model.fit_predict(X_scaled)
    
    # Add cluster labels to the original data
    df_with_clusters = df.copy()
    df_with_clusters['Cluster'] = clusters
    
    # Calculate silhouette score
    silhouette = silhouette_score(X_scaled, clusters) if n_clusters > 1 and n_clusters < len(X) else 0
    
    # Calculate cluster stats (only numeric columns)
    numeric_cols = df_with_clusters.select_dtypes(include=[np.number]).columns.tolist()
    if 'Cluster' not in numeric_cols:
        numeric_cols.append('Cluster')
    cluster_stats = df_with_clusters[numeric_cols].groupby('Cluster').mean()
    
    # Prepare results
    results = {
        'model': model,
        'scaler': scaler,
        'clusters': clusters,
        'df_with_clusters': df_with_clusters,
        'silhouette': silhouette,
        'inertia': model.inertia_,
        'cluster_centers': model.cluster_centers_,
        'cluster_stats': cluster_stats,
        'model_summary': {
            'Number of Clusters': n_clusters,
            'Silhouette Score': round(silhouette, 4),
            'Inertia': round(model.inertia_, 4),
            'Cluster Sizes': df_with_clusters['Cluster'].value_counts().to_dict()
        }
    }
    
    return results


# Function to generate sample data if needed
def generate_sample_data(dataset_type="stock"):
    if dataset_type == "stock":
        # Generate sample stock data
        date_range = pd.date_range(start="2022-01-01", periods=500, freq='B')
        price = 100 + np.cumsum(np.random.normal(0, 1, 500)) * 0.5
        volume = np.random.randint(1000, 100000, 500)
        
        # Add some features
        ma_10 = pd.Series(price).rolling(window=10).mean().values
        ma_30 = pd.Series(price).rolling(window=30).mean().values
        rsi = np.random.uniform(30, 70, 500)  # Simplified RSI
        
        df = pd.DataFrame({
            'Date': date_range,
            'Price': price,
            'Volume': volume,
            'MA_10': ma_10,
            'MA_30': ma_30,
            'RSI': rsi,
            'Market_Sentiment': np.random.choice(['Positive', 'Neutral', 'Negative'], 500),
            'Trading_Day': date_range.dayofweek
        })
        
        return df
    
    elif dataset_type == "binary":
        # Generate sample data for logistic regression
        n_samples = 500
        x1 = np.random.normal(0, 1, n_samples)
        x2 = np.random.normal(0, 1, n_samples)
        y_prob = 1 / (1 + np.exp(-(2 * x1 - 3 * x2 + np.random.normal(0, 0.5, n_samples))))
        y = (y_prob > 0.5).astype(int)
        
        df = pd.DataFrame({
            'Feature1': x1,
            'Feature2': x2,
            'Feature3': np.random.normal(0, 1, n_samples),
            'Feature4': np.random.normal(0, 1, n_samples),
            'Target': y,
            'Category': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
            'Date': pd.date_range(start="2022-01-01", periods=n_samples)
        })
        
        return df
    
    elif dataset_type == "cluster":
        # Generate sample data for clustering
        n_samples = 500
        n_clusters = 4
        centers = np.array([
            [2, 2],
            [-2, -2],
            [2, -2],
            [-2, 2]
        ])
        cluster_std = 0.5
        
        X, _ = make_blobs(n_samples=n_samples, centers=centers, cluster_std=cluster_std, random_state=42)
        
        df = pd.DataFrame({
            'Feature1': X[:, 0],
            'Feature2': X[:, 1],
            'Feature3': np.random.normal(0, 1, n_samples),
            'Feature4': np.random.normal(0, 1, n_samples),
            'Feature5': np.random.normal(0, 1, n_samples),
            'ID': range(n_samples),
            'Date': pd.date_range(start="2022-01-01", periods=n_samples)
        })
        
        return df
    
    else:
        # Default to stock data
        return generate_sample_data("stock")
