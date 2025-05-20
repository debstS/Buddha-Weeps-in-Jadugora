"""
MLOps Pipeline Implementation for Jadugora Uranium Mining Impact Assessment

This script sets up a cloud-based MLOps pipeline for automated model training,
evaluation, and deployment of the environmental and health impact models.
"""

import os
import sys
import logging
import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/mlops_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Ensure necessary directories exist
os.makedirs("deployment", exist_ok=True)
os.makedirs("deployment/models", exist_ok=True)
os.makedirs("deployment/api", exist_ok=True)
os.makedirs("deployment/monitoring", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# MLflow setup
MLFLOW_TRACKING_URI = "sqlite:///deployment/mlflow.db"
EXPERIMENT_NAME = "jadugora_impact_assessment"

def setup_mlflow():
    """
    Set up MLflow tracking server and experiment.
    
    Returns:
        str: Active experiment ID
    """
    logger.info("Setting up MLflow tracking...")
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    
    # Create or get experiment
    try:
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
            logger.info(f"Created new experiment with ID: {experiment_id}")
        else:
            experiment_id = experiment.experiment_id
            logger.info(f"Using existing experiment with ID: {experiment_id}")
    except Exception as e:
        logger.error(f"Error setting up MLflow experiment: {e}")
        experiment_id = "0"  # Default experiment
    
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    return experiment_id

def load_models():
    """
    Load trained models from the models directory.
    
    Returns:
        dict: Dictionary of loaded models
    """
    logger.info("Loading trained models...")
    
    models = {}
    model_files = [f for f in os.listdir("models") if f.endswith(".pkl")]
    
    for model_file in model_files:
        model_name = model_file.replace(".pkl", "")
        try:
            with open(os.path.join("models", model_file), "rb") as f:
                models[model_name] = pickle.load(f)
            logger.info(f"Loaded model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
    
    return models

def register_models_with_mlflow(models, datasets):
    """
    Register models with MLflow for version tracking.
    
    Args:
        models (dict): Dictionary of trained models
        datasets (dict): Dictionary of datasets used for training
    
    Returns:
        dict: Dictionary of registered model versions
    """
    logger.info("Registering models with MLflow...")
    
    registered_models = {}
    
    # Environmental impact models
    env_targets = {
        'radiation_level': 'residential_area',
        'water_uranium': 'uranium_concentration_ppb',
        'soil_uranium': 'uranium_ppm'
    }
    
    # Health impact models
    health_targets = {
        'cancer_rate': 'cancer_cases',
        'respiratory_disease_rate': 'respiratory_disease',
        'skin_disorders_rate': 'skin_disorders',
        'kidney_disease_rate': 'kidney_disease'
    }
    
    # Register environmental models
    for target_name, target_col in env_targets.items():
        for model_type in ['linear_regression', 'ridge', 'lasso', 'random_forest', 'gradient_boosting']:
            model_key = f"{target_name}_{model_type}"
            if model_key in models:
                try:
                    # Start MLflow run
                    with mlflow.start_run(run_name=model_key) as run:
                        # Log model parameters
                        if hasattr(models[model_key], 'named_steps') and 'model' in models[model_key].named_steps:
                            model_params = models[model_key].named_steps['model'].get_params()
                            for param_name, param_value in model_params.items():
                                if isinstance(param_value, (int, float, str, bool)):
                                    mlflow.log_param(param_name, param_value)
                        
                        # Evaluate model
                        X = datasets['env_health'].drop(columns=list(env_targets.values()) + list(health_targets.values()))
                        y = datasets['env_health'][target_col]
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        y_pred = models[model_key].predict(X_test)
                        
                        # Log metrics
                        mse = mean_squared_error(y_test, y_pred)
                        rmse = np.sqrt(mse)
                        r2 = r2_score(y_test, y_pred)
                        mlflow.log_metric("mse", mse)
                        mlflow.log_metric("rmse", rmse)
                        mlflow.log_metric("r2", r2)
                        
                        # Log model
                        mlflow.sklearn.log_model(models[model_key], model_key)
                        
                        # Register model
                        model_uri = f"runs:/{run.info.run_id}/{model_key}"
                        registered_model = mlflow.register_model(model_uri, model_key)
                        registered_models[model_key] = registered_model.version
                        
                        logger.info(f"Registered model {model_key} with version {registered_model.version}")
                except Exception as e:
                    logger.error(f"Error registering model {model_key}: {e}")
    
    # Register health models
    for target_name, target_col in health_targets.items():
        for model_type in ['linear_regression', 'ridge', 'lasso', 'random_forest', 'gradient_boosting']:
            model_key = f"{target_name}_{model_type}"
            if model_key in models:
                try:
                    # Start MLflow run
                    with mlflow.start_run(run_name=model_key) as run:
                        # Log model parameters
                        if hasattr(models[model_key], 'named_steps') and 'model' in models[model_key].named_steps:
                            model_params = models[model_key].named_steps['model'].get_params()
                            for param_name, param_value in model_params.items():
                                if isinstance(param_value, (int, float, str, bool)):
                                    mlflow.log_param(param_name, param_value)
                        
                        # Evaluate model
                        env_features = ['residential_area', 'ph_level', 'heavy_metals_ppm', 
                                      'uranium_concentration_ppb', 'uranium_ppm', 'radium_ppm', 
                                      'lead_ppm', 'arsenic_ppm']
                        
                        mining_features = ['ore_extracted_tons', 'uranium_produced_kg', 
                                         'waste_generated_tons', 'tailings_volume_cubic_m', 
                                         'water_used_million_liters']
                        
                        all_features = env_features + mining_features
                        
                        X = datasets['mining_health'][all_features]
                        y = datasets['mining_health'][target_col]
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        y_pred = models[model_key].predict(X_test)
                        
                        # Log metrics
                        mse = mean_squared_error(y_test, y_pred)
                        rmse = np.sqrt(mse)
                        r2 = r2_score(y_test, y_pred)
                        mlflow.log_metric("mse", mse)
                        mlflow.log_metric("rmse", rmse)
                        mlflow.log_metric("r2", r2)
                        
                        # Log model
                        mlflow.sklearn.log_model(models[model_key], model_key)
                        
                        # Register model
                        model_uri = f"runs:/{run.info.run_id}/{model_key}"
                        registered_model = mlflow.register_model(model_uri, model_key)
                        registered_models[model_key] = registered_model.version
                        
                        logger.info(f"Registered model {model_key} with version {registered_model.version}")
                except Exception as e:
                    logger.error(f"Error registering model {model_key}: {e}")
    
    return registered_models

def create_model_api():
    """
    Create a Flask API for model serving.
    
    Returns:
        bool: Success status
    """
    logger.info("Creating model serving API...")
    
    api_code = """
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
import mlflow
import mlflow.sklearn

app = Flask(__name__)

# Load models
@app.before_first_request
def load_models():
    global models
    models = {}
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    
    # Load environmental impact models
    env_model_names = [
        "radiation_level_lasso",
        "water_uranium_gradient_boosting",
        "soil_uranium_lasso"
    ]
    
    # Load health impact models
    health_model_names = [
        "cancer_rate_random_forest",
        "respiratory_disease_rate_random_forest",
        "skin_disorders_rate_gradient_boosting",
        "kidney_disease_rate_lasso"
    ]
    
    # Load all models
    for model_name in env_model_names + health_model_names:
        try:
            models[model_name] = mlflow.sklearn.load_model(f"models:/{model_name}/latest")
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            # Fallback to local model file
            try:
                with open(f"../models/{model_name}.pkl", "rb") as f:
                    models[model_name] = pickle.load(f)
            except Exception as e:
                print(f"Error loading local model {model_name}: {e}")

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})

@app.route('/predict/environmental', methods=['POST'])
def predict_environmental():
    try:
        # Get input data
        data = request.json
        
        # Convert to DataFrame
        input_df = pd.DataFrame(data, index=[0])
        
        # Make predictions
        predictions = {}
        
        if "radiation_level_lasso" in models:
            predictions["radiation_level"] = float(models["radiation_level_lasso"].predict(input_df)[0])
        
        if "water_uranium_gradient_boosting" in models:
            predictions["water_uranium"] = float(models["water_uranium_gradient_boosting"].predict(input_df)[0])
        
        if "soil_uranium_lasso" in models:
            predictions["soil_uranium"] = float(models["soil_uranium_lasso"].predict(input_df)[0])
        
        return jsonify({"predictions": predictions})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/predict/health', methods=['POST'])
def predict_health():
    try:
        # Get input data
        data = request.json
        
        # Convert to DataFrame
        input_df = pd.DataFrame(data, index=[0])
        
        # Make predictions
        predictions = {}
        
        if "cancer_rate_random_forest" in models:
            predictions["cancer_rate"] = float(models["cancer_rate_random_forest"].predict(input_df)[0])
        
        if "respiratory_disease_rate_random_forest" in models:
            predictions["respiratory_disease_rate"] = float(models["respiratory_disease_rate_random_forest"].predict(input_df)[0])
        
        if "skin_disorders_rate_gradient_boosting" in models:
            predictions["skin_disorders_rate"] = float(models["skin_disorders_rate_gradient_boosting"].predict(input_df)[0])
        
        if "kidney_disease_rate_lasso" in models:
            predictions["kidney_disease_rate"] = float(models["kidney_disease_rate_lasso"].predict(input_df)[0])
        
        return jsonify({"predictions": predictions})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
"""
    
    try:
        with open("deployment/api/app.py", "w") as f:
            f.write(api_code)
        
        # Create requirements.txt
        requirements = """
flask==2.0.1
pandas==1.3.3
numpy==1.21.2
scikit-learn==1.0
mlflow==1.20.2
gunicorn==20.1.0
"""
        
        with open("deployment/api/requirements.txt", "w") as f:
            f.write(requirements)
        
        # Create Dockerfile
        dockerfile = """
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Copy MLflow database and models
COPY mlflow.db .
COPY ../models ./models

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
"""
        
        with open("deployment/api/Dockerfile", "w") as f:
            f.write(dockerfile)
        
        logger.info("Model API created successfully")
        return True
    except Exception as e:
        logger.error(f"Error creating model API: {e}")
        return False

def create_monitoring_dashboard():
    """
    Create a monitoring dashboard for model performance.
    
    Returns:
        bool: Success status
    """
    logger.info("Creating monitoring dashboard...")
    
    dashboard_code = """
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import sqlite3
import os

# Initialize Dash app
app = dash.Dash(__name__)

# Define layout
app.layout = html.Div([
    html.H1("Jadugora Impact Assessment - Model Monitoring Dashboard"),
    
    html.Div([
        html.H2("Model Performance Metrics"),
        html.Div([
            html.Label("Select Model Type:"),
            dcc.Dropdown(
                id='model-type-dropdown',
                options=[
                    {'label': 'Environmental Impact Models', 'value': 'environmental'},
                    {'label': 'Health Impact Models', 'value': 'health'}
                ],
                value='environmental'
            )
        ], style={'width': '30%', 'margin': '10px'}),
        
        html.Div([
            html.Label("Select Metric:"),
            dcc.Dropdown(
                id='metric-dropdown',
                options=[
                    {'label': 'R² Score', 'value': 'r2'},
                    {'label': 'RMSE', 'value': 'rmse'},
                    {'label': 'MSE', 'value': 'mse'}
                ],
                value='r2'
            )
        ], style={'width': '30%', 'margin': '10px'}),
        
        dcc.Graph(id='metrics-graph')
    ]),
    
    html.Div([
        html.H2("Model Predictions vs Actual Values"),
        html.Div([
            html.Label("Select Target:"),
            dcc.Dropdown(
                id='target-dropdown',
                options=[
                    {'label': 'Radiation Level', 'value': 'radiation_level'},
                    {'label': 'Water Uranium', 'value': 'water_uranium'},
                    {'label': 'Soil Uranium', 'value': 'soil_uranium'},
                    {'label': 'Cancer Rate', 'value': 'cancer_rate'},
                    {'label': 'Respiratory Disease Rate', 'value': 'respiratory_disease_rate'},
                    {'label': 'Skin Disorders Rate', 'value': 'skin_disorders_rate'},
                    {'label': 'Kidney Disease Rate', 'value': 'kidney_disease_rate'}
                ],
                value='radiation_level'
            )
        ], style={'width': '30%', 'margin': '10px'}),
        
        dcc.Graph(id='predictions-graph')
    ]),
    
    html.Div([
        html.H2("Model Drift Detection"),
        html.P("This section monitors changes in model performance over time to detect potential model drift."),
        dcc.Graph(id='drift-graph')
    ]),
    
    html.Div([
        html.H2("Data Quality Monitoring"),
        html.P("This section monitors the quality of input data for potential issues."),
        dcc.Graph(id='data-quality-graph')
    ])
])

# Define callback for metrics graph
@app.callback(
    Output('metrics-graph', 'figure'),
    [Input('model-type-dropdown', 'value'),
     Input('metric-dropdown', 'value')]
)
def update_metrics_graph(model_type, metric):
    # In a real implementation, this would query MLflow or a database
    # For demonstration, we'll use mock data
    
    if model_type == 'environmental':
        models = ['radiation_level', 'water_uranium', 'soil_uranium']
        if metric == 'r2':
            values = [-5.7076, -0.4412, -1.0718]
        elif metric == 'rmse':
            values = [0.0325, 0.7523, 0.4848]
        else:  # mse
            values = [0.0011, 0.5660, 0.2350]
    else:  # health
        models = ['cancer_rate', 'respiratory_disease_rate', 'skin_disorders_rate', 'kidney_disease_rate']
        if metric == 'r2':
            values = [-1.6535, -1.6535, 0.3541, -0.1674]
        elif metric == 'rmse':
            values = [10.5882, 10.5882, 2.0092, 3.2414]
        else:  # mse
            values = [112.1100, 112.1100, 4.0369, 10.5067]
    
    df = pd.DataFrame({
        'Model': models,
        'Value': values
    })
    
    fig = px.bar(df, x='Model', y='Value', title=f"{metric.upper()} for {model_type.title()} Models")
    return fig

# Define callback for predictions graph
@app.callback(
    Output('predictions-graph', 'figure'),
    [Input('target-dropdown', 'value')]
)
def update_predictions_graph(target):
    # In a real implementation, this would load actual prediction data
    # For demonstration, we'll use mock data
    
    # Generate some sample data
    np.random.seed(42)
    years = list(range(2015, 2025))
    actual = np.random.normal(size=len(years))
    predicted = actual + np.random.normal(scale=0.2, size=len(years))
    
    # Scale based on target
    if target in ['radiation_level']:
        actual = 0.5 + 0.3 * actual
        predicted = 0.5 + 0.3 * predicted
    elif target in ['water_uranium', 'soil_uranium']:
        actual = 5 + 2 * actual
        predicted = 5 + 2 * predicted
    else:  # health metrics
        actual = 20 + 10 * actual
        predicted = 20 + 10 * predicted
    
    df = pd.DataFrame({
        'Year': years,
        'Actual': actual,
        'Predicted': predicted
    })
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Year'], y=df['Actual'], mode='lines+markers', name='Actual'))
    fig.add_trace(go.Scatter(x=df['Year'], y=df['Predicted'], mode='lines+markers', name='Predicted'))
    fig.update_layout(title=f"Actual vs Predicted {target.replace('_', ' ').title()} Over Time",
                     xaxis_title="Year",
                     yaxis_title="Value")
    return fig

# Define callback for drift graph
@app.callback(
    Output('drift-graph', 'figure'),
    [Input('target-dropdown', 'value')]
)
def update_drift_graph(target):
    # In a real implementation, this would analyze model performance over time
    # For demonstration, we'll use mock data
    
    # Generate some sample data
    np.random.seed(42)
    months = pd.date_range(start='2024-01-01', end='2025-05-01', freq='M')
    r2_scores = 0.8 + np.cumsum(np.random.normal(scale=0.02, size=len(months))) * -0.05
    
    df = pd.DataFrame({
        'Month': months,
        'R² Score': r2_scores
    })
    
    fig = px.line(df, x='Month', y='R² Score', 
                 title=f"Model Performance Drift for {target.replace('_', ' ').title()}")
    
    # Add threshold line
    fig.add_shape(
        type="line",
        x0=months.min(),
        y0=0.7,
        x1=months.max(),
        y1=0.7,
        line=dict(
            color="Red",
            width=2,
            dash="dash",
        )
    )
    
    fig.add_annotation(
        x=months.max(),
        y=0.7,
        text="Performance Threshold",
        showarrow=False,
        yshift=10
    )
    
    return fig

# Define callback for data quality graph
@app.callback(
    Output('data-quality-graph', 'figure'),
    [Input('model-type-dropdown', 'value')]
)
def update_data_quality_graph(model_type):
    # In a real implementation, this would analyze input data quality
    # For demonstration, we'll use mock data
    
    if model_type == 'environmental':
        features = ['mine_proximity', 'residential_area', 'ph_level', 'heavy_metals_ppm', 
                   'uranium_concentration_ppb', 'uranium_ppm', 'radium_ppm']
        missing_pct = np.random.uniform(low=0, high=2, size=len(features))
        outlier_pct = np.random.uniform(low=0, high=5, size=len(features))
    else:  # health
        features = ['cancer_cases', 'respiratory_disease', 'skin_disorders', 'kidney_disease']
        missing_pct = np.random.uniform(low=0, high=1, size=len(features))
        outlier_pct = np.random.uniform(low=0, high=3, size=len(features))
    
    df = pd.DataFrame({
        'Feature': features,
        'Missing Values (%)': missing_pct,
        'Outliers (%)': outlier_pct
    })
    
    fig = go.Figure(data=[
        go.Bar(name='Missing Values (%)', x=df['Feature'], y=df['Missing Values (%)']),
        go.Bar(name='Outliers (%)', x=df['Feature'], y=df['Outliers (%)'])
    ])
    
    fig.update_layout(
        title=f"Data Quality Metrics for {model_type.title()} Features",
        xaxis_title="Feature",
        yaxis_title="Percentage (%)",
        barmode='group'
    )
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)
"""
    
    try:
        with open("deployment/monitoring/dashboard.py", "w") as f:
            f.write(dashboard_code)
        
        # Create requirements.txt
        requirements = """
dash==2.0.0
plotly==5.5.0
pandas==1.3.3
numpy==1.21.2
"""
        
        with open("deployment/monitoring/requirements.txt", "w") as f:
            f.write(requirements)
        
        # Create Dockerfile
        dockerfile = """
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8050

CMD ["python", "dashboard.py"]
"""
        
        with open("deployment/monitoring/Dockerfile", "w") as f:
            f.write(dockerfile)
        
        logger.info("Monitoring dashboard created successfully")
        return True
    except Exception as e:
        logger.error(f"Error creating monitoring dashboard: {e}")
        return False

def create_ci_cd_pipeline():
    """
    Create CI/CD pipeline configuration for automated deployment.
    
    Returns:
        bool: Success status
    """
    logger.info("Creating CI/CD pipeline configuration...")
    
    # GitHub Actions workflow
    github_workflow = """
name: Jadugora Impact Assessment CI/CD

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 0 * * 0'  # Weekly on Sundays at midnight

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
        pip install pytest pytest-cov
    
    - name: Run tests
      run: |
        pytest --cov=src tests/
    
    - name: Upload test results
      uses: actions/upload-artifact@v2
      with:
        name: test-results
        path: coverage.xml
  
  train_models:
    needs: test
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
    
    - name: Run data collection
      run: |
        python src/data_collection.py
    
    - name: Run exploratory analysis
      run: |
        python src/exploratory_analysis.py
    
    - name: Fix datasets
      run: |
        python src/fix_datasets.py
    
    - name: Train models
      run: |
        python src/model_development.py
    
    - name: Upload models
      uses: actions/upload-artifact@v2
      with:
        name: models
        path: models/
  
  deploy_api:
    needs: train_models
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Download models
      uses: actions/download-artifact@v2
      with:
        name: models
        path: models/
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v1
    
    - name: Login to DockerHub
      uses: docker/login-action@v1
      with:
        username: ${{ secrets.DOCKERHUB_USERNAME }}
        password: ${{ secrets.DOCKERHUB_TOKEN }}
    
    - name: Build and push API
      uses: docker/build-push-action@v2
      with:
        context: ./deployment/api
        push: true
        tags: ${{ secrets.DOCKERHUB_USERNAME }}/jadugora-impact-api:latest
    
    - name: Build and push Dashboard
      uses: docker/build-push-action@v2
      with:
        context: ./deployment/monitoring
        push: true
        tags: ${{ secrets.DOCKERHUB_USERNAME }}/jadugora-impact-dashboard:latest
    
  deploy_cloud:
    needs: deploy_api
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install awscli
    
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v1
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-east-1
    
    - name: Deploy to AWS
      run: |
        aws cloudformation deploy \\
          --template-file deployment/cloudformation.yaml \\
          --stack-name jadugora-impact-assessment \\
          --capabilities CAPABILITY_IAM \\
          --parameter-overrides \\
            DockerImageApi=${{ secrets.DOCKERHUB_USERNAME }}/jadugora-impact-api:latest \\
            DockerImageDashboard=${{ secrets.DOCKERHUB_USERNAME }}/jadugora-impact-dashboard:latest
"""
    
    # AWS CloudFormation template
    cloudformation_template = """
AWSTemplateFormatVersion: '2010-09-09'
Description: 'Jadugora Impact Assessment MLOps Infrastructure'

Parameters:
  DockerImageApi:
    Type: String
    Description: Docker image for the API
  
  DockerImageDashboard:
    Type: String
    Description: Docker image for the dashboard

Resources:
  # VPC and Networking
  VPC:
    Type: AWS::EC2::VPC
    Properties:
      CidrBlock: 10.0.0.0/16
      EnableDnsSupport: true
      EnableDnsHostnames: true
      Tags:
        - Key: Name
          Value: JadugoraVPC

  PublicSubnet1:
    Type: AWS::EC2::Subnet
    Properties:
      VpcId: !Ref VPC
      CidrBlock: 10.0.1.0/24
      AvailabilityZone: !Select [0, !GetAZs '']
      MapPublicIpOnLaunch: true
      Tags:
        - Key: Name
          Value: JadugoraPublicSubnet1

  PublicSubnet2:
    Type: AWS::EC2::Subnet
    Properties:
      VpcId: !Ref VPC
      CidrBlock: 10.0.2.0/24
      AvailabilityZone: !Select [1, !GetAZs '']
      MapPublicIpOnLaunch: true
      Tags:
        - Key: Name
          Value: JadugoraPublicSubnet2

  InternetGateway:
    Type: AWS::EC2::InternetGateway
    Properties:
      Tags:
        - Key: Name
          Value: JadugoraIGW

  AttachGateway:
    Type: AWS::EC2::VPCGatewayAttachment
    Properties:
      VpcId: !Ref VPC
      InternetGatewayId: !Ref InternetGateway

  RouteTable:
    Type: AWS::EC2::RouteTable
    Properties:
      VpcId: !Ref VPC
      Tags:
        - Key: Name
          Value: JadugoraRouteTable

  Route:
    Type: AWS::EC2::Route
    DependsOn: AttachGateway
    Properties:
      RouteTableId: !Ref RouteTable
      DestinationCidrBlock: 0.0.0.0/0
      GatewayId: !Ref InternetGateway

  SubnetRouteTableAssociation1:
    Type: AWS::EC2::SubnetRouteTableAssociation
    Properties:
      SubnetId: !Ref PublicSubnet1
      RouteTableId: !Ref RouteTable

  SubnetRouteTableAssociation2:
    Type: AWS::EC2::SubnetRouteTableAssociation
    Properties:
      SubnetId: !Ref PublicSubnet2
      RouteTableId: !Ref RouteTable

  # ECS Cluster
  ECSCluster:
    Type: AWS::ECS::Cluster
    Properties:
      ClusterName: JadugoraCluster

  # API Service
  ApiTaskDefinition:
    Type: AWS::ECS::TaskDefinition
    Properties:
      Family: jadugora-api
      NetworkMode: awsvpc
      RequiresCompatibilities:
        - FARGATE
      Cpu: '256'
      Memory: '512'
      ExecutionRoleArn: !GetAtt ECSTaskExecutionRole.Arn
      ContainerDefinitions:
        - Name: api
          Image: !Ref DockerImageApi
          Essential: true
          PortMappings:
            - ContainerPort: 5000
              HostPort: 5000
              Protocol: tcp
          LogConfiguration:
            LogDriver: awslogs
            Options:
              awslogs-group: !Ref ApiLogGroup
              awslogs-region: !Ref AWS::Region
              awslogs-stream-prefix: api

  ApiLogGroup:
    Type: AWS::Logs::LogGroup
    Properties:
      LogGroupName: /ecs/jadugora-api
      RetentionInDays: 30

  ApiService:
    Type: AWS::ECS::Service
    DependsOn: ApiLoadBalancerListener
    Properties:
      ServiceName: jadugora-api
      Cluster: !Ref ECSCluster
      TaskDefinition: !Ref ApiTaskDefinition
      LaunchType: FARGATE
      DesiredCount: 1
      NetworkConfiguration:
        AwsvpcConfiguration:
          AssignPublicIp: ENABLED
          Subnets:
            - !Ref PublicSubnet1
            - !Ref PublicSubnet2
          SecurityGroups:
            - !Ref ApiSecurityGroup
      LoadBalancers:
        - ContainerName: api
          ContainerPort: 5000
          TargetGroupArn: !Ref ApiTargetGroup

  ApiSecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupDescription: Security group for API
      VpcId: !Ref VPC
      SecurityGroupIngress:
        - IpProtocol: tcp
          FromPort: 5000
          ToPort: 5000
          CidrIp: 0.0.0.0/0

  ApiLoadBalancer:
    Type: AWS::ElasticLoadBalancingV2::LoadBalancer
    Properties:
      Name: jadugora-api-lb
      Subnets:
        - !Ref PublicSubnet1
        - !Ref PublicSubnet2
      SecurityGroups:
        - !Ref ApiLoadBalancerSecurityGroup

  ApiLoadBalancerSecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupDescription: Security group for API load balancer
      VpcId: !Ref VPC
      SecurityGroupIngress:
        - IpProtocol: tcp
          FromPort: 80
          ToPort: 80
          CidrIp: 0.0.0.0/0

  ApiTargetGroup:
    Type: AWS::ElasticLoadBalancingV2::TargetGroup
    Properties:
      Name: jadugora-api-tg
      Port: 5000
      Protocol: HTTP
      TargetType: ip
      VpcId: !Ref VPC
      HealthCheckPath: /health
      HealthCheckIntervalSeconds: 30
      HealthCheckTimeoutSeconds: 5
      HealthyThresholdCount: 2
      UnhealthyThresholdCount: 2

  ApiLoadBalancerListener:
    Type: AWS::ElasticLoadBalancingV2::Listener
    Properties:
      LoadBalancerArn: !Ref ApiLoadBalancer
      Port: 80
      Protocol: HTTP
      DefaultActions:
        - Type: forward
          TargetGroupArn: !Ref ApiTargetGroup

  # Dashboard Service
  DashboardTaskDefinition:
    Type: AWS::ECS::TaskDefinition
    Properties:
      Family: jadugora-dashboard
      NetworkMode: awsvpc
      RequiresCompatibilities:
        - FARGATE
      Cpu: '256'
      Memory: '512'
      ExecutionRoleArn: !GetAtt ECSTaskExecutionRole.Arn
      ContainerDefinitions:
        - Name: dashboard
          Image: !Ref DockerImageDashboard
          Essential: true
          PortMappings:
            - ContainerPort: 8050
              HostPort: 8050
              Protocol: tcp
          LogConfiguration:
            LogDriver: awslogs
            Options:
              awslogs-group: !Ref DashboardLogGroup
              awslogs-region: !Ref AWS::Region
              awslogs-stream-prefix: dashboard

  DashboardLogGroup:
    Type: AWS::Logs::LogGroup
    Properties:
      LogGroupName: /ecs/jadugora-dashboard
      RetentionInDays: 30

  DashboardService:
    Type: AWS::ECS::Service
    DependsOn: DashboardLoadBalancerListener
    Properties:
      ServiceName: jadugora-dashboard
      Cluster: !Ref ECSCluster
      TaskDefinition: !Ref DashboardTaskDefinition
      LaunchType: FARGATE
      DesiredCount: 1
      NetworkConfiguration:
        AwsvpcConfiguration:
          AssignPublicIp: ENABLED
          Subnets:
            - !Ref PublicSubnet1
            - !Ref PublicSubnet2
          SecurityGroups:
            - !Ref DashboardSecurityGroup
      LoadBalancers:
        - ContainerName: dashboard
          ContainerPort: 8050
          TargetGroupArn: !Ref DashboardTargetGroup

  DashboardSecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupDescription: Security group for Dashboard
      VpcId: !Ref VPC
      SecurityGroupIngress:
        - IpProtocol: tcp
          FromPort: 8050
          ToPort: 8050
          CidrIp: 0.0.0.0/0

  DashboardLoadBalancer:
    Type: AWS::ElasticLoadBalancingV2::LoadBalancer
    Properties:
      Name: jadugora-dashboard-lb
      Subnets:
        - !Ref PublicSubnet1
        - !Ref PublicSubnet2
      SecurityGroups:
        - !Ref DashboardLoadBalancerSecurityGroup

  DashboardLoadBalancerSecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupDescription: Security group for Dashboard load balancer
      VpcId: !Ref VPC
      SecurityGroupIngress:
        - IpProtocol: tcp
          FromPort: 80
          ToPort: 80
          CidrIp: 0.0.0.0/0

  DashboardTargetGroup:
    Type: AWS::ElasticLoadBalancingV2::TargetGroup
    Properties:
      Name: jadugora-dashboard-tg
      Port: 8050
      Protocol: HTTP
      TargetType: ip
      VpcId: !Ref VPC
      HealthCheckPath: /
      HealthCheckIntervalSeconds: 30
      HealthCheckTimeoutSeconds: 5
      HealthyThresholdCount: 2
      UnhealthyThresholdCount: 2

  DashboardLoadBalancerListener:
    Type: AWS::ElasticLoadBalancingV2::Listener
    Properties:
      LoadBalancerArn: !Ref DashboardLoadBalancer
      Port: 80
      Protocol: HTTP
      DefaultActions:
        - Type: forward
          TargetGroupArn: !Ref DashboardTargetGroup

  # IAM Roles
  ECSTaskExecutionRole:
    Type: AWS::IAM::Role
    Properties:
      AssumeRolePolicyDocument:
        Statement:
          - Effect: Allow
            Principal:
              Service: ecs-tasks.amazonaws.com
            Action: sts:AssumeRole
      ManagedPolicyArns:
        - arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy

Outputs:
  ApiEndpoint:
    Description: URL of the API
    Value: !Join ['', ['http://', !GetAtt ApiLoadBalancer.DNSName]]
  
  DashboardEndpoint:
    Description: URL of the Dashboard
    Value: !Join ['', ['http://', !GetAtt DashboardLoadBalancer.DNSName]]
"""
    
    try:
        # Create GitHub Actions workflow
        os.makedirs(".github/workflows", exist_ok=True)
        with open(".github/workflows/ci-cd.yml", "w") as f:
            f.write(github_workflow)
        
        # Create CloudFormation template
        with open("deployment/cloudformation.yaml", "w") as f:
            f.write(cloudformation_template)
        
        logger.info("CI/CD pipeline configuration created successfully")
        return True
    except Exception as e:
        logger.error(f"Error creating CI/CD pipeline configuration: {e}")
        return False

def create_deployment_documentation():
    """
    Create documentation for the MLOps pipeline and deployment.
    
    Returns:
        bool: Success status
    """
    logger.info("Creating deployment documentation...")
    
    readme_content = """# Jadugora Uranium Mining Impact Assessment - MLOps Pipeline

## Overview

This repository contains the MLOps pipeline for the Jadugora Uranium Mining Impact Assessment project. The pipeline automates the training, evaluation, and deployment of machine learning models that assess the environmental and health impacts of uranium mining in Jadugora, Jharkhand, India.

## Architecture

The MLOps pipeline consists of the following components:

1. **Data Collection and Processing**: Scripts for collecting and processing environmental, health, and socioeconomic data.
2. **Exploratory Data Analysis**: Scripts for analyzing and visualizing the data.
3. **Model Development**: Scripts for training and evaluating machine learning models.
4. **Model Registry**: MLflow for tracking and versioning models.
5. **Model Serving API**: Flask API for serving model predictions.
6. **Monitoring Dashboard**: Dash application for monitoring model performance and data quality.
7. **CI/CD Pipeline**: GitHub Actions workflow for automating the deployment process.
8. **Cloud Infrastructure**: AWS CloudFormation template for deploying the infrastructure.

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Docker
- AWS CLI (for cloud deployment)

### Local Development

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/jadugora-impact-assessment.git
   cd jadugora-impact-assessment
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the data collection script:
   ```
   python src/data_collection.py
   ```

4. Run the exploratory analysis script:
   ```
   python src/exploratory_analysis.py
   ```

5. Run the model development script:
   ```
   python src/model_development.py
   ```

6. Run the MLOps pipeline setup:
   ```
   python src/mlops_pipeline.py
   ```

### Docker Deployment

1. Build the API Docker image:
   ```
   cd deployment/api
   docker build -t jadugora-impact-api .
   ```

2. Build the Dashboard Docker image:
   ```
   cd deployment/monitoring
   docker build -t jadugora-impact-dashboard .
   ```

3. Run the containers:
   ```
   docker run -p 5000:5000 jadugora-impact-api
   docker run -p 8050:8050 jadugora-impact-dashboard
   ```

### Cloud Deployment

1. Configure AWS credentials:
   ```
   aws configure
   ```

2. Deploy the CloudFormation stack:
   ```
   aws cloudformation deploy \\
     --template-file deployment/cloudformation.yaml \\
     --stack-name jadugora-impact-assessment \\
     --capabilities CAPABILITY_IAM \\
     --parameter-overrides \\
       DockerImageApi=yourusername/jadugora-impact-api:latest \\
       DockerImageDashboard=yourusername/jadugora-impact-dashboard:latest
   ```

## API Documentation

### Environmental Impact Prediction

**Endpoint**: `/predict/environmental`

**Method**: POST

**Request Body**:
```json
{
  "ore_extracted_tons": 350000,
  "uranium_produced_kg": 70000,
  "waste_generated_tons": 300000,
  "tailings_volume_cubic_m": 250000,
  "water_used_million_liters": 1000
}
```

**Response**:
```json
{
  "predictions": {
    "radiation_level": 0.85,
    "water_uranium": 16.2,
    "soil_uranium": 7.5
  }
}
```

### Health Impact Prediction

**Endpoint**: `/predict/health`

**Method**: POST

**Request Body**:
```json
{
  "residential_area": 0.85,
  "ph_level": 6.2,
  "heavy_metals_ppm": 3.5,
  "uranium_concentration_ppb": 16.2,
  "uranium_ppm": 7.5,
  "radium_ppm": 1.8,
  "lead_ppm": 25.3,
  "arsenic_ppm": 12.5,
  "ore_extracted_tons": 350000,
  "uranium_produced_kg": 70000,
  "waste_generated_tons": 300000,
  "tailings_volume_cubic_m": 250000,
  "water_used_million_liters": 1000
}
```

**Response**:
```json
{
  "predictions": {
    "cancer_rate": 28.5,
    "respiratory_disease_rate": 65.2,
    "skin_disorders_rate": 45.8,
    "kidney_disease_rate": 30.1
  }
}
```

## Monitoring Dashboard

The monitoring dashboard provides visualizations for:

1. Model performance metrics
2. Actual vs. predicted values
3. Model drift detection
4. Data quality monitoring

Access the dashboard at: http://localhost:8050 (local) or the AWS endpoint (cloud).

## CI/CD Pipeline

The CI/CD pipeline automates the following steps:

1. Run tests
2. Train models
3. Build and push Docker images
4. Deploy to AWS

The pipeline is triggered on:
- Push to main branch
- Pull requests to main branch
- Weekly schedule (Sundays at midnight)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
"""
    
    try:
        with open("deployment/README.md", "w") as f:
            f.write(readme_content)
        
        logger.info("Deployment documentation created successfully")
        return True
    except Exception as e:
        logger.error(f"Error creating deployment documentation: {e}")
        return False

def main():
    """
    Main function to execute all MLOps pipeline setup steps.
    """
    logger.info("Starting MLOps pipeline setup")
    
    try:
        # Set up MLflow
        experiment_id = setup_mlflow()
        
        # Load datasets
        datasets = {}
        try:
            # Load fixed datasets
            mining_health_df = pd.read_csv("data/processed/mining_health_data_fixed.csv")
            env_health_df = pd.read_csv("data/processed/environmental_health_data_fixed.csv")
            
            datasets['mining_health'] = mining_health_df
            datasets['env_health'] = env_health_df
            
            logger.info("Datasets loaded successfully")
        except Exception as e:
            logger.error(f"Error loading datasets: {e}")
            raise
        
        # Load trained models
        models = load_models()
        
        # Register models with MLflow
        registered_models = register_models_with_mlflow(models, datasets)
        
        # Create model API
        api_created = create_model_api()
        
        # Create monitoring dashboard
        dashboard_created = create_monitoring_dashboard()
        
        # Create CI/CD pipeline
        cicd_created = create_ci_cd_pipeline()
        
        # Create deployment documentation
        docs_created = create_deployment_documentation()
        
        # Check if all steps completed successfully
        if api_created and dashboard_created and cicd_created and docs_created:
            logger.info("MLOps pipeline setup completed successfully")
            print("MLOps pipeline setup completed successfully. Results saved to deployment/ directory.")
        else:
            logger.warning("MLOps pipeline setup completed with warnings")
            print("MLOps pipeline setup completed with warnings. Check logs for details.")
    except Exception as e:
        logger.error(f"Error in MLOps pipeline setup: {e}")
        print(f"Error in MLOps pipeline setup: {e}")
        raise

if __name__ == "__main__":
    main()
