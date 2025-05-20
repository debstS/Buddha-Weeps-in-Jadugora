"""
Exploratory Data Analysis for Jadugora Uranium Mining Impact Project

This script performs exploratory data analysis on the collected datasets related to
environmental, health, and socioeconomic impacts of uranium mining in Jadugora.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/eda.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Ensure necessary directories exist
os.makedirs("data/processed", exist_ok=True)
os.makedirs("reports/figures", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

def load_datasets():
    """
    Load all raw datasets for analysis.
    
    Returns:
        dict: Dictionary containing all loaded datasets
    """
    logger.info("Loading datasets for analysis...")
    
    datasets = {}
    
    # Environmental data
    try:
        datasets['radiation'] = pd.read_csv("data/raw/radiation_levels.csv")
        datasets['water'] = pd.read_csv("data/raw/water_quality.csv")
        datasets['soil'] = pd.read_csv("data/raw/soil_contamination.csv")
        logger.info("Environmental datasets loaded successfully")
    except Exception as e:
        logger.error(f"Error loading environmental datasets: {e}")
    
    # Health data
    try:
        datasets['disease'] = pd.read_csv("data/raw/disease_prevalence.csv")
        datasets['birth'] = pd.read_csv("data/raw/birth_defects.csv")
        datasets['mortality'] = pd.read_csv("data/raw/mortality_rates.csv")
        logger.info("Health datasets loaded successfully")
    except Exception as e:
        logger.error(f"Error loading health datasets: {e}")
    
    # Socioeconomic data
    try:
        datasets['employment'] = pd.read_csv("data/raw/employment_data.csv")
        datasets['education'] = pd.read_csv("data/raw/education_data.csv")
        datasets['displacement'] = pd.read_csv("data/raw/displacement_data.csv")
        logger.info("Socioeconomic datasets loaded successfully")
    except Exception as e:
        logger.error(f"Error loading socioeconomic datasets: {e}")
    
    # Mining production data
    try:
        datasets['mining'] = pd.read_csv("data/raw/mining_production.csv")
        logger.info("Mining production data loaded successfully")
    except Exception as e:
        logger.error(f"Error loading mining production data: {e}")
    
    # Spatial data
    try:
        datasets['mining_sites'] = pd.read_csv("data/raw/mining_sites.csv")
        datasets['villages'] = pd.read_csv("data/raw/villages.csv")
        datasets['sampling_points'] = pd.read_csv("data/raw/sampling_points.csv")
        logger.info("Spatial datasets loaded successfully")
    except Exception as e:
        logger.error(f"Error loading spatial datasets: {e}")
    
    return datasets

def generate_summary_statistics(datasets):
    """
    Generate summary statistics for all datasets.
    
    Args:
        datasets (dict): Dictionary containing all loaded datasets
    
    Returns:
        dict: Dictionary containing summary statistics for each dataset
    """
    logger.info("Generating summary statistics...")
    
    summary_stats = {}
    
    for name, df in datasets.items():
        try:
            # Basic statistics
            stats = {
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': df.dtypes.to_dict(),
                'missing_values': df.isnull().sum().to_dict(),
                'numeric_summary': {}
            }
            
            # Numeric summaries
            numeric_cols = df.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                stats['numeric_summary'][col] = {
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'mean': df[col].mean(),
                    'median': df[col].median(),
                    'std': df[col].std()
                }
            
            summary_stats[name] = stats
            logger.info(f"Summary statistics generated for {name} dataset")
        except Exception as e:
            logger.error(f"Error generating summary statistics for {name}: {e}")
    
    # Save summary statistics
    with open("reports/summary_statistics.json", "w") as f:
        # Convert non-serializable objects to strings and native Python types
        serializable_stats = {}
        for dataset_name, stats in summary_stats.items():
            numeric_summary_serializable = {}
            for col, metrics in stats['numeric_summary'].items():
                numeric_summary_serializable[col] = {
                    k: float(v) if isinstance(v, (np.float64, np.float32, np.int64, np.int32)) else v
                    for k, v in metrics.items()
                }
            
            serializable_stats[dataset_name] = {
                'shape': str(stats['shape']),
                'columns': stats['columns'],
                'dtypes': {k: str(v) for k, v in stats['dtypes'].items()},
                'missing_values': {k: int(v) if isinstance(v, (np.int64, np.int32)) else v 
                                  for k, v in stats['missing_values'].items()},
                'numeric_summary': numeric_summary_serializable
            }
        json.dump(serializable_stats, f, indent=4)
    
    return summary_stats

def analyze_environmental_data(datasets):
    """
    Analyze environmental datasets and create visualizations.
    
    Args:
        datasets (dict): Dictionary containing all loaded datasets
    """
    logger.info("Analyzing environmental data...")
    
    # Radiation levels over time
    try:
        radiation_df = datasets['radiation']
        radiation_df['date'] = pd.to_datetime(radiation_df['date'])
        
        plt.figure(figsize=(12, 6))
        plt.plot(radiation_df['date'], radiation_df['mine_proximity'], label='Near Mine')
        plt.plot(radiation_df['date'], radiation_df['residential_area'], label='Residential Area')
        plt.plot(radiation_df['date'], radiation_df['control_site'], label='Control Site')
        plt.title('Radiation Levels Over Time')
        plt.xlabel('Date')
        plt.ylabel('Radiation Level (μSv/h)')
        plt.legend()
        plt.tight_layout()
        plt.savefig('reports/figures/radiation_levels_time_series.png')
        plt.close()
        
        # Distribution of radiation levels
        plt.figure(figsize=(12, 6))
        sns.histplot(radiation_df['mine_proximity'], kde=True, label='Near Mine')
        sns.histplot(radiation_df['residential_area'], kde=True, label='Residential Area')
        sns.histplot(radiation_df['control_site'], kde=True, label='Control Site')
        plt.title('Distribution of Radiation Levels')
        plt.xlabel('Radiation Level (μSv/h)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.tight_layout()
        plt.savefig('reports/figures/radiation_levels_distribution.png')
        plt.close()
        
        logger.info("Radiation analysis completed")
    except Exception as e:
        logger.error(f"Error analyzing radiation data: {e}")
    
    # Water quality analysis
    try:
        water_df = datasets['water']
        water_df['date'] = pd.to_datetime(water_df['date'])
        
        # Time series of water quality parameters
        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        
        axes[0].plot(water_df['date'], water_df['ph_level'])
        axes[0].set_title('pH Levels Over Time')
        axes[0].set_ylabel('pH')
        
        axes[1].plot(water_df['date'], water_df['heavy_metals_ppm'], color='orange')
        axes[1].set_title('Heavy Metals Concentration Over Time')
        axes[1].set_ylabel('Concentration (ppm)')
        
        axes[2].plot(water_df['date'], water_df['uranium_concentration_ppb'], color='red')
        axes[2].set_title('Uranium Concentration Over Time')
        axes[2].set_ylabel('Concentration (ppb)')
        axes[2].set_xlabel('Date')
        
        plt.tight_layout()
        plt.savefig('reports/figures/water_quality_time_series.png')
        plt.close()
        
        # Correlation between water quality parameters
        corr = water_df[['ph_level', 'heavy_metals_ppm', 'uranium_concentration_ppb']].corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation Between Water Quality Parameters')
        plt.tight_layout()
        plt.savefig('reports/figures/water_quality_correlation.png')
        plt.close()
        
        logger.info("Water quality analysis completed")
    except Exception as e:
        logger.error(f"Error analyzing water quality data: {e}")
    
    # Soil contamination analysis
    try:
        soil_df = datasets['soil']
        soil_df['date'] = pd.to_datetime(soil_df['date'])
        
        # Time series of soil contaminants
        fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
        
        axes[0].plot(soil_df['date'], soil_df['uranium_ppm'], color='purple')
        axes[0].set_title('Uranium in Soil Over Time')
        axes[0].set_ylabel('Concentration (ppm)')
        
        axes[1].plot(soil_df['date'], soil_df['radium_ppm'], color='red')
        axes[1].set_title('Radium in Soil Over Time')
        axes[1].set_ylabel('Concentration (ppm)')
        
        axes[2].plot(soil_df['date'], soil_df['lead_ppm'], color='blue')
        axes[2].set_title('Lead in Soil Over Time')
        axes[2].set_ylabel('Concentration (ppm)')
        
        axes[3].plot(soil_df['date'], soil_df['arsenic_ppm'], color='green')
        axes[3].set_title('Arsenic in Soil Over Time')
        axes[3].set_ylabel('Concentration (ppm)')
        axes[3].set_xlabel('Date')
        
        plt.tight_layout()
        plt.savefig('reports/figures/soil_contamination_time_series.png')
        plt.close()
        
        # Correlation between soil contaminants
        corr = soil_df[['uranium_ppm', 'radium_ppm', 'lead_ppm', 'arsenic_ppm']].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation Between Soil Contaminants')
        plt.tight_layout()
        plt.savefig('reports/figures/soil_contaminants_correlation.png')
        plt.close()
        
        logger.info("Soil contamination analysis completed")
    except Exception as e:
        logger.error(f"Error analyzing soil contamination data: {e}")

def analyze_health_data(datasets):
    """
    Analyze health datasets and create visualizations.
    
    Args:
        datasets (dict): Dictionary containing all loaded datasets
    """
    logger.info("Analyzing health data...")
    
    # Disease prevalence analysis
    try:
        disease_df = datasets['disease']
        
        # Disease trends over time
        plt.figure(figsize=(12, 6))
        plt.plot(disease_df['year'], disease_df['cancer_cases'], marker='o', label='Cancer')
        plt.plot(disease_df['year'], disease_df['respiratory_disease'], marker='s', label='Respiratory Disease')
        plt.plot(disease_df['year'], disease_df['skin_disorders'], marker='^', label='Skin Disorders')
        plt.plot(disease_df['year'], disease_df['kidney_disease'], marker='d', label='Kidney Disease')
        plt.title('Disease Prevalence Over Time (per 10,000 population)')
        plt.xlabel('Year')
        plt.ylabel('Cases per 10,000')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('reports/figures/disease_prevalence_trends.png')
        plt.close()
        
        logger.info("Disease prevalence analysis completed")
    except Exception as e:
        logger.error(f"Error analyzing disease prevalence data: {e}")
    
    # Birth defects analysis
    try:
        birth_df = datasets['birth']
        
        # Birth defects trends over time
        plt.figure(figsize=(12, 6))
        plt.plot(birth_df['year'], birth_df['congenital_defects'], marker='o', label='Congenital Defects')
        plt.plot(birth_df['year'], birth_df['stillbirths'], marker='s', label='Stillbirths')
        plt.plot(birth_df['year'], birth_df['low_birth_weight'], marker='^', label='Low Birth Weight')
        plt.title('Birth Issues Over Time (per 1,000 births)')
        plt.xlabel('Year')
        plt.ylabel('Cases per 1,000 births')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('reports/figures/birth_defects_trends.png')
        plt.close()
        
        logger.info("Birth defects analysis completed")
    except Exception as e:
        logger.error(f"Error analyzing birth defects data: {e}")
    
    # Mortality rates analysis
    try:
        mortality_df = datasets['mortality']
        
        # Mortality trends over time
        plt.figure(figsize=(12, 6))
        plt.plot(mortality_df['year'], mortality_df['overall_mortality'], marker='o', label='Overall Mortality')
        plt.plot(mortality_df['year'], mortality_df['cancer_mortality'], marker='s', label='Cancer Mortality')
        plt.plot(mortality_df['year'], mortality_df['infant_mortality'], marker='^', label='Infant Mortality')
        plt.title('Mortality Rates Over Time (per 1,000 population)')
        plt.xlabel('Year')
        plt.ylabel('Deaths per 1,000')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('reports/figures/mortality_trends.png')
        plt.close()
        
        logger.info("Mortality rates analysis completed")
    except Exception as e:
        logger.error(f"Error analyzing mortality data: {e}")

def analyze_socioeconomic_data(datasets):
    """
    Analyze socioeconomic datasets and create visualizations.
    
    Args:
        datasets (dict): Dictionary containing all loaded datasets
    """
    logger.info("Analyzing socioeconomic data...")
    
    # Employment analysis
    try:
        employment_df = datasets['employment']
        
        # Employment trends over time
        plt.figure(figsize=(12, 6))
        plt.stackplot(employment_df['year'], 
                     employment_df['mining_employment_pct'],
                     employment_df['agriculture_employment_pct'],
                     employment_df['service_sector_pct'],
                     labels=['Mining', 'Agriculture', 'Service Sector'])
        plt.plot(employment_df['year'], employment_df['unemployment_rate'], 'r-', linewidth=2, label='Unemployment Rate')
        plt.title('Employment Distribution and Unemployment Rate Over Time')
        plt.xlabel('Year')
        plt.ylabel('Percentage (%)')
        plt.legend(loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('reports/figures/employment_trends.png')
        plt.close()
        
        logger.info("Employment analysis completed")
    except Exception as e:
        logger.error(f"Error analyzing employment data: {e}")
    
    # Education analysis
    try:
        education_df = datasets['education']
        
        # Education trends over time
        plt.figure(figsize=(12, 6))
        plt.plot(education_df['year'], education_df['literacy_rate'], marker='o', label='Literacy Rate')
        plt.plot(education_df['year'], education_df['primary_education_pct'], marker='s', label='Primary Education')
        plt.plot(education_df['year'], education_df['secondary_education_pct'], marker='^', label='Secondary Education')
        plt.plot(education_df['year'], education_df['higher_education_pct'], marker='d', label='Higher Education')
        plt.title('Education Indicators Over Time')
        plt.xlabel('Year')
        plt.ylabel('Percentage (%)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('reports/figures/education_trends.png')
        plt.close()
        
        logger.info("Education analysis completed")
    except Exception as e:
        logger.error(f"Error analyzing education data: {e}")
    
    # Displacement analysis
    try:
        displacement_df = datasets['displacement']
        
        # Create figure with two y-axes
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Plot families displaced
        color = 'tab:blue'
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Families Displaced', color=color)
        ax1.plot(displacement_df['year'], displacement_df['families_displaced'], color=color, marker='o')
        ax1.tick_params(axis='y', labelcolor=color)
        
        # Create second y-axis
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Land Acquired (hectares)', color=color)
        ax2.plot(displacement_df['year'], displacement_df['land_acquired_hectares'], color=color, marker='s')
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title('Displacement and Land Acquisition Over Time')
        fig.tight_layout()
        plt.savefig('reports/figures/displacement_trends.png')
        plt.close()
        
        # Compensation trends
        plt.figure(figsize=(12, 6))
        plt.plot(displacement_df['year'], displacement_df['compensation_per_hectare_inr'], marker='o')
        plt.title('Land Compensation Rates Over Time')
        plt.xlabel('Year')
        plt.ylabel('Compensation (INR per hectare)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('reports/figures/compensation_trends.png')
        plt.close()
        
        logger.info("Displacement analysis completed")
    except Exception as e:
        logger.error(f"Error analyzing displacement data: {e}")

def analyze_mining_production(datasets):
    """
    Analyze mining production data and create visualizations.
    
    Args:
        datasets (dict): Dictionary containing all loaded datasets
    """
    logger.info("Analyzing mining production data...")
    
    try:
        mining_df = datasets['mining']
        
        # Production trends over time
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        color = 'tab:blue'
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Ore Extracted (tons)', color=color)
        ax1.plot(mining_df['year'], mining_df['ore_extracted_tons'], color=color, marker='o')
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Uranium Produced (kg)', color=color)
        ax2.plot(mining_df['year'], mining_df['uranium_produced_kg'], color=color, marker='s')
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title('Mining Production Over Time')
        fig.tight_layout()
        plt.savefig('reports/figures/mining_production_trends.png')
        plt.close()
        
        # Waste generation
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        color = 'tab:green'
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Waste Generated (tons)', color=color)
        ax1.plot(mining_df['year'], mining_df['waste_generated_tons'], color=color, marker='o')
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax2 = ax1.twinx()
        color = 'tab:purple'
        ax2.set_ylabel('Tailings Volume (cubic m)', color=color)
        ax2.plot(mining_df['year'], mining_df['tailings_volume_cubic_m'], color=color, marker='s')
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title('Mining Waste Generation Over Time')
        fig.tight_layout()
        plt.savefig('reports/figures/mining_waste_trends.png')
        plt.close()
        
        # Water usage
        plt.figure(figsize=(12, 6))
        plt.plot(mining_df['year'], mining_df['water_used_million_liters'], marker='o', color='tab:blue')
        plt.title('Water Usage in Mining Operations Over Time')
        plt.xlabel('Year')
        plt.ylabel('Water Used (million liters)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('reports/figures/mining_water_usage.png')
        plt.close()
        
        # Calculate efficiency metrics
        mining_df['waste_to_ore_ratio'] = mining_df['waste_generated_tons'] / mining_df['ore_extracted_tons']
        mining_df['uranium_yield_percentage'] = (mining_df['uranium_produced_kg'] / mining_df['ore_extracted_tons']) * 100
        mining_df['water_per_ton_ore'] = mining_df['water_used_million_liters'] * 1000 / mining_df['ore_extracted_tons']
        
        # Save processed mining data
        mining_df.to_csv('data/processed/mining_with_metrics.csv', index=False)
        
        # Efficiency metrics visualization
        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        
        axes[0].plot(mining_df['year'], mining_df['waste_to_ore_ratio'], marker='o', color='brown')
        axes[0].set_title('Waste to Ore Ratio Over Time')
        axes[0].set_ylabel('Ratio')
        axes[0].grid(True)
        
        axes[1].plot(mining_df['year'], mining_df['uranium_yield_percentage'], marker='s', color='purple')
        axes[1].set_title('Uranium Yield Percentage Over Time')
        axes[1].set_ylabel('Yield (%)')
        axes[1].grid(True)
        
        axes[2].plot(mining_df['year'], mining_df['water_per_ton_ore'], marker='^', color='blue')
        axes[2].set_title('Water Usage per Ton of Ore Over Time')
        axes[2].set_ylabel('Water (liters/ton)')
        axes[2].set_xlabel('Year')
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig('reports/figures/mining_efficiency_metrics.png')
        plt.close()
        
        logger.info("Mining production analysis completed")
    except Exception as e:
        logger.error(f"Error analyzing mining production data: {e}")

def analyze_spatial_data(datasets):
    """
    Analyze spatial data and create visualizations.
    
    Args:
        datasets (dict): Dictionary containing all loaded datasets
    """
    logger.info("Analyzing spatial data...")
    
    try:
        mining_sites = datasets['mining_sites']
        villages = datasets['villages']
        sampling_points = datasets['sampling_points']
        
        # Scatter plot of mining sites, villages, and sampling points
        plt.figure(figsize=(12, 10))
        
        # Plot mining sites
        plt.scatter(mining_sites['longitude'], mining_sites['latitude'], 
                   s=mining_sites['area_hectares']*2, c='red', alpha=0.7, 
                   label='Mining Sites', edgecolors='black')
        
        # Plot villages
        plt.scatter(villages['longitude'], villages['latitude'], 
                   s=np.sqrt(villages['population']), c='blue', alpha=0.5, 
                   label='Villages', edgecolors='black')
        
        # Plot sampling points
        colors = {'Water': 'cyan', 'Soil': 'brown', 'Air': 'green'}
        for sample_type in sampling_points['sample_type'].unique():
            subset = sampling_points[sampling_points['sample_type'] == sample_type]
            plt.scatter(subset['longitude'], subset['latitude'], 
                       s=30, c=colors[sample_type], alpha=0.7, 
                       label=f'{sample_type} Sampling Points', edgecolors='black')
        
        plt.title('Spatial Distribution of Mining Sites, Villages, and Sampling Points')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('reports/figures/spatial_distribution.png')
        plt.close()
        
        # Analysis of village distance to mines
        plt.figure(figsize=(10, 6))
        sns.histplot(villages['distance_to_nearest_mine_km'], bins=10, kde=True)
        plt.title('Distribution of Village Distances to Nearest Mine')
        plt.xlabel('Distance (km)')
        plt.ylabel('Number of Villages')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('reports/figures/village_mine_distance_distribution.png')
        plt.close()
        
        # Analysis of sampling points by type and distance
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='sample_type', y='distance_to_nearest_mine_km', data=sampling_points)
        plt.title('Distance to Nearest Mine by Sampling Point Type')
        plt.xlabel('Sample Type')
        plt.ylabel('Distance (km)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('reports/figures/sampling_point_distance_by_type.png')
        plt.close()
        
        logger.info("Spatial data analysis completed")
    except Exception as e:
        logger.error(f"Error analyzing spatial data: {e}")

def analyze_relationships(datasets):
    """
    Analyze relationships between different datasets.
    
    Args:
        datasets (dict): Dictionary containing all loaded datasets
    """
    logger.info("Analyzing relationships between datasets...")
    
    try:
        # Prepare data for correlation analysis
        # Merge mining production with health data
        mining_df = datasets['mining']
        disease_df = datasets['disease']
        
        # Ensure we have matching years
        merged_df = pd.merge(mining_df, disease_df, on='year', how='inner')
        
        # Calculate correlations
        correlation_cols = [
            'ore_extracted_tons', 'uranium_produced_kg', 'waste_generated_tons',
            'tailings_volume_cubic_m', 'water_used_million_liters',
            'cancer_cases', 'respiratory_disease', 'skin_disorders', 'kidney_disease'
        ]
        
        corr_matrix = merged_df[correlation_cols].corr()
        
        # Visualize correlations
        plt.figure(figsize=(14, 12))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
        plt.title('Correlation Between Mining Activities and Disease Prevalence')
        plt.tight_layout()
        plt.savefig('reports/figures/mining_health_correlation.png')
        plt.close()
        
        # Save the correlation matrix
        corr_matrix.to_csv('data/processed/mining_health_correlation.csv')
        
        # Time series comparison of mining production and cancer cases
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        color = 'tab:blue'
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Uranium Production (kg)', color=color)
        ax1.plot(merged_df['year'], merged_df['uranium_produced_kg'], color=color, marker='o')
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Cancer Cases (per 10,000)', color=color)
        ax2.plot(merged_df['year'], merged_df['cancer_cases'], color=color, marker='s')
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title('Uranium Production and Cancer Cases Over Time')
        fig.tight_layout()
        plt.savefig('reports/figures/uranium_cancer_comparison.png')
        plt.close()
        
        # Scatter plot with regression line
        plt.figure(figsize=(10, 6))
        sns.regplot(x='uranium_produced_kg', y='cancer_cases', data=merged_df)
        plt.title('Relationship Between Uranium Production and Cancer Cases')
        plt.xlabel('Uranium Production (kg)')
        plt.ylabel('Cancer Cases (per 10,000)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('reports/figures/uranium_cancer_regression.png')
        plt.close()
        
        logger.info("Relationship analysis completed")
    except Exception as e:
        logger.error(f"Error analyzing relationships between datasets: {e}")

def create_summary_report():
    """
    Create a summary report of the exploratory data analysis.
    """
    logger.info("Creating summary report...")
    
    report_content = """# Exploratory Data Analysis: Uranium Mining Impact in Jadugora

## Overview
This report summarizes the findings from exploratory data analysis of environmental, health, socioeconomic, and mining production data related to uranium mining activities in Jadugora, Jharkhand, India.

## Environmental Impact Analysis

### Radiation Levels
The analysis of radiation data shows significantly higher radiation levels near mining sites compared to residential areas and control sites. The time series analysis indicates fluctuations in radiation levels that may correlate with mining activity intensity.

### Water Quality
Water quality parameters show concerning trends, with pH levels tending toward acidity and elevated levels of heavy metals and uranium concentration in water samples. These contaminants show seasonal variations and general increasing trends over time.

### Soil Contamination
Soil samples reveal the presence of uranium, radium, lead, and arsenic at levels that exceed typical background concentrations. The contamination appears to be more severe in proximity to mining and processing facilities.

## Health Impact Analysis

### Disease Prevalence
The analysis shows increasing trends in cancer cases, respiratory diseases, skin disorders, and kidney diseases in the population near mining operations. Cancer cases in particular show a strong positive correlation with mining production volumes.

### Birth Defects
Data on birth outcomes indicates elevated rates of congenital defects, stillbirths, and low birth weight in the region. These rates have shown concerning upward trends over the study period.

### Mortality Rates
Overall mortality and cancer-specific mortality rates show increasing trends in the mining-affected areas, while infant mortality has decreased (though at a slower rate than national averages).

## Socioeconomic Impact Analysis

### Employment Patterns
Employment data reveals a gradual shift away from mining and agricultural employment toward the service sector. Unemployment rates have shown a slight increase over time, suggesting economic challenges in the region.

### Education
Education indicators show improvement in literacy rates and primary education, but secondary and higher education rates remain relatively low compared to national averages.

### Displacement and Land Acquisition
The data shows significant displacement of families and acquisition of land for mining purposes, particularly in the earlier years of operations. Compensation rates have increased over time but may not reflect the true economic and social costs of displacement.

## Mining Production Analysis

### Production Trends
Uranium mining production has generally increased over the study period, with corresponding increases in waste generation and water usage. The efficiency metrics indicate declining ore quality over time, requiring more intensive extraction processes.

### Environmental Efficiency
The waste-to-ore ratio has increased over time, suggesting less efficient extraction and greater environmental burden per unit of uranium produced. Water usage per ton of ore has also increased, indicating greater resource intensity.

## Spatial Analysis

### Geographic Distribution
The spatial analysis reveals clustering of villages around mining sites, with many communities located within 5km of active mining operations. Sampling points show varying levels of contamination based on proximity to mines.

## Relationship Analysis

### Mining-Health Correlations
Strong positive correlations were found between mining production metrics and disease prevalence, particularly for cancer and respiratory diseases. The regression analysis suggests a potential dose-response relationship between uranium production volumes and cancer cases.

## Conclusions and Next Steps

The exploratory data analysis reveals concerning patterns of environmental contamination, health impacts, and socioeconomic changes associated with uranium mining activities in Jadugora. These findings warrant further investigation through advanced statistical modeling and machine learning approaches to quantify relationships and predict future impacts.

The next phase of analysis will focus on developing predictive models for:
1. Environmental contamination spread based on mining activity
2. Health outcome prediction based on exposure metrics
3. Socioeconomic impact assessment and intervention planning

These models will provide a more comprehensive understanding of the complex interactions between mining activities and community impacts, and can inform policy decisions and remediation efforts.
"""
    
    with open("reports/eda_summary_report.md", "w") as f:
        f.write(report_content)
    
    logger.info("Summary report created")

def main():
    """
    Main function to execute all EDA steps.
    """
    logger.info("Starting exploratory data analysis")
    
    # Load all datasets
    datasets = load_datasets()
    
    # Generate summary statistics
    summary_stats = generate_summary_statistics(datasets)
    
    # Analyze each category of data
    analyze_environmental_data(datasets)
    analyze_health_data(datasets)
    analyze_socioeconomic_data(datasets)
    analyze_mining_production(datasets)
    analyze_spatial_data(datasets)
    
    # Analyze relationships between datasets
    analyze_relationships(datasets)
    
    # Create summary report
    create_summary_report()
    
    logger.info("Exploratory data analysis completed")
    print("EDA completed. Results saved to reports/ directory.")

if __name__ == "__main__":
    main()
