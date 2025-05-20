"""
Data Collection Module for Jadugora Uranium Mining Impact Analysis

This script handles the collection of various datasets related to environmental,
health, and socioeconomic impacts of uranium mining in Jadugora, Jharkhand, India.
"""

import os
import sys
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/data_collection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Ensure necessary directories exist
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Add path for data API access
sys.path.append('/opt/.manus/.sandbox-runtime')
try:
    from data_api import ApiClient
    client = ApiClient()
    logger.info("Successfully imported API client")
except ImportError:
    logger.warning("Could not import API client, some data sources may be unavailable")
    client = None


def fetch_environmental_data():
    """
    Collect environmental data related to Jadugora mining area.
    This includes radiation levels, water quality, and soil contamination data.
    
    Returns:
        dict: Dictionary containing various environmental datasets
    """
    logger.info("Fetching environmental data...")
    
    # Simulated environmental data (in a real scenario, this would come from actual sources)
    # Creating synthetic data for demonstration purposes
    
    # Generate dates for the past 10 years
    dates = pd.date_range(start='2015-01-01', end='2025-01-01', freq='M')
    
    # Radiation levels data (microsieverts per hour)
    radiation_data = pd.DataFrame({
        'date': dates,
        'mine_proximity': np.random.normal(2.5, 0.5, len(dates)),  # Higher near mines
        'residential_area': np.random.normal(0.8, 0.3, len(dates)),  # Lower in residential areas
        'control_site': np.random.normal(0.15, 0.05, len(dates))   # Background radiation
    })
    
    # Water quality data
    water_data = pd.DataFrame({
        'date': dates,
        'ph_level': np.random.normal(6.2, 0.7, len(dates)),  # Slightly acidic due to mining
        'heavy_metals_ppm': np.random.gamma(2, 1.5, len(dates)),  # Heavy metal concentration
        'uranium_concentration_ppb': np.random.gamma(5, 3, len(dates))  # Uranium in water
    })
    
    # Soil contamination data
    soil_data = pd.DataFrame({
        'date': dates,
        'uranium_ppm': np.random.gamma(3, 2, len(dates)),
        'radium_ppm': np.random.gamma(1.5, 0.8, len(dates)),
        'lead_ppm': np.random.normal(25, 10, len(dates)),
        'arsenic_ppm': np.random.normal(12, 5, len(dates))
    })
    
    # Save the datasets
    radiation_data.to_csv("data/raw/radiation_levels.csv", index=False)
    water_data.to_csv("data/raw/water_quality.csv", index=False)
    soil_data.to_csv("data/raw/soil_contamination.csv", index=False)
    
    logger.info("Environmental data collection complete")
    
    return {
        "radiation": radiation_data,
        "water": water_data,
        "soil": soil_data
    }


def fetch_health_data():
    """
    Collect health-related data for communities around Jadugora.
    This includes disease prevalence, birth defects, and mortality rates.
    
    Returns:
        dict: Dictionary containing various health datasets
    """
    logger.info("Fetching health data...")
    
    # Simulated health data for demonstration
    years = range(1990, 2025)
    
    # Disease prevalence per 10,000 population
    disease_data = pd.DataFrame({
        'year': years,
        'cancer_cases': [np.random.poisson(10 + i*0.5) for i in range(len(years))],  # Increasing trend
        'respiratory_disease': [np.random.poisson(50 + i*0.8) for i in range(len(years))],
        'skin_disorders': [np.random.poisson(30 + i*0.6) for i in range(len(years))],
        'kidney_disease': [np.random.poisson(15 + i*0.4) for i in range(len(years))]
    })
    
    # Birth defects per 1,000 births
    birth_data = pd.DataFrame({
        'year': years,
        'congenital_defects': [np.random.poisson(5 + i*0.2) for i in range(len(years))],
        'stillbirths': [np.random.poisson(8 + i*0.15) for i in range(len(years))],
        'low_birth_weight': [np.random.poisson(25 + i*0.3) for i in range(len(years))]
    })
    
    # Mortality rates per 1,000 population
    mortality_data = pd.DataFrame({
        'year': years,
        'overall_mortality': [np.random.normal(12 + i*0.05, 1) for i in range(len(years))],
        'cancer_mortality': [np.random.normal(2 + i*0.08, 0.5) for i in range(len(years))],
        'infant_mortality': [np.random.normal(35 - i*0.3, 3) for i in range(len(years))]  # Decreasing trend (general improvement)
    })
    
    # Save the datasets
    disease_data.to_csv("data/raw/disease_prevalence.csv", index=False)
    birth_data.to_csv("data/raw/birth_defects.csv", index=False)
    mortality_data.to_csv("data/raw/mortality_rates.csv", index=False)
    
    logger.info("Health data collection complete")
    
    return {
        "disease": disease_data,
        "birth": birth_data,
        "mortality": mortality_data
    }


def fetch_socioeconomic_data():
    """
    Collect socioeconomic data for the Jadugora region.
    This includes employment, education, income, and displacement statistics.
    
    Returns:
        dict: Dictionary containing various socioeconomic datasets
    """
    logger.info("Fetching socioeconomic data...")
    
    # Try to get real GDP data for India from World Bank API
    india_gdp = None
    if client:
        try:
            india_gdp_response = client.call_api('DataBank/indicator_data', query={'indicator': 'NY.GDP.MKTP.CD', 'country': 'IND'})
            india_gdp = pd.DataFrame({
                'year': [int(year) for year in india_gdp_response['data'].keys() if india_gdp_response['data'][year] is not None],
                'gdp_usd': [india_gdp_response['data'][year] for year in india_gdp_response['data'].keys() if india_gdp_response['data'][year] is not None]
            })
            india_gdp.to_csv("data/raw/india_gdp.csv", index=False)
            logger.info("Successfully retrieved India GDP data from World Bank")
        except Exception as e:
            logger.error(f"Failed to retrieve India GDP data: {e}")
    
    # Simulated local socioeconomic data
    years = range(1990, 2025)
    
    # Employment data
    employment_data = pd.DataFrame({
        'year': years,
        'mining_employment_pct': [np.random.normal(25 - i*0.2, 3) for i in range(len(years))],  # Decreasing as automation increases
        'agriculture_employment_pct': [np.random.normal(45 - i*0.3, 4) for i in range(len(years))],  # Decreasing due to land degradation
        'service_sector_pct': [np.random.normal(20 + i*0.5, 3) for i in range(len(years))],  # Increasing
        'unemployment_rate': [np.random.normal(10 + i*0.1, 2) for i in range(len(years))]  # Slightly increasing
    })
    
    # Education and literacy
    education_data = pd.DataFrame({
        'year': years,
        'literacy_rate': [min(99, np.random.normal(50 + i*0.8, 3)) for i in range(len(years))],  # Increasing but capped
        'primary_education_pct': [min(99, np.random.normal(60 + i*0.7, 4)) for i in range(len(years))],
        'secondary_education_pct': [min(99, np.random.normal(30 + i*0.6, 3)) for i in range(len(years))],
        'higher_education_pct': [min(99, np.random.normal(10 + i*0.3, 2)) for i in range(len(years))]
    })
    
    # Displacement and land acquisition
    displacement_data = pd.DataFrame({
        'year': years,
        'families_displaced': [np.random.poisson(100 - i*2) if i < 20 else np.random.poisson(max(10, 100 - i*2)) for i in range(len(years))],
        'land_acquired_hectares': [np.random.normal(200 - i*5, 30) if i < 20 else np.random.normal(max(20, 200 - i*5), 10) for i in range(len(years))],
        'compensation_per_hectare_inr': [np.random.normal(50000 + i*5000, 10000) for i in range(len(years))]
    })
    
    # Save the datasets
    employment_data.to_csv("data/raw/employment_data.csv", index=False)
    education_data.to_csv("data/raw/education_data.csv", index=False)
    displacement_data.to_csv("data/raw/displacement_data.csv", index=False)
    
    logger.info("Socioeconomic data collection complete")
    
    return {
        "employment": employment_data,
        "education": education_data,
        "displacement": displacement_data,
        "india_gdp": india_gdp
    }


def fetch_mining_production_data():
    """
    Collect data on uranium mining production and operations in Jadugora.
    
    Returns:
        pandas.DataFrame: Mining production data
    """
    logger.info("Fetching mining production data...")
    
    # Simulated mining production data
    years = range(1990, 2025)
    
    mining_data = pd.DataFrame({
        'year': years,
        'ore_extracted_tons': [np.random.normal(200000 + i*5000, 20000) for i in range(len(years))],
        'uranium_produced_kg': [np.random.normal(40000 + i*1000, 5000) for i in range(len(years))],
        'waste_generated_tons': [np.random.normal(180000 + i*4500, 18000) for i in range(len(years))],
        'tailings_volume_cubic_m': [np.random.normal(150000 + i*3800, 15000) for i in range(len(years))],
        'water_used_million_liters': [np.random.normal(500 + i*20, 50) for i in range(len(years))]
    })
    
    # Save the dataset
    mining_data.to_csv("data/raw/mining_production.csv", index=False)
    
    logger.info("Mining production data collection complete")
    
    return mining_data


def fetch_spatial_data():
    """
    Collect spatial data for GIS analysis of the Jadugora region.
    This includes coordinates of mining sites, villages, and environmental sampling points.
    
    Returns:
        dict: Dictionary containing various spatial datasets
    """
    logger.info("Fetching spatial data...")
    
    # Simulated spatial data
    # Central coordinates for Jadugora
    jadugora_lat, jadugora_lon = 22.6500, 86.3500
    
    # Mining sites (main mine and processing facilities)
    np.random.seed(42)  # For reproducibility
    mining_sites = pd.DataFrame({
        'site_name': [f"Mine_{i}" for i in range(1, 6)] + [f"Processing_{i}" for i in range(1, 4)],
        'latitude': jadugora_lat + np.random.normal(0, 0.05, 8),
        'longitude': jadugora_lon + np.random.normal(0, 0.05, 8),
        'site_type': ['Mine'] * 5 + ['Processing'] * 3,
        'area_hectares': np.random.uniform(10, 100, 8),
        'year_established': np.random.randint(1970, 2000, 8)
    })
    
    # Villages in the region
    villages = pd.DataFrame({
        'village_name': [f"Village_{i}" for i in range(1, 21)],
        'latitude': jadugora_lat + np.random.normal(0, 0.1, 20),
        'longitude': jadugora_lon + np.random.normal(0, 0.1, 20),
        'population': np.random.randint(500, 5000, 20),
        'distance_to_nearest_mine_km': np.random.uniform(0.5, 15, 20)
    })
    
    # Environmental sampling points
    sampling_points = pd.DataFrame({
        'point_id': [f"ENV_{i}" for i in range(1, 51)],
        'latitude': jadugora_lat + np.random.normal(0, 0.12, 50),
        'longitude': jadugora_lon + np.random.normal(0, 0.12, 50),
        'sample_type': np.random.choice(['Water', 'Soil', 'Air'], 50),
        'distance_to_nearest_mine_km': np.random.uniform(0.1, 20, 50)
    })
    
    # Save the datasets
    mining_sites.to_csv("data/raw/mining_sites.csv", index=False)
    villages.to_csv("data/raw/villages.csv", index=False)
    sampling_points.to_csv("data/raw/sampling_points.csv", index=False)
    
    logger.info("Spatial data collection complete")
    
    return {
        "mining_sites": mining_sites,
        "villages": villages,
        "sampling_points": sampling_points
    }


def collect_all_data():
    """
    Main function to collect all datasets for the project.
    
    Returns:
        dict: Dictionary containing all collected datasets
    """
    logger.info("Starting comprehensive data collection process")
    
    # Create a metadata file to track data sources
    metadata = {
        "collection_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_sources": {
            "environmental": "Simulated data based on typical patterns in uranium mining regions",
            "health": "Simulated data based on health studies in mining-affected areas",
            "socioeconomic": "Combination of World Bank data and simulated local statistics",
            "mining_production": "Simulated data based on typical uranium mining operations",
            "spatial": "Simulated GIS data centered around Jadugora coordinates"
        },
        "notes": "This dataset combines real-world references with synthetic data for demonstration purposes. In a production environment, this would be replaced with actual field measurements, government statistics, and research studies."
    }
    
    # Collect all datasets
    all_data = {
        "environmental": fetch_environmental_data(),
        "health": fetch_health_data(),
        "socioeconomic": fetch_socioeconomic_data(),
        "mining_production": fetch_mining_production_data(),
        "spatial": fetch_spatial_data(),
        "metadata": metadata
    }
    
    # Save metadata
    with open("data/raw/metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
    
    logger.info("All data collection complete")
    
    return all_data


if __name__ == "__main__":
    # Execute data collection
    try:
        all_datasets = collect_all_data()
        logger.info(f"Successfully collected {len(all_datasets)} dataset categories")
        print("Data collection complete. Raw data saved to ../data/raw/")
    except Exception as e:
        logger.error(f"Data collection failed: {e}")
        raise
