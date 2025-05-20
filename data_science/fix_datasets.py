"""
Fix for dataset merging issues in the model development pipeline.
This script cleans and standardizes column names in the merged datasets.
"""

import pandas as pd
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/data_fix.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def fix_merged_datasets():
    """
    Fix column naming issues in merged datasets to ensure consistency
    for machine learning model development.
    """
    logger.info("Starting to fix merged datasets...")
    
    # Ensure directories exist
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    try:
        # Load the merged datasets
        mining_health_df = pd.read_csv("data/processed/mining_health_data.csv")
        env_health_df = pd.read_csv("data/processed/environmental_health_data.csv")
        
        logger.info(f"Original mining_health_df columns: {mining_health_df.columns.tolist()}")
        logger.info(f"Original env_health_df columns: {env_health_df.columns.tolist()}")
        
        # Fix column naming in mining_health_df
        # Identify duplicated columns (those with _x and _y suffixes)
        duplicated_cols = [col for col in mining_health_df.columns if col.endswith('_x')]
        
        # Create a mapping for renaming
        rename_mapping = {}
        for col in duplicated_cols:
            base_col = col[:-2]  # Remove the _x suffix
            rename_mapping[col] = base_col
            
            # Drop the corresponding _y column if it exists
            y_col = f"{base_col}_y"
            if y_col in mining_health_df.columns:
                mining_health_df = mining_health_df.drop(columns=[y_col])
        
        # Rename the columns
        mining_health_df = mining_health_df.rename(columns=rename_mapping)
        
        logger.info(f"Fixed mining_health_df columns: {mining_health_df.columns.tolist()}")
        
        # Fix column naming in env_health_df if needed
        duplicated_cols_env = [col for col in env_health_df.columns if col.endswith('_x')]
        
        if duplicated_cols_env:
            # Create a mapping for renaming
            rename_mapping_env = {}
            for col in duplicated_cols_env:
                base_col = col[:-2]  # Remove the _x suffix
                rename_mapping_env[col] = base_col
                
                # Drop the corresponding _y column if it exists
                y_col = f"{base_col}_y"
                if y_col in env_health_df.columns:
                    env_health_df = env_health_df.drop(columns=[y_col])
            
            # Rename the columns
            env_health_df = env_health_df.rename(columns=rename_mapping_env)
            
            logger.info(f"Fixed env_health_df columns: {env_health_df.columns.tolist()}")
        
        # Save the fixed datasets
        mining_health_df.to_csv("data/processed/mining_health_data_fixed.csv", index=False)
        env_health_df.to_csv("data/processed/environmental_health_data_fixed.csv", index=False)
        
        logger.info("Successfully fixed and saved merged datasets")
        
        return True
    except Exception as e:
        logger.error(f"Error fixing merged datasets: {e}")
        return False

if __name__ == "__main__":
    success = fix_merged_datasets()
    if success:
        print("Dataset fixing completed successfully. Fixed datasets saved to data/processed/ directory.")
    else:
        print("Failed to fix datasets. Check logs for details.")
