import os
from pathlib import Path

# Data file paths
PRODUCT_MASTER_FILE = "/home/chinmay/Desktop/vaspp/data/product_master.csv"
TIME_SERIES_FILE = "/home/chinmay/Desktop/vaspp/data/ts_perishability.csv"
IOT_DATA_FILE = "/home/chinmay/Desktop/vaspp/data/iot_data.csv"
SUPPLIER_DATA_FILE = "/home/chinmay/Desktop/vaspp/data/supplier_data.csv"
SALES_DATA_FILE = "/home/chinmay/Desktop/vaspp/data/sales_data.csv"

# Ollama model configuration
OLLAMA_MODEL = "gemma3:27b"  

# Alert thresholds
EXPIRY_ALERT_THRESHOLD = 3  # Days
WASTE_PERCENTAGE_ALERT = 10.0  # Percentage
SHELF_LIFE_UTILIZATION_ALERT = 0.8  # 80%
TEMP_DEVIATION_ALERT = 2.0  # Degrees Celsius
HUMIDITY_DEVIATION_ALERT = 10.0  # Percentage points