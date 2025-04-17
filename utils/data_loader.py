import pandas as pd
from datetime import datetime

def load_product_master_data(file_path):
    """Load product master data from CSV file"""
    df = pd.read_csv(file_path)
    # Create column name mapping
    column_mapping = {
        'initial_shelf_life': 'Initial_Shelf_Life',
        'Storage_temp': 'Storage_Temp',
        'Unit_stroage_temp': 'Unit_Storage_Temp',
        'Critical_humidity': 'Critical_Humidity',
        'Minmum_order_qty': 'Minmum_Order_Qty'
    }
    # Rename columns
    df = df.rename(columns=column_mapping)
    return df

def load_time_series_data(file_path):
    """Load time series perishability data from CSV file"""
    df = pd.read_csv(file_path)
    # Create column name mapping
    column_mapping = {
        'days remaining': 'days_remaining',
        'temp_deviation(Celsius)': 'temp_deviation'
    }
    # Rename columns
    df = df.rename(columns=column_mapping)
    # Convert timestamp string to datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    return df

def load_iot_data(file_path):
    """Load IoT sensor data from CSV file"""
    df = pd.read_csv(file_path)
    # Create column name mapping
    column_mapping = {
        'TIESTAMP': 'TIMESTAMP'
    }
    # Rename columns
    df = df.rename(columns=column_mapping)
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
    return df

def load_supplier_data(file_path):
    """Load supplier and transportation data from CSV file"""
    df = pd.read_csv(file_path)
    # Create column name mapping
    column_mapping = {
        'Supplier/vendor': 'Supplier_vendor'
    }
    # Rename columns
    df = df.rename(columns=column_mapping)
    return df

def load_sales_data(file_path):
    """Load sales data from CSV file"""
    try:
        df = pd.read_csv(file_path)
        
        # Create column name mapping if needed
        column_mapping = {
            'Sales_order': 'order_id',
            'SO_ITEM': 'order_item',
            'createdon': 'date',
            'Delivered date': 'delivery_date'
        }
        
        # Rename columns
        df = df.rename(columns=column_mapping)
        
        # Convert date columns to datetime if they contain valid date strings
        date_columns = ['date', 'delivery_date']
        for col in date_columns:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except:
                    # If conversion fails, keep as is
                    pass
        
        # Add warehouse_id column if not present (assuming Plant is the warehouse)
        if 'warehouse_id' not in df.columns and 'Plant' in df.columns:
            df['warehouse_id'] = df['Plant']
        
        return df
    except Exception as e:
        print(f"Warning: Error loading sales data: {e}")
        return None

def parse_lead_time_to_hours(lead_time_str):
    """Convert lead time string (e.g., '48 Hrs') to hours (int)"""
    if 'Hrs' in lead_time_str:
        return int(lead_time_str.split(' ')[0])
    return 0