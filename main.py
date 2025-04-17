import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import argparse
from datetime import datetime
import json
import os

from agents.inventory_monitoring_agent import InventoryMonitoringAgent
from agents.environmental_monitoring_agent import EnvironmentalMonitoringAgent
from agents.demand_prediction_agent import DemandPredictionAgent  # Add import
from utils.data_loader import (
    load_product_master_data,
    load_time_series_data,
    load_iot_data,
    load_supplier_data,
    load_sales_data  # Add this function
)
import config

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Perishable Goods Monitoring System')
    parser.add_argument('--analyze', choices=['inventory', 'environment', 'demand', 'all'], 
                        default='all', help='Type of analysis to perform')  # Add 'demand' option
    parser.add_argument('--sku', type=str, help='SKU ID to analyze')
    parser.add_argument('--warehouse', type=str, help='Warehouse ID to analyze')
    parser.add_argument('--export', action='store_true', help='Export alerts to CSV')
    parser.add_argument('--export-format', choices=['csv', 'json'], default='csv',
                        help='Export format (csv or json)')
    parser.add_argument('--forecast-days', type=int, default=7,
                        help='Number of days to forecast for demand prediction')  # Add this argument
    return parser.parse_args()

def load_data():
    """Load all required data"""
    print("Loading data...")
    try:
        product_data = load_product_master_data(config.PRODUCT_MASTER_FILE)
        time_series_data = load_time_series_data(config.TIME_SERIES_FILE)
        iot_data = load_iot_data(config.IOT_DATA_FILE)
        supplier_data = load_supplier_data(config.SUPPLIER_DATA_FILE)
        sales_data = load_sales_data(config.SALES_DATA_FILE) if hasattr(config, 'SALES_DATA_FILE') else None  # Add this line
        
        print(f"Loaded {len(product_data)} products")
        print(f"Loaded {len(time_series_data)} time series records")
        print(f"Loaded {len(iot_data)} IoT records")
        print(f"Loaded {len(supplier_data)} supplier records")
        if sales_data is not None:
            print(f"Loaded {len(sales_data)} sales records")
        
        return product_data, time_series_data, iot_data, supplier_data, sales_data  # Add sales_data
    except Exception as e:
        print(f"Error loading data: {e}")
        exit(1)

def export_alerts(alerts, alert_type, export_format='csv'):
    """Export alerts to CSV or JSON file"""
    if not alerts:
        print(f"No {alert_type} alerts to export.")
        return
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{alert_type}_alerts_{timestamp}"
    
    alerts_data = [alert.dict() for alert in alerts]
    
    if export_format == 'csv':
        filename += '.csv'
        pd.DataFrame(alerts_data).to_csv(filename, index=False)
    else:  # json
        filename += '.json'
        with open(filename, 'w') as f:
            json.dump(alerts_data, f, default=str, indent=2)
    
    print(f"Exported {len(alerts)} {alert_type} alerts to {filename}")

def analyze_inventory(inventory_agent, args):
    """Run inventory analysis"""
    # Existing implementation, no changes needed
    print("\nRunning inventory analysis...")
    
    # Generate inventory alerts
    inventory_alerts = inventory_agent.generate_alerts()
    print(f"Generated {len(inventory_alerts)} inventory alerts")
    
    # Display alerts
    if inventory_alerts:
        alerts_df = pd.DataFrame([alert.dict() for alert in inventory_alerts])
        print("\nInventory Alerts:")
        print(alerts_df[['SKU_ID', 'warehouse_id', 'severity', 'message']].head())
        
        # Export alerts if requested
        if args.export:
            export_alerts(inventory_alerts, "inventory", args.export_format)
    
    # Generate specific analysis if SKU and warehouse are provided
    if args.sku and args.warehouse:
        analysis = inventory_agent.get_inventory_analysis(args.sku, args.warehouse)
        print(f"\nDetailed Inventory Analysis for SKU {args.sku} at Warehouse {args.warehouse}:")
        print(analysis)

def analyze_environment(environment_agent, args):
    """Run environmental analysis"""
    # Existing implementation, no changes needed
    print("\nRunning environmental analysis...")
    
    # Generate environmental alerts
    environmental_alerts = environment_agent.generate_alerts()
    print(f"Generated {len(environmental_alerts)} environmental alerts")
    
    # Display alerts
    if environmental_alerts:
        alerts_df = pd.DataFrame([alert.dict() for alert in environmental_alerts])
        print("\nEnvironmental Alerts:")
        print(alerts_df[['SKU_ID', 'warehouse_id', 'severity', 'message']].head())
        
        # Export alerts if requested
        if args.export:
            export_alerts(environmental_alerts, "environmental", args.export_format)
    
    # Generate specific analysis if SKU is provided
    if args.sku:
        analysis = environment_agent.get_environmental_analysis(args.sku, args.warehouse)
        warehouse_text = f" at Warehouse {args.warehouse}" if args.warehouse else ""
        print(f"\nDetailed Environmental Analysis for SKU {args.sku}{warehouse_text}:")
        print(analysis)

def analyze_demand(demand_agent, args):
    """Run demand prediction analysis"""
    print("\nRunning demand prediction analysis...")
    
    # Generate demand alerts
    demand_alerts = demand_agent.generate_alerts()
    print(f"Generated {len(demand_alerts)} demand alerts")
    
    # Display alerts
    if demand_alerts:
        alerts_df = pd.DataFrame([alert.dict() for alert in demand_alerts])
        print("\nDemand Prediction Alerts:")
        print(alerts_df[['SKU_ID', 'warehouse_id', 'severity', 'message']].head())
        
        # Export alerts if requested
        if args.export:
            export_alerts(demand_alerts, "demand", args.export_format)
    
    # Generate specific analysis if SKU and warehouse are provided
    if args.sku and args.warehouse:
        analysis = demand_agent.get_demand_analysis(args.sku, args.warehouse)
        print(f"\nDetailed Demand Analysis for SKU {args.sku} at Warehouse {args.warehouse}:")
        print(analysis)
        
        # Generate and display forecast if requested
        forecast = demand_agent.forecast_demand(args.sku, args.warehouse, args.forecast_days)
        if forecast:
            print(f"\nDemand Forecast for next {args.forecast_days} days:")
            for f in forecast:
                print(f"  {f.forecast_date.strftime('%Y-%m-%d')}: {f.quantity:.2f} {f.unit} (Confidence: {f.confidence:.2f})")

def main():
    """Main function"""
    print("Perishable Inventory Monitoring System")
    print("======================================")
    
    # Parse arguments
    args = parse_arguments()
    
    # Load data
    product_data, time_series_data, iot_data, supplier_data, sales_data = load_data()
    
    # Initialize agents
    inventory_agent = InventoryMonitoringAgent(product_data, time_series_data)
    environment_agent = EnvironmentalMonitoringAgent(product_data, time_series_data, iot_data)
    demand_agent = DemandPredictionAgent(product_data, time_series_data, sales_data)  
    
    # Run analysis based on arguments
    if args.analyze in ['inventory', 'all']:
        analyze_inventory(inventory_agent, args)
        
    if args.analyze in ['environment', 'all']:
        analyze_environment(environment_agent, args)
    
    if args.analyze in ['demand', 'all']:  # Add this block
        analyze_demand(demand_agent, args)
    
    print("\nAnalysis complete.")

if __name__ == "__main__":
    main()