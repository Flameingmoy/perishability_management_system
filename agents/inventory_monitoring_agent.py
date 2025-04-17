import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from models.data_models import ProductAlert
import config

class InventoryMonitoringAgent:
    def __init__(self, product_data, time_series_data):
        """
        Initialize the inventory monitoring agent.
        
        Args:
            product_data: DataFrame containing product master data
            time_series_data: DataFrame containing time series perishability data
        """
        self.product_data = product_data
        self.time_series_data = time_series_data
        self.llm = OllamaLLM(model=config.OLLAMA_MODEL)
        
        # Initialize prompt template for LLM analysis
        self.analysis_template = PromptTemplate(
            input_variables=["product_info", "inventory_status", "metrics", "alerts"],
            template="""
            Based on the following information about a perishable product:
            
            Product Info:
            {product_info}
            
            Current Inventory Status:
            {inventory_status}
            
            Key Metrics:
            {metrics}
            
            Current Alerts:
            {alerts}
            
            Please provide:
            1. An analysis of the current inventory situation
            2. Recommendations for immediate actions to reduce waste
            3. Suggestions for optimal inventory management
            
            Your response should be concise and actionable.
            """
        )
        
        self.analysis_chain = LLMChain(llm=self.llm, prompt=self.analysis_template)
    
    def calculate_waste_percentage(self, sku_id, warehouse_id=None):
        """
        Calculate waste percentage using the formula:
        Waste Percentage = (Waste_Qty / (Current_Stock + Waste_Qty)) × 100
        
        Args:
            sku_id: The SKU ID to calculate waste percentage for
            warehouse_id: Optional warehouse ID to filter by
        
        Returns:
            Waste percentage as a float
        """
        df = self.time_series_data
        if warehouse_id:
            df = df[df['warehous_id'] == warehouse_id]
        
        df = df[df['SKU_id'] == sku_id]
        
        if df.empty:
            return 0.0
            
        total_waste = df['waste_qty'].sum()
        total_current_stock = df['current_stock'].sum()
        
        if total_waste + total_current_stock == 0:
            return 0.0
            
        waste_percentage = (total_waste / (total_current_stock + total_waste)) * 100
        return waste_percentage
    
    def calculate_shelf_life_utilization(self, sku_id, warehouse_id=None):
        """
        Calculate shelf-life utilization using the formula:
        Shelf-Life Utilization = (Initial_Shelf_Life - Days_Remaining) / Initial_Shelf_Life
        
        Args:
            sku_id: The SKU ID to calculate shelf-life utilization for
            warehouse_id: Optional warehouse ID to filter by
        
        Returns:
            Shelf-life utilization as a float (between 0 and 1)
        """
        ts_df = self.time_series_data
        if warehouse_id:
            ts_df = ts_df[ts_df['warehous_id'] == warehouse_id]
        
        ts_df = ts_df[ts_df['SKU_id'] == sku_id]
        
        if ts_df.empty:
            return 0.0
        
        product_df = self.product_data[self.product_data['SKU_ID'] == sku_id]
        
        if product_df.empty:
            return 0.0
        
        initial_shelf_life = product_df.iloc[0]['Initial_Shelf_Life']
        days_remaining = ts_df.iloc[0]['days_remaining']
        
        utilization = (initial_shelf_life - days_remaining) / initial_shelf_life
        return max(0.0, min(1.0, utilization))  # Ensure result is between 0 and 1
    
    def identify_critical_inventory(self):
        """
        Identify inventory that is nearing expiry or has high waste percentages.
        
        Returns:
            DataFrame containing critical inventory items
        """
        results = []
        
        # Get unique SKU and warehouse combinations
        unique_combinations = self.time_series_data[['SKU_id', 'warehous_id']].drop_duplicates()
        
        for _, row in unique_combinations.iterrows():
            sku_id = row['SKU_id']
            warehouse_id = row['warehous_id']
            
            # Get time series data for this SKU and warehouse
            ts_df = self.time_series_data[
                (self.time_series_data['SKU_id'] == sku_id) & 
                (self.time_series_data['warehous_id'] == warehouse_id)
            ]
            
            if ts_df.empty:
                continue
                
            # Get latest record
            latest = ts_df.iloc[0]
            
            # Get product data
            product_data = self.product_data[self.product_data['SKU_ID'] == sku_id]
            if product_data.empty:
                continue
                
            product = product_data.iloc[0]
            
            # Calculate metrics
            waste_percentage = self.calculate_waste_percentage(sku_id, warehouse_id)
            shelf_life_utilization = self.calculate_shelf_life_utilization(sku_id, warehouse_id)
            
            # Check if this is critical inventory
            is_critical = False
            reasons = []
            
            if latest['days_remaining'] <= config.EXPIRY_ALERT_THRESHOLD:
                is_critical = True
                reasons.append(f"Only {latest['days_remaining']} days remaining until expiry")
                
            if waste_percentage >= config.WASTE_PERCENTAGE_ALERT:
                is_critical = True
                reasons.append(f"High waste percentage: {waste_percentage:.2f}%")
                
            if shelf_life_utilization >= config.SHELF_LIFE_UTILIZATION_ALERT:
                is_critical = True
                reasons.append(f"High shelf life utilization: {shelf_life_utilization:.2f}")
                
            if is_critical:
                results.append({
                    'SKU_ID': sku_id,
                    'warehous_id': warehouse_id,
                    'current_stock': latest['current_stock'],
                    'unit': latest['unit_curr_stock'],
                    'days_remaining': latest['days_remaining'],
                    'waste_percentage': waste_percentage,
                    'shelf_life_utilization': shelf_life_utilization,
                    'reasons': ', '.join(reasons)
                })
                
        return pd.DataFrame(results)
    
    def generate_alerts(self):
        """
        Generate alerts for critical inventory items.
        
        Returns:
            List of ProductAlert objects
        """
        critical_inventory = self.identify_critical_inventory()
        alerts = []
        
        for _, item in critical_inventory.iterrows():
            # Determine alert severity based on days remaining
            severity = "LOW"
            if item['days_remaining'] <= 1:
                severity = "HIGH"
            elif item['days_remaining'] <= 3:
                severity = "MEDIUM"
                
            # Determine recommended action
            recommended_action = ""
            if item['days_remaining'] <= 1:
                recommended_action = "Immediate markdown or transfer to high-demand location"
            elif item['days_remaining'] <= 3:
                recommended_action = "Apply discount to accelerate sales"
            else:
                recommended_action = "Monitor closely"
                
            alert = ProductAlert(
                SKU_ID=item['SKU_ID'],
                warehouse_id=item['warehous_id'],
                alert_type="EXPIRY_RISK",
                severity=severity,
                message=item['reasons'],
                timestamp=datetime.now(),
                recommended_action=recommended_action
            )
            alerts.append(alert)
            
        return alerts
    
    def get_inventory_analysis(self, sku_id, warehouse_id):
        """
        Get LLM analysis for a specific SKU and warehouse.
        
        Args:
            sku_id: SKU ID to analyze
            warehouse_id: Warehouse ID to analyze
        
        Returns:
            Analysis text from the LLM
        """
        # Get product info
        product_info = self.product_data[self.product_data['SKU_ID'] == sku_id]
        if product_info.empty:
            return "Product not found."
        
        product_info = product_info.iloc[0].to_dict()
        
        # Get inventory status
        inventory_status = self.time_series_data[
            (self.time_series_data['SKU_id'] == sku_id) & 
            (self.time_series_data['warehous_id'] == warehouse_id)
        ]
        
        if inventory_status.empty:
            return "Inventory data not found."
            
        inventory_status = inventory_status.iloc[0].to_dict()
        
        # Calculate metrics
        waste_percentage = self.calculate_waste_percentage(sku_id, warehouse_id)
        shelf_life_utilization = self.calculate_shelf_life_utilization(sku_id, warehouse_id)
        
        metrics = {
            "waste_percentage": f"{waste_percentage:.2f}%",
            "shelf_life_utilization": f"{shelf_life_utilization:.2f}",
            "days_to_expiry": inventory_status['days_remaining']
        }
        
        # Generate alerts
        critical_inventory = self.identify_critical_inventory()
        relevant_alerts = critical_inventory[
            (critical_inventory['SKU_ID'] == sku_id) & 
            (critical_inventory['warehous_id'] == warehouse_id)
        ]
        
        alerts = []
        if not relevant_alerts.empty:
            alerts = relevant_alerts.iloc[0]['reasons']
        
        # Get analysis from LLM
        analysis = self.analysis_chain.run(
            product_info=str(product_info),
            inventory_status=str(inventory_status),
            metrics=str(metrics),
            alerts=str(alerts)
        )
        
        return analysis