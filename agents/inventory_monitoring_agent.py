import pandas as pd
import numpy as np
from datetime import datetime
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import config
from models.data_models import ProductAlert

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
        
        # LLM for analysis
        try:
            self.llm = OllamaLLM(model=config.OLLAMA_MODEL)  # Using the new model specified in config
            
            # Analysis prompt
            self.analysis_prompt = PromptTemplate(
                input_variables=["sku_id", "warehouse_id", "inventory_data", "product_data"],
                template="""
                Analyze the inventory situation for product {sku_id} in warehouse {warehouse_id}.
                
                Product data:
                {product_data}
                
                Inventory data:
                {inventory_data}
                
                Provide a detailed analysis including:
                1. Current stock level and shelf-life remaining
                2. Waste percentage and implications
                3. Shelf-life utilization rate
                4. Recommendations for inventory management
                5. Potential risks and mitigation strategies
                
                Analysis:
                """
            )
            
            self.analysis_chain = LLMChain(llm=self.llm, prompt=self.analysis_prompt)
        except Exception as e:
            print(f"Warning: Could not initialize LLM for analysis: {e}")
            self.llm = None
    
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
        try:
            # Filter data by SKU and warehouse
            filtered_data = self.time_series_data[self.time_series_data['SKU_id'] == sku_id]
            if warehouse_id:
                filtered_data = filtered_data[filtered_data['warehous_id'] == warehouse_id]
            
            if filtered_data.empty:
                return 0.0
            
            # Sum waste quantity and current stock
            total_waste = filtered_data['waste_qty'].sum()
            total_stock = filtered_data['current_stock'].sum()
            
            # Calculate waste percentage
            if total_stock + total_waste > 0:
                waste_percentage = (total_waste / (total_stock + total_waste)) * 100
                return round(waste_percentage, 2)
            else:
                return 0.0
        except Exception as e:
            print(f"Error calculating waste percentage: {e}")
            return 0.0
    
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
        try:
            # Get initial shelf life from product data
            product_info = self.product_data[self.product_data['SKU_ID'] == sku_id]
            if product_info.empty:
                return 0.0
            
            initial_shelf_life = product_info['Initial_Shelf_Life'].values[0]
            
            # Filter time series data
            filtered_data = self.time_series_data[self.time_series_data['SKU_id'] == sku_id]
            if warehouse_id:
                filtered_data = filtered_data[filtered_data['warehous_id'] == warehouse_id]
            
            if filtered_data.empty:
                return 0.0
            
            # Calculate average days remaining
            avg_days_remaining = filtered_data['days_remaining'].mean()
            
            # Calculate utilization
            if initial_shelf_life > 0:
                utilization = (initial_shelf_life - avg_days_remaining) / initial_shelf_life
                return min(1.0, max(0.0, utilization))  # Ensure value is between 0 and 1
            else:
                return 0.0
        except Exception as e:
            print(f"Error calculating shelf-life utilization: {e}")
            return 0.0
    
    def identify_critical_inventory(self):
        """
        Identify inventory that is nearing expiry or has high waste percentages.
        
        Returns:
            DataFrame containing critical inventory items
        """
        try:
            # Initialize empty list to store critical items
            critical_items = []
            
            # Get unique combinations of SKU_id and warehouse_id
            unique_combinations = self.time_series_data[['SKU_id', 'warehous_id']].drop_duplicates()
            
            for _, row in unique_combinations.iterrows():
                sku_id = row['SKU_id']
                warehouse_id = row['warehous_id']
                
                # Calculate metrics
                waste_percentage = self.calculate_waste_percentage(sku_id, warehouse_id)
                shelf_life_utilization = self.calculate_shelf_life_utilization(sku_id, warehouse_id)
                
                # Get days remaining
                filtered_data = self.time_series_data[
                    (self.time_series_data['SKU_id'] == sku_id) & 
                    (self.time_series_data['warehous_id'] == warehouse_id)
                ]
                days_remaining = filtered_data['days_remaining'].mean() if not filtered_data.empty else 0
                current_stock = filtered_data['current_stock'].sum() if not filtered_data.empty else 0
                
                # Determine if item is critical
                is_critical = (
                    days_remaining <= config.EXPIRY_ALERT_THRESHOLD or
                    waste_percentage >= config.WASTE_PERCENTAGE_ALERT or
                    shelf_life_utilization >= config.SHELF_LIFE_UTILIZATION_ALERT
                )
                
                if is_critical:
                    critical_items.append({
                        'SKU_ID': sku_id,
                        'warehouse_id': warehouse_id,
                        'days_remaining': days_remaining,
                        'current_stock': current_stock,
                        'waste_percentage': waste_percentage,
                        'shelf_life_utilization': shelf_life_utilization
                    })
            
            return pd.DataFrame(critical_items)
        except Exception as e:
            print(f"Error identifying critical inventory: {e}")
            return pd.DataFrame()
    
    def generate_alerts(self):
        """
        Generate alerts for critical inventory items.
        
        Returns:
            List of ProductAlert objects
        """
        try:
            # Get critical inventory
            critical_inventory = self.identify_critical_inventory()
            
            if critical_inventory.empty:
                return []
            
            # Initialize list to store alerts
            alerts = []
            
            for _, item in critical_inventory.iterrows():
                # Determine alert type and severity
                if item['days_remaining'] <= config.EXPIRY_ALERT_THRESHOLD:
                    alert_type = "EXPIRY"
                    severity = "HIGH" if item['days_remaining'] <= 1 else "MEDIUM"
                    message = f"Product is nearing expiry with only {item['days_remaining']:.1f} days remaining."
                    action = "Consider immediate price reduction or redistribution to high-demand locations."
                elif item['waste_percentage'] >= config.WASTE_PERCENTAGE_ALERT:
                    alert_type = "WASTE"
                    severity = "HIGH" if item['waste_percentage'] >= 20 else "MEDIUM"
                    message = f"High waste percentage detected: {item['waste_percentage']:.1f}%."
                    action = "Review storage conditions and handling procedures. Consider adjusting order quantities."
                elif item['shelf_life_utilization'] >= config.SHELF_LIFE_UTILIZATION_ALERT:
                    alert_type = "SHELF_LIFE"
                    severity = "MEDIUM"
                    message = f"Product has used {item['shelf_life_utilization'] * 100:.1f}% of its shelf life."
                    action = "Monitor closely and consider promotional pricing to increase turnover."
                else:
                    alert_type = "GENERAL"
                    severity = "LOW"
                    message = "Potential inventory optimization opportunity."
                    action = "Review inventory levels against forecasted demand."
                
                # Create alert
                alert = ProductAlert(
                    SKU_ID=item['SKU_ID'],
                    warehouse_id=item['warehouse_id'],
                    alert_type=alert_type,
                    severity=severity,
                    message=message,
                    timestamp=datetime.now(),
                    recommended_action=action
                )
                
                alerts.append(alert)
            
            return alerts
        except Exception as e:
            print(f"Error generating alerts: {e}")
            return []
    
    def get_inventory_analysis(self, sku_id, warehouse_id):
        """
        Get LLM analysis for a specific SKU and warehouse.
        
        Args:
            sku_id: SKU ID to analyze
            warehouse_id: Warehouse ID to analyze
        
        Returns:
            Analysis text from the LLM
        """
        try:
            if not self.llm:
                return "LLM analysis is not available."
            
            # Get product data
            product_info = self.product_data[self.product_data['SKU_ID'] == sku_id]
            if product_info.empty:
                return f"No product information found for SKU {sku_id}."
            
            # Get inventory data
            inventory_data = self.time_series_data[
                (self.time_series_data['SKU_id'] == sku_id) & 
                (self.time_series_data['warehous_id'] == warehouse_id)
            ]
            if inventory_data.empty:
                return f"No inventory data found for SKU {sku_id} in warehouse {warehouse_id}."
            
            # Calculate additional metrics
            waste_percentage = self.calculate_waste_percentage(sku_id, warehouse_id)
            shelf_life_utilization = self.calculate_shelf_life_utilization(sku_id, warehouse_id)
            
            # Add calculated metrics to inventory data
            inventory_data_with_metrics = inventory_data.copy()
            inventory_data_with_metrics['waste_percentage'] = waste_percentage
            inventory_data_with_metrics['shelf_life_utilization'] = shelf_life_utilization * 100  # Convert to percentage
            
            # Run analysis
            result = self.analysis_chain.run(
                sku_id=sku_id,
                warehouse_id=warehouse_id,
                inventory_data=inventory_data_with_metrics.to_string(),
                product_data=product_info.to_string()
            )
            
            return result
        except Exception as e:
            return f"Error generating inventory analysis: {e}"