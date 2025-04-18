import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import config
from models.data_models import ProductAlert, DemandForecast

class DemandPredictionAgent:
    """
    Agent responsible for forecasting demand for perishable products and identifying
    inventory at risk of expiration before being sold.
    """
    
    def __init__(self, product_data, time_series_data, sales_data=None, promotion_data=None):
        """
        Initialize the demand prediction agent.
        
        Args:
            product_data: DataFrame containing product master data
            time_series_data: DataFrame containing time series perishability data
            sales_data: Optional DataFrame containing historical sales data
            promotion_data: Optional DataFrame containing promotion calendar
        """
        self.product_data = product_data
        self.time_series_data = time_series_data
        self.sales_data = sales_data
        self.promotion_data = promotion_data
        
        # LLM for analysis
        try:
            self.llm = OllamaLLM(model=config.OLLAMA_MODEL)  # Using the new model specified in config
            
            # Analysis prompt
            self.analysis_prompt = PromptTemplate(
                input_variables=["sku_id", "warehouse_id", "product_data", "historical_data", "forecast_data"],
                template="""
                Analyze the demand forecast for product {sku_id} in warehouse {warehouse_id}.
                
                Product data:
                {product_data}
                
                Historical data:
                {historical_data}
                
                Forecast data:
                {forecast_data}
                
                Provide a detailed analysis including:
                1. Historical demand patterns
                2. Forecast for upcoming period
                3. Factors affecting demand
                4. Risk of expired inventory
                5. Recommendations for inventory planning
                
                Analysis:
                """
            )
            
            self.analysis_chain = LLMChain(llm=self.llm, prompt=self.analysis_prompt)
        except Exception as e:
            print(f"Warning: Could not initialize LLM for analysis: {e}")
            self.llm = None
    
    def get_historical_sales(self, sku_id, warehouse_id=None, days=30):
        """
        Get historical sales data for a specific SKU and optionally a specific warehouse.
        If sales_data is not available, use time_series_data to estimate sales.
        
        Args:
            sku_id: The SKU ID to get historical sales for
            warehouse_id: Optional warehouse ID to filter by
            days: Number of days of historical data to retrieve
            
        Returns:
            DataFrame containing historical sales data
        """
        try:
            if self.sales_data is not None and not self.sales_data.empty:
                # Use actual sales data if available
                sales = self.sales_data.copy()
                
                # Filter by SKU and warehouse
                filtered_sales = sales[sales['SKU_ID'] == sku_id]
                if warehouse_id:
                    filtered_sales = filtered_sales[filtered_sales['warehouse_id'] == warehouse_id]
                
                # Filter by date (last N days)
                if 'date' in filtered_sales.columns:
                    cutoff_date = datetime.now() - timedelta(days=days)
                    filtered_sales = filtered_sales[filtered_sales['date'] >= cutoff_date]
                
                return filtered_sales
            else:
                # Use time_series_data to estimate sales
                ts_data = self.time_series_data.copy()
                
                # Filter by SKU and warehouse
                filtered_data = ts_data[ts_data['SKU_id'] == sku_id]
                if warehouse_id:
                    filtered_data = filtered_data[filtered_data['warehous_id'] == warehouse_id]
                
                if filtered_data.empty:
                    return pd.DataFrame()
                
                # Estimate sales from demand forecast
                if 'Demand_forecast' in filtered_data.columns:
                    # Create dates (assume data is from the past 30 days)
                    start_date = datetime.now() - timedelta(days=days)
                    dates = [(start_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days)]
                    
                    # Create estimated sales (assuming 90% of demand is fulfilled)
                    demand_value = filtered_data['Demand_forecast'].mean()
                    sales_values = [demand_value * 0.9 * (1 + 0.1 * np.random.randn()) for _ in range(days)]
                    
                    # Create a DataFrame
                    estimated_sales = pd.DataFrame({
                        'date': dates,
                        'SKU_ID': sku_id,
                        'warehouse_id': warehouse_id if warehouse_id else "Unknown",
                        'quantity': sales_values,
                        'estimated': True
                    })
                    
                    return estimated_sales
                else:
                    return pd.DataFrame()
        except Exception as e:
            print(f"Error getting historical sales: {e}")
            return pd.DataFrame()
    
    def detect_seasonality(self, sku_id, warehouse_id=None):
        """
        Detect seasonal patterns in sales data.
        
        Args:
            sku_id: The SKU ID to analyze
            warehouse_id: Optional warehouse ID to filter by
            
        Returns:
            Dictionary containing seasonality information
        """
        try:
            # Get historical sales
            historical_sales = self.get_historical_sales(sku_id, warehouse_id, days=365)
            
            if historical_sales.empty:
                return {"has_seasonality": False, "pattern": "Unknown"}
            
            # Check if we have actual or estimated data
            is_estimated = 'estimated' in historical_sales.columns and historical_sales['estimated'].any()
            
            # If we have actual sales data with dates, attempt to detect seasonality
            if not is_estimated and 'date' in historical_sales.columns:
                # Convert to datetime if not already
                if historical_sales['date'].dtype != 'datetime64[ns]':
                    historical_sales['date'] = pd.to_datetime(historical_sales['date'])
                
                # Extract month and day of week
                historical_sales['month'] = historical_sales['date'].dt.month
                historical_sales['day_of_week'] = historical_sales['date'].dt.dayofweek
                
                # Group by month and calculate average sales
                monthly_sales = historical_sales.groupby('month')['quantity'].mean().reset_index()
                
                # Check for monthly patterns (simple approach)
                monthly_std = monthly_sales['quantity'].std()
                monthly_mean = monthly_sales['quantity'].mean()
                
                # If standard deviation is significant compared to mean, assume seasonality
                if monthly_std > 0.2 * monthly_mean:
                    # Find peak months (months with sales above average)
                    peak_months = monthly_sales[monthly_sales['quantity'] > monthly_mean]['month'].tolist()
                    
                    # Convert month numbers to names
                    month_names = {
                        1: 'January', 2: 'February', 3: 'March', 4: 'April', 
                        5: 'May', 6: 'June', 7: 'July', 8: 'August',
                        9: 'September', 10: 'October', 11: 'November', 12: 'December'
                    }
                    peak_month_names = [month_names[m] for m in peak_months]
                    
                    return {
                        "has_seasonality": True,
                        "pattern": "Monthly",
                        "peak_periods": peak_month_names,
                        "confidence": min(1.0, monthly_std / monthly_mean + 0.2)
                    }
                
                # Check for weekly patterns
                weekly_sales = historical_sales.groupby('day_of_week')['quantity'].mean().reset_index()
                weekly_std = weekly_sales['quantity'].std()
                weekly_mean = weekly_sales['quantity'].mean()
                
                if weekly_std > 0.15 * weekly_mean:
                    # Find peak days (days with sales above average)
                    peak_days = weekly_sales[weekly_sales['quantity'] > weekly_mean]['day_of_week'].tolist()
                    
                    # Convert day numbers to names
                    day_names = {
                        0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 
                        3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'
                    }
                    peak_day_names = [day_names[d] for d in peak_days]
                    
                    return {
                        "has_seasonality": True,
                        "pattern": "Weekly",
                        "peak_periods": peak_day_names,
                        "confidence": min(1.0, weekly_std / weekly_mean + 0.2)
                    }
                
                return {"has_seasonality": False, "pattern": "None detected", "confidence": 0.5}
            else:
                # For estimated data or no date information
                return {"has_seasonality": False, "pattern": "Unknown", "confidence": 0.0}
        except Exception as e:
            print(f"Error detecting seasonality: {e}")
            return {"has_seasonality": False, "pattern": "Error", "confidence": 0.0}
    
    def forecast_demand(self, sku_id, warehouse_id=None, days_ahead=7):
        """
        Forecast demand for a specific SKU and warehouse for the specified number of days.
        
        Args:
            sku_id: The SKU ID to forecast demand for
            warehouse_id: Optional warehouse ID to filter by
            days_ahead: Number of days to forecast
            
        Returns:
            List of DemandForecast objects
        """
        try:
            # Get historical sales
            historical_sales = self.get_historical_sales(sku_id, warehouse_id, days=30)
            
            # Get product data
            product_info = self.product_data[self.product_data['SKU_ID'] == sku_id].iloc[0] if not self.product_data[self.product_data['SKU_ID'] == sku_id].empty else None
            
            # Get current stock and forecast from time series data
            ts_data = None
            if warehouse_id:
                ts_filter = (self.time_series_data['SKU_id'] == sku_id) & (self.time_series_data['warehous_id'] == warehouse_id)
            else:
                ts_filter = self.time_series_data['SKU_id'] == sku_id
                
            if not self.time_series_data[ts_filter].empty:
                ts_data = self.time_series_data[ts_filter].iloc[0]
            
            # Determine base demand
            if not historical_sales.empty and 'quantity' in historical_sales.columns:
                base_demand = historical_sales['quantity'].mean()
                unit = historical_sales['unit'].iloc[0] if 'unit' in historical_sales.columns else 'units'
            elif ts_data is not None and 'Demand_forecast' in ts_data:
                base_demand = ts_data['Demand_forecast']
                unit = ts_data['unit_Demand_forecast'] if 'unit_Demand_forecast' in ts_data else 'units'
            else:
                # Fallback to default
                base_demand = 100
                unit = 'units'
            
            # Check for seasonality
            seasonality = self.detect_seasonality(sku_id, warehouse_id)
            
            # Generate forecasts for each day
            forecasts = []
            
            current_date = datetime.now()
            
            for day in range(days_ahead):
                forecast_date = current_date + timedelta(days=day+1)
                
                # Apply seasonality factor if detected
                seasonality_factor = 1.0
                if seasonality["has_seasonality"]:
                    if seasonality["pattern"] == "Weekly":
                        day_of_week = forecast_date.weekday()
                        day_name = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][day_of_week]
                        if day_name in seasonality.get("peak_periods", []):
                            seasonality_factor = 1.2
                        else:
                            seasonality_factor = 0.9
                    elif seasonality["pattern"] == "Monthly":
                        month_name = ['January', 'February', 'March', 'April', 'May', 'June', 
                                     'July', 'August', 'September', 'October', 'November', 'December'][forecast_date.month - 1]
                        if month_name in seasonality.get("peak_periods", []):
                            seasonality_factor = 1.3
                        else:
                            seasonality_factor = 0.8
                
                # Apply day-specific factor (weekend boost)
                day_factor = 1.2 if forecast_date.weekday() >= 5 else 1.0
                
                # Apply trend factor (slight increase over time)
                trend_factor = 1.0 + (day * 0.01)
                
                # Apply random noise for realistic variation
                noise_factor = 1.0 + (0.05 * np.random.randn())
                
                # Calculate forecast quantity
                quantity = base_demand * seasonality_factor * day_factor * trend_factor * noise_factor
                
                # Determine confidence level
                base_confidence = 0.9  # Start with high confidence
                time_decay = day * 0.02  # Confidence decreases with forecast horizon
                confidence = max(0.5, base_confidence - time_decay)
                
                # Create forecast object
                forecast = DemandForecast(
                    SKU_ID=sku_id,
                    warehouse_id=warehouse_id if warehouse_id else "ALL",
                    forecast_date=forecast_date,
                    quantity=round(quantity, 2),
                    unit=unit,
                    confidence=confidence
                )
                
                forecasts.append(forecast)
            
            return forecasts
        except Exception as e:
            print(f"Error forecasting demand: {e}")
            return []
    
    def identify_at_risk_inventory(self):
        """
        Identify inventory that is at risk of expiring before being sold.
        
        Returns:
            DataFrame containing at-risk inventory items
        """
        try:
            # Initialize empty list for at-risk items
            at_risk_items = []
            
            # Get unique combinations of SKU_id and warehouse_id
            unique_combinations = self.time_series_data[['SKU_id', 'warehous_id']].drop_duplicates()
            
            for _, row in unique_combinations.iterrows():
                sku_id = row['SKU_id']
                warehouse_id = row['warehous_id']
                
                # Get product info
                product_info = self.product_data[self.product_data['SKU_ID'] == sku_id]
                if product_info.empty:
                    continue
                
                # Get current stock and days remaining
                ts_data = self.time_series_data[
                    (self.time_series_data['SKU_id'] == sku_id) & 
                    (self.time_series_data['warehous_id'] == warehouse_id)
                ]
                
                if ts_data.empty:
                    continue
                
                current_stock = ts_data['current_stock'].sum()
                days_remaining = ts_data['days_remaining'].mean()
                
                # Forecast demand for the period until expiry
                forecast_days = min(int(days_remaining) + 1, 30)  # Cap at 30 days
                forecasts = self.forecast_demand(sku_id, warehouse_id, days_ahead=forecast_days)
                
                if not forecasts:
                    continue
                
                # Calculate total forecasted demand until expiry
                total_demand = sum(f.quantity for f in forecasts if (f.forecast_date - datetime.now()).days <= days_remaining)
                
                # Determine if stock is at risk (current stock > forecasted demand)
                excess_stock = current_stock - total_demand
                at_risk = excess_stock > 0
                
                if at_risk:
                    at_risk_items.append({
                        'SKU_ID': sku_id,
                        'warehouse_id': warehouse_id,
                        'current_stock': current_stock,
                        'days_remaining': days_remaining,
                        'forecasted_demand': total_demand,
                        'excess_stock': excess_stock,
                        'excess_percentage': (excess_stock / current_stock * 100) if current_stock > 0 else 0
                    })
            
            return pd.DataFrame(at_risk_items)
        except Exception as e:
            print(f"Error identifying at-risk inventory: {e}")
            return pd.DataFrame()
    
    def generate_alerts(self):
        """
        Generate alerts for inventory at risk of expiring before being sold.
        
        Returns:
            List of ProductAlert objects
        """
        try:
            # Get at-risk inventory
            at_risk_inventory = self.identify_at_risk_inventory()
            
            if at_risk_inventory.empty:
                return []
            
            # Initialize list for alerts
            alerts = []
            
            for _, item in at_risk_inventory.iterrows():
                # Determine alert severity based on excess percentage and days remaining
                excess_pct = item['excess_percentage']
                days_remaining = item['days_remaining']
                
                if excess_pct > 50 and days_remaining < 7:
                    severity = "HIGH"
                elif excess_pct > 30 or days_remaining < 5:
                    severity = "MEDIUM"
                else:
                    severity = "LOW"
                
                # Create alert message
                message = (f"At risk of {item['excess_stock']:.1f} units expiring. " +
                          f"Current stock: {item['current_stock']:.1f}, " +
                          f"Forecasted demand: {item['forecasted_demand']:.1f}, " +
                          f"Days until expiry: {item['days_remaining']:.1f}")
                
                # Create recommended action
                if severity == "HIGH":
                    action = "Implement immediate price reduction or consider redistribution to higher-demand locations."
                elif severity == "MEDIUM":
                    action = "Consider promotional pricing or bundling with fast-moving products."
                else:
                    action = "Monitor closely and consider price optimization for affected stock."
                
                # Create alert
                alert = ProductAlert(
                    SKU_ID=item['SKU_ID'],
                    warehouse_id=item['warehouse_id'],
                    alert_type="EXCESS_INVENTORY",
                    severity=severity,
                    message=message,
                    timestamp=datetime.now(),
                    recommended_action=action
                )
                
                alerts.append(alert)
            
            return alerts
        except Exception as e:
            print(f"Error generating demand alerts: {e}")
            return []
    
    def get_demand_analysis(self, sku_id, warehouse_id):
        """
        Get LLM analysis of demand forecast for a specific SKU and warehouse.
        
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
            
            # Get historical sales
            historical_sales = self.get_historical_sales(sku_id, warehouse_id, days=30)
            if historical_sales.empty:
                historical_data_str = "No historical sales data available."
            else:
                historical_data_str = historical_sales.to_string()
            
            # Get demand forecast
            forecasts = self.forecast_demand(sku_id, warehouse_id, days_ahead=14)
            if not forecasts:
                forecast_data_str = "No forecast data available."
            else:
                forecast_df = pd.DataFrame([f.dict() for f in forecasts])
                forecast_data_str = forecast_df.to_string()
            
            # Run analysis
            result = self.analysis_chain.run(
                sku_id=sku_id,
                warehouse_id=warehouse_id,
                product_data=product_info.to_string(),
                historical_data=historical_data_str,
                forecast_data=forecast_data_str
            )
            
            return result
        except Exception as e:
            return f"Error generating demand analysis: {e}"