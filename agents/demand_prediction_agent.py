# agents/demand_prediction_agent.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings("ignore")

from models.data_models import ProductAlert, DemandForecast
from utils.time_series_utils import extract_seasonality, detect_anomalies
import config

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
        self.llm = OllamaLLM(model=config.OLLAMA_MODEL)
        
        # Initialize the demand forecast analysis prompt
        self.forecast_analysis_template = PromptTemplate(
            input_variables=["product_info", "historical_sales", "current_inventory", "forecast_data", "expiry_risk"],
            template="""
            Based on the following information about a perishable product:
            
            Product Info:
            {product_info}
            
            Historical Sales:
            {historical_sales}
            
            Current Inventory:
            {current_inventory}
            
            Forecast Data:
            {forecast_data}
            
            Expiry Risk Assessment:
            {expiry_risk}
            
            Please provide:
            1. An analysis of the demand forecast vs current inventory levels
            2. Identify any mismatch between supply and expected demand
            3. Suggest inventory management strategies based on the forecast
            4. Highlight any at-risk inventory that may not be sold before expiry
            
            Your response should be concise, actionable, and focus on preventing waste.
            """
        )
        
        self.forecast_analysis_chain = LLMChain(llm=self.llm, prompt=self.forecast_analysis_template)
    
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
        if self.sales_data is not None:
            # Use actual sales data if available
            sales = self.sales_data[self.sales_data['SKU_ID'] == sku_id].copy()
            if warehouse_id:
                # Filter by warehouse if specified
                sales = sales[sales['warehouse_id'] == warehouse_id]
            
            # Convert dates and sort
            sales['date'] = pd.to_datetime(sales['date'])
            sales = sales.sort_values('date', ascending=False)
            
            # Limit to requested number of days
            cutoff_date = datetime.now() - timedelta(days=days)
            sales = sales[sales['date'] >= cutoff_date]
            
            return sales
        else:
            # Use time series data to estimate sales
            # Assume change in inventory levels represents sales
            ts_data = self.time_series_data[self.time_series_data['SKU_ID'] == sku_id].copy()
            if warehouse_id:
                ts_data = ts_data[ts_data['warehous_id'] == warehouse_id]
            
            # Convert timestamps
            ts_data['Timestamp'] = pd.to_datetime(ts_data['Timestamp'])
            ts_data = ts_data.sort_values('Timestamp')
            
            # Create a time series of inventory changes
            ts_data['prev_stock'] = ts_data['current_stock'].shift(1)
            ts_data['sales_estimate'] = ts_data['prev_stock'] - ts_data['current_stock'] + ts_data['waste_qty']
            ts_data['sales_estimate'] = ts_data['sales_estimate'].clip(lower=0)  # Ensure no negative sales
            
            # Keep only necessary columns
            sales_estimate = ts_data[['Timestamp', 'SKU_ID', 'warehous_id', 'sales_estimate', 'unit_curr_stock']]
            sales_estimate = sales_estimate.rename(columns={
                'Timestamp': 'date',
                'sales_estimate': 'quantity', 
                'unit_curr_stock': 'unit'
            })
            
            # Limit to requested number of days
            cutoff_date = datetime.now() - timedelta(days=days)
            sales_estimate = sales_estimate[sales_estimate['date'] >= cutoff_date]
            
            return sales_estimate
    
    def detect_seasonality(self, sku_id, warehouse_id=None):
        """
        Detect seasonal patterns in sales data.
        
        Args:
            sku_id: The SKU ID to analyze
            warehouse_id: Optional warehouse ID to filter by
            
        Returns:
            Dictionary containing seasonality information
        """
        # Get historical sales data
        sales_data = self.get_historical_sales(sku_id, warehouse_id, days=90)  # Use 90 days for seasonality
        
        if len(sales_data) < 14:  # Need at least 2 weeks of data
            return {"weekly_pattern": None, "detected": False}
        
        # Extract date features
        sales_data['dayofweek'] = sales_data['date'].dt.dayofweek  # Monday=0, Sunday=6
        sales_data['month'] = sales_data['date'].dt.month
        
        # Analyze weekly patterns
        weekly_pattern = sales_data.groupby('dayofweek')['quantity'].mean().to_dict()
        
        # Calculate coefficient of variation to determine if seasonality exists
        values = list(weekly_pattern.values())
        cv = np.std(values) / np.mean(values) if np.mean(values) > 0 else 0
        
        # Return seasonality information
        return {
            "weekly_pattern": weekly_pattern,
            "detected": cv > 0.2,  # Threshold for determining significant seasonality
            "strength": cv
        }
    
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
        # Get historical sales data
        sales_data = self.get_historical_sales(sku_id, warehouse_id, days=30)
        
        if len(sales_data) < 5:  # Need at least 5 data points for forecasting
            # If insufficient data, use the demand forecast from time series data
            ts_data = self.time_series_data[
                (self.time_series_data['SKU_id'] == sku_id) & 
                (self.time_series_data['warehous_id'] == warehouse_id if warehouse_id else True)
            ]
            
            if not ts_data.empty:
                # Get the most recent demand forecast
                latest = ts_data.iloc[0]
                daily_forecast = latest['Demand_forecast'] / 7  # Assume weekly forecast
                
                forecasts = []
                for i in range(days_ahead):
                    forecast_date = datetime.now() + timedelta(days=i+1)
                    forecasts.append(DemandForecast(
                        SKU_ID=sku_id,
                        warehouse_id=warehouse_id if warehouse_id else "ALL",
                        forecast_date=forecast_date,
                        quantity=daily_forecast,
                        unit=latest['unit_Demand_forecast'],
                        confidence=0.5  # Low confidence due to lack of data
                    ))
                return forecasts
            else:
                # No data available
                return []
        
        # Prepare time series data
        if 'date' in sales_data.columns:
            sales_data = sales_data.set_index('date')
            sales_ts = sales_data['quantity'].resample('D').sum().fillna(0)
        else:
            # For testing, create a simple time series
            dates = pd.date_range(end=datetime.now(), periods=len(sales_data), freq='D')
            sales_ts = pd.Series(sales_data['quantity'].values, index=dates)
        
        # Ensure we have enough data for forecasting
        if len(sales_ts) < 5:
            return []
            
        try:
            # Simple ARIMA model for forecasting
            model = ARIMA(sales_ts, order=(1, 0, 1))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=days_ahead)
            
            # Get seasonality information
            seasonality = self.detect_seasonality(sku_id, warehouse_id)
            
            # Adjust forecast based on seasonality if detected
            if seasonality["detected"]:
                for i in range(days_ahead):
                    forecast_date = datetime.now() + timedelta(days=i+1)
                    dow = forecast_date.weekday()
                    if dow in seasonality["weekly_pattern"]:
                        # Apply seasonal adjustment factor
                        avg_daily = sales_ts.mean()
                        seasonal_factor = seasonality["weekly_pattern"][dow] / avg_daily if avg_daily > 0 else 1
                        forecast[i] = forecast[i] * seasonal_factor
            
            # Ensure no negative forecasts
            forecast = np.maximum(forecast, 0)
            
            # Create DemandForecast objects
            forecasts = []
            for i in range(days_ahead):
                forecast_date = datetime.now() + timedelta(days=i+1)
                forecasts.append(DemandForecast(
                    SKU_ID=sku_id,
                    warehouse_id=warehouse_id if warehouse_id else "ALL",
                    forecast_date=forecast_date,
                    quantity=float(forecast[i]),
                    unit=sales_data['unit'].iloc[0] if 'unit' in sales_data.columns else "units",
                    confidence=0.7  # Medium confidence
                ))
            
            return forecasts
            
        except Exception as e:
            print(f"Error forecasting demand for {sku_id}: {e}")
            return []
    
    def identify_at_risk_inventory(self):
        """
        Identify inventory that is at risk of expiring before being sold.
        
        Returns:
            DataFrame containing at-risk inventory items
        """
        results = []
        
        # Get unique SKU and warehouse combinations
        unique_combinations = self.time_series_data[['SKU_id', 'warehous_id']].drop_duplicates()
        
        for _, row in unique_combinations.iterrows():
            sku_id = row['SKU_id']
            warehouse_id = row['warehous_id']
            
            # Get time series data for this SKU and warehouse
            ts_data = self.time_series_data[
                (self.time_series_data['SKU_id'] == sku_id) & 
                (self.time_series_data['warehous_id'] == warehouse_id)
            ]
            
            if ts_data.empty:
                continue
                
            # Get latest record
            latest = ts_data.iloc[0]
            
            # Forecast demand for the remaining shelf life
            days_remaining = int(latest['days_remaining'])
            if days_remaining <= 0:
                continue  # Already expired
                
            forecasts = self.forecast_demand(sku_id, warehouse_id, days_ahead=days_remaining)
            
            if not forecasts:
                continue
                
            # Calculate total forecasted demand
            total_forecasted_demand = sum(f.quantity for f in forecasts)
            
            # Compare with current stock
            current_stock = latest['current_stock']
            
            # Calculate excess inventory
            excess_inventory = current_stock - total_forecasted_demand
            
            # If excess inventory is positive, this inventory is at risk
            if excess_inventory > 0:
                risk_percentage = (excess_inventory / current_stock) * 100
                
                # Only consider significant risk
                if risk_percentage >= 10:  # At least 10% at risk
                    results.append({
                        'SKU_ID': sku_id,
                        'warehouse_id': warehouse_id,
                        'current_stock': current_stock,
                        'unit': latest['unit_curr_stock'],
                        'days_remaining': days_remaining,
                        'forecasted_demand': total_forecasted_demand,
                        'excess_inventory': excess_inventory,
                        'risk_percentage': risk_percentage,
                        'average_daily_demand': total_forecasted_demand / days_remaining if days_remaining > 0 else 0
                    })
        
        return pd.DataFrame(results)
    
    def generate_alerts(self):
        """
        Generate alerts for inventory at risk of expiring before being sold.
        
        Returns:
            List of ProductAlert objects
        """
        at_risk_inventory = self.identify_at_risk_inventory()
        alerts = []
        
        for _, item in at_risk_inventory.iterrows():
            # Determine alert severity based on risk percentage and days remaining
            severity = "LOW"
            if item['risk_percentage'] >= 50 or item['days_remaining'] <= 2:
                severity = "HIGH"
            elif item['risk_percentage'] >= 25 or item['days_remaining'] <= 5:
                severity = "MEDIUM"
                
            # Calculate estimated days of inventory
            avg_daily_demand = item['average_daily_demand']
            estimated_days = float('inf') if avg_daily_demand == 0 else item['current_stock'] / avg_daily_demand
            
            # Create message
            message = (
                f"Excess inventory risk: {item['risk_percentage']:.1f}% of stock " +
                f"({item['excess_inventory']:.1f} {item['unit']}) likely to expire before sale. " +
                f"Current stock would last {estimated_days:.1f} days at forecasted demand rate."
            )
            
            # Determine recommended action
            recommended_action = ""
            if severity == "HIGH":
                recommended_action = "Consider immediate markdown or transfer to high-demand location"
            elif severity == "MEDIUM":
                recommended_action = "Consider promotional pricing or bundling"
            else:
                recommended_action = "Monitor demand patterns closely"
                
            alert = ProductAlert(
                SKU_ID=item['SKU_ID'],
                warehouse_id=item['warehouse_id'],
                alert_type="DEMAND_RISK",
                severity=severity,
                message=message,
                timestamp=datetime.now(),
                recommended_action=recommended_action
            )
            alerts.append(alert)
            
        return alerts
    
    def get_demand_analysis(self, sku_id, warehouse_id):
        """
        Get LLM analysis of demand forecast for a specific SKU and warehouse.
        
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
        
        # Get historical sales data
        historical_sales = self.get_historical_sales(sku_id, warehouse_id, days=14)
        if historical_sales.empty:
            historical_sales_text = "No historical sales data available."
        else:
            if isinstance(historical_sales.index, pd.DatetimeIndex):
                historical_sales.reset_index(inplace=True)
            historical_sales_text = historical_sales.to_string()
        
        # Get current inventory
        inventory_status = self.time_series_data[
            (self.time_series_data['SKU_id'] == sku_id) & 
            (self.time_series_data['warehous_id'] == warehouse_id)
        ]
        
        if inventory_status.empty:
            return "Inventory data not found."
            
        inventory_status = inventory_status.iloc[0].to_dict()
        
        # Generate demand forecast
        days_remaining = inventory_status['days_remaining']
        forecast = self.forecast_demand(sku_id, warehouse_id, days_ahead=min(7, days_remaining))
        
        forecast_text = "\n".join([
            f"Date: {f.forecast_date.strftime('%Y-%m-%d')}, " +
            f"Quantity: {f.quantity:.2f} {f.unit}, " +
            f"Confidence: {f.confidence:.2f}"
            for f in forecast
        ])
        
        if not forecast:
            forecast_text = "Unable to generate forecast."
        
        # Calculate expiry risk
        at_risk_inventory = self.identify_at_risk_inventory()
        at_risk = at_risk_inventory[
            (at_risk_inventory['SKU_ID'] == sku_id) & 
            (at_risk_inventory['warehouse_id'] == warehouse_id)
        ]
        
        if at_risk.empty:
            expiry_risk_text = "No significant expiry risk detected."
        else:
            item = at_risk.iloc[0]
            expiry_risk_text = (
                f"EXPIRY RISK: {item['risk_percentage']:.1f}% of current stock " +
                f"({item['excess_inventory']:.1f} {item['unit']}) may expire before sale. " +
                f"Average daily demand: {item['average_daily_demand']:.1f} {item['unit']}. " +
                f"Current stock would last {item['current_stock'] / item['average_daily_demand']:.1f} days " +
                f"at forecasted demand rate, but only {item['days_remaining']} days until expiry."
            )
        
        # Get analysis from LLM
        analysis = self.forecast_analysis_chain.run(
            product_info=str(product_info),
            historical_sales=historical_sales_text,
            current_inventory=str(inventory_status),
            forecast_data=forecast_text,
            expiry_risk=expiry_risk_text
        )
        
        return analysis