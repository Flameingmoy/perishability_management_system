import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from models.data_models import ProductAlert
import config

class EnvironmentalMonitoringAgent:
    def __init__(self, product_data, time_series_data, iot_data):
        """
        Initialize the environmental monitoring agent.
        
        Args:
            product_data: DataFrame containing product master data
            time_series_data: DataFrame containing time series perishability data
            iot_data: DataFrame containing IoT sensor data
        """
        self.product_data = product_data
        self.time_series_data = time_series_data
        self.iot_data = iot_data
        self.llm = OllamaLLM(model=config.OLLAMA_MODEL)
        
        # Initialize prompt template for LLM analysis
        self.analysis_template = PromptTemplate(
            input_variables=["product_info", "environmental_data", "deviations", "risk_assessment"],
            template="""
            Based on the following information about environmental conditions for a perishable product:
            
            Product Info:
            {product_info}
            
            Environmental Data:
            {environmental_data}
            
            Environmental Deviations:
            {deviations}
            
            Risk Assessment:
            {risk_assessment}
            
            Please provide:
            1. An analysis of the current environmental conditions
            2. How these conditions are affecting product quality and shelf-life
            3. Recommendations for immediate corrective actions
            4. Long-term strategies to improve environmental conditions
            
            Your response should be concise and actionable.
            """
        )
        
        self.analysis_chain = LLMChain(llm=self.llm, prompt=self.analysis_template)
        
        # Clean and convert data types
        self._preprocess_data()
    
    def _preprocess_data(self):
        """
        Preprocess data to ensure proper types
        """
        # Convert Critical_Humidity in product data (removing % sign and converting to float)
        if 'Critical_Humidity' in self.product_data.columns:
            self.product_data['Critical_Humidity'] = self.product_data['Critical_Humidity'].apply(
                lambda x: float(str(x).replace('%', '')) / 100 if isinstance(x, str) else x
            )
        
        # Ensure TEMP_HUMIDITY is numeric in IoT data
        if 'TEMP_HUMIDITY' in self.iot_data.columns:
            self.iot_data['TEMP_HUMIDITY'] = pd.to_numeric(self.iot_data['TEMP_HUMIDITY'], errors='coerce')
            
        # Ensure numeric temperature deviation
        if 'temp_deviation' in self.time_series_data.columns:
            self.time_series_data['temp_deviation'] = pd.to_numeric(self.time_series_data['temp_deviation'], errors='coerce')
    
    def calculate_critical_alert_rate(self, sku_id=None):
        """
        Calculate critical alert rate using the formula:
        % batches with Temp_Deviation > 2°C or Humidity > Critical_Humidity
        
        Args:
            sku_id: Optional SKU ID to filter by
        
        Returns:
            Critical alert rate as a percentage
        """
        ts_df = self.time_series_data.copy()
        if sku_id:
            ts_df = ts_df[ts_df['SKU_id'] == sku_id]
        
        if ts_df.empty:
            return 0.0
            
        # Join with product data to get Critical_Humidity values
        merged_df = pd.merge(
            ts_df, 
            self.product_data, 
            left_on='SKU_id', 
            right_on='SKU_ID'
        )
        
        if merged_df.empty:
            return 0.0
        
        # Join with IoT data to get humidity values
        merged_df = pd.merge(
            merged_df,
            self.iot_data,
            left_on='SKU_id',
            right_on='SKU_ID',
            how='left'
        )
        
        if merged_df.empty or 'TEMP_HUMIDITY' not in merged_df.columns:
            return 0.0
        
        # Get invalid rows for debugging
        invalid_rows = merged_df[pd.isna(merged_df['Critical_Humidity']) | pd.isna(merged_df['TEMP_HUMIDITY'])].copy()
        if not invalid_rows.empty:
            print(f"Warning: Found {len(invalid_rows)} rows with invalid humidity data")
        
        # Filter out invalid rows
        valid_df = merged_df.dropna(subset=['Critical_Humidity', 'TEMP_HUMIDITY']).copy()
        if valid_df.empty:
            return 0.0
        
        # Ensure numeric types
        valid_df['Critical_Humidity'] = pd.to_numeric(valid_df['Critical_Humidity'], errors='coerce')
        valid_df['TEMP_HUMIDITY'] = pd.to_numeric(valid_df['TEMP_HUMIDITY'], errors='coerce')
        valid_df['temp_deviation'] = pd.to_numeric(valid_df['temp_deviation'], errors='coerce')
        
        # Apply filtering after handling missing values
        temp_deviation_filter = abs(valid_df['temp_deviation']) > config.TEMP_DEVIATION_ALERT
        
        # Handle humidity comparison safely
        humidity_filter = pd.Series(False, index=valid_df.index)  # Default to False
        
        try:
            # First ensure we have numeric values to compare
            mask = pd.notna(valid_df['TEMP_HUMIDITY']) & pd.notna(valid_df['Critical_Humidity'])
            if mask.any():
                humidity_filter[mask] = valid_df.loc[mask, 'TEMP_HUMIDITY'] > valid_df.loc[mask, 'Critical_Humidity'] * 100
        except Exception as e:
            print(f"Warning: Error in humidity comparison: {e}")
        
        # Count batches with critical deviations
        critical_batches = valid_df[temp_deviation_filter | humidity_filter]
        
        critical_alert_rate = (len(critical_batches) / len(valid_df)) * 100 if len(valid_df) > 0 else 0.0
        return critical_alert_rate
    
    def calculate_temp_compliance_rate(self, sku_id=None):
        """
        Calculate temperature compliance rate using the formula:
        Temp Compliance Rate = % readings within ±0.5°C of ideal
        
        Args:
            sku_id: Optional SKU ID to filter by
        
        Returns:
            Temperature compliance rate as a percentage
        """
        ts_df = self.time_series_data
        if sku_id:
            ts_df = ts_df[ts_df['SKU_id'] == sku_id]
        
        if ts_df.empty:
            return 100.0  # Default to 100% if no data
        
        # Ensure numeric type for temp_deviation
        ts_df['temp_deviation'] = pd.to_numeric(ts_df['temp_deviation'], errors='coerce')
        
        # Filter out NaN values
        valid_df = ts_df.dropna(subset=['temp_deviation'])
        if valid_df.empty:
            return 100.0
            
        # Count readings within compliance range
        compliant_readings = valid_df[abs(valid_df['temp_deviation']) <= 0.5]
        
        compliance_rate = (len(compliant_readings) / len(valid_df)) * 100
        return compliance_rate
    
    def calculate_quality_degradation_index(self, sku_id=None):
        """
        Calculate Quality Degradation Index using the formula:
        Quality Degradation Index = Σ(Temp_Deviation × Time)
        
        Args:
            sku_id: Optional SKU ID to filter by
        
        Returns:
            Quality Degradation Index as a float
        """
        ts_df = self.time_series_data
        if sku_id:
            ts_df = ts_df[ts_df['SKU_id'] == sku_id]
        
        if ts_df.empty:
            return 0.0
        
        # Ensure numeric type for temp_deviation
        ts_df['temp_deviation'] = pd.to_numeric(ts_df['temp_deviation'], errors='coerce')
        
        # Filter out NaN values
        valid_df = ts_df.dropna(subset=['temp_deviation'])
        if valid_df.empty:
            return 0.0
            
        # Assume each reading represents 1 time unit (e.g., 1 hour)
        # Sum of absolute temperature deviations × time
        degradation_index = (abs(valid_df['temp_deviation']) * 1).sum()
        return degradation_index
    
    def calculate_shelf_life_decay(self, sku_id):
        """
        Calculate adjusted shelf life using the formula:
        Adjusted_Life = Initial_Shelf_Life × (1 - 0.05×Temp_Deviation + 0.03×Humidity_Change)
        
        Args:
            sku_id: SKU ID to calculate for
        
        Returns:
            Dictionary with original and adjusted shelf life
        """
        # Get product data
        product_df = self.product_data[self.product_data['SKU_ID'] == sku_id]
        if product_df.empty:
            return {"original": 0, "adjusted": 0, "adjustment_factor": 1.0}
        
        # Convert Initial_Shelf_Life to float if needed
        initial_shelf_life = product_df.iloc[0].get('Initial_Shelf_Life')
        if initial_shelf_life is None:
            initial_shelf_life = product_df.iloc[0].get('initial_shelf_life', 0)
        
        try:
            initial_shelf_life = float(initial_shelf_life)
        except (ValueError, TypeError):
            print(f"Warning: Invalid shelf life for SKU {sku_id}: {initial_shelf_life}")
            initial_shelf_life = 0
            
        # Get time series data
        ts_df = self.time_series_data[self.time_series_data['SKU_id'] == sku_id]
        if ts_df.empty:
            return {"original": initial_shelf_life, "adjusted": initial_shelf_life, "adjustment_factor": 1.0}
            
        # Get IoT data
        iot_df = self.iot_data[self.iot_data['SKU_ID'] == sku_id]
        if iot_df.empty:
            return {"original": initial_shelf_life, "adjusted": initial_shelf_life, "adjustment_factor": 1.0}
        
        # Ensure numeric type for temp_deviation
        ts_df['temp_deviation'] = pd.to_numeric(ts_df['temp_deviation'], errors='coerce')
            
        # Calculate average temperature deviation
        avg_temp_deviation = abs(ts_df['temp_deviation'].mean())
        
        # Calculate humidity change (using standard humidity as 70%)
        if 'TEMP_HUMIDITY' in iot_df.columns:
            # Ensure numeric type
            iot_df['TEMP_HUMIDITY'] = pd.to_numeric(iot_df['TEMP_HUMIDITY'], errors='coerce')
            avg_humidity = iot_df['TEMP_HUMIDITY'].mean()
            humidity_change = abs(avg_humidity - 70) / 100  # Convert to decimal
        else:
            humidity_change = 0.0
        
        # Calculate adjusted shelf life
        adjustment_factor = 1 - (0.05 * avg_temp_deviation) + (0.03 * humidity_change)
        adjusted_shelf_life = initial_shelf_life * adjustment_factor
        
        return {
            "original": initial_shelf_life,
            "adjusted": adjusted_shelf_life,
            "adjustment_factor": adjustment_factor
        }
    
    def identify_environmental_issues(self):
        """
        Identify products with environmental issues.
        
        Returns:
            DataFrame with environmental issues
        """
        results = []
        
        # Get unique SKUs
        unique_skus = self.time_series_data['SKU_id'].unique()
        
        for sku_id in unique_skus:
            try:
                # Get product data
                product_df = self.product_data[self.product_data['SKU_ID'] == sku_id]
                if product_df.empty:
                    continue
                    
                product = product_df.iloc[0].to_dict()
                
                # Get time series data
                ts_df = self.time_series_data[self.time_series_data['SKU_id'] == sku_id]
                if ts_df.empty:
                    continue
                    
                # Get IoT data
                iot_df = self.iot_data[self.iot_data['SKU_ID'] == sku_id]
                if iot_df.empty:
                    continue
                
                # Ensure data types are correct
                ts_df['temp_deviation'] = pd.to_numeric(ts_df['temp_deviation'], errors='coerce')
                
                if 'TEMP_HUMIDITY' in iot_df.columns:
                    iot_df['TEMP_HUMIDITY'] = pd.to_numeric(iot_df['TEMP_HUMIDITY'], errors='coerce')
                
                if 'Critical_Humidity' in product:
                    try:
                        product['Critical_Humidity'] = float(str(product['Critical_Humidity']).replace('%', '')) / 100 if isinstance(product['Critical_Humidity'], str) else product['Critical_Humidity']
                    except (ValueError, TypeError):
                        product['Critical_Humidity'] = 0.7  # Default value
                    
                # Calculate metrics
                critical_alert_rate = self.calculate_critical_alert_rate(sku_id)
                temp_compliance_rate = self.calculate_temp_compliance_rate(sku_id)
                quality_degradation_index = self.calculate_quality_degradation_index(sku_id)
                shelf_life_decay = self.calculate_shelf_life_decay(sku_id)
                
                # Check for environmental issues
                has_issues = False
                issues = []
                
                # Check temperature deviation
                avg_temp_deviation = abs(ts_df['temp_deviation'].mean())
                if avg_temp_deviation > config.TEMP_DEVIATION_ALERT:
                    has_issues = True
                    issues.append(f"High temperature deviation: {avg_temp_deviation:.2f}°C")
                    
                # Check humidity
                if 'TEMP_HUMIDITY' in iot_df.columns and 'Critical_Humidity' in product:
                    avg_humidity = iot_df['TEMP_HUMIDITY'].mean()
                    critical_humidity = product['Critical_Humidity'] * 100  # Convert to percentage
                    
                    if not pd.isna(avg_humidity) and not pd.isna(critical_humidity) and avg_humidity > critical_humidity:
                        has_issues = True
                        issues.append(f"High humidity: {avg_humidity:.2f}% (Critical: {critical_humidity:.2f}%)")
                    
                # Check critical alert rate
                if critical_alert_rate > 20.0:  # 20% threshold
                    has_issues = True
                    issues.append(f"High critical alert rate: {critical_alert_rate:.2f}%")
                    
                # Check temperature compliance rate
                if temp_compliance_rate < 80.0:  # 80% threshold
                    has_issues = True
                    issues.append(f"Low temperature compliance rate: {temp_compliance_rate:.2f}%")
                    
                # Check quality degradation
                if quality_degradation_index > 10.0:  # Arbitrary threshold
                    has_issues = True
                    issues.append(f"High quality degradation index: {quality_degradation_index:.2f}")
                    
                # Check shelf life adjustment
                if shelf_life_decay["adjustment_factor"] < 0.8:  # 20% reduction threshold
                    has_issues = True
                    reduction = (1 - shelf_life_decay["adjustment_factor"]) * 100
                    issues.append(f"Significant shelf life reduction: {reduction:.2f}%")
                    
                if has_issues:
                    for warehouse_id in ts_df['warehous_id'].unique():
                        warehouse_ts = ts_df[ts_df['warehous_id'] == warehouse_id]
                        if warehouse_ts.empty:
                            continue
                            
                        results.append({
                            'SKU_ID': sku_id,
                            'warehouse_id': warehouse_id,
                            'product_category': product.get('Product_Category', 'Unknown'),
                            'critical_alert_rate': critical_alert_rate,
                            'temp_compliance_rate': temp_compliance_rate,
                            'quality_degradation_index': quality_degradation_index,
                            'original_shelf_life': shelf_life_decay["original"],
                            'adjusted_shelf_life': shelf_life_decay["adjusted"],
                            'shelf_life_reduction': (1 - shelf_life_decay["adjustment_factor"]) * 100,
                            'issues': ', '.join(issues)
                        })
            except Exception as e:
                print(f"Error processing SKU {sku_id}: {e}")
                continue
                
        return pd.DataFrame(results) if results else pd.DataFrame()
    
    def generate_alerts(self):
        """
        Generate alerts for environmental issues.
        
        Returns:
            List of ProductAlert objects
        """
        try:
            environmental_issues = self.identify_environmental_issues()
            alerts = []
            
            if environmental_issues.empty:
                print("No environmental issues found")
                return []
            
            for _, issue in environmental_issues.iterrows():
                try:
                    # Determine alert severity
                    severity = "LOW"
                    if issue['critical_alert_rate'] > 50.0 or issue['shelf_life_reduction'] > 30.0:
                        severity = "HIGH"
                    elif issue['critical_alert_rate'] > 20.0 or issue['shelf_life_reduction'] > 15.0:
                        severity = "MEDIUM"
                        
                    # Determine recommended action
                    if severity == "HIGH":
                        recommended_action = "Immediate relocation to compliant storage or markdown for quick sale"
                    elif severity == "MEDIUM":
                        recommended_action = "Adjust storage conditions and monitor closely"
                    else:
                        recommended_action = "Monitor and maintain storage conditions"
                        
                    alert = ProductAlert(
                        SKU_ID=issue['SKU_ID'],
                        warehouse_id=issue['warehouse_id'],
                        alert_type="ENVIRONMENTAL_RISK",
                        severity=severity,
                        message=issue['issues'],
                        timestamp=datetime.now(),
                        recommended_action=recommended_action
                    )
                    alerts.append(alert)
                except Exception as e:
                    print(f"Error generating alert for issue: {e}")
                    continue
                
            return alerts
        except Exception as e:
            print(f"Error in generate_alerts: {e}")
            return []
    
    def get_environmental_analysis(self, sku_id, warehouse_id=None):
        """
        Get LLM analysis for environmental conditions of a specific SKU.
        
        Args:
            sku_id: SKU ID to analyze
            warehouse_id: Optional warehouse ID to filter by
        
        Returns:
            Analysis text from the LLM
        """
        try:
            # Get product info
            product_info = self.product_data[self.product_data['SKU_ID'] == sku_id]
            if product_info.empty:
                return "Product not found."
            
            product_info = product_info.iloc[0].to_dict()
            
            # Get environmental data
            ts_df = self.time_series_data[self.time_series_data['SKU_id'] == sku_id]
            if warehouse_id:
                ts_df = ts_df[ts_df['warehous_id'] == warehouse_id]
            
            if ts_df.empty:
                return "Environmental data not found."
                
            iot_df = self.iot_data[self.iot_data['SKU_ID'] == sku_id]
            
            if iot_df.empty:
                return "IoT data not found."
            
            # Ensure numeric types
            ts_df['temp_deviation'] = pd.to_numeric(ts_df['temp_deviation'], errors='coerce')
            if 'TEMP_HUMIDITY' in iot_df.columns:
                iot_df['TEMP_HUMIDITY'] = pd.to_numeric(iot_df['TEMP_HUMIDITY'], errors='coerce')
            
            environmental_data = {}
            environmental_data["avg_temp_deviation"] = ts_df['temp_deviation'].mean()
            environmental_data["max_temp_deviation"] = ts_df['temp_deviation'].max()
            
            if 'TEMP_HUMIDITY' in iot_df.columns:
                environmental_data["avg_humidity"] = iot_df['TEMP_HUMIDITY'].mean()
            else:
                environmental_data["avg_humidity"] = "Data not available"
                
            if 'SHOCK_EVENTS' in iot_df.columns:
                environmental_data["shock_events"] = iot_df['SHOCK_EVENTS'].sum()
            else:
                environmental_data["shock_events"] = "Data not available"
            
            # Calculate deviations
            deviations = {
                "critical_alert_rate": self.calculate_critical_alert_rate(sku_id),
                "temp_compliance_rate": self.calculate_temp_compliance_rate(sku_id),
                "quality_degradation_index": self.calculate_quality_degradation_index(sku_id)
            }
            
            # Calculate risk assessment
            shelf_life_decay = self.calculate_shelf_life_decay(sku_id)
            shelf_life_reduction = (1 - shelf_life_decay["adjustment_factor"]) * 100
            
            risk_assessment = {
                "original_shelf_life": shelf_life_decay["original"],
                "adjusted_shelf_life": shelf_life_decay["adjusted"],
                "shelf_life_reduction_percentage": shelf_life_reduction,
                "risk_level": "HIGH" if shelf_life_reduction > 30.0 else "MEDIUM" if shelf_life_reduction > 15.0 else "LOW"
            }
            
            # Get analysis from LLM
            analysis = self.analysis_chain.run(
                product_info=str(product_info),
                environmental_data=str(environmental_data),
                deviations=str(deviations),
                risk_assessment=str(risk_assessment)
            )
            
            return analysis
        except Exception as e:
            return f"Error generating environmental analysis: {e}"