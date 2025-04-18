import pandas as pd
import numpy as np
from datetime import datetime
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import config
from models.data_models import ProductAlert

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
        
        # Preprocess data
        self._preprocess_data()
        
        # LLM for analysis
        try:
            self.llm = OllamaLLM(model=config.OLLAMA_MODEL)  # Using the new model specified in config
            
            # Analysis prompt
            self.analysis_prompt = PromptTemplate(
                input_variables=["sku_id", "product_data", "iot_data", "environmental_metrics"],
                template="""
                Analyze the environmental conditions for product {sku_id}.
                
                Product data:
                {product_data}
                
                IoT sensor data:
                {iot_data}
                
                Environmental metrics:
                {environmental_metrics}
                
                Provide a detailed analysis including:
                1. Temperature compliance rate and deviations
                2. Humidity conditions and impact on product quality
                3. Quality degradation assessment
                4. Recommended storage condition adjustments
                5. Potential environmental risks and mitigation strategies
                
                Analysis:
                """
            )
            
            self.analysis_chain = LLMChain(llm=self.llm, prompt=self.analysis_prompt)
        except Exception as e:
            print(f"Warning: Could not initialize LLM for analysis: {e}")
            self.llm = None
    
    def _preprocess_data(self):
        """
        Preprocess data to ensure proper types
        """
        try:
            # Convert 'TIMESTAMP' column to datetime if it exists
            if 'TIMESTAMP' in self.iot_data.columns:
                if self.iot_data['TIMESTAMP'].dtype == 'object':
                    self.iot_data['TIMESTAMP'] = pd.to_datetime(self.iot_data['TIMESTAMP'], errors='coerce')
            
            # Ensure numeric columns are properly typed
            numeric_columns = ['TEMP_HUMIDITY', 'SHOCK_EVENTS']
            for col in numeric_columns:
                if col in self.iot_data.columns:
                    self.iot_data[col] = pd.to_numeric(self.iot_data[col], errors='coerce')
            
            # Handle missing values
            self.iot_data = self.iot_data.fillna({
                'TEMP_HUMIDITY': self.iot_data['TEMP_HUMIDITY'].mean() if not self.iot_data.empty else 0,
                'SHOCK_EVENTS': 0
            })
        except Exception as e:
            print(f"Error preprocessing IoT data: {e}")
    
    def calculate_critical_alert_rate(self, sku_id=None):
        """
        Calculate critical alert rate using the formula:
        % batches with Temp_Deviation > 2°C or Humidity > Critical_Humidity
        
        Args:
            sku_id: Optional SKU ID to filter by
        
        Returns:
            Critical alert rate as a percentage
        """
        try:
            # Get relevant data
            temp_data = self.time_series_data.copy()
            humidity_data = self.iot_data.copy()
            
            if sku_id:
                temp_data = temp_data[temp_data['SKU_id'] == sku_id]
                humidity_data = humidity_data[humidity_data['SKU_ID'] == sku_id]
            
            if temp_data.empty or humidity_data.empty:
                return 0.0
            
            # Get product info to determine critical humidity levels
            if sku_id:
                product_info = self.product_data[self.product_data['SKU_ID'] == sku_id]
                if not product_info.empty and 'Critical_Humidity' in product_info.columns:
                    critical_humidity = product_info['Critical_Humidity'].values[0]
                else:
                    critical_humidity = 80.0  # Default if not found
            else:
                critical_humidity = 80.0  # Default if no specific SKU
            
            # Count critical temperature deviations
            critical_temp_count = len(temp_data[temp_data['temp_deviation'] > config.TEMP_DEVIATION_ALERT])
            
            # Count critical humidity deviations
            critical_humidity_count = len(humidity_data[humidity_data['TEMP_HUMIDITY'] > critical_humidity])
            
            # Total number of records
            total_records = len(temp_data) + len(humidity_data)
            
            # Calculate alert rate
            if total_records > 0:
                alert_rate = ((critical_temp_count + critical_humidity_count) / total_records) * 100
                return round(alert_rate, 2)
            else:
                return 0.0
        except Exception as e:
            print(f"Error calculating critical alert rate: {e}")
            return 0.0
    
    def calculate_temp_compliance_rate(self, sku_id=None):
        """
        Calculate temperature compliance rate using the formula:
        Temp Compliance Rate = % readings within ±0.5°C of ideal
        
        Args:
            sku_id: Optional SKU ID to filter by
        
        Returns:
            Temperature compliance rate as a percentage
        """
        try:
            # Get temperature deviation data
            temp_data = self.time_series_data.copy()
            
            if sku_id:
                temp_data = temp_data[temp_data['SKU_id'] == sku_id]
                
                # Get ideal temperature from product data
                product_info = self.product_data[self.product_data['SKU_ID'] == sku_id]
                if not product_info.empty and 'Storage_Temp' in product_info.columns:
                    ideal_temp = product_info['Storage_Temp'].values[0]
                else:
                    ideal_temp = 0.0  # Default
            else:
                ideal_temp = 0.0  # Default if no specific SKU
            
            if temp_data.empty:
                return 0.0
            
            # Count compliant readings (within ±0.5°C)
            compliant_count = len(temp_data[abs(temp_data['temp_deviation']) <= 0.5])
            
            # Calculate compliance rate
            compliance_rate = (compliant_count / len(temp_data)) * 100
            return round(compliance_rate, 2)
        except Exception as e:
            print(f"Error calculating temperature compliance rate: {e}")
            return 0.0
    
    def calculate_quality_degradation_index(self, sku_id=None):
        """
        Calculate Quality Degradation Index using the formula:
        Quality Degradation Index = Σ(Temp_Deviation × Time)
        
        Args:
            sku_id: Optional SKU ID to filter by
        
        Returns:
            Quality Degradation Index as a float
        """
        try:
            # Get temperature deviation data
            temp_data = self.time_series_data.copy()
            
            if sku_id:
                temp_data = temp_data[temp_data['SKU_id'] == sku_id]
            
            if temp_data.empty:
                return 0.0
            
            # Calculate quality degradation
            # Assume 1 unit of time per reading for simplicity
            time_factor = 1.0
            
            # Sum of absolute temperature deviations × time
            degradation_index = sum(abs(temp_data['temp_deviation']) * time_factor)
            
            return round(degradation_index, 2)
        except Exception as e:
            print(f"Error calculating quality degradation index: {e}")
            return 0.0
    
    def calculate_shelf_life_decay(self, sku_id):
        """
        Calculate adjusted shelf life using the formula:
        Adjusted_Life = Initial_Shelf_Life × (1 - 0.05×Temp_Deviation + 0.03×Humidity_Change)
        
        Args:
            sku_id: SKU ID to calculate for
        
        Returns:
            Dictionary with original and adjusted shelf life
        """
        try:
            # Get product data
            product_info = self.product_data[self.product_data['SKU_ID'] == sku_id]
            
            if product_info.empty or 'Initial_Shelf_Life' not in product_info.columns:
                return {"original": 0, "adjusted": 0}
            
            # Get initial shelf life
            initial_shelf_life = product_info['Initial_Shelf_Life'].values[0]
            
            # Get temperature deviation data
            temp_data = self.time_series_data[self.time_series_data['SKU_id'] == sku_id]
            
            # Get humidity data
            humidity_data = self.iot_data[self.iot_data['SKU_ID'] == sku_id]
            
            if temp_data.empty or humidity_data.empty:
                return {"original": initial_shelf_life, "adjusted": initial_shelf_life}
            
            # Calculate average temperature deviation
            avg_temp_deviation = abs(temp_data['temp_deviation'].mean())
            
            # Calculate humidity change
            # For simplicity, compare current humidity with critical humidity
            critical_humidity = product_info['Critical_Humidity'].values[0] if 'Critical_Humidity' in product_info.columns else 70.0
            current_humidity = humidity_data['TEMP_HUMIDITY'].mean()
            humidity_change = abs(current_humidity - critical_humidity) / 100.0  # Normalize to 0-1 range
            
            # Calculate adjusted shelf life
            temp_factor = 0.05 * avg_temp_deviation
            humidity_factor = 0.03 * humidity_change
            
            adjustment_factor = 1 - temp_factor + humidity_factor
            adjusted_shelf_life = initial_shelf_life * adjustment_factor
            
            return {
                "original": initial_shelf_life,
                "adjusted": round(adjusted_shelf_life, 1)
            }
        except Exception as e:
            print(f"Error calculating shelf life decay: {e}")
            return {"original": 0, "adjusted": 0}
    
    def identify_environmental_issues(self):
        """
        Identify products with environmental issues.
        
        Returns:
            DataFrame with environmental issues
        """
        try:
            # Initialize empty list to store issues
            issues = []
            
            # Get unique SKUs
            unique_skus = self.product_data['SKU_ID'].unique()
            
            for sku_id in unique_skus:
                # Calculate metrics
                critical_alert_rate = self.calculate_critical_alert_rate(sku_id)
                temp_compliance_rate = self.calculate_temp_compliance_rate(sku_id)
                quality_degradation = self.calculate_quality_degradation_index(sku_id)
                shelf_life_info = self.calculate_shelf_life_decay(sku_id)
                
                # Get SKU data
                sku_temp_data = self.time_series_data[self.time_series_data['SKU_id'] == sku_id]
                sku_humidity_data = self.iot_data[self.iot_data['SKU_ID'] == sku_id]
                
                # Get warehouses for this SKU
                warehouses = sku_temp_data['warehous_id'].unique() if not sku_temp_data.empty else []
                
                # For each warehouse, check if there are environmental issues
                for warehouse_id in warehouses:
                    # Filter data by warehouse
                    warehouse_temp_data = sku_temp_data[sku_temp_data['warehous_id'] == warehouse_id]
                    
                    if warehouse_temp_data.empty:
                        continue
                    
                    # Get max temperature deviation for this warehouse
                    max_temp_deviation = warehouse_temp_data['temp_deviation'].max()
                    
                    # Calculate shelf life reduction percentage
                    shelf_life_reduction = ((shelf_life_info['original'] - shelf_life_info['adjusted']) / 
                                           shelf_life_info['original'] * 100) if shelf_life_info['original'] > 0 else 0
                    
                    # Check for issues
                    has_issues = (
                        critical_alert_rate > 10.0 or
                        temp_compliance_rate < 80.0 or
                        quality_degradation > 5.0 or
                        shelf_life_reduction > 10.0 or
                        abs(max_temp_deviation) > config.TEMP_DEVIATION_ALERT
                    )
                    
                    if has_issues:
                        issues.append({
                            'SKU_ID': sku_id,
                            'warehouse_id': warehouse_id,
                            'critical_alert_rate': critical_alert_rate,
                            'temp_compliance_rate': temp_compliance_rate,
                            'quality_degradation': quality_degradation,
                            'max_temp_deviation': max_temp_deviation,
                            'shelf_life_reduction': shelf_life_reduction
                        })
            
            return pd.DataFrame(issues)
        except Exception as e:
            print(f"Error identifying environmental issues: {e}")
            return pd.DataFrame()
    
    def generate_alerts(self):
        """
        Generate alerts for environmental issues.
        
        Returns:
            List of ProductAlert objects
        """
        try:
            # Get environmental issues
            issues = self.identify_environmental_issues()
            
            if issues.empty:
                return []
            
            # Initialize list to store alerts
            alerts = []
            
            for _, issue in issues.iterrows():
                # Determine alert type and severity
                if issue['max_temp_deviation'] > config.TEMP_DEVIATION_ALERT:
                    alert_type = "TEMPERATURE"
                    severity = "HIGH" if abs(issue['max_temp_deviation']) > 5.0 else "MEDIUM"
                    message = f"Temperature deviation of {issue['max_temp_deviation']:.1f}°C detected."
                    action = "Adjust storage temperature settings and verify refrigeration equipment."
                elif issue['shelf_life_reduction'] > 10.0:
                    alert_type = "SHELF_LIFE"
                    severity = "HIGH" if issue['shelf_life_reduction'] > 20.0 else "MEDIUM"
                    message = f"Environmental conditions reducing shelf life by {issue['shelf_life_reduction']:.1f}%."
                    action = "Review storage conditions and reduce temperature fluctuations."
                elif issue['temp_compliance_rate'] < 80.0:
                    alert_type = "COMPLIANCE"
                    severity = "MEDIUM"
                    message = f"Temperature compliance rate is low at {issue['temp_compliance_rate']:.1f}%."
                    action = "Check temperature control systems and calibrate sensors."
                elif issue['quality_degradation'] > 5.0:
                    alert_type = "QUALITY"
                    severity = "MEDIUM"
                    message = f"Quality degradation index is elevated at {issue['quality_degradation']:.1f}."
                    action = "Implement more frequent quality checks and improve environmental stability."
                else:
                    alert_type = "GENERAL"
                    severity = "LOW"
                    message = "Potential environmental optimization opportunity."
                    action = "Monitor environmental conditions and adjust as needed."
                
                # Create alert
                alert = ProductAlert(
                    SKU_ID=issue['SKU_ID'],
                    warehouse_id=issue['warehouse_id'],
                    alert_type=alert_type,
                    severity=severity,
                    message=message,
                    timestamp=datetime.now(),
                    recommended_action=action
                )
                
                alerts.append(alert)
            
            return alerts
        except Exception as e:
            print(f"Error generating environmental alerts: {e}")
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
            if not self.llm:
                return "LLM analysis is not available."
            
            # Get product data
            product_info = self.product_data[self.product_data['SKU_ID'] == sku_id]
            if product_info.empty:
                return f"No product information found for SKU {sku_id}."
            
            # Get IoT data
            iot_data = self.iot_data[self.iot_data['SKU_ID'] == sku_id]
            if iot_data.empty:
                return f"No IoT sensor data found for SKU {sku_id}."
            
            # Calculate metrics
            metrics = {
                'critical_alert_rate': self.calculate_critical_alert_rate(sku_id),
                'temp_compliance_rate': self.calculate_temp_compliance_rate(sku_id),
                'quality_degradation': self.calculate_quality_degradation_index(sku_id),
                'shelf_life_decay': self.calculate_shelf_life_decay(sku_id)
            }
            
            # Format metrics as string
            metrics_str = "\n".join([f"{k}: {v}" for k, v in metrics.items()])
            
            # Run analysis
            result = self.analysis_chain.run(
                sku_id=sku_id,
                product_data=product_info.to_string(),
                iot_data=iot_data.to_string(),
                environmental_metrics=metrics_str
            )
            
            return result
        except Exception as e:
            return f"Error generating environmental analysis: {e}"