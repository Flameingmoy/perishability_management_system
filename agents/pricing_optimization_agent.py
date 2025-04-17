# agents/pricing_optimization_agent.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import warnings
warnings.filterwarnings("ignore")

from models.data_models import ProductAlert, PriceRecommendation
import config

class PricingOptimizationAgent:
    """
    Agent responsible for optimizing pricing of perishable products to reduce waste.
    Calculates optimal discount levels and recommends timing for marking down products.
    """
    
    def __init__(self, product_data, time_series_data, price_data=None, demand_prediction_agent=None):
        """
        Initialize the pricing optimization agent.
        
        Args:
            product_data: DataFrame containing product master data
            time_series_data: DataFrame containing time series perishability data
            price_data: Optional DataFrame containing price elasticity data
            demand_prediction_agent: Optional DemandPredictionAgent for demand forecasts
        """
        self.product_data = product_data
        self.time_series_data = time_series_data
        self.price_data = price_data
        self.demand_prediction_agent = demand_prediction_agent
        self.llm = OllamaLLM(model=config.OLLAMA_MODEL)
        
        # Default price elasticity by product category if specific data not available
        self.default_elasticity = {
            'Dairy': -1.5,    # More elastic
            'Fresh': -2.0,    # Very elastic
            'Frozen': -1.2,   # Less elastic
            'Bakery': -2.2,   # Very elastic
            'Produce': -1.8,  # More elastic
            'Meat': -1.3,     # Less elastic
            'Seafood': -1.6,  # More elastic
            'Deli': -1.4      # Moderate elasticity
        }
        
        # Progressive discount strategy templates by remaining shelf life
        self.discount_templates = {
            # days_remaining: [(min_discount, max_discount), ...]
            1: [(30, 70)],                           # 1 day left - large discount range
            2: [(20, 50)],                           # 2 days left
            3: [(15, 35)],                           # 3 days left
            4: [(10, 25)],                           # 4 days left
            5: [(5, 15)],                            # 5 days left
            6: [(0, 10)],                            # 6 days left
            7: [(0, 5)]                              # 7 days left
        }
        
        # Initialize prompt template for price recommendation analysis
        self.price_analysis_template = PromptTemplate(
            input_variables=["product_info", "inventory_status", "price_elasticity", "discount_recommendation", "expected_impact"],
            template="""
            Based on the following information about a perishable product:
            
            Product Info:
            {product_info}
            
            Current Inventory Status:
            {inventory_status}
            
            Price Elasticity Information:
            {price_elasticity}
            
            Discount Recommendation:
            {discount_recommendation}
            
            Expected Impact:
            {expected_impact}
            
            Please provide:
            1. An analysis of the recommended pricing strategy
            2. The rationale behind the discount level
            3. The expected impact on sales velocity and waste reduction
            4. Any additional considerations for implementation
            
            Your response should be concise, actionable, and focused on maximizing value recovery.
            """
        )
        
        self.price_analysis_chain = LLMChain(llm=self.llm, prompt=self.price_analysis_template)
    
    def get_price_elasticity(self, sku_id):
        """
        Get price elasticity for a specific SKU.
        
        Args:
            sku_id: The SKU ID to get price elasticity for
            
        Returns:
            Price elasticity value (negative number)
        """
        # If price data is available, look up elasticity
        if self.price_data is not None and 'price_elasticity' in self.price_data.columns:
            sku_price_data = self.price_data[self.price_data['SKU_ID'] == sku_id]
            if not sku_price_data.empty and not pd.isna(sku_price_data.iloc[0]['price_elasticity']):
                return sku_price_data.iloc[0]['price_elasticity']
        
        # Otherwise, use default elasticity based on product category
        product_info = self.product_data[self.product_data['SKU_ID'] == sku_id]
        if not product_info.empty and 'Product_Category' in product_info.columns:
            category = product_info.iloc[0]['Product_Category']
            if category in self.default_elasticity:
                return self.default_elasticity[category]
        
        # Default value if category not found
        return -1.5  # Moderate elasticity as fallback
    
    def calculate_optimal_discount(self, sku_id, warehouse_id, base_price=None):
        """
        Calculate the optimal discount for a product based on days remaining,
        price elasticity, and waste risk.
        
        Args:
            sku_id: The SKU ID to calculate discount for
            warehouse_id: The warehouse ID
            base_price: Optional base price (if not provided, will be set to 1.0 and discount percentage returned)
            
        Returns:
            Dictionary containing discount recommendation
        """
        # Get current inventory status
        ts_data = self.time_series_data[
            (self.time_series_data['SKU_id'] == sku_id) & 
            (self.time_series_data['warehous_id'] == warehouse_id)
        ]
        
        if ts_data.empty:
            return None
            
        # Get latest inventory record
        latest = ts_data.iloc[0]
        days_remaining = latest['days_remaining']
        
        # Get price elasticity
        price_elasticity = self.get_price_elasticity(sku_id)
        
        # Determine appropriate discount range based on days remaining
        if days_remaining in self.discount_templates:
            discount_ranges = self.discount_templates[days_remaining]
        elif days_remaining <= 0:
            # Already expired, maximum discount
            discount_ranges = [(70, 90)]
        elif days_remaining < min(self.discount_templates.keys()):
            # Use the smallest days_remaining template
            discount_ranges = self.discount_templates[min(self.discount_templates.keys())]
        elif days_remaining > max(self.discount_templates.keys()):
            # No discount needed yet
            return {
                "sku_id": sku_id,
                "warehouse_id": warehouse_id,
                "discount_percentage": 0,
                "days_remaining": days_remaining,
                "base_price": base_price if base_price else 1.0,
                "recommended_price": base_price if base_price else 1.0,
                "expected_sales_lift": 0
            }
        
        # Check if we have demand prediction agent to help with forecasting
        if self.demand_prediction_agent:
            # If demand prediction agent is available, check if current inventory
            # exceeds forecasted demand for the remaining shelf life
            forecasts = self.demand_prediction_agent.forecast_demand(
                sku_id, warehouse_id, days_ahead=days_remaining
            )
            
            if forecasts:
                total_forecasted_demand = sum(f.quantity for f in forecasts)
                current_stock = latest['current_stock']
                
                # Calculate excess inventory ratio
                excess_ratio = max(0, (current_stock - total_forecasted_demand) / current_stock) if current_stock > 0 else 0
                
                # Adjust discount based on excess ratio
                if excess_ratio > 0:
                    min_discount, max_discount = discount_ranges[0]
                    
                    # Scale discount within range based on excess ratio
                    # Higher excess ratio means higher discount within the range
                    discount_percentage = min_discount + excess_ratio * (max_discount - min_discount)
                    
                    # Ensure discount is within the range
                    discount_percentage = max(min_discount, min(max_discount, discount_percentage))
                else:
                    # No excess inventory
                    discount_percentage = 0
            else:
                # No forecast available, use middle of range
                min_discount, max_discount = discount_ranges[0]
                discount_percentage = (min_discount + max_discount) / 2
        else:
            # No demand prediction agent available, use middle of range
            min_discount, max_discount = discount_ranges[0]
            discount_percentage = (min_discount + max_discount) / 2
        
        # Calculate expected sales lift based on price elasticity
        # Sales increase = Price elasticity * Price decrease percentage
        expected_sales_lift = abs(price_elasticity) * (discount_percentage / 100)
        
        # Calculate recommended price
        if base_price:
            recommended_price = base_price * (1 - discount_percentage / 100)
        else:
            recommended_price = 1.0 * (1 - discount_percentage / 100)
            base_price = 1.0
        
        return {
            "sku_id": sku_id,
            "warehouse_id": warehouse_id,
            "discount_percentage": discount_percentage,
            "days_remaining": days_remaining,
            "base_price": base_price,
            "recommended_price": recommended_price,
            "expected_sales_lift": expected_sales_lift
        }
    
    def get_progressive_discount_strategy(self, sku_id, warehouse_id, base_price=None):
        """
        Generate a progressive discount strategy for a product based on its
        remaining shelf life.
        
        Args:
            sku_id: The SKU ID to create strategy for
            warehouse_id: The warehouse ID
            base_price: Optional base price (if not provided, will return percentages)
            
        Returns:
            Dictionary containing progressive discount strategy
        """
        # Get current inventory status
        ts_data = self.time_series_data[
            (self.time_series_data['SKU_id'] == sku_id) & 
            (self.time_series_data['warehous_id'] == warehouse_id)
        ]
        
        if ts_data.empty:
            return None
            
        # Get latest inventory record
        latest = ts_data.iloc[0]
        current_days_remaining = latest['days_remaining']
        
        if current_days_remaining <= 0:
            return None  # Already expired
        
        # Generate discount recommendations for each remaining day
        strategy = []
        for day in range(int(current_days_remaining), 0, -1):
            # Create a temporary record with modified days_remaining
            temp_record = latest.copy()
            temp_record['days_remaining'] = day
            
            # Calculate discount for this day
            discount_info = self.calculate_optimal_discount(sku_id, warehouse_id, base_price)
            
            if discount_info:
                # Add date to the strategy
                discount_date = datetime.now() + timedelta(days=current_days_remaining - day)
                discount_info['date'] = discount_date
                strategy.append(discount_info)
        
        return {
            "sku_id": sku_id,
            "warehouse_id": warehouse_id,
            "current_days_remaining": current_days_remaining,
            "strategy": strategy
        }
    
    def generate_price_recommendations(self):
        """
        Generate price recommendations for all products that need discounting.
        
        Returns:
            List of PriceRecommendation objects
        """
        recommendations = []
        
        # Get unique SKU and warehouse combinations
        unique_combinations = self.time_series_data[['SKU_id', 'warehous_id']].drop_duplicates()
        
        for _, row in unique_combinations.iterrows():
            sku_id = row['SKU_id']
            warehouse_id = row['warehous_id']
            
            # Get current inventory status
            ts_data = self.time_series_data[
                (self.time_series_data['SKU_id'] == sku_id) & 
                (self.time_series_data['warehous_id'] == warehouse_id)
            ]
            
            if ts_data.empty:
                continue
                
            # Get latest record
            latest = ts_data.iloc[0]
            days_remaining = latest['days_remaining']
            
            # Only recommend discounts for products with short shelf life remaining
            if days_remaining <= 7:  # Threshold for considering discount
                # Get base price if available, otherwise use 1.0
                base_price = 1.0
                if self.price_data is not None and 'base_price' in self.price_data.columns:
                    price_data = self.price_data[self.price_data['SKU_ID'] == sku_id]
                    if not price_data.empty and not pd.isna(price_data.iloc[0]['base_price']):
                        base_price = price_data.iloc[0]['base_price']
                
                # Calculate optimal discount
                discount_info = self.calculate_optimal_discount(sku_id, warehouse_id, base_price)
                
                if discount_info and discount_info['discount_percentage'] > 0:
                    # Create expiry date
                    expiry_date = datetime.now() + timedelta(days=days_remaining)
                    
                    # Create reasoning
                    if self.demand_prediction_agent:
                        forecasts = self.demand_prediction_agent.forecast_demand(
                            sku_id, warehouse_id, days_ahead=days_remaining
                        )
                        
                        if forecasts:
                            total_forecasted_demand = sum(f.quantity for f in forecasts)
                            current_stock = latest['current_stock']
                            
                            if current_stock > total_forecasted_demand:
                                excess = current_stock - total_forecasted_demand
                                reasoning = (
                                    f"Current stock ({current_stock:.1f} {latest['unit_curr_stock']}) exceeds " +
                                    f"forecasted demand ({total_forecasted_demand:.1f} {latest['unit_curr_stock']}) " +
                                    f"by {excess:.1f} {latest['unit_curr_stock']} before expiry. " +
                                    f"A {discount_info['discount_percentage']:.1f}% discount is recommended to " +
                                    f"accelerate sales and prevent waste."
                                )
                            else:
                                reasoning = (
                                    f"Product has {days_remaining} days remaining until expiry. " +
                                    f"A {discount_info['discount_percentage']:.1f}% discount is recommended based on " +
                                    f"standard progressive markdown strategy."
                                )
                        else:
                            reasoning = (
                                f"Product has {days_remaining} days remaining until expiry. " +
                                f"A {discount_info['discount_percentage']:.1f}% discount is recommended based on " +
                                f"standard progressive markdown strategy."
                            )
                    else:
                        reasoning = (
                            f"Product has {days_remaining} days remaining until expiry. " +
                            f"A {discount_info['discount_percentage']:.1f}% discount is recommended based on " +
                            f"standard progressive markdown strategy."
                        )
                    
                    # Create price recommendation
                    recommendation = PriceRecommendation(
                        SKU_ID=sku_id,
                        warehouse_id=warehouse_id,
                        current_price=base_price,
                        recommended_price=discount_info['recommended_price'],
                        discount_percentage=discount_info['discount_percentage'],
                        expected_sales_lift=discount_info['expected_sales_lift'],
                        expiry_date=expiry_date,
                        timestamp=datetime.now(),
                        reasoning=reasoning
                    )
                    
                    recommendations.append(recommendation)
        
        return recommendations
    
    def generate_alerts(self):
        """
        Generate alerts for products that need immediate price adjustment.
        
        Returns:
            List of ProductAlert objects
        """
        price_recommendations = self.generate_price_recommendations()
        alerts = []
        
        for rec in price_recommendations:
            # Determine alert severity based on discount percentage and days until expiry
            expiry_delta = (rec.expiry_date - datetime.now()).days
            
            severity = "LOW"
            if expiry_delta <= 2 or rec.discount_percentage >= 50:
                severity = "HIGH"
            elif expiry_delta <= 4 or rec.discount_percentage >= 25:
                severity = "MEDIUM"
            
            # Create message
            message = (
                f"Price adjustment recommended: {rec.discount_percentage:.1f}% discount " +
                f"(from {rec.current_price:.2f} to {rec.recommended_price:.2f}). " +
                f"Expected sales increase: {rec.expected_sales_lift * 100:.1f}%. " +
                f"Product expires in {expiry_delta} days."
            )
            
            # Determine recommended action
            if severity == "HIGH":
                recommended_action = "Implement price reduction immediately"
            elif severity == "MEDIUM":
                recommended_action = "Schedule price reduction within 24 hours"
            else:
                recommended_action = "Consider price reduction within 48 hours"
            
            alert = ProductAlert(
                SKU_ID=rec.SKU_ID,
                warehouse_id=rec.warehouse_id,
                alert_type="PRICE_ADJUSTMENT",
                severity=severity,
                message=message,
                timestamp=datetime.now(),
                recommended_action=recommended_action
            )
            
            alerts.append(alert)
        
        return alerts
    
    def get_price_analysis(self, sku_id, warehouse_id):
        """
        Get LLM analysis of pricing recommendations for a specific SKU and warehouse.
        
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
        
        # Get current inventory status
        inventory_status = self.time_series_data[
            (self.time_series_data['SKU_id'] == sku_id) & 
            (self.time_series_data['warehous_id'] == warehouse_id)
        ]
        
        if inventory_status.empty:
            return "Inventory data not found."
            
        inventory_status = inventory_status.iloc[0].to_dict()
        
        # Get price elasticity
        price_elasticity = self.get_price_elasticity(sku_id)
        price_elasticity_text = (
            f"Price elasticity: {price_elasticity}. " +
            f"A 10% price reduction is expected to increase sales by {abs(price_elasticity) * 10:.1f}%."
        )
        
        # Get discount recommendation
        discount_info = self.calculate_optimal_discount(sku_id, warehouse_id)
        
        if not discount_info:
            return "Unable to generate discount recommendation."
            
        discount_text = (
            f"Recommended discount: {discount_info['discount_percentage']:.1f}%. " +
            f"Original price: {discount_info['base_price']:.2f}, " +
            f"Recommended price: {discount_info['recommended_price']:.2f}."
        )
        
        # Calculate expected impact
        days_remaining = inventory_status['days_remaining']
        current_stock = inventory_status['current_stock']
        unit = inventory_status['unit_curr_stock']
        
        # Get additional impact info from demand prediction agent if available
        if self.demand_prediction_agent:
            forecasts = self.demand_prediction_agent.forecast_demand(
                sku_id, warehouse_id, days_ahead=int(days_remaining)
            )
            
            if forecasts:
                total_forecasted_demand = sum(f.quantity for f in forecasts)
                
                # Calculate expected demand with discount
                expected_demand_with_discount = total_forecasted_demand * (1 + discount_info['expected_sales_lift'])
                
                # Calculate waste reduction
                potential_waste = max(0, current_stock - total_forecasted_demand)
                expected_waste_with_discount = max(0, current_stock - expected_demand_with_discount)
                waste_reduction = potential_waste - expected_waste_with_discount
                
                impact_text = (
                    f"Current stock: {current_stock:.1f} {unit}. " +
                    f"Forecasted demand without discount: {total_forecasted_demand:.1f} {unit}. " +
                    f"Expected demand with {discount_info['discount_percentage']:.1f}% discount: {expected_demand_with_discount:.1f} {unit}. " +
                    f"Potential waste reduction: {waste_reduction:.1f} {unit}."
                )
            else:
                # Simple impact calculation without demand forecast
                expected_sales_increase = discount_info['expected_sales_lift'] * 100
                impact_text = (
                    f"Current stock: {current_stock:.1f} {unit}. " +
                    f"Days until expiry: {days_remaining:.1f}. " +
                    f"Expected sales increase with {discount_info['discount_percentage']:.1f}% discount: {expected_sales_increase:.1f}%."
                )
        else:
            # Simple impact calculation without demand forecast
            expected_sales_increase = discount_info['expected_sales_lift'] * 100
            impact_text = (
                f"Current stock: {current_stock:.1f} {unit}. " +
                f"Days until expiry: {days_remaining:.1f}. " +
                f"Expected sales increase with {discount_info['discount_percentage']:.1f}% discount: {expected_sales_increase:.1f}%."
            )
        
        # Get analysis from LLM
        analysis = self.price_analysis_chain.run(
            product_info=str(product_info),
            inventory_status=str(inventory_status),
            price_elasticity=price_elasticity_text,
            discount_recommendation=discount_text,
            expected_impact=impact_text
        )
        
        return analysis