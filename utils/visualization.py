import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def plot_expiry_distribution(alerts_df):
    """
    Create a distribution chart of days remaining before expiry.
    
    Args:
        alerts_df: DataFrame containing alert data with days_remaining column
    
    Returns:
        Plotly figure showing expiry distribution
    """
    if 'days_remaining' not in alerts_df.columns:
        # Create a sample chart with a message if data is missing
        fig = go.Figure()
        fig.add_annotation(
            text="Days remaining data not available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
        
    # Create a histogram of days remaining
    fig = px.histogram(
        alerts_df, 
        x='days_remaining',
        nbins=10,
        color='severity',
        title="Distribution of Days Remaining Before Expiry",
        labels={'days_remaining': 'Days Remaining', 'count': 'Number of Products'},
        color_discrete_map={'HIGH': '#EF4444', 'MEDIUM': '#F59E0B', 'LOW': '#10B981'}
    )
    
    fig.update_layout(
        xaxis_title="Days Remaining Before Expiry",
        yaxis_title="Number of Products",
        legend_title="Severity"
    )
    
    return fig

def plot_waste_percentage(alerts_df):
    """
    Create a bar chart of waste percentage by product.
    
    Args:
        alerts_df: DataFrame containing alert data with waste_percentage column
    
    Returns:
        Plotly figure showing waste percentage by product
    """
    if 'waste_percentage' not in alerts_df.columns or 'SKU_ID' not in alerts_df.columns:
        # Create a sample chart with a message if data is missing
        fig = go.Figure()
        fig.add_annotation(
            text="Waste percentage data not available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
        
    # Get top 10 products by waste percentage
    top_waste = alerts_df.sort_values('waste_percentage', ascending=False).head(10)
    
    # Create a bar chart
    fig = px.bar(
        top_waste,
        x='SKU_ID',
        y='waste_percentage',
        color='waste_percentage',
        color_continuous_scale=px.colors.sequential.Reds,
        title="Top 10 Products by Waste Percentage",
        labels={'waste_percentage': 'Waste %', 'SKU_ID': 'Product SKU'}
    )
    
    fig.update_layout(
        xaxis_title="Product SKU",
        yaxis_title="Waste Percentage (%)",
        xaxis={'categoryorder':'total descending'}
    )
    
    return fig

def plot_temp_compliance_rate(alerts_df):
    """
    Create a gauge chart showing temperature compliance rate.
    
    Args:
        alerts_df: DataFrame containing alert data
    
    Returns:
        Plotly figure showing temperature compliance rate
    """
    # Calculate compliance rate (this is a placeholder - in real application, 
    # you would extract this from your data or calculate it)
    # For this example, we'll assume alerts with severity != 'HIGH' are compliant
    if 'severity' not in alerts_df.columns:
        compliance_rate = 0
    else:
        compliant = alerts_df[alerts_df['severity'] != 'HIGH'].shape[0]
        total = alerts_df.shape[0]
        compliance_rate = (compliant / total * 100) if total > 0 else 0
    
    # Create gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=compliance_rate,
        title={'text': "Temperature Compliance Rate (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#10B981"},
            'steps': [
                {'range': [0, 50], 'color': "#EF4444"},
                {'range': [50, 80], 'color': "#F59E0B"},
                {'range': [80, 100], 'color': "#10B981"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    return fig

def plot_demand_forecast(forecast_data):
    """
    Create a line chart showing demand forecast over time.
    
    Args:
        forecast_data: List of DemandForecast objects
    
    Returns:
        Plotly figure showing demand forecast
    """
    if not forecast_data:
        # Create a sample chart with a message if data is missing
        fig = go.Figure()
        fig.add_annotation(
            text="Forecast data not available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    try:
        # Convert forecast data to DataFrame
        dates = [f.forecast_date for f in forecast_data]
        quantities = [f.quantity for f in forecast_data]
        confidence = [f.confidence for f in forecast_data]
        
        # Calculate confidence intervals
        upper_bound = [q + (1.96 * (1 - c) * q) for q, c in zip(quantities, confidence)]
        lower_bound = [max(0, q - (1.96 * (1 - c) * q)) for q, c in zip(quantities, confidence)]
        
        # Create DataFrame
        df = pd.DataFrame({
            'date': dates,
            'quantity': quantities,
            'upper_bound': upper_bound,
            'lower_bound': lower_bound
        })
        
        # Create line chart
        fig = go.Figure()
        
        # Add main forecast line
        fig.add_trace(go.Scatter(
            x=df['date'], 
            y=df['quantity'],
            mode='lines+markers',
            name='Forecast',
            line=dict(color='#4299E1', width=2)
        ))
        
        # Add confidence interval as shaded area
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['upper_bound'],
            mode='lines',
            line=dict(width=0),
            name='Upper Bound',
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['lower_bound'],
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(66, 153, 225, 0.2)',
            name='Confidence Interval'
        ))
        
        fig.update_layout(
            title="Demand Forecast",
            xaxis_title="Date",
            yaxis_title=f"Quantity ({forecast_data[0].unit if forecast_data else 'units'})",
            hovermode="x unified"
        )
        
        return fig
    except Exception as e:
        # Create error chart
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating forecast chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig