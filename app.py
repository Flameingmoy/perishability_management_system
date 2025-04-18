import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import io
import numpy as np
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Import our own agent implementations
from agents.inventory_monitoring_agent import InventoryMonitoringAgent
from agents.environmental_monitoring_agent import EnvironmentalMonitoringAgent
from agents.demand_prediction_agent import DemandPredictionAgent

# Import utilities
from utils.data_loader import (
    load_product_master_data,
    load_time_series_data,
    load_iot_data,
    load_supplier_data,
    load_sales_data
)
from utils.visualization import (
    plot_expiry_distribution, 
    plot_waste_percentage,
    plot_temp_compliance_rate,
    plot_demand_forecast
)

import config

# Page configuration
st.set_page_config(
    page_title="Perishable Goods Analytics",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define color scheme based on style guide
PRIMARY_COLOR = "#2D3748"  # slate blue
SECONDARY_COLOR = "#4A5568"  # grey blue
BACKGROUND_COLOR = "#F7FAFC"  # off-white
TEXT_COLOR = "#1A202C"  # dark grey
ACCENT_COLOR = "#4299E1"  # bright blue

# Apply custom CSS for styling
st.markdown(f"""
<style>
    .main {{
        background-color: {BACKGROUND_COLOR};
        color: {TEXT_COLOR};
    }}
    .stButton>button {{
        background-color: {PRIMARY_COLOR};
        color: white;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 16px;
    }}
    .stTabs [data-baseweb="tab"] {{
        height: 50px;
        white-space: pre-wrap;
        background-color: {SECONDARY_COLOR};
        color: white;
        border-radius: 4px 4px 0 0;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {ACCENT_COLOR};
    }}
    .metric-card {{
        background-color: white;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
        margin-bottom: 16px;
    }}
    .chat-container {{
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 16px;
        height: 400px;
        overflow-y: auto;
        margin-bottom: 16px;
        background-color: white;
    }}
    .user-message {{
        background-color: {ACCENT_COLOR};
        color: white;
        border-radius: 16px 16px 0 16px;
        padding: 8px 12px;
        margin: 8px 0;
        max-width: 70%;
        margin-left: auto;
    }}
    .bot-message {{
        background-color: #f0f0f0;
        border-radius: 16px 16px 16px 0;
        padding: 8px 12px;
        margin: 8px 0;
        max-width: 70%;
    }}
</style>
""", unsafe_allow_html=True)

# Initialize session state variables if they don't exist
if 'inventory_alerts' not in st.session_state:
    st.session_state.inventory_alerts = None
if 'environmental_alerts' not in st.session_state:
    st.session_state.environmental_alerts = None
if 'demand_alerts' not in st.session_state:
    st.session_state.demand_alerts = None
if 'selected_sku' not in st.session_state:
    st.session_state.selected_sku = None
if 'selected_warehouse' not in st.session_state:
    st.session_state.selected_warehouse = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
if 'export_files' not in st.session_state:
    st.session_state.export_files = []
if 'scm_knowledge_base' not in st.session_state:
    st.session_state.scm_knowledge_base = None
if 'ollama_chain' not in st.session_state:
    st.session_state.ollama_chain = None

def init_scm_knowledge_base():
    """Initialize the SCM knowledge base from the SCM Scenarios PDF"""
    if st.session_state.scm_knowledge_base is None:
        try:
            # Load SCM scenarios content
            scm_text = """
            SCM Scenarios

            Logic & Reason: If raw materials are unavailable, alternative procurement strategies are needed.

            Batch Size Efficiency
            Definition: Large Batch / Small Batch - Optimal batch size for manufacturing.
            Logic & Reason: Large batches reduce cost per unit, while small batches reduce waste.

            Machine Utilization
            Definition: High / Low - How efficiently manufacturing machines are being used.
            Logic & Reason: Low utilization may indicate an opportunity to optimize production scheduling.

            Outsourcing Feasibility
            Definition: Yes / No - Whether the product can be outsourced instead of inhouse production.
            Logic & Reason: If inhouse production is inefficient or costly, outsourcing is an option.

            Bundle Offer Available
            Definition: Yes / No - If the item can be bundled with another product to boost sales.
            Logic & Reason: Helps move slowmoving stock by combining it with popular items.

            Marketing Push Needed
            Definition: Yes / No - Whether a marketing campaign is needed.
            Logic & Reason: If sales are low, promotions, social media ads, or discounts may be required.

            Seasonal Demand
            Definition: Winter / Summer / Festive / No Effect - Determines if demand fluctuates by season.
            Logic & Reason: Ensures stock availability during peak demand periods.

            Competitor Price Influence
            Definition: Lower / Higher / Same - Compares pricing with competitors.
            Logic & Reason: If competitors sell cheaper, pricing adjustments or promotions may be needed.

            Customer Reviews
            Definition: Good / Average / Poor - Customer feedback rating.
            Logic & Reason: Helps understand customer perception and improve product quality.

            Decision Logic

            If stock is not available and the item is fastmoving:
             Supplier On Time: Move from another store (if available).
             Supplier Delayed: Raise PO if critical or move from a store.
             If stock is available: No action is required.
             Items with "Move from Another Store": Do not have a PO raised date.
             Supplier Lead Time: Considered for PO orders to calculate Expected Delivery Date.
             Supplier Delay: Lead times may extend (not accounted for in this default table).
             Supplier Codes: Help track which supplier is responsible for replenishment.
             Delayed suppliers: Affect the recommendation—either moving stock from another store or raising a PO.
             Expected Delivery Date: PO Raised Date + Supplier Lead Time (applicable only if PO is raised).

             If stock is not available and the item is slowmoving:
             Eligible for discount: Yes if sales have not increased.
             Sales increase: Yes → No discount needed.
             If sales haven't increased: No → Discount is recommended to boost sales.
             If sales did not increase after applying a discount: Further discount is suggested (Yes).
             If sales still don't increase after additional discounts: Product is flagged for discontinuation (Yes).
             Fastmoving items: Not considered for discontinuation.
             Alternative Product Available: If another product can replace this item, it reduces dependency on slowmoving or discontinued products.
             Excess Stock Flag: If a store has too much stock, consider redistribution instead of ordering more.
             Supplier Reliability: Helps in deciding whether to switch suppliers based on past performance.
             Stock Transfer Needed: Instead of raising a PO, move excess stock from another store with overstock.
             Customer Complaints: If a product has high complaints, discontinuation should be prioritized.
             Alternative Product Available: If an alternative exists, avoid raising a PO and consider switching.
             Excess Stock: If another store has excess stock, transfer it instead of ordering more.

            Additional Manufacturing & Sales Strategies

             ManufacturingRelated Enhancements
            1. Production Lead Time (Days): How long it takes to manufacture the product if it's produced inhouse.
            2. Raw Material Availability: Available / Shortage - Whether raw materials are available to produce the item.
            3. Batch Size Efficiency: Large Batch / Small Batch - Whether producing in large or small batches optimizes cost and efficiency.
            4. Machine Utilization: High / Low - If manufacturing machines are underutilized, consider shifting production.
            5. Outsourcing Feasibility: Yes / No - If production is slow, can this item be outsourced to a supplier instead?

             Sales & Marketing Strategies for Increasing Demand
            1. Bundle Offer Available: Yes / No - Whether the item can be sold as a combo deal to boost sales.
            2. Marketing Push Needed: Yes / No - If sales are low, consider promoting it through discounts, ads, or campaigns.
            3. Seasonal Demand: Winter / Summer / Festive / No Effect - If demand varies by season, production should align.
            4. Competitor Price Influence: Lower / Higher / Same - If competitors sell cheaper, adjust pricing accordingly.
            5. Customer Reviews: Good / Average / Poor - Positive reviews increase trust and sales, while negative ones indicate a need for quality improvements.

             How These Insights Help
             Better Inventory Planning: Avoid unnecessary purchases by tracking stock transfers and excess inventory.
             Optimized Manufacturing: Consider outsourcing, batch size adjustments, or machine utilization improvements.
             Boosting Sales: Use discounts, marketing, bundling, and seasonal strategies to improve product movement.
             Better Supplier Management: Identify reliable suppliers and avoid those with frequent delays.
             CustomerCentric Approach: Stop products with high complaints and invest in improving product quality.

            Improved DecisionMaking
            If Total Stock is high but Sold is low: Apply discounts or consider stopping the product.
            If Available Stock is 0: Check if it's fastmoving and raise a PO or transfer stock.
            If Supplier Delay is frequent: Consider alternative suppliers.
            """
            
            # Create text splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            
            # Split text into chunks
            chunks = text_splitter.split_text(scm_text)
            
            # Create embeddings
            embeddings = OllamaEmbeddings(model=config.OLLAMA_EMBEDDING_MODEL)
            
            # Create vector store
            vectorstore = FAISS.from_texts(chunks, embeddings)
            
            # Create retriever
            st.session_state.scm_knowledge_base = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            )
            
            # Initialize Ollama
            llm = OllamaLLM(model=config.OLLAMA_MODEL)
            
            # Create a QA chain
            st.session_state.ollama_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=st.session_state.scm_knowledge_base,
                return_source_documents=True
            )
            
        except Exception as e:
            st.error(f"Error initializing SCM knowledge base: {e}")

def load_data():
    """Load all required data and return initialized agents"""
    with st.spinner("Loading data..."):
        try:
            product_data = load_product_master_data(config.PRODUCT_MASTER_FILE)
            time_series_data = load_time_series_data(config.TIME_SERIES_FILE)
            iot_data = load_iot_data(config.IOT_DATA_FILE)
            supplier_data = load_supplier_data(config.SUPPLIER_DATA_FILE)
            sales_data = load_sales_data(config.SALES_DATA_FILE) if hasattr(config, 'SALES_DATA_FILE') else None
            
            # Initialize agents
            inventory_agent = InventoryMonitoringAgent(product_data, time_series_data)
            environment_agent = EnvironmentalMonitoringAgent(product_data, time_series_data, iot_data)
            demand_agent = DemandPredictionAgent(product_data, time_series_data, sales_data)
            
            return {
                "product_data": product_data,
                "time_series_data": time_series_data,
                "iot_data": iot_data,
                "supplier_data": supplier_data,
                "sales_data": sales_data,
                "inventory_agent": inventory_agent,
                "environment_agent": environment_agent, 
                "demand_agent": demand_agent
            }
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None

def export_data(data, name, format='csv'):
    """Export data to a file and return the filename"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{name}_{timestamp}.{format}"
    
    if format == 'csv':
        data.to_csv(filename, index=False)
    else:  # json
        data.to_json(filename, orient='records', date_format='iso')
    
    return filename

def run_analysis(data_objects, analysis_type, selected_sku, selected_warehouse, forecast_days=7):
    """Run the selected analysis and update session state with results"""
    
    if analysis_type == 'inventory' or analysis_type == 'all':
        inventory_agent = data_objects['inventory_agent']
        
        # Generate alerts
        inventory_alerts = inventory_agent.generate_alerts()
        if inventory_alerts:
            st.session_state.inventory_alerts = pd.DataFrame([alert.dict() for alert in inventory_alerts])
        
        # Generate specific analysis if SKU and warehouse are provided
        if selected_sku and selected_warehouse:
            analysis = inventory_agent.get_inventory_analysis(selected_sku, selected_warehouse)
            st.session_state.analysis_results['inventory'] = analysis
    
    if analysis_type == 'environment' or analysis_type == 'all':
        environment_agent = data_objects['environment_agent']
        
        # Generate alerts
        environmental_alerts = environment_agent.generate_alerts()
        if environmental_alerts:
            st.session_state.environmental_alerts = pd.DataFrame([alert.dict() for alert in environmental_alerts])
        
        # Generate specific analysis if SKU is provided
        if selected_sku:
            analysis = environment_agent.get_environmental_analysis(selected_sku, selected_warehouse)
            st.session_state.analysis_results['environment'] = analysis
    
    if analysis_type == 'demand' or analysis_type == 'all':
        demand_agent = data_objects['demand_agent']
        
        # Generate alerts
        demand_alerts = demand_agent.generate_alerts()
        if demand_alerts:
            st.session_state.demand_alerts = pd.DataFrame([alert.dict() for alert in demand_alerts])
        
        # Generate specific analysis if SKU and warehouse are provided
        if selected_sku and selected_warehouse:
            analysis = demand_agent.get_demand_analysis(selected_sku, selected_warehouse)
            st.session_state.analysis_results['demand'] = analysis
            
            # Generate forecast
            forecast = demand_agent.forecast_demand(selected_sku, selected_warehouse, forecast_days)
            if forecast:
                st.session_state.analysis_results['forecast'] = forecast

def get_data_summary():
    """Get summaries of current data to provide context to the chatbot"""
    inventory_summary = "No inventory data available."
    environmental_summary = "No environmental data available."
    demand_summary = "No demand data available."
    
    # Summarize inventory alerts
    if 'inventory_alerts' in st.session_state and st.session_state.inventory_alerts is not None and not st.session_state.inventory_alerts.empty:
        df = st.session_state.inventory_alerts
        high_severity = len(df[df['severity'] == 'HIGH'])
        medium_severity = len(df[df['severity'] == 'MEDIUM'])
        low_severity = len(df[df['severity'] == 'LOW'])
        unique_skus = df['SKU_ID'].nunique()
        unique_warehouses = df['warehouse_id'].nunique() if 'warehouse_id' in df.columns else 0
        
        inventory_summary = f"""
        Inventory alerts: {len(df)} total alerts
        - {high_severity} high severity alerts
        - {medium_severity} medium severity alerts
        - {low_severity} low severity alerts
        - Affecting {unique_skus} unique products across {unique_warehouses} warehouses
        """
    
    # Summarize environmental alerts
    if 'environmental_alerts' in st.session_state and st.session_state.environmental_alerts is not None and not st.session_state.environmental_alerts.empty:
        df = st.session_state.environmental_alerts
        high_severity = len(df[df['severity'] == 'HIGH'])
        unique_skus = df['SKU_ID'].nunique()
        
        environmental_summary = f"""
        Environmental alerts: {len(df)} total alerts
        - {high_severity} critical environmental alerts
        - Affecting {unique_skus} unique products
        - Most common issues: temperature deviations and humidity concerns
        """
    
    # Summarize demand alerts
    if 'demand_alerts' in st.session_state and st.session_state.demand_alerts is not None and not st.session_state.demand_alerts.empty:
        df = st.session_state.demand_alerts
        high_severity = len(df[df['severity'] == 'HIGH'])
        unique_skus = df['SKU_ID'].nunique()
        
        demand_summary = f"""
        Demand alerts: {len(df)} total alerts
        - {high_severity} critical demand alerts
        - Affecting {unique_skus} unique products
        - Issues include: supply-demand mismatches and potential overstocking
        """
    
    # Add forecast information if available
    if 'analysis_results' in st.session_state and 'forecast' in st.session_state.analysis_results:
        forecast = st.session_state.analysis_results['forecast']
        if forecast:
            forecast_days = len(forecast)
            product = forecast[0].SKU_ID
            warehouse = forecast[0].warehouse_id
            demand_summary += f"\nDetailed {forecast_days}-day forecast available for product {product} in warehouse {warehouse}."
    
    return inventory_summary, environmental_summary, demand_summary

def query_ollama(question):
    """Query the Ollama model with the user's question and knowledge base"""
    try:
        # Initialize SCM knowledge base if not already done
        init_scm_knowledge_base()
        
        # Get data summaries for context
        inventory_summary, environmental_summary, demand_summary = get_data_summary()
        
        # Format chat history
        chat_history = ""
        for message in st.session_state.chat_history[-5:]:  # Last 5 messages for context
            role = "Human" if message["role"] == "user" else "AI"
            chat_history += f"{role}: {message['content']}\n"
        
        # Create enhanced question with context
        enhanced_question = f"""
        Previous conversation:
        {chat_history}
        
        Available inventory data summary:
        {inventory_summary}
        
        Available environmental monitoring data summary:
        {environmental_summary}
        
        Available demand prediction data summary:
        {demand_summary}
        
        Human question: {question}
        
        Please provide a helpful answer based on the data and supply chain management knowledge.
        """
        
        # Query the model
        result = st.session_state.ollama_chain({"query": enhanced_question})
        
        return result["result"]
    
    except Exception as e:
        st.error(f"Error querying Ollama: {e}")
        return f"I'm sorry, I encountered an error: {str(e)}. Please try again or check if Ollama is running correctly."

def display_chat_interface():
    """Display the chat interface for interacting with the Ollama model"""
    st.markdown("""
    ### AI Assistant for Perishable Goods Analysis
    Ask questions about the analysis results, get insights about the data, or request recommendations.
    """)
    
    # Check if SCM knowledge base is initialized
    init_scm_knowledge_base()
    
    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"<div class='user-message'>{message['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='bot-message'>{message['content']}</div>", unsafe_allow_html=True)
    
    # Chat input
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Ask a question about perishable goods management:", key="user_input")
        submit_button = st.form_submit_button("Send")
        
        if submit_button and user_input:
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            # Get response from model
            with st.spinner("Thinking..."):
                response = query_ollama(user_input)
            
            # Add assistant response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            
            # Rerun to update the chat display
            st.rerun()
    
    # Option to clear chat history
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
    
    # Export chat history option
    if st.session_state.chat_history and st.button("Export Conversation"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"conversation_{timestamp}.txt"
        
        with open(filename, "w") as f:
            for message in st.session_state.chat_history:
                role = "Human" if message["role"] == "user" else "AI"
                f.write(f"{role}: {message['content']}\n\n")
        
        with open(filename, "rb") as f:
            st.download_button(
                label="Download Conversation",
                data=f,
                file_name=filename,
                mime="text/plain"
            )

def display_dashboard(data_objects):
    """Display the main dashboard interface"""
    st.title("🔄 Perishable Goods Analytics Dashboard")
    
    # Sidebar for analysis controls
    with st.sidebar:
        st.header("Analysis Controls")
        
        # Analysis type selection
        analysis_type = st.selectbox(
            "Select Analysis Type",
            options=['all', 'inventory', 'environment', 'demand'],
            index=0
        )
        
        # Product and warehouse selection
        products = data_objects['product_data']
        unique_skus = products['SKU_ID'].unique()
        selected_sku = st.selectbox("Select Product SKU", options=[''] + list(unique_skus))
        
        time_series = data_objects['time_series_data']
        unique_warehouses = time_series['warehous_id'].unique()
        selected_warehouse = st.selectbox("Select Warehouse", options=[''] + list(unique_warehouses))
        
        # Forecast days selection for demand analysis
        forecast_days = st.slider("Forecast Days", min_value=1, max_value=30, value=7)
        
        # Run analysis button
        if st.button("Run Analysis"):
            st.session_state.selected_sku = selected_sku if selected_sku else None
            st.session_state.selected_warehouse = selected_warehouse if selected_warehouse else None
            run_analysis(data_objects, analysis_type, st.session_state.selected_sku, 
                        st.session_state.selected_warehouse, forecast_days)
            st.success("Analysis complete!")
        
        # Export options
        st.header("Export Options")
        export_format = st.radio("Export Format", options=['csv', 'json'])
        
        if st.button("Export All Alerts"):
            export_files = []
            
            if st.session_state.inventory_alerts is not None and not st.session_state.inventory_alerts.empty:
                filename = export_data(st.session_state.inventory_alerts, "inventory_alerts", export_format)
                export_files.append({"name": "Inventory Alerts", "filename": filename})
                
            if st.session_state.environmental_alerts is not None and not st.session_state.environmental_alerts.empty:
                filename = export_data(st.session_state.environmental_alerts, "environmental_alerts", export_format)
                export_files.append({"name": "Environmental Alerts", "filename": filename})
                
            if st.session_state.demand_alerts is not None and not st.session_state.demand_alerts.empty:
                filename = export_data(st.session_state.demand_alerts, "demand_alerts", export_format)
                export_files.append({"name": "Demand Alerts", "filename": filename})
            
            st.session_state.export_files = export_files
            if export_files:
                st.success(f"Exported {len(export_files)} files!")
    
    # Main content area - tabs for different views
    tabs = st.tabs(["Overview", "Inventory Analysis", "Environmental Analysis", "Demand Analysis", "AI Insights"])
    
    # Overview Tab
    with tabs[0]:
        st.header("System Overview")
        
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        
        with metrics_col1:
            inventory_count = 0
            if st.session_state.inventory_alerts is not None:
                inventory_count = len(st.session_state.inventory_alerts)
            st.metric("Inventory Alerts", inventory_count)
        
        with metrics_col2:
            environmental_count = 0
            if st.session_state.environmental_alerts is not None:
                environmental_count = len(st.session_state.environmental_alerts)
            st.metric("Environmental Alerts", environmental_count)
        
        with metrics_col3:
            demand_count = 0
            if st.session_state.demand_alerts is not None:
                demand_count = len(st.session_state.demand_alerts)
            st.metric("Demand Alerts", demand_count)
        
        # Display recently exported files
        if st.session_state.export_files:
            st.subheader("Recent Exports")
            for file_info in st.session_state.export_files:
                with open(file_info["filename"], "rb") as f:
                    st.download_button(
                        label=f"Download {file_info['name']}",
                        data=f,
                        file_name=file_info["filename"],
                        mime="text/csv" if file_info["filename"].endswith("csv") else "application/json"
                    )
    
    # Inventory Analysis Tab
    with tabs[1]:
        st.header("Inventory Analysis")
        
        if st.session_state.inventory_alerts is not None and not st.session_state.inventory_alerts.empty:
            # Summary metrics
            try:
                high_severity = len(st.session_state.inventory_alerts[st.session_state.inventory_alerts['severity'] == 'HIGH'])
                medium_severity = len(st.session_state.inventory_alerts[st.session_state.inventory_alerts['severity'] == 'MEDIUM'])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("High Severity Alerts", high_severity)
                with col2:
                    st.metric("Medium Severity Alerts", medium_severity)
                
                # Expiry distribution chart
                st.subheader("Expiry Distribution")
                fig = plot_expiry_distribution(st.session_state.inventory_alerts)
                st.plotly_chart(fig, use_container_width=True)
                
                # Waste percentage by product
                st.subheader("Waste Percentage by Product")
                try:
                    if 'waste_percentage' in st.session_state.inventory_alerts.columns:
                        waste_fig = plot_waste_percentage(st.session_state.inventory_alerts)
                        st.plotly_chart(waste_fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not generate waste percentage chart: {e}")
                
                # Alerts table
                st.subheader("Inventory Alerts")
                st.dataframe(st.session_state.inventory_alerts[['SKU_ID', 'warehouse_id', 'severity', 'message', 'recommended_action']])
            except Exception as e:
                st.error(f"Error displaying inventory analysis: {e}")
        else:
            st.info("No inventory analysis data available. Run an analysis to view results.")
        
        # Show specific analysis if available
        if 'inventory' in st.session_state.analysis_results:
            st.subheader(f"Analysis for SKU: {st.session_state.selected_sku}")
            st.markdown(st.session_state.analysis_results['inventory'])
    
    # Environmental Analysis Tab
    with tabs[2]:
        st.header("Environmental Analysis")
        
        if st.session_state.environmental_alerts is not None and not st.session_state.environmental_alerts.empty:
            try:
                # Summary metrics
                high_severity = len(st.session_state.environmental_alerts[st.session_state.environmental_alerts['severity'] == 'HIGH'])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Critical Environmental Alerts", high_severity)
                with col2:
                    st.metric("Total Products Affected", st.session_state.environmental_alerts['SKU_ID'].nunique())
                
                # Temperature compliance chart
                st.subheader("Temperature Compliance Rate")
                temp_fig = plot_temp_compliance_rate(st.session_state.environmental_alerts)
                st.plotly_chart(temp_fig, use_container_width=True)
                
                # Alerts table
                st.subheader("Environmental Alerts")
                st.dataframe(st.session_state.environmental_alerts[['SKU_ID', 'warehouse_id', 'severity', 'message', 'recommended_action']])
            except Exception as e:
                st.error(f"Error displaying environmental analysis: {e}")
        else:
            st.info("No environmental analysis data available. Run an analysis to view results.")
        
        # Show specific analysis if available  
        if 'environment' in st.session_state.analysis_results:
            st.subheader(f"Analysis for SKU: {st.session_state.selected_sku}")
            st.markdown(st.session_state.analysis_results['environment'])
    
    # Demand Analysis Tab
    with tabs[3]:
        st.header("Demand Analysis")
        
        if st.session_state.demand_alerts is not None and not st.session_state.demand_alerts.empty:
            try:
                # Summary metrics
                high_severity = len(st.session_state.demand_alerts[st.session_state.demand_alerts['severity'] == 'HIGH'])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Critical Demand Alerts", high_severity)
                with col2:
                    st.metric("Products with Demand Issues", st.session_state.demand_alerts['SKU_ID'].nunique())
                
                # Demand forecast chart
                if 'forecast' in st.session_state.analysis_results:
                    st.subheader("Demand Forecast")
                    forecast_fig = plot_demand_forecast(st.session_state.analysis_results['forecast'])
                    st.plotly_chart(forecast_fig, use_container_width=True)
                
                # Alerts table
                st.subheader("Demand Alerts")
                st.dataframe(st.session_state.demand_alerts[['SKU_ID', 'warehouse_id', 'severity', 'message', 'recommended_action']])
            except Exception as e:
                st.error(f"Error displaying demand analysis: {e}")
        else:
            st.info("No demand analysis data available. Run an analysis to view results.")
        
        # Show specific analysis if available
        if 'demand' in st.session_state.analysis_results:
            st.subheader(f"Analysis for SKU: {st.session_state.selected_sku}")
            st.markdown(st.session_state.analysis_results['demand'])
    
    # AI Insights Tab
    with tabs[4]:
        st.header("AI Insights with Ollama")
        display_chat_interface()

def main():
    """Main application function"""
    # Initialize SCM knowledge base
    init_scm_knowledge_base()
    
    # Load data
    data_objects = load_data()
    
    if data_objects:
        # Display the dashboard
        display_dashboard(data_objects)
    else:
        st.error("Failed to load data. Please check the data files and try again.")

if __name__ == "__main__":
    main()