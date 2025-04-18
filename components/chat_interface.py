import streamlit as st
import pandas as pd
import os
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from datetime import datetime

import config

def init_chat_session():
    """Initialize chat session state variables if they don't exist"""
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
    if 'ollama_model' not in st.session_state:
        st.session_state.ollama_model = OllamaLLM(model=config.OLLAMA_MODEL)
    if 'prompt_template' not in st.session_state:
        # Create prompt template for chat interactions
        st.session_state.prompt_template = PromptTemplate(
            input_variables=["chat_history", "inventory_data", "environmental_data", "demand_data", "question"],
            template="""
            You are an AI assistant specializing in perishable goods inventory management.
            
            Previous conversation:
            {chat_history}
            
            Available inventory data summary:
            {inventory_data}
            
            Available environmental monitoring data summary:
            {environmental_data}
            
            Available demand prediction data summary:
            {demand_data}
            
            Human: {question}
            AI: 
            """
        )
    if 'llm_chain' not in st.session_state:
        # Initialize the LLM chain
        st.session_state.llm_chain = LLMChain(
            llm=st.session_state.ollama_model, 
            prompt=st.session_state.prompt_template
        )

def format_chat_history():
    """Format the chat history for inclusion in the prompt"""
    formatted_history = ""
    for message in st.session_state.chat_messages:
        role = "Human" if message["role"] == "user" else "AI"
        formatted_history += f"{role}: {message['content']}\n"
    return formatted_history

def get_data_summary():
    """Get summaries of current data to provide context to the LLM"""
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
        unique_warehouses = df['warehouse_id'].nunique() if 'warehouse_id' in df.columns else df['warehous_id'].nunique()
        
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
    """Query the Ollama model with the user's question and chat history context"""
    chat_history = format_chat_history()
    inventory_summary, environmental_summary, demand_summary = get_data_summary()
    
    # Call the LLM chain with all context
    response = st.session_state.llm_chain.run(
        chat_history=chat_history,
        inventory_data=inventory_summary,
        environmental_data=environmental_summary,
        demand_data=demand_summary,
        question=question
    )
    
    return response

def display_chat_interface():
    """Display the chat interface for interacting with the Ollama model"""
    init_chat_session()
    
    # Display introductory message about the AI assistant
    st.markdown("""
    ### AI Assistant for Perishable Goods Analysis
    Ask questions about the analysis results, get insights about the data, or request recommendations.
    """)
    
    # Display chat messages
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the analysis..."):
        # Add user message to chat history
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response with a spinner while processing
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Process the query with Ollama
                response = query_ollama(prompt)
            
            # Display the response
            st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.chat_messages.append({"role": "assistant", "content": response})
    
    # Option to clear chat history
    if st.button("Clear Chat History"):
        st.session_state.chat_messages = []
        st.rerun()
    
    # Export chat history option
    if st.session_state.chat_messages and st.button("Export Conversation"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"conversation_{timestamp}.txt"
        
        with open(filename, "w") as f:
            for message in st.session_state.chat_messages:
                role = "Human" if message["role"] == "user" else "AI"
                f.write(f"{role}: {message['content']}\n\n")
        
        with open(filename, "rb") as f:
            st.download_button(
                label="Download Conversation",
                data=f,
                file_name=filename,
                mime="text/plain"
            )
