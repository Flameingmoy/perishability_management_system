Perishability management system : an agentic workflow that run inventory, environment and demand forecast analysis of a supply chain . This application can be useful for managing a supply chain which has a lot of volatile/perishable products.

Tech Stack: Langchain for calling agents and connecting them to the data, Ollama for the llm and embeddings , specifically using granite3.3:8b for the llm and snowflake-arctic embbedings v2 for the embeddings, streamlit for the front end ui and plotly for dashboard creation.
```mermaid
flowchart TD
    %% User
    User["User"]:::user

    %% Frontend Layer
    subgraph "Frontend Layer"
        UI1["app.py"]:::frontend
        UI2["main.py"]:::frontend
        Chat["Chat Interface"]:::frontend
        Viz["Visualization Helpers"]:::frontend
    end

    %% Orchestration Layer
    Orchestrator["Agent Orchestrator"]:::agent

    %% Agent Layer
    subgraph "Agent Layer"
        Demand["DemandPredictionAgent"]:::agent
        Inventory["InventoryMonitoringAgent"]:::agent
        Environmental["EnvironmentalMonitoringAgent"]:::agent
    end

    %% Utilities & Models
    subgraph "Utilities & Models"
        DataLoader["Data Loader"]:::util
        TimeSeries["Time Series Utils"]:::util
        DataModels["Data Models"]:::data
        Config["config.py"]:::util
    end

    %% Data Layer
    subgraph "Data Layer"
        CSV["CSV Data Sources"]:::data
    end

    %% External Services
    subgraph "External Services"
        Ollama["Ollama LLM"]:::external
        Arctic["Snowflake Arctic v2"]:::external
    end

    %% Flow Connections
    User --> UI1
    User --> UI2
    UI1 --> Chat
    UI1 --> Viz
    UI2 --> Orchestrator
    Chat --> Orchestrator
    Orchestrator --> Demand
    Orchestrator --> Inventory
    Orchestrator --> Environmental
    Demand --> DataLoader
    Inventory --> DataLoader
    Environmental --> DataLoader
    DataLoader --> CSV
    Demand --> TimeSeries
    Inventory --> TimeSeries
    Environmental --> TimeSeries
    Demand --> DataModels
    Inventory --> DataModels
    Environmental --> DataModels
    Demand --> Ollama
    Inventory --> Ollama
    Environmental --> Ollama
    Demand --> Arctic
    Inventory --> Arctic
    Environmental --> Arctic
    Demand --> Viz
    Inventory --> Viz
    Environmental --> Viz
    Config --> Orchestrator
    Config --> Demand
    Config --> Inventory
    Config --> Environmental

    %% Click Events
    click UI1 "https://github.com/flameingmoy/perishability_management_system/blob/main/app.py"
    click UI2 "https://github.com/flameingmoy/perishability_management_system/blob/main/main.py"
    click Chat "https://github.com/flameingmoy/perishability_management_system/blob/main/components/chat_interface.py"
    click Viz "https://github.com/flameingmoy/perishability_management_system/blob/main/utils/visualization.py"
    click Demand "https://github.com/flameingmoy/perishability_management_system/blob/main/agents/demand_prediction_agent.py"
    click Inventory "https://github.com/flameingmoy/perishability_management_system/blob/main/agents/inventory_monitoring_agent.py"
    click Environmental "https://github.com/flameingmoy/perishability_management_system/blob/main/agents/environmental_monitoring_agent.py"
    click DataLoader "https://github.com/flameingmoy/perishability_management_system/blob/main/utils/data_loader.py"
    click TimeSeries "https://github.com/flameingmoy/perishability_management_system/blob/main/utils/time_series_utils.py"
    click DataModels "https://github.com/flameingmoy/perishability_management_system/blob/main/models/data_models.py"
    click Config "https://github.com/flameingmoy/perishability_management_system/blob/main/config.py"
    click README "https://github.com/flameingmoy/perishability_management_system/blob/main/README.md"
    click SCM "https://github.com/flameingmoy/perishability_management_system/blob/main/SCM Scenarios.pdf"
    click CSV "https://github.com/flameingmoy/perishability_management_system/tree/main/data/"

    %% Styling
    classDef frontend fill:#D0E8FF,stroke:#0366d6,color:#0366d6;
    classDef agent fill:#DFF5D0,stroke:#2a7d1a,color:#2a7d1a;
    classDef util fill:#E8E8E8,stroke:#666666,color:#333333;
    classDef data fill:#FFE8D0,stroke:#d66e00,color:#d66e00;
    classDef external fill:#E8D0FF,stroke:#6a1a9a,color:#6a1a9a;
    classDef user fill:#FFFFFF,stroke:#000000,color:#000000,stroke-dasharray: 5 5;
```mermaid
