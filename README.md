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
```mermaid
flowchart TD
    input[HSI Patch\n3D Tensor\n(63 bands × patch_size × patch_size)]
    
    %% CNN Feature Extractor
    input --> conv1[Conv2D\n63→64 channels\n3×3 kernel]
    conv1 --> conv2[Conv2D\n64→128 channels\n3×3 kernel]
    conv2 --> conv3[Conv2D\n128→embed_dim channels\n3×3 kernel]
    conv3 --> pool[Global Average Pooling]
    pool --> flatten[Flatten to 1D]
    
    %% Transformer
    flatten --> add_seq[Add Sequence Dimension]
    add_seq --> tr_block[Transformer Blocks]
    
    %% Transformer Block Details
    tr_block --> |"× N layers"| norm1[Layer Normalization]
    norm1 --> mha[Multi-Head Attention]
    mha --> add1[Add & Normalize]
    add1 --> norm2[Layer Normalization]
    norm2 --> mlp[MLP\nembed_dim → mlp_dim → embed_dim]
    mlp --> add2[Add & Normalize]
    
    %% Final Processing
    add2 --> final_norm[Layer Normalization]
    final_norm --> global_pool[Global Average Pooling]
    
    %% Classification Head
    global_pool --> fc[Fully Connected\nembed_dim → num_classes]
    fc --> output[Output\nClass Probabilities]
    
    %% Styling
    classDef cnn fill:#c6e5ff,stroke:#333
    classDef transformer fill:#ffe6cc,stroke:#333
    classDef general fill:#f9f9f9,stroke:#333
    
    class conv1,conv2,conv3,pool,flatten cnn
    class add_seq,norm1,mha,add1,norm2,mlp,add2,final_norm,global_pool,tr_block transformer
    class input,fc,output general
```mermaid
