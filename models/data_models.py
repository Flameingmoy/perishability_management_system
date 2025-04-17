from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class ProductMaster(BaseModel):
    SKU_ID: str
    Product_Category: str
    Initial_Shelf_Life: int
    Storage_Temp: float
    Unit_Storage_Temp: str
    Critical_Humidity: float
    Minmum_Order_Qty: int
    UNIT_orderqty: str

class TimeSeriesData(BaseModel):
    Timestamp: datetime
    warehous_id: str
    SKU_id: str
    current_stock: float
    unit_curr_stock: str
    days_remaining: int
    temp_deviation: float
    waste_qty: float
    unit_waste_qty: str
    Demand_forecast: float
    unit_Demand_forecast: str

class IoTData(BaseModel):
    DEVICE_ID: str
    SKU_ID: str
    TIMESTAMP: datetime
    TEMP_HUMIDITY: float
    SHOCK_EVENTS: int

class SupplierData(BaseModel):
    Supplier_vendor: str
    SKU_ID: str
    LEAD_TIME: str
    transport_mode: str
    TEMP_COMPLIANCE: float
    CO2E_PER_UNIT: str

class ProductAlert(BaseModel):
    SKU_ID: str
    warehouse_id: str
    alert_type: str
    severity: str  # HIGH, MEDIUM, LOW
    message: str
    timestamp: datetime
    recommended_action: str


class DemandForecast(BaseModel):
    SKU_ID: str
    warehouse_id: str
    forecast_date: datetime
    quantity: float
    unit: str
    confidence: float  # 0-1 representing confidence level

class PriceRecommendation(BaseModel):
    SKU_ID: str
    warehouse_id: str
    current_price: float
    recommended_price: float
    discount_percentage: float
    expected_sales_lift: float
    expiry_date: datetime
    timestamp: datetime
    reasoning: str