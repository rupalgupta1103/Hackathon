"""
Pydantic schemas for API request/response validation
"""

from pydantic import BaseModel, Field, validator, ConfigDict
from datetime import datetime, date
from typing import Optional, Dict, List, Any, Generic, TypeVar
from enum import Enum

# Type variable for pagination
T = TypeVar('T')

# ==================== Enums ====================

class MealType(str, Enum):
    BREAKFAST = "breakfast"
    LUNCH = "lunch"
    SNACKS = "snacks"
    DINNER = "dinner"

class WasteReason(str, Enum):
    OVERPRODUCTION = "overproduction"
    QUALITY_ISSUE = "quality_issue"
    SPOILAGE = "spoilage"
    STUDENT_ABSENCE = "student_absence"
    OTHER = "other"

class MenuCategory(str, Enum):
    MAIN_COURSE = "main_course"
    BREAD = "bread"
    DESSERT = "dessert"
    BEVERAGE = "beverage"
    SIDE_DISH = "side_dish"

# ==================== Request Schemas ====================

class DemandPredictionRequest(BaseModel):
    """Request schema for demand prediction"""
    date: date
    temperature: float = Field(..., ge=-10, le=50, description="Temperature in Celsius")
    rainfall: float = Field(0, ge=0, description="Rainfall in mm")
    is_holiday: bool = False
    is_exam: bool = False
    special_event: Optional[str] = Field(None, max_length=100)
    
    @validator('date')
    def date_not_in_past(cls, v):
        if v < date.today():
            raise ValueError('Prediction date cannot be in the past')
        return v

class ProcurementRequest(BaseModel):
    """Request schema for procurement optimization"""
    predictions: Dict[str, Dict[str, float]] = Field(..., description="Predicted demand by meal and dish")
    budget_constraint: Optional[float] = Field(None, gt=0, description="Maximum budget in INR")
    current_stock: Optional[Dict[str, float]] = Field(None, description="Current stock levels in kg")
    
    @validator('predictions')
    def validate_predictions(cls, v):
        if not v:
            raise ValueError('Predictions cannot be empty')
        return v

class WasteLogRequest(BaseModel):
    """Request schema for logging waste"""
    meal_type: MealType
    dish_name: str = Field(..., min_length=1, max_length=100)
    waste_quantity: float = Field(..., gt=0, le=1000, description="Waste quantity in kg")
    waste_reason: WasteReason
    notes: Optional[str] = Field(None, max_length=500)
    
    @validator('waste_quantity')
    def validate_waste_quantity(cls, v):
        if v <= 0:
            raise ValueError('Waste quantity must be positive')
        return v

class MenuOptimizationRequest(BaseModel):
    """Request schema for menu optimization"""
    historical_consumption: Dict[str, Dict[str, float]] = Field(..., description="Historical consumption data")
    dietary_preferences: Optional[Dict[str, List[str]]] = Field(None, description="Dietary preferences by category")
    budget_per_meal: Optional[float] = Field(None, gt=0, description="Budget per meal in INR")
    sustainability_focus: bool = Field(False, description="Prioritize sustainable options")

class MealConsumptionCreate(BaseModel):
    """Request schema for creating meal consumption record"""
    date: datetime = Field(default_factory=datetime.utcnow)
    meal_type: MealType
    dish_name: str = Field(..., min_length=1, max_length=100)
    quantity_prepared: float = Field(..., gt=0, le=10000)
    quantity_consumed: float = Field(..., ge=0, le=10000)
    student_count: int = Field(..., ge=0, le=10000)
    temperature: Optional[float] = None
    rainfall: Optional[float] = 0
    is_holiday: bool = False
    special_event: Optional[str] = None
    
    @validator('quantity_consumed')
    def validate_consumption(cls, v, values):
        if 'quantity_prepared' in values and v > values['quantity_prepared']:
            raise ValueError('Consumed quantity cannot exceed prepared quantity')
        return v

class MenuItemCreate(BaseModel):
    """Request schema for creating menu item"""
    dish_name: str = Field(..., min_length=1, max_length=100)
    category: MenuCategory
    ingredients: Dict[str, float] = Field(..., description="Ingredients with quantities in kg")
    base_price: float = Field(..., gt=0, le=10000)
    preparation_time: int = Field(30, ge=5, le=240, description="Preparation time in minutes")
    is_vegetarian: bool = True
    is_vegan: bool = False
    is_gluten_free: bool = False
    seasonal_months: Optional[List[int]] = Field(None, description="Months when available (1-12)")
    
    @validator('seasonal_months')
    def validate_months(cls, v):
        if v:
            for month in v:
                if month < 1 or month > 12:
                    raise ValueError('Month must be between 1 and 12')
        return v

class ProcurementCreate(BaseModel):
    """Request schema for creating procurement record"""
    date: datetime = Field(default_factory=datetime.utcnow)
    ingredient_name: str = Field(..., min_length=1, max_length=100)
    quantity: float = Field(..., gt=0, le=100000)
    unit_price: float = Field(..., gt=0, le=10000)
    vendor: str = Field(..., min_length=1, max_length=100)
    vendor_contact: Optional[str] = None
    predicted_quantity: float = Field(..., ge=0)

class WasteLogCreate(BaseModel):
    """Request schema for creating waste log"""
    date: datetime = Field(default_factory=datetime.utcnow)
    meal_id: Optional[int] = None
    meal_type: MealType
    dish_name: str = Field(..., min_length=1, max_length=100)
    waste_quantity: float = Field(..., gt=0, le=1000)
    waste_reason: WasteReason
    notes: Optional[str] = None

# ==================== Response Schemas ====================

class DemandPredictionResponse(BaseModel):
    """Response schema for demand prediction"""
    date: date
    predictions: Dict[str, Dict[str, int]] = Field(..., description="Predicted demand by meal and dish")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence score")
    model_used: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "date": "2024-01-15",
            "predictions": {
                "breakfast": {"rice": 150, "chapati": 200},
                "lunch": {"rice": 500, "dal": 450}
            },
            "confidence": 0.85,
            "model_used": "XGBoost",
            "timestamp": "2024-01-14T10:30:00"
        }
    })

class ProcurementResponse(BaseModel):
    """Response schema for procurement optimization"""
    optimal_quantities: Dict[str, float] = Field(..., description="Optimal quantities to procure in kg")
    total_cost: float = Field(..., description="Total optimized cost in INR")
    traditional_cost: float = Field(..., description="Traditional procurement cost in INR")
    savings: float = Field(..., description="Cost savings in INR")
    savings_percentage: float = Field(..., ge=0, le=100)
    status: str
    recommendations: Dict[str, Dict[str, Any]] = Field(..., description="Detailed procurement recommendations")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class SustainabilityMetricsResponse(BaseModel):
    """Response schema for sustainability metrics"""
    metrics: Dict[str, Any] = Field(..., description="Detailed sustainability metrics")
    sustainability_score: Dict[str, Any] = Field(..., description="Sustainability scores")
    report: str = Field(..., description="Human-readable sustainability report")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class WasteLogResponse(BaseModel):
    """Response schema for waste log creation"""
    message: str
    id: int
    co2_footprint: float
    water_footprint: float
    cost_loss: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class HealthCheckResponse(BaseModel):
    """Response schema for health check"""
    status: str
    message: str
    version: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "status": "healthy",
            "message": "Smart Mess Optimization System API",
            "version": "1.0.0",
            "timestamp": "2024-01-14T10:30:00"
        }
    })

# ==================== Database Response Schemas ====================

class MealConsumptionResponse(BaseModel):
    """Response schema for meal consumption data"""
    id: int
    date: datetime
    meal_type: str
    dish_name: str
    quantity_prepared: float
    quantity_consumed: float
    quantity_wasted: float
    student_count: int
    day_of_week: int
    is_weekend: bool
    is_holiday: bool
    temperature: Optional[float]
    rainfall: Optional[float]
    special_event: Optional[str]
    created_at: datetime
    updated_at: datetime
    
    model_config = ConfigDict(from_attributes=True)

class MenuItemResponse(BaseModel):
    """Response schema for menu item data"""
    id: int
    dish_name: str
    category: str
    ingredients: str
    base_price: float
    preparation_time: int
    popularity_score: float
    sustainability_score: float
    nutritional_score: float
    is_vegetarian: bool
    is_vegan: bool
    is_gluten_free: bool
    seasonal_months: Optional[str]
    created_at: datetime
    updated_at: datetime
    
    model_config = ConfigDict(from_attributes=True)

class ProcurementDataResponse(BaseModel):
    """Response schema for procurement data"""
    id: int
    date: datetime
    ingredient_name: str
    quantity: float
    unit_price: float
    total_cost: float
    vendor: str
    vendor_contact: Optional[str]
    predicted_quantity: float
    actual_used: Optional[float]
    quality_rating: Optional[int]
    delivery_delay: Optional[int]
    created_at: datetime
    updated_at: datetime
    
    model_config = ConfigDict(from_attributes=True)

class WasteLogDataResponse(BaseModel):
    """Response schema for waste log data"""
    id: int
    date: datetime
    meal_id: Optional[int]
    meal_type: str
    dish_name: str
    waste_quantity: float
    waste_reason: str
    co2_footprint: float
    water_footprint: float
    cost_loss: float
    notes: Optional[str]
    created_at: datetime
    updated_at: datetime
    
    model_config = ConfigDict(from_attributes=True)

# ==================== Common Response Schemas ====================

class ErrorResponse(BaseModel):
    """Error response schema"""
    error: str
    detail: Optional[str] = None
    status_code: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "error": "Not Found",
            "detail": "Resource with id 123 not found",
            "status_code": 404,
            "timestamp": "2024-01-14T10:30:00"
        }
    })

class PaginatedResponse(BaseModel, Generic[T]):
    """Paginated response schema"""
    items: List[T]
    total: int
    page: int
    size: int
    pages: int
    has_next: bool
    has_prev: bool
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "items": [],
            "total": 100,
            "page": 1,
            "size": 10,
            "pages": 10,
            "has_next": True,
            "has_prev": False
        }
    })

class DateRangeRequest(BaseModel):
    """Date range request schema"""
    start_date: date
    end_date: date
    
    @validator('end_date')
    def validate_dates(cls, v, values):
        if 'start_date' in values and v < values['start_date']:
            raise ValueError('End date must be after start date')
        return v