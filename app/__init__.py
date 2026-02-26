"""
Utilities module for Smart Mess Optimization System
Provides helper functions, common utilities, and shared functionality
"""

from app.utils.helpers import (
    # Date/time utilities
    date_to_str, str_to_date, get_date_range, get_week_dates,
    get_month_dates, is_weekend, is_holiday_check,
    
    # Math utilities
    safe_divide, calculate_percentage, round_decimal,
    calculate_moving_average, calculate_growth_rate,
    
    # String utilities
    slugify, truncate_text, sanitize_input,
    format_currency, format_number, format_percentage,
    
    # Data utilities
    chunk_list, flatten_list, deduplicate_list,
    safe_json_loads, safe_json_dumps,
    
    # Validation utilities
    validate_email, validate_phone, validate_date_range,
    validate_positive_number, validate_in_range,
    
    # File utilities
    ensure_dir, safe_file_write, safe_file_read,
    get_file_extension, get_file_size,
    
    # Cache utilities
    memoize, timed_cache, async_cache
)

from app.utils.logger import (
    setup_logger, get_logger, log_function_call,
    log_execution_time, LogContext, LogLevel
)

from app.utils.decorators import (
    retry_on_failure, handle_exceptions, timeit,
    singleton, rate_limit, validate_input
)

from app.utils.notifications import (
    send_email, send_slack_message, send_sms,
    NotificationManager, NotificationType
)

from app.utils.validators import (
    DataValidator, SchemaValidator, ValidationError,
    validate_schema, validate_model
)

from app.utils.converters import (
    kg_to_g, g_to_kg, liters_to_ml, ml_to_liters,
    celsius_to_fahrenheit, fahrenheit_to_celsius,
    currency_converter, unit_converter
)

from app.utils.analytics import (
    calculate_statistics, detect_outliers, calculate_correlation,
    time_series_decompose, calculate_seasonality
)

__version__ = '1.0.0'
__all__ = [
    # Date/time utilities
    'date_to_str',
    'str_to_date',
    'get_date_range',
    'get_week_dates',
    'get_month_dates',
    'is_weekend',
    'is_holiday_check',
    
    # Math utilities
    'safe_divide',
    'calculate_percentage',
    'round_decimal',
    'calculate_moving_average',
    'calculate_growth_rate',
    
    # String utilities
    'slugify',
    'truncate_text',
    'sanitize_input',
    'format_currency',
    'format_number',
    'format_percentage',
    
    # Data utilities
    'chunk_list',
    'flatten_list',
    'deduplicate_list',
    'safe_json_loads',
    'safe_json_dumps',
    
    # Validation utilities
    'validate_email',
    'validate_phone',
    'validate_date_range',
    'validate_positive_number',
    'validate_in_range',
    
    # File utilities
    'ensure_dir',
    'safe_file_write',
    'safe_file_read',
    'get_file_extension',
    'get_file_size',
    
    # Cache utilities
    'memoize',
    'timed_cache',
    'async_cache',
    
    # Logging
    'setup_logger',
    'get_logger',
    'log_function_call',
    'log_execution_time',
    'LogContext',
    'LogLevel',
    
    # Decorators
    'retry_on_failure',
    'handle_exceptions',
    'timeit',
    'singleton',
    'rate_limit',
    'validate_input',
    
    # Notifications
    'send_email',
    'send_slack_message',
    'send_sms',
    'NotificationManager',
    'NotificationType',
    
    # Validators
    'DataValidator',
    'SchemaValidator',
    'ValidationError',
    'validate_schema',
    'validate_model',
    
    # Converters
    'kg_to_g',
    'g_to_kg',
    'liters_to_ml',
    'ml_to_liters',
    'celsius_to_fahrenheit',
    'fahrenheit_to_celsius',
    'currency_converter',
    'unit_converter',
    
    # Analytics
    'calculate_statistics',
    'detect_outliers',
    'calculate_correlation',
    'time_series_decompose',
    'calculate_seasonality'
]

# Package metadata
__author__ = 'Smart Mess Optimization Team'
__description__ = 'Common utilities and helper functions'
__license__ = 'MIT'

# ==================== Constants ====================

# Unit conversion factors
UNIT_CONVERSIONS = {
    'weight': {
        'kg_to_g': 1000,
        'g_to_kg': 0.001,
        'kg_to_lb': 2.20462,
        'lb_to_kg': 0.453592,
        'g_to_lb': 0.00220462,
        'lb_to_g': 453.592
    },
    'volume': {
        'l_to_ml': 1000,
        'ml_to_l': 0.001,
        'l_to_gal': 0.264172,
        'gal_to_l': 3.78541,
        'ml_to_fl_oz': 0.033814,
        'fl_oz_to_ml': 29.5735
    },
    'temperature': {
        'c_to_f': lambda c: (c * 9/5) + 32,
        'f_to_c': lambda f: (f - 32) * 5/9
    }
}

# Currency conversion rates (example - would be fetched from API in production)
CURRENCY_RATES = {
    'USD': 1.0,
    'INR': 83.0,  # 1 USD = 83 INR
    'EUR': 0.92,   # 1 USD = 0.92 EUR
    'GBP': 0.79,   # 1 USD = 0.79 GBP
    'JPY': 150.0,  # 1 USD = 150 JPY
    'CNY': 7.2,    # 1 USD = 7.2 CNY
    'AUD': 1.52,   # 1 USD = 1.52 AUD
    'CAD': 1.35    # 1 USD = 1.35 CAD
}

# Date formats
DATE_FORMATS = {
    'iso': '%Y-%m-%d',
    'us': '%m/%d/%Y',
    'eu': '%d/%m/%Y',
    'full': '%Y-%m-%d %H:%M:%S',
    'time': '%H:%M:%S',
    'month_day': '%B %d',
    'month_year': '%B %Y'
}

# Validation patterns
VALIDATION_PATTERNS = {
    'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
    'phone': r'^\+?[1-9]\d{1,14}$',
    'indian_phone': r'^\+91[6-9]\d{9}$',
    'pincode': r'^\d{6}$',
    'gst': r'^\d{2}[A-Z]{5}\d{4}[A-Z]{1}[1-9A-Z]{1}[Z]{1}[0-9A-Z]{1}$',
    'pan': r'^[A-Z]{5}[0-9]{4}[A-Z]{1}$',
    'aadhaar': r'^\d{12}$'
}

# ==================== Helper Functions ====================

def get_date_utils() -> dict:
    """
    Get date utility functions
    """
    return {
        'formats': DATE_FORMATS,
        'now': datetime.now,
        'today': date.today,
        'weekday_names': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
        'month_names': ['January', 'February', 'March', 'April', 'May', 'June',
                       'July', 'August', 'September', 'October', 'November', 'December']
    }


def get_conversion_factor(from_unit: str, to_unit: str, category: str = 'weight') -> float:
    """
    Get conversion factor between units
    
    Args:
        from_unit: Source unit
        to_unit: Target unit
        category: Unit category (weight, volume, temperature)
    
    Returns:
        Conversion factor
    """
    if category == 'temperature':
        # Temperature conversions need functions, not simple factors
        raise ValueError("Temperature conversions require functions, use temperature_converter()")
    
    conversion_key = f"{from_unit}_to_{to_unit}"
    return UNIT_CONVERSIONS.get(category, {}).get(conversion_key, 1.0)


def temperature_converter(value: float, from_unit: str, to_unit: str) -> float:
    """
    Convert temperature between units
    
    Args:
        value: Temperature value
        from_unit: Source unit ('c', 'f')
        to_unit: Target unit ('c', 'f')
    
    Returns:
        Converted temperature
    """
    from_unit = from_unit.lower()
    to_unit = to_unit.lower()
    
    if from_unit == to_unit:
        return value
    
    if from_unit == 'c' and to_unit == 'f':
        return UNIT_CONVERSIONS['temperature']['c_to_f'](value)
    elif from_unit == 'f' and to_unit == 'c':
        return UNIT_CONVERSIONS['temperature']['f_to_c'](value)
    else:
        raise ValueError(f"Unsupported temperature conversion: {from_unit} to {to_unit}")


def currency_converter(amount: float, from_currency: str, to_currency: str = 'INR') -> float:
    """
    Convert currency
    
    Args:
        amount: Amount to convert
        from_currency: Source currency code
        to_currency: Target currency code
    
    Returns:
        Converted amount
    """
    from_currency = from_currency.upper()
    to_currency = to_currency.upper()
    
    if from_currency not in CURRENCY_RATES or to_currency not in CURRENCY_RATES:
        raise ValueError(f"Unsupported currency: {from_currency} or {to_currency}")
    
    # Convert to USD first (base currency)
    usd_amount = amount / CURRENCY_RATES[from_currency]
    
    # Convert from USD to target
    converted = usd_amount * CURRENCY_RATES[to_currency]
    
    return round(converted, 2)


def format_currency_indian(amount: float) -> str:
    """
    Format amount in Indian currency format
    
    Args:
        amount: Amount in INR
    
    Returns:
        Formatted string (e.g., ₹1,23,456.78)
    """
    from app.utils.helpers import format_currency_indian as _format_indian
    return _format_indian(amount)


def validate_indian_details(**kwargs) -> dict:
    """
    Validate Indian-specific details
    
    Args:
        **kwargs: Fields to validate (pan, aadhaar, gst, pincode, phone)
    
    Returns:
        Dictionary with validation results
    """
    from app.utils.validators import validate_indian_pan, validate_indian_aadhaar
    from app.utils.validators import validate_indian_gst, validate_indian_pincode
    from app.utils.validators import validate_indian_phone
    
    results = {}
    
    if 'pan' in kwargs:
        results['pan_valid'] = validate_indian_pan(kwargs['pan'])
    
    if 'aadhaar' in kwargs:
        results['aadhaar_valid'] = validate_indian_aadhaar(kwargs['aadhaar'])
    
    if 'gst' in kwargs:
        results['gst_valid'] = validate_indian_gst(kwargs['gst'])
    
    if 'pincode' in kwargs:
        results['pincode_valid'] = validate_indian_pincode(kwargs['pincode'])
    
    if 'phone' in kwargs:
        results['phone_valid'] = validate_indian_phone(kwargs['phone'])
    
    return results


def calculate_business_days(start_date: date, end_date: date, holidays: List[date] = None) -> int:
    """
    Calculate number of business days between dates
    
    Args:
        start_date: Start date
        end_date: End date
        holidays: List of holiday dates
    
    Returns:
        Number of business days
    """
    business_days = 0
    current = start_date
    holidays = holidays or []
    
    while current <= end_date:
        if current.weekday() < 5 and current not in holidays:  # Monday=0, Friday=4
            business_days += 1
        current += timedelta(days=1)
    
    return business_days


def parse_ingredients_string(ingredients_str: str) -> Dict[str, float]:
    """
    Parse ingredients string to dictionary
    
    Args:
        ingredients_str: JSON string or formatted string
    
    Returns:
        Dictionary of ingredient: quantity
    """
    import json
    
    try:
        # Try parsing as JSON
        return json.loads(ingredients_str)
    except:
        # Try parsing as "ingredient:quantity" format
        result = {}
        try:
            parts = ingredients_str.split(',')
            for part in parts:
                if ':' in part:
                    ing, qty = part.split(':')
                    result[ing.strip()] = float(qty.strip())
        except:
            pass
    
    return result


def generate_meal_id(date: date, meal_type: str, dish_name: str) -> str:
    """
    Generate unique meal ID
    
    Args:
        date: Meal date
        meal_type: Type of meal
        dish_name: Dish name
    
    Returns:
        Unique meal ID
    """
    date_str = date.strftime('%Y%m%d')
    meal_code = meal_type[:3].upper()
    dish_code = dish_name[:5].upper().replace(' ', '')
    return f"{date_str}_{meal_code}_{dish_code}"


def calculate_waste_metrics(waste_kg: float, food_type: str = 'average') -> dict:
    """
    Calculate environmental impact of waste
    
    Args:
        waste_kg: Waste quantity in kg
        food_type: Type of food
    
    Returns:
        Dictionary with environmental metrics
    """
    from app.sustainability import get_environmental_factor
    
    co2_factor = get_environmental_factor(food_type, 'co2_emissions')
    water_factor = get_environmental_factor(food_type, 'water_footprint')
    
    return {
        'waste_kg': round(waste_kg, 2),
        'co2_kg': round(waste_kg * co2_factor, 2),
        'water_liters': round(waste_kg * water_factor, 2),
        'cost_loss': round(waste_kg * 30, 2),  # Assuming ₹30/kg waste cost
        'food_type': food_type
    }


def paginate_results(items: List, page: int = 1, per_page: int = 10) -> dict:
    """
    Paginate a list of items
    
    Args:
        items: List to paginate
        page: Page number (1-indexed)
        per_page: Items per page
    
    Returns:
        Paginated result dictionary
    """
    total = len(items)
    total_pages = (total + per_page - 1) // per_page
    
    start = (page - 1) * per_page
    end = start + per_page
    
    paginated_items = items[start:end] if start < total else []
    
    return {
        'items': paginated_items,
        'page': page,
        'per_page': per_page,
        'total': total,
        'total_pages': total_pages,
        'has_next': page < total_pages,
        'has_prev': page > 1
    }


def time_it(func):
    """
    Decorator to measure function execution time
    
    Args:
        func: Function to measure
    
    Returns:
        Wrapped function
    """
    from functools import wraps
    import time
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        
        logger = get_logger(__name__)
        logger.debug(f"{func.__name__} took {end - start:.4f} seconds")
        
        return result
    
    return wrapper


def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Retry decorator with exponential backoff
    
    Args:
        max_attempts: Maximum number of attempts
        delay: Initial delay in seconds
        backoff: Backoff multiplier
    
    Returns:
        Decorated function
    """
    from functools import wraps
    import time
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        time.sleep(current_delay)
                        current_delay *= backoff
            
            raise last_exception
        
        return wrapper
    
    return decorator


def singleton(cls):
    """
    Singleton class decorator
    
    Args:
        cls: Class to make singleton
    
    Returns:
        Singleton class
    """
    instances = {}
    
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return get_instance


class ContextManager:
    """
    Context manager for common operations
    """
    
    @staticmethod
    def timer(name: str = "Operation"):
        """
        Context manager for timing operations
        
        Args:
            name: Operation name
        """
        import time
        
        class TimerContext:
            def __enter__(self):
                self.start = time.time()
                return self
            
            def __exit__(self, *args):
                self.end = time.time()
                logger = get_logger(__name__)
                logger.info(f"{name} took {self.end - self.start:.2f} seconds")
        
        return TimerContext()
    
    @staticmethod
    def db_transaction(db_session):
        """
        Context manager for database transactions
        
        Args:
            db_session: Database session
        """
        class TransactionContext:
            def __enter__(self):
                return db_session
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                if exc_type:
                    db_session.rollback()
                else:
                    db_session.commit()
        
        return TransactionContext()


# ==================== Initialize Module ====================

import logging
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional, Tuple

# Set up module logger
utils_logger = logging.getLogger(__name__)
utils_logger.info(f"Utils module initialized (version {__version__})")
utils_logger.info(f"Available currency conversions: {', '.join(CURRENCY_RATES.keys())}")
utils_logger.info(f"Unit conversion categories: {', '.join(UNIT_CONVERSIONS.keys())}")

# Export context manager
__all__.append('ContextManager')