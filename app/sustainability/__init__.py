"""
Sustainability module for Smart Mess Optimization System
Tracks environmental impact, calculates sustainability metrics, and generates reports
"""

from app.sustainability.metrics_calculator import SustainabilityMetrics
from app.sustainability.waste_monitor import WasteMonitor
from app.sustainability.carbon_calculator import CarbonCalculator
from app.sustainability.water_footprint import WaterFootprintCalculator
from app.sustainability.impact_reporter import ImpactReporter
from app.sustainability.sdg_tracker import SDGTracker
from app.sustainability.certification_manager import CertificationManager

__version__ = '1.0.0'
__all__ = [
    # Main calculators
    'SustainabilityMetrics',
    'WasteMonitor',
    'CarbonCalculator',
    'WaterFootprintCalculator',
    
    # Reporting and tracking
    'ImpactReporter',
    'SDGTracker',
    'CertificationManager'
]

# Package metadata
__author__ = 'Smart Mess Optimization Team'
__description__ = 'Sustainability tracking and environmental impact calculation'
__license__ = 'MIT'

# ==================== Sustainability Constants ====================

# Environmental impact factors (per kg of food)
ENVIRONMENTAL_FACTORS = {
    'co2_emissions': {
        'rice': 2.5,        # kg CO2 per kg
        'wheat': 1.8,        # kg CO2 per kg
        'vegetables': 1.2,    # kg CO2 per kg
        'fruits': 1.1,        # kg CO2 per kg
        'dairy': 3.5,         # kg CO2 per kg
        'meat': 15.0,         # kg CO2 per kg
        'pulses': 1.5,        # kg CO2 per kg
        'oils': 2.8,          # kg CO2 per kg
        'average': 2.5,       # Default average
    },
    'water_footprint': {
        'rice': 2500,         # liters per kg
        'wheat': 1800,        # liters per kg
        'vegetables': 300,     # liters per kg
        'fruits': 900,         # liters per kg
        'dairy': 1000,         # liters per kg
        'meat': 15000,         # liters per kg
        'pulses': 4000,        # liters per kg
        'oils': 5000,          # liters per kg
        'average': 1000,       # Default average
    },
    'land_use': {
        'rice': 2.5,          # m² per kg
        'wheat': 2.0,          # m² per kg
        'vegetables': 0.5,     # m² per kg
        'fruits': 1.0,         # m² per kg
        'dairy': 8.0,          # m² per kg
        'meat': 50.0,          # m² per kg
        'pulses': 7.0,         # m² per kg
        'average': 5.0,        # Default average
    }
}

# Waste categories
WASTE_CATEGORIES = {
    'preparation': 'Waste during food preparation',
    'serving': 'Waste during serving (plate waste)',
    'spoilage': 'Food spoiled before use',
    'overproduction': 'Excess food prepared',
    'quality_issues': 'Food rejected due to quality',
    'storage_loss': 'Loss during storage',
    'transport_damage': 'Damage during transport'
}

# Sustainability goals aligned with SDG 12
SUSTAINABILITY_GOALS = {
    'waste_reduction': {
        'target': 50,          # 50% reduction by 2030
        'baseline_year': 2020,
        'unit': 'percentage',
        'sdg_target': '12.3'   # Halve per capita food waste
    },
    'co2_reduction': {
        'target': 30,           # 30% reduction by 2030
        'baseline_year': 2020,
        'unit': 'percentage',
        'sdg_target': '12.4'    # Environmentally sound management
    },
    'water_conservation': {
        'target': 20,           # 20% reduction by 2030
        'baseline_year': 2020,
        'unit': 'percentage',
        'sdg_target': '12.2'    # Sustainable management of natural resources
    },
    'sustainable_procurement': {
        'target': 50,           # 50% sustainable sourcing by 2025
        'baseline_year': 2020,
        'unit': 'percentage',
        'sdg_target': '12.7'    # Sustainable procurement practices
    }
}

# Certification standards
CERTIFICATION_STANDARDS = {
    'iso_14001': {
        'name': 'ISO 14001 Environmental Management',
        'description': 'International standard for environmental management systems',
        'requirements': ['Waste tracking', 'Emission monitoring', 'Continuous improvement']
    },
    'green_restaurant': {
        'name': 'Green Restaurant Certification',
        'description': 'Certification for sustainable restaurant operations',
        'requirements': ['Waste reduction', 'Energy efficiency', 'Sustainable sourcing']
    },
    'plastic_free': {
        'name': 'Plastic-Free Certification',
        'description': 'Elimination of single-use plastics',
        'requirements': ['No plastic packaging', 'Compostable alternatives']
    },
    'zero_waste': {
        'name': 'Zero Waste Certification',
        'description': '90% diversion from landfill',
        'requirements': ['Waste audit', 'Recycling program', 'Composting']
    }
}

# ==================== Helper Functions ====================

def get_environmental_factor(food_type: str, factor_type: str = 'co2_emissions') -> float:
    """
    Get environmental factor for a specific food type
    
    Args:
        food_type: Type of food (rice, vegetables, etc.)
        factor_type: Type of factor (co2_emissions, water_footprint, land_use)
    
    Returns:
        Environmental factor value
    """
    factors = ENVIRONMENTAL_FACTORS.get(factor_type, {})
    return factors.get(food_type.lower(), factors.get('average', 0))


def calculate_co2_savings(waste_reduced_kg: float, food_type: str = 'average') -> float:
    """
    Calculate CO2 savings from waste reduction
    
    Args:
        waste_reduced_kg: Amount of waste reduced in kg
        food_type: Type of food
    
    Returns:
        CO2 savings in kg
    """
    co2_factor = get_environmental_factor(food_type, 'co2_emissions')
    return waste_reduced_kg * co2_factor


def calculate_water_savings(waste_reduced_kg: float, food_type: str = 'average') -> float:
    """
    Calculate water savings from waste reduction
    
    Args:
        waste_reduced_kg: Amount of waste reduced in kg
        food_type: Type of food
    
    Returns:
        Water savings in liters
    """
    water_factor = get_environmental_factor(food_type, 'water_footprint')
    return waste_reduced_kg * water_factor


def get_sustainability_score(metrics: dict) -> dict:
    """
    Calculate comprehensive sustainability score
    
    Args:
        metrics: Dictionary of sustainability metrics
    
    Returns:
        Dictionary with scores and ratings
    """
    scores = {}
    
    # Waste management score (0-100)
    if 'waste_percentage' in metrics:
        waste_score = max(0, 100 - (metrics['waste_percentage'] * 2))
        scores['waste_management'] = min(100, waste_score)
    
    # Carbon efficiency score
    if 'co2_per_meal' in metrics:
        baseline_co2 = 2.5  # kg CO2 per meal baseline
        carbon_score = (baseline_co2 / max(metrics['co2_per_meal'], 0.1)) * 100
        scores['carbon_efficiency'] = min(100, carbon_score)
    
    # Water efficiency score
    if 'water_per_meal' in metrics:
        baseline_water = 1000  # liters per meal baseline
        water_score = (baseline_water / max(metrics['water_per_meal'], 1)) * 100
        scores['water_efficiency'] = min(100, water_score)
    
    # Procurement sustainability score
    if 'sustainable_procurement_percentage' in metrics:
        scores['sustainable_procurement'] = metrics['sustainable_procurement_percentage']
    
    # Calculate overall score
    if scores:
        overall_score = sum(scores.values()) / len(scores)
        scores['overall'] = overall_score
        scores['rating'] = get_sustainability_rating(overall_score)
    
    return scores


def get_sustainability_rating(score: float) -> str:
    """
    Get sustainability rating based on score
    
    Args:
        score: Sustainability score (0-100)
    
    Returns:
        Rating string
    """
    if score >= 90:
        return "Excellent 🌟"
    elif score >= 75:
        return "Good ✅"
    elif score >= 60:
        return "Fair ⚠️"
    elif score >= 40:
        return "Poor 📉"
    else:
        return "Critical 🚨"


def get_sdg_alignment(metrics: dict) -> dict:
    """
    Check alignment with SDG goals
    
    Args:
        metrics: Dictionary of sustainability metrics
    
    Returns:
        Dictionary with SDG alignment status
    """
    sdg_alignment = {
        'sdg_12_2': {  # Sustainable management of natural resources
            'target': '12.2',
            'description': 'Sustainable management and efficient use of natural resources',
            'status': 'on_track' if metrics.get('water_efficiency', 0) > 70 else 'needs_improvement',
            'progress': metrics.get('water_efficiency', 0)
        },
        'sdg_12_3': {  # Halve food waste
            'target': '12.3',
            'description': 'Halve per capita global food waste',
            'status': 'on_track' if metrics.get('waste_reduction', 0) > 25 else 'needs_improvement',
            'progress': metrics.get('waste_reduction', 0)
        },
        'sdg_12_4': {  # Environmentally sound management
            'target': '12.4',
            'description': 'Environmentally sound management of chemicals and waste',
            'status': 'on_track' if metrics.get('co2_reduction', 0) > 15 else 'needs_improvement',
            'progress': metrics.get('co2_reduction', 0)
        },
        'sdg_12_5': {  # Substantially reduce waste generation
            'target': '12.5',
            'description': 'Substantially reduce waste generation through prevention and recycling',
            'status': 'on_track' if metrics.get('recycling_rate', 0) > 50 else 'needs_improvement',
            'progress': metrics.get('recycling_rate', 0)
        },
        'sdg_12_7': {  # Sustainable procurement
            'target': '12.7',
            'description': 'Promote sustainable procurement practices',
            'status': 'on_track' if metrics.get('sustainable_procurement', 0) > 40 else 'needs_improvement',
            'progress': metrics.get('sustainable_procurement', 0)
        }
    }
    
    return sdg_alignment


def generate_impact_summary(metrics: dict, time_period: str = 'monthly') -> str:
    """
    Generate human-readable impact summary
    
    Args:
        metrics: Dictionary of sustainability metrics
        time_period: Period for the summary (daily, weekly, monthly, yearly)
    
    Returns:
        Formatted impact summary string
    """
    lines = []
    lines.append("🌍 ENVIRONMENTAL IMPACT SUMMARY")
    lines.append("=" * 40)
    
    if time_period == 'daily':
        lines.append(f"📊 Period: Today")
    elif time_period == 'weekly':
        lines.append(f"📊 Period: Last 7 Days")
    elif time_period == 'monthly':
        lines.append(f"📊 Period: This Month")
    elif time_period == 'yearly':
        lines.append(f"📊 Period: This Year")
    
    lines.append("")
    
    # Waste impact
    if 'total_waste' in metrics:
        lines.append(f"🗑️  Food Waste: {metrics['total_waste']:.1f} kg")
        lines.append(f"   • CO₂ Footprint: {metrics.get('total_co2', 0):.1f} kg CO₂")
        lines.append(f"   • Water Footprint: {metrics.get('total_water', 0):,.0f} liters")
        lines.append(f"   • Financial Loss: ₹{metrics.get('total_cost', 0):,.0f}")
    
    lines.append("")
    
    # Reduction achievements
    if 'waste_reduced' in metrics:
        lines.append(f"✅ ACHIEVEMENTS")
        lines.append(f"   • Waste Reduced: {metrics['waste_reduced']:.1f} kg")
        lines.append(f"   • CO₂ Saved: {metrics.get('co2_saved', 0):.1f} kg")
        lines.append(f"   • Water Saved: {metrics.get('water_saved', 0):,.0f} liters")
        lines.append(f"   • Cost Saved: ₹{metrics.get('cost_saved', 0):,.0f}")
    
    lines.append("")
    
    # Sustainability score
    if 'sustainability_score' in metrics:
        score = metrics['sustainability_score']
        rating = get_sustainability_rating(score)
        lines.append(f"🎯 Sustainability Score: {score:.1f}/100 - {rating}")
    
    lines.append("")
    lines.append("📈 SDG 12 PROGRESS")
    
    sdg_alignment = get_sdg_alignment(metrics)
    for sdg, data in sdg_alignment.items():
        status_icon = "✅" if data['status'] == 'on_track' else "⚠️"
        lines.append(f"   {status_icon} {data['target']}: {data['progress']:.1f}%")
    
    lines.append("")
    lines.append("🎯 NEXT TARGETS")
    
    if metrics.get('waste_percentage', 0) > 10:
        lines.append("   • Reduce waste below 10%")
    if metrics.get('sustainable_procurement', 0) < 50:
        lines.append("   • Increase sustainable procurement to 50%")
    if metrics.get('recycling_rate', 0) < 60:
        lines.append("   • Improve recycling rate to 60%")
    
    return "\n".join(lines)


def get_certification_readiness(metrics: dict) -> dict:
    """
    Check readiness for sustainability certifications
    
    Args:
        metrics: Dictionary of sustainability metrics
    
    Returns:
        Dictionary with certification readiness status
    """
    readiness = {}
    
    for cert_id, cert_info in CERTIFICATION_STANDARDS.items():
        requirements_met = 0
        total_requirements = len(cert_info['requirements'])
        
        # Check against metrics
        if 'Waste tracking' in cert_info['requirements'] and 'waste_tracking' in metrics:
            if metrics['waste_tracking']:
                requirements_met += 1
        
        if 'Emission monitoring' in cert_info['requirements'] and 'co2_tracking' in metrics:
            if metrics['co2_tracking']:
                requirements_met += 1
        
        if 'Waste reduction' in cert_info['requirements'] and 'waste_reduction' in metrics:
            if metrics['waste_reduction'] > 20:
                requirements_met += 1
        
        if 'Recycling program' in cert_info['requirements'] and 'recycling_rate' in metrics:
            if metrics['recycling_rate'] > 30:
                requirements_met += 1
        
        if 'Composting' in cert_info['requirements'] and 'composting' in metrics:
            if metrics['composting']:
                requirements_met += 1
        
        readiness[cert_id] = {
            'name': cert_info['name'],
            'readiness_percentage': (requirements_met / total_requirements) * 100 if total_requirements > 0 else 0,
            'requirements_met': requirements_met,
            'total_requirements': total_requirements,
            'status': 'ready' if requirements_met == total_requirements else 'in_progress'
        }
    
    return readiness


# ==================== Initialize Module ====================

import logging

# Set up module logger
sustainability_logger = logging.getLogger(__name__)
sustainability_logger.info(f"Sustainability module initialized (version {__version__})")
sustainability_logger.info("Environmental factors loaded for CO2, water, and land use tracking")
sustainability_logger.info(f"SDG 12 goals tracked: {', '.join([g['sdg_target'] for g in SUSTAINABILITY_GOALS.values()])}")

# Export additional utilities
__all__.extend([
    'ENVIRONMENTAL_FACTORS',
    'WASTE_CATEGORIES',
    'SUSTAINABILITY_GOALS',
    'CERTIFICATION_STANDARDS',
    'get_environmental_factor',
    'calculate_co2_savings',
    'calculate_water_savings',
    'get_sustainability_score',
    'get_sustainability_rating',
    'get_sdg_alignment',
    'generate_impact_summary',
    'get_certification_readiness'
])