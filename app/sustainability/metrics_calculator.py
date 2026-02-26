import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

class SustainabilityMetrics:
    def __init__(self, config):
        self.config = config
        self.co2_factor = config.CO2_PER_KG_FOOD
        self.water_factor = config.WATER_PER_KG_FOOD
        self.cost_per_kg = config.WASTE_COST_PER_KG
        
    def calculate_waste_metrics(self, 
                               waste_logs: List[Dict],
                               time_period_days: int = 30) -> Dict:
        """Calculate comprehensive waste metrics"""
        
        if not waste_logs:
            return self.get_empty_metrics()
        
        df = pd.DataFrame(waste_logs)
        
        # Basic waste statistics
        total_waste_kg = df['waste_quantity'].sum()
        avg_daily_waste = total_waste_kg / time_period_days
        
        # Cost impact
        total_cost_loss = df['cost_loss'].sum() if 'cost_loss' in df.columns else total_waste_kg * self.cost_per_kg
        monthly_cost_loss = total_cost_loss
        
        # Environmental impact
        total_co2 = total_waste_kg * self.co2_factor
        total_water = total_waste_kg * self.water_factor
        
        # Waste breakdown by meal type
        waste_by_meal = df.groupby('meal_type')['waste_quantity'].sum().to_dict()
        
        # Waste breakdown by reason
        waste_by_reason = df.groupby('waste_reason')['waste_quantity'].sum().to_dict()
        
        # Calculate potential savings with 10% reduction
        potential_savings = {
            'cost': monthly_cost_loss * 0.1,
            'co2': total_co2 * 0.1,
            'water': total_water * 0.1
        }
        
        return {
            'total_waste_kg': total_waste_kg,
            'avg_daily_waste_kg': avg_daily_waste,
            'total_cost_loss': monthly_cost_loss,
            'annual_cost_loss': monthly_cost_loss * 12,
            'total_co2_kg': total_co2,
            'annual_co2_kg': total_co2 * 12,
            'total_water_liters': total_water,
            'annual_water_liters': total_water * 12,
            'waste_by_meal': waste_by_meal,
            'waste_by_reason': waste_by_reason,
            'potential_savings_10pc': potential_savings
        }
    
    def calculate_sustainability_score(self, 
                                     waste_metrics: Dict,
                                     procurement_efficiency: float,
                                     renewable_percentage: float = 0) -> Dict:
        """Calculate overall sustainability score (0-100)"""
        
        # Waste reduction score (40% weight)
        if waste_metrics['total_waste_kg'] > 0:
            waste_score = max(0, 100 - (waste_metrics['avg_daily_waste_kg'] / 10))
        else:
            waste_score = 100
        
        # Procurement efficiency score (30% weight)
        procurement_score = min(100, procurement_efficiency * 100)
        
        # Renewable/eco-friendly score (30% weight)
        renewable_score = renewable_percentage * 100
        
        # Weighted average
        overall_score = (
            waste_score * 0.4 +
            procurement_score * 0.3 +
            renewable_score * 0.3
        )
        
        return {
            'overall_score': round(overall_score, 2),
            'waste_score': round(waste_score, 2),
            'procurement_score': round(procurement_score, 2),
            'renewable_score': round(renewable_score, 2),
            'rating': self.get_sustainability_rating(overall_score)
        }
    
    def get_sustainability_rating(self, score: float) -> str:
        """Convert score to rating"""
        if score >= 90:
            return "Excellent 🌟"
        elif score >= 75:
            return "Good ✅"
        elif score >= 60:
            return "Fair ⚠️"
        else:
            return "Needs Improvement ❌"
    
    def calculate_roi(self, 
                     initial_investment: float,
                     monthly_savings: float,
                     months: int = 12) -> Dict:
        """Calculate ROI for the system"""
        
        total_savings = monthly_savings * months
        net_profit = total_savings - initial_investment
        roi_percentage = (net_profit / initial_investment) * 100 if initial_investment > 0 else 0
        
        payback_months = initial_investment / monthly_savings if monthly_savings > 0 else float('inf')
        
        return {
            'initial_investment': initial_investment,
            'monthly_savings': monthly_savings,
            'annual_savings': total_savings,
            'net_profit': net_profit,
            'roi_percentage': round(roi_percentage, 2),
            'payback_months': round(payback_months, 1)
        }
    
    def get_empty_metrics(self) -> Dict:
        """Return empty metrics structure"""
        return {
            'total_waste_kg': 0,
            'avg_daily_waste_kg': 0,
            'total_cost_loss': 0,
            'annual_cost_loss': 0,
            'total_co2_kg': 0,
            'annual_co2_kg': 0,
            'total_water_liters': 0,
            'annual_water_liters': 0,
            'waste_by_meal': {},
            'waste_by_reason': {},
            'potential_savings_10pc': {
                'cost': 0,
                'co2': 0,
                'water': 0
            }
        }
    
    def generate_impact_report(self, metrics: Dict) -> str:
        """Generate a human-readable impact report"""
        
        report = f"""
        🌱 SUSTAINABILITY IMPACT REPORT 🌱
        ================================
        
        📊 Waste Statistics:
        • Total Waste: {metrics['total_waste_kg']:.1f} kg
        • Daily Average: {metrics['avg_daily_waste_kg']:.1f} kg/day
        
        💰 Financial Impact:
        • Monthly Loss: ₹{metrics['total_cost_loss']:,.0f}
        • Annual Loss: ₹{metrics['annual_cost_loss']:,.0f}
        
        🌍 Environmental Impact:
        • CO₂ Footprint: {metrics['total_co2_kg']:.1f} kg CO₂/month
        • Water Footprint: {metrics['total_water_liters']:,.0f} liters/month
        
        🎯 With 10% Reduction:
        • Save: ₹{metrics['potential_savings_10pc']['cost']:,.0f}/month
        • Prevent: {metrics['potential_savings_10pc']['co2']:.1f} kg CO₂/month
        • Conserve: {metrics['potential_savings_10pc']['water']:,.0f} liters/month
        
        For a 2000-student hostel, even 10% waste reduction saves 
        ₹{metrics['annual_cost_loss']*0.1:,.0f} annually and prevents 
        {metrics['annual_co2_kg']*0.1:.1f} kg CO₂ emissions.
        """
        
        return report