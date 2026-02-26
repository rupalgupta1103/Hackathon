"""
Complex queries and analytics for the Smart Mess Optimization System
Provides advanced database queries for reporting, analytics, and insights
"""

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, desc, asc, extract, case, text
from sqlalchemy.sql import label
from datetime import datetime, timedelta, date
from typing import List, Optional, Dict, Any, Tuple, Union
import logging
import json
from collections import defaultdict

from app.database import models
from app.config import config

logger = logging.getLogger(__name__)


class QueryBuilder:
    """
    Advanced query builder for complex database analytics
    """
    
    def __init__(self, db: Session):
        self.db = db
    
    # ==================== Daily Analytics ====================
    
    def get_daily_consumption_summary(self, target_date: Union[datetime, date]) -> Dict[str, Any]:
        """
        Get comprehensive summary of consumption for a specific date
        """
        # Convert to datetime range
        if isinstance(target_date, date) and not isinstance(target_date, datetime):
            start_date = datetime.combine(target_date, datetime.min.time())
            end_date = datetime.combine(target_date, datetime.max.time())
        else:
            start_date = target_date
            end_date = target_date + timedelta(days=1) - timedelta(microseconds=1)
        
        # Get all meals for the date
        meals = self.db.query(models.MealConsumption).filter(
            and_(
                models.MealConsumption.date >= start_date,
                models.MealConsumption.date <= end_date
            )
        ).all()
        
        if not meals:
            return {
                'date': target_date if isinstance(target_date, date) else target_date.date(),
                'has_data': False,
                'total_meals': 0,
                'total_students': 0,
                'total_prepared': 0,
                'total_consumed': 0,
                'total_wasted': 0,
                'waste_percentage': 0,
                'by_meal_type': {},
                'top_dishes': [],
                'waste_by_reason': {}
            }
        
        # Aggregate totals
        total_prepared = sum(m.quantity_prepared for m in meals)
        total_consumed = sum(m.quantity_consumed for m in meals)
        total_wasted = sum(m.quantity_wasted for m in meals)
        total_students = sum(m.student_count for m in meals)
        
        # Group by meal type
        by_meal_type = {}
        for meal in meals:
            if meal.meal_type not in by_meal_type:
                by_meal_type[meal.meal_type] = {
                    'prepared': 0,
                    'consumed': 0,
                    'wasted': 0,
                    'students': 0,
                    'dishes': []
                }
            
            by_meal_type[meal.meal_type]['prepared'] += meal.quantity_prepared
            by_meal_type[meal.meal_type]['consumed'] += meal.quantity_consumed
            by_meal_type[meal.meal_type]['wasted'] += meal.quantity_wasted
            by_meal_type[meal.meal_type]['students'] += meal.student_count
            by_meal_type[meal.meal_type]['dishes'].append({
                'dish_name': meal.dish_name,
                'prepared': meal.quantity_prepared,
                'consumed': meal.quantity_consumed,
                'wasted': meal.quantity_wasted,
                'consumption_rate': meal.consumption_rate
            })
        
        # Get waste logs for the date
        waste_logs = self.db.query(models.WasteLog).filter(
            and_(
                models.WasteLog.date >= start_date,
                models.WasteLog.date <= end_date
            )
        ).all()
        
        waste_by_reason = {}
        for log in waste_logs:
            waste_by_reason[log.waste_reason] = waste_by_reason.get(log.waste_reason, 0) + log.waste_quantity
        
        # Get top dishes by consumption
        top_dishes = sorted(
            [{'dish': m.dish_name, 'consumed': m.quantity_consumed} for m in meals],
            key=lambda x: x['consumed'],
            reverse=True
        )[:5]
        
        return {
            'date': target_date if isinstance(target_date, date) else target_date.date(),
            'has_data': True,
            'total_meals': len(meals),
            'total_students': total_students,
            'total_prepared': round(total_prepared, 2),
            'total_consumed': round(total_consumed, 2),
            'total_wasted': round(total_wasted, 2),
            'waste_percentage': round((total_wasted / total_prepared * 100) if total_prepared > 0 else 0, 2),
            'avg_students_per_meal': round(total_students / len(meals) if meals else 0, 2),
            'by_meal_type': by_meal_type,
            'top_dishes': top_dishes,
            'waste_by_reason': waste_by_reason,
            'environmental_impact': {
                'co2_footprint': round(total_wasted * config.CO2_PER_KG_FOOD, 2),
                'water_footprint': round(total_wasted * config.WATER_PER_KG_FOOD, 2),
                'cost_loss': round(total_wasted * config.WASTE_COST_PER_KG, 2)
            }
        }
    
    def get_daily_comparison(self, date1: Union[datetime, date], date2: Union[datetime, date]) -> Dict[str, Any]:
        """
        Compare consumption between two dates
        """
        summary1 = self.get_daily_consumption_summary(date1)
        summary2 = self.get_daily_consumption_summary(date2)
        
        return {
            'date1': summary1,
            'date2': summary2,
            'comparison': {
                'students_change': summary2['total_students'] - summary1['total_students'],
                'students_change_pct': self._calculate_change_percentage(
                    summary1['total_students'], summary2['total_students']
                ),
                'waste_change': summary2['total_wasted'] - summary1['total_wasted'],
                'waste_change_pct': self._calculate_change_percentage(
                    summary1['total_wasted'], summary2['total_wasted']
                ),
                'waste_percentage_change': summary2['waste_percentage'] - summary1['waste_percentage']
            }
        }
    
    # ==================== Weekly/Monthly Analytics ====================
    
    def get_weekly_summary(self, end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Get summary for the last 7 days
        """
        if end_date is None:
            end_date = datetime.utcnow()
        
        start_date = end_date - timedelta(days=7)
        
        meals = self.db.query(models.MealConsumption).filter(
            and_(
                models.MealConsumption.date >= start_date,
                models.MealConsumption.date <= end_date
            )
        ).all()
        
        # Group by day
        daily_stats = {}
        for meal in meals:
            day = meal.date.date()
            if day not in daily_stats:
                daily_stats[day] = {
                    'total_students': 0,
                    'total_prepared': 0,
                    'total_consumed': 0,
                    'total_wasted': 0
                }
            
            daily_stats[day]['total_students'] += meal.student_count
            daily_stats[day]['total_prepared'] += meal.quantity_prepared
            daily_stats[day]['total_consumed'] += meal.quantity_consumed
            daily_stats[day]['total_wasted'] += meal.quantity_wasted
        
        # Calculate trends
        days = sorted(daily_stats.keys())
        waste_trend = [daily_stats[day]['total_wasted'] for day in days]
        
        # Calculate averages
        avg_daily_waste = sum(d['total_wasted'] for d in daily_stats.values()) / max(len(daily_stats), 1)
        
        return {
            'period': {
                'start': start_date.date(),
                'end': end_date.date(),
                'days': (end_date - start_date).days
            },
            'daily_stats': daily_stats,
            'totals': {
                'total_students': sum(d['total_students'] for d in daily_stats.values()),
                'total_prepared': sum(d['total_prepared'] for d in daily_stats.values()),
                'total_consumed': sum(d['total_consumed'] for d in daily_stats.values()),
                'total_wasted': sum(d['total_wasted'] for d in daily_stats.values())
            },
            'averages': {
                'daily_students': round(sum(d['total_students'] for d in daily_stats.values()) / max(len(daily_stats), 1), 2),
                'daily_waste': round(avg_daily_waste, 2),
                'waste_percentage': round(
                    (sum(d['total_wasted'] for d in daily_stats.values()) / 
                     max(sum(d['total_prepared'] for d in daily_stats.values()), 1)) * 100, 2
                )
            },
            'trends': {
                'waste_trend': waste_trend,
                'waste_increasing': len(waste_trend) > 1 and waste_trend[-1] > waste_trend[0]
            }
        }
    
    def get_monthly_summary(self, year: int, month: int) -> Dict[str, Any]:
        """
        Get summary for a specific month
        """
        start_date = datetime(year, month, 1)
        if month == 12:
            end_date = datetime(year + 1, 1, 1) - timedelta(microseconds=1)
        else:
            end_date = datetime(year, month + 1, 1) - timedelta(microseconds=1)
        
        meals = self.db.query(models.MealConsumption).filter(
            and_(
                models.MealConsumption.date >= start_date,
                models.MealConsumption.date <= end_date
            )
        ).all()
        
        # Group by day of week
        dow_stats = {i: {'total_students': 0, 'total_wasted': 0, 'count': 0} for i in range(7)}
        
        for meal in meals:
            dow = meal.day_of_week
            dow_stats[dow]['total_students'] += meal.student_count
            dow_stats[dow]['total_wasted'] += meal.quantity_wasted
            dow_stats[dow]['count'] += 1
        
        # Calculate day-of-week averages
        dow_averages = {}
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        for i, day_name in enumerate(days):
            if dow_stats[i]['count'] > 0:
                dow_averages[day_name] = {
                    'avg_students': round(dow_stats[i]['total_students'] / dow_stats[i]['count'], 2),
                    'avg_waste': round(dow_stats[i]['total_wasted'] / dow_stats[i]['count'], 2)
                }
        
        return {
            'month': f"{year}-{month:02d}",
            'total_days': (end_date - start_date).days,
            'total_meals': len(meals),
            'total_students': sum(m.student_count for m in meals),
            'total_prepared': sum(m.quantity_prepared for m in meals),
            'total_consumed': sum(m.quantity_consumed for m in meals),
            'total_wasted': sum(m.quantity_wasted for m in meals),
            'waste_percentage': round(
                (sum(m.quantity_wasted for m in meals) / max(sum(m.quantity_prepared for m in meals), 1)) * 100, 2
            ),
            'day_of_week_averages': dow_averages,
            'environmental_impact': {
                'total_co2': round(sum(m.quantity_wasted for m in meals) * config.CO2_PER_KG_FOOD, 2),
                'total_water': round(sum(m.quantity_wasted for m in meals) * config.WATER_PER_KG_FOOD, 2),
                'total_cost_loss': round(sum(m.quantity_wasted for m in meals) * config.WASTE_COST_PER_KG, 2)
            }
        }
    
    # ==================== Waste Analytics ====================
    
    def get_waste_analysis(self, days: int = 30) -> Dict[str, Any]:
        """
        Comprehensive waste analysis for the last N days
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Get waste logs
        waste_logs = self.db.query(models.WasteLog).filter(
            models.WasteLog.date >= cutoff_date
        ).all()
        
        if not waste_logs:
            return {
                'period_days': days,
                'has_data': False,
                'total_waste': 0,
                'avg_daily_waste': 0
            }
        
        # Basic statistics
        total_waste = sum(log.waste_quantity for log in waste_logs)
        avg_daily_waste = total_waste / days
        
        # Waste by reason
        waste_by_reason = {}
        for log in waste_logs:
            waste_by_reason[log.waste_reason] = waste_by_reason.get(log.waste_reason, 0) + log.waste_quantity
        
        # Waste by meal type
        waste_by_meal = {}
        for log in waste_logs:
            waste_by_meal[log.meal_type] = waste_by_meal.get(log.meal_type, 0) + log.waste_quantity
        
        # Waste by dish
        waste_by_dish = {}
        for log in waste_logs:
            if log.dish_name not in waste_by_dish:
                waste_by_dish[log.dish_name] = {
                    'total_waste': 0,
                    'incidents': 0,
                    'avg_waste_per_incident': 0
                }
            waste_by_dish[log.dish_name]['total_waste'] += log.waste_quantity
            waste_by_dish[log.dish_name]['incidents'] += 1
        
        for dish in waste_by_dish:
            waste_by_dish[dish]['avg_waste_per_incident'] = (
                waste_by_dish[dish]['total_waste'] / waste_by_dish[dish]['incidents']
            )
        
        # Top 10 wasted dishes
        top_wasted = sorted(
            [{'dish': k, 'waste': v['total_waste']} for k, v in waste_by_dish.items()],
            key=lambda x: x['waste'],
            reverse=True
        )[:10]
        
        # Daily waste trend
        daily_trend = {}
        for log in waste_logs:
            day = log.date.date()
            daily_trend[day] = daily_trend.get(day, 0) + log.waste_quantity
        
        # Calculate if waste is increasing
        sorted_days = sorted(daily_trend.keys())
        if len(sorted_days) >= 7:
            first_week_avg = sum(daily_trend[day] for day in sorted_days[:7]) / 7
            last_week_avg = sum(daily_trend[day] for day in sorted_days[-7:]) / 7
            trend_direction = 'increasing' if last_week_avg > first_week_avg else 'decreasing'
            trend_percentage = ((last_week_avg - first_week_avg) / first_week_avg * 100) if first_week_avg > 0 else 0
        else:
            trend_direction = 'insufficient_data'
            trend_percentage = 0
        
        return {
            'period_days': days,
            'has_data': True,
            'total_waste': round(total_waste, 2),
            'avg_daily_waste': round(avg_daily_waste, 2),
            'waste_by_reason': {k: round(v, 2) for k, v in waste_by_reason.items()},
            'waste_by_meal': {k: round(v, 2) for k, v in waste_by_meal.items()},
            'top_wasted_dishes': top_wasted,
            'daily_trend': {str(k): round(v, 2) for k, v in daily_trend.items()},
            'trend': {
                'direction': trend_direction,
                'percentage': round(trend_percentage, 2)
            },
            'environmental_impact': {
                'total_co2': round(total_waste * config.CO2_PER_KG_FOOD, 2),
                'total_water': round(total_waste * config.WATER_PER_KG_FOOD, 2),
                'total_cost': round(total_waste * config.WASTE_COST_PER_KG, 2)
            },
            'recommendations': self._generate_waste_recommendations(waste_by_reason, top_wasted)
        }
    
    def _generate_waste_recommendations(self, waste_by_reason: Dict, top_wasted: List) -> List[str]:
        """Generate recommendations based on waste analysis"""
        recommendations = []
        
        # Recommendations based on waste reasons
        if waste_by_reason.get('overproduction', 0) > waste_by_reason.get('total', 1) * 0.3:
            recommendations.append("Consider reducing portion sizes or preparation quantities")
        
        if waste_by_reason.get('quality_issue', 0) > 0:
            recommendations.append("Review ingredient quality and storage conditions")
        
        if waste_by_reason.get('spoilage', 0) > 0:
            recommendations.append("Improve inventory rotation (FIFO) and storage practices")
        
        # Recommendations based on top wasted dishes
        if top_wasted and top_wasted[0]['waste'] > 0:
            dish = top_wasted[0]['dish']
            recommendations.append(f"Review popularity of '{dish}' - consider replacing or modifying recipe")
        
        return recommendations
    
    # ==================== Dish Popularity Analytics ====================
    
    def get_dish_popularity_ranking(self, days: int = 30) -> List[Dict[str, Any]]:
        """
        Get ranking of dishes by popularity (consumption rate)
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Get consumption data for each dish
        dish_stats = self.db.query(
            models.MealConsumption.dish_name,
            func.sum(models.MealConsumption.quantity_prepared).label('total_prepared'),
            func.sum(models.MealConsumption.quantity_consumed).label('total_consumed'),
            func.avg(models.MealConsumption.student_count).label('avg_students'),
            func.count(models.MealConsumption.id).label('times_served')
        ).filter(
            models.MealConsumption.date >= cutoff_date
        ).group_by(
            models.MealConsumption.dish_name
        ).all()
        
        rankings = []
        for dish in dish_stats:
            if dish.total_prepared > 0:
                consumption_rate = (dish.total_consumed / dish.total_prepared * 100)
                
                # Get waste data for this dish
                waste_total = self.db.query(
                    func.sum(models.WasteLog.waste_quantity)
                ).filter(
                    and_(
                        models.WasteLog.dish_name == dish.dish_name,
                        models.WasteLog.date >= cutoff_date
                    )
                ).scalar() or 0
                
                rankings.append({
                    'dish_name': dish.dish_name,
                    'total_prepared': round(float(dish.total_prepared), 2),
                    'total_consumed': round(float(dish.total_consumed), 2),
                    'consumption_rate': round(consumption_rate, 2),
                    'avg_students': round(float(dish.avg_students), 2),
                    'times_served': dish.times_served,
                    'total_waste': round(float(waste_total), 2),
                    'waste_percentage': round((waste_total / dish.total_prepared * 100) if dish.total_prepared > 0 else 0, 2),
                    'popularity_score': round(consumption_rate / 100, 3)
                })
        
        # Sort by consumption rate descending
        rankings.sort(key=lambda x: x['consumption_rate'], reverse=True)
        
        return rankings
    
    def get_dish_performance_trend(self, dish_name: str, days: int = 90) -> Dict[str, Any]:
        """
        Get performance trend for a specific dish
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Get all instances of this dish
        meals = self.db.query(models.MealConsumption).filter(
            and_(
                models.MealConsumption.dish_name == dish_name,
                models.MealConsumption.date >= cutoff_date
            )
        ).order_by(models.MealConsumption.date).all()
        
        if not meals:
            return {'dish_name': dish_name, 'has_data': False}
        
        # Calculate trends
        dates = []
        consumption_rates = []
        waste_amounts = []
        student_counts = []
        
        for meal in meals:
            dates.append(meal.date.strftime('%Y-%m-%d'))
            consumption_rates.append(meal.consumption_rate)
            waste_amounts.append(meal.quantity_wasted)
            student_counts.append(meal.student_count)
        
        # Calculate moving average
        moving_avg = []
        window = min(7, len(consumption_rates))
        for i in range(len(consumption_rates)):
            start = max(0, i - window + 1)
            avg = sum(consumption_rates[start:i+1]) / (i - start + 1)
            moving_avg.append(round(avg, 2))
        
        return {
            'dish_name': dish_name,
            'has_data': True,
            'total_times_served': len(meals),
            'avg_consumption_rate': round(sum(consumption_rates) / len(consumption_rates), 2),
            'avg_students': round(sum(student_counts) / len(student_counts), 2),
            'total_waste': round(sum(waste_amounts), 2),
            'trend': {
                'dates': dates,
                'consumption_rates': consumption_rates,
                'moving_avg': moving_avg,
                'waste_amounts': waste_amounts,
                'student_counts': student_counts
            },
            'performance': 'improving' if consumption_rates[-1] > consumption_rates[0] else 'declining'
        }
    
    # ==================== Procurement Analytics ====================
    
    def get_procurement_analysis(self, days: int = 30) -> Dict[str, Any]:
        """
        Analyze procurement efficiency and patterns
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        procurements = self.db.query(models.Procurement).filter(
            models.Procurement.date >= cutoff_date
        ).all()
        
        if not procurements:
            return {'period_days': days, 'has_data': False}
        
        # Basic statistics
        total_spent = sum(p.total_cost for p in procurements)
        total_quantity = sum(p.quantity for p in procurements)
        
        # Group by ingredient
        by_ingredient = {}
        for p in procurements:
            if p.ingredient_name not in by_ingredient:
                by_ingredient[p.ingredient_name] = {
                    'total_quantity': 0,
                    'total_cost': 0,
                    'avg_price': 0,
                    'purchase_count': 0,
                    'vendors': set()
                }
            
            by_ingredient[p.ingredient_name]['total_quantity'] += p.quantity
            by_ingredient[p.ingredient_name]['total_cost'] += p.total_cost
            by_ingredient[p.ingredient_name]['purchase_count'] += 1
            by_ingredient[p.ingredient_name]['vendors'].add(p.vendor)
        
        # Calculate averages and clean up
        for ingredient, data in by_ingredient.items():
            data['avg_price'] = data['total_cost'] / data['total_quantity'] if data['total_quantity'] > 0 else 0
            data['vendors'] = list(data['vendors'])
        
        # Prediction accuracy
        predictions = [p for p in procurements if p.actual_used is not None]
        if predictions:
            accuracy_sum = sum(
                min(p.actual_used, p.predicted_quantity) / max(p.predicted_quantity, 1) * 100
                for p in predictions
            )
            avg_accuracy = accuracy_sum / len(predictions)
        else:
            avg_accuracy = None
        
        # Vendor analysis
        vendor_stats = {}
        for p in procurements:
            if p.vendor not in vendor_stats:
                vendor_stats[p.vendor] = {
                    'total_spent': 0,
                    'purchase_count': 0,
                    'avg_quality': []
                }
            vendor_stats[p.vendor]['total_spent'] += p.total_cost
            vendor_stats[p.vendor]['purchase_count'] += 1
            if p.quality_rating:
                vendor_stats[p.vendor]['avg_quality'].append(p.quality_rating)
        
        for vendor in vendor_stats:
            if vendor_stats[vendor]['avg_quality']:
                vendor_stats[vendor]['avg_quality'] = sum(vendor_stats[vendor]['avg_quality']) / len(vendor_stats[vendor]['avg_quality'])
            else:
                vendor_stats[vendor]['avg_quality'] = None
        
        return {
            'period_days': days,
            'has_data': True,
            'total_spent': round(total_spent, 2),
            'total_quantity': round(total_quantity, 2),
            'avg_price_per_kg': round(total_spent / total_quantity if total_quantity > 0 else 0, 2),
            'purchase_count': len(procurements),
            'by_ingredient': {
                k: {
                    'total_quantity': round(v['total_quantity'], 2),
                    'total_cost': round(v['total_cost'], 2),
                    'avg_price': round(v['avg_price'], 2),
                    'purchase_count': v['purchase_count'],
                    'vendors': v['vendors']
                }
                for k, v in by_ingredient.items()
            },
            'prediction_accuracy': round(avg_accuracy, 2) if avg_accuracy else None,
            'vendor_analysis': vendor_stats
        }
    
    def get_ingredient_usage_forecast(self, days_ahead: int = 7) -> Dict[str, float]:
        """
        Forecast ingredient usage based on historical patterns
        """
        # Get historical usage (last 60 days)
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=60)
        
        procurements = self.db.query(models.Procurement).filter(
            and_(
                models.Procurement.date >= start_date,
                models.Procurement.date <= end_date
            )
        ).all()
        
        # Calculate daily usage patterns
        usage_by_ingredient = defaultdict(list)
        for p in procurements:
            if p.actual_used:
                usage_by_ingredient[p.ingredient_name].append(p.actual_used)
            else:
                usage_by_ingredient[p.ingredient_name].append(p.quantity)
        
        # Calculate forecast with trend adjustment
        forecast = {}
        for ingredient, quantities in usage_by_ingredient.items():
            if quantities:
                # Simple moving average
                recent = quantities[-14:]  # Last 14 days
                avg_daily = sum(recent) / len(recent) if recent else sum(quantities) / len(quantities)
                
                # Add trend factor (increase forecast for trending ingredients)
                if len(quantities) >= 14:
                    first_week = sum(quantities[:7]) / 7
                    last_week = sum(quantities[-7:]) / 7
                    trend_factor = max(0.8, min(1.2, last_week / first_week if first_week > 0 else 1))
                else:
                    trend_factor = 1
                
                forecast[ingredient] = round(avg_daily * days_ahead * trend_factor, 2)
        
        return forecast
    
    # ==================== Sustainability Analytics ====================
    
    def get_sustainability_kpis(self) -> Dict[str, Any]:
        """
        Get key sustainability indicators
        """
        now = datetime.utcnow()
        
        # Current month
        month_start = datetime(now.year, now.month, 1)
        
        # Previous month
        if now.month == 1:
            prev_month_start = datetime(now.year - 1, 12, 1)
            prev_month_end = datetime(now.year, now.month, 1) - timedelta(days=1)
        else:
            prev_month_start = datetime(now.year, now.month - 1, 1)
            prev_month_end = month_start - timedelta(days=1)
        
        # Current month waste
        current_waste = self.db.query(
            func.sum(models.WasteLog.waste_quantity).label('total_waste'),
            func.sum(models.WasteLog.co2_footprint).label('total_co2'),
            func.sum(models.WasteLog.cost_loss).label('total_cost')
        ).filter(models.WasteLog.date >= month_start).first()
        
        # Previous month waste
        prev_waste = self.db.query(
            func.sum(models.WasteLog.waste_quantity).label('total_waste')
        ).filter(
            and_(
                models.WasteLog.date >= prev_month_start,
                models.WasteLog.date <= prev_month_end
            )
        ).first()
        
        # Current month procurement
        current_procurement = self.db.query(
            func.sum(models.Procurement.total_cost).label('total_cost')
        ).filter(models.Procurement.date >= month_start).first()
        
        # Calculate improvements
        waste_reduction = 0
        if prev_waste.total_waste and prev_waste.total_waste > 0:
            waste_reduction = ((prev_waste.total_waste - (current_waste.total_waste or 0)) / prev_waste.total_waste) * 100
        
        # Get waste by reason for current month
        waste_by_reason = self.db.query(
            models.WasteLog.waste_reason,
            func.sum(models.WasteLog.waste_quantity).label('total')
        ).filter(models.WasteLog.date >= month_start
        ).group_by(models.WasteLog.waste_reason).all()
        
        # Calculate sustainability score (0-100)
        waste_score = max(0, 100 - ((current_waste.total_waste or 0) / 10))
        procurement_score = 70  # Placeholder - would need more data
        overall_score = (waste_score * 0.6 + procurement_score * 0.4)
        
        return {
            'current_month': {
                'total_waste': round(float(current_waste.total_waste or 0), 2),
                'total_co2': round(float(current_waste.total_co2 or 0), 2),
                'total_cost': round(float(current_waste.total_cost or 0), 2),
                'procurement_cost': round(float(current_procurement.total_cost or 0), 2),
                'waste_by_reason': {r.waste_reason: round(r.total, 2) for r in waste_by_reason}
            },
            'improvements': {
                'waste_reduction_percentage': round(waste_reduction, 2),
                'waste_reduction_amount': round((prev_waste.total_waste or 0) - (current_waste.total_waste or 0), 2)
            },
            'projections': {
                'annual_waste': round(float(current_waste.total_waste or 0) * 12, 2),
                'annual_co2': round(float(current_waste.total_co2 or 0) * 12, 2),
                'annual_savings_potential': round(float(current_waste.total_cost or 0) * 12 * 0.2, 2)  # 20% reduction potential
            },
            'sustainability_score': {
                'overall': round(overall_score, 2),
                'waste_score': round(waste_score, 2),
                'procurement_score': round(procurement_score, 2),
                'rating': self._get_sustainability_rating(overall_score)
            }
        }
    
    def _get_sustainability_rating(self, score: float) -> str:
        """Get sustainability rating based on score"""
        if score >= 90:
            return "Excellent"
        elif score >= 75:
            return "Good"
        elif score >= 60:
            return "Fair"
        elif score >= 40:
            return "Poor"
        else:
            return "Critical"
    
    # ==================== Pattern Analysis ====================
    
    def get_meal_pattern_analysis(self) -> Dict[str, Any]:
        """
        Analyze meal patterns and attendance trends
        """
        # Get last 90 days of data
        cutoff_date = datetime.utcnow() - timedelta(days=90)
        
        meals = self.db.query(models.MealConsumption).filter(
            models.MealConsumption.date >= cutoff_date
        ).all()
        
        # Group by day of week and meal type
        patterns = {}
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        for meal in meals:
            day_name = days[meal.day_of_week]
            if day_name not in patterns:
                patterns[day_name] = {}
            
            if meal.meal_type not in patterns[day_name]:
                patterns[day_name][meal.meal_type] = {
                    'total_students': 0,
                    'total_waste': 0,
                    'count': 0,
                    'dishes': set()
                }
            
            patterns[day_name][meal.meal_type]['total_students'] += meal.student_count
            patterns[day_name][meal.meal_type]['total_waste'] += meal.quantity_wasted
            patterns[day_name][meal.meal_type]['count'] += 1
            patterns[day_name][meal.meal_type]['dishes'].add(meal.dish_name)
        
        # Calculate averages and clean up
        for day in patterns:
            for meal_type in patterns[day]:
                count = patterns[day][meal_type]['count']
                if count > 0:
                    patterns[day][meal_type]['avg_students'] = round(
                        patterns[day][meal_type]['total_students'] / count, 2
                    )
                    patterns[day][meal_type]['avg_waste'] = round(
                        patterns[day][meal_type]['total_waste'] / count, 2
                    )
                    patterns[day][meal_type]['unique_dishes'] = len(patterns[day][meal_type]['dishes'])
                
                # Remove raw data
                del patterns[day][meal_type]['total_students']
                del patterns[day][meal_type]['total_waste']
                del patterns[day][meal_type]['count']
                del patterns[day][meal_type]['dishes']
        
        # Find best and worst days
        avg_attendance = {}
        for day in patterns:
            total_students = sum(m.get('avg_students', 0) for m in patterns[day].values())
            avg_attendance[day] = total_students / len(patterns[day]) if patterns[day] else 0
        
        best_day = max(avg_attendance, key=avg_attendance.get) if avg_attendance else None
        worst_day = min(avg_attendance, key=avg_attendance.get) if avg_attendance else None
        
        return {
            'patterns': patterns,
            'insights': {
                'best_attendance_day': best_day,
                'worst_attendance_day': worst_day,
                'average_daily_attendance': round(sum(avg_attendance.values()) / len(avg_attendance) if avg_attendance else 0, 2),
                'weekend_vs_weekday': self