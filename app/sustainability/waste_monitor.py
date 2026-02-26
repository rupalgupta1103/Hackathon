"""
Waste Monitor Module
Real-time tracking, analysis, and reporting of food waste
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, Counter
import logging
from enum import Enum
from dataclasses import dataclass, field
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.config import config

logger = logging.getLogger(__name__)


class WasteCategory(str, Enum):
    """Categories of food waste"""
    PREPARATION = "preparation"  # Waste during food prep (peels, trimmings)
    SERVING = "serving"  # Plate waste, uneaten food
    SPOILAGE = "spoilage"  # Food spoiled before use
    OVERPRODUCTION = "overproduction"  # Excess food prepared
    QUALITY_ISSUE = "quality_issue"  # Rejected due to quality
    STORAGE_LOSS = "storage_loss"  # Lost during storage
    TRANSPORT_DAMAGE = "transport_damage"  # Damaged during transport
    EXPIRY = "expiry"  # Expired before use


class WasteSeverity(str, Enum):
    """Severity levels for waste incidents"""
    LOW = "low"  # < 1 kg
    MEDIUM = "medium"  # 1-5 kg
    HIGH = "high"  # 5-20 kg
    CRITICAL = "critical"  # > 20 kg


@dataclass
class WasteIncident:
    """Represents a single waste incident"""
    id: str
    timestamp: datetime
    category: WasteCategory
    meal_type: str
    dish_name: str
    quantity_kg: float
    reason: str
    severity: WasteSeverity
    cost_loss: float = 0.0
    co2_footprint: float = 0.0
    water_footprint: float = 0.0
    notes: Optional[str] = None
    reported_by: Optional[str] = None
    actions_taken: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Calculate environmental impact after initialization"""
        if self.cost_loss == 0:
            self.cost_loss = self.quantity_kg * config.WASTE_COST_PER_KG
        if self.co2_footprint == 0:
            self.co2_footprint = self.quantity_kg * config.CO2_PER_KG_FOOD
        if self.water_footprint == 0:
            self.water_footprint = self.quantity_kg * config.WATER_PER_KG_FOOD


@dataclass
class WasteSummary:
    """Summary of waste for a time period"""
    period_start: datetime
    period_end: datetime
    total_waste_kg: float = 0.0
    total_cost_loss: float = 0.0
    total_co2: float = 0.0
    total_water: float = 0.0
    by_category: Dict[WasteCategory, float] = field(default_factory=dict)
    by_meal: Dict[str, float] = field(default_factory=dict)
    by_dish: Dict[str, float] = field(default_factory=dict)
    by_severity: Dict[WasteSeverity, int] = field(default_factory=dict)
    incidents_count: int = 0
    top_reasons: List[Tuple[str, float]] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'period': {
                'start': self.period_start.isoformat(),
                'end': self.period_end.isoformat()
            },
            'total_waste_kg': round(self.total_waste_kg, 2),
            'total_cost_loss': round(self.total_cost_loss, 2),
            'total_co2_kg': round(self.total_co2, 2),
            'total_water_liters': round(self.total_water, 2),
            'by_category': {k.value: round(v, 2) for k, v in self.by_category.items()},
            'by_meal': {k: round(v, 2) for k, v in self.by_meal.items()},
            'by_dish': dict(sorted(self.by_dish.items(), key=lambda x: x[1], reverse=True)[:10]),
            'incidents_count': self.incidents_count,
            'top_reasons': [(r, round(w, 2)) for r, w in self.top_reasons[:5]]
        }


class WasteMonitor:
    """
    Real-time waste monitoring and analysis system
    """
    
    def __init__(self, db_session=None):
        self.db = db_session
        self.incidents: List[WasteIncident] = []
        self.alerts_enabled = True
        self.waste_thresholds = {
            WasteSeverity.LOW: 1.0,
            WasteSeverity.MEDIUM: 5.0,
            WasteSeverity.HIGH: 20.0,
            WasteSeverity.CRITICAL: float('inf')
        }
        
        # Alert thresholds
        self.alert_thresholds = {
            'daily_waste': 50.0,  # Alert if daily waste > 50 kg
            'per_incident': 10.0,  # Alert if single incident > 10 kg
            'cost_per_day': 5000,  # Alert if daily cost loss > ₹5000
            'repetition': 3  # Alert if same dish wasted > 3 times in week
        }
        
        logger.info("Waste Monitor initialized")
    
    def log_incident(self, 
                     category: WasteCategory,
                     meal_type: str,
                     dish_name: str,
                     quantity_kg: float,
                     reason: str,
                     notes: Optional[str] = None,
                     reported_by: Optional[str] = None) -> WasteIncident:
        """
        Log a new waste incident
        
        Args:
            category: Category of waste
            meal_type: Type of meal (breakfast, lunch, dinner)
            dish_name: Name of the dish
            quantity_kg: Quantity wasted in kg
            reason: Reason for waste
            notes: Additional notes
            reported_by: Person reporting the incident
            
        Returns:
            Created WasteIncident object
        """
        # Determine severity
        severity = self._determine_severity(quantity_kg)
        
        # Create incident
        incident = WasteIncident(
            id=f"W{datetime.utcnow().strftime('%Y%m%d%H%M%S')}{len(self.incidents)}",
            timestamp=datetime.utcnow(),
            category=category,
            meal_type=meal_type,
            dish_name=dish_name,
            quantity_kg=quantity_kg,
            reason=reason,
            severity=severity,
            notes=notes,
            reported_by=reported_by
        )
        
        self.incidents.append(incident)
        
        # Save to database if available
        if self.db:
            self._save_to_db(incident)
        
        # Check for alerts
        if self.alerts_enabled:
            self._check_alerts(incident)
        
        logger.info(f"Logged waste incident: {incident.id} - {quantity_kg}kg of {dish_name} ({category.value})")
        
        return incident
    
    def _determine_severity(self, quantity_kg: float) -> WasteSeverity:
        """Determine severity level based on quantity"""
        if quantity_kg < self.waste_thresholds[WasteSeverity.LOW]:
            return WasteSeverity.LOW
        elif quantity_kg < self.waste_thresholds[WasteSeverity.MEDIUM]:
            return WasteSeverity.MEDIUM
        elif quantity_kg < self.waste_thresholds[WasteSeverity.HIGH]:
            return WasteSeverity.HIGH
        else:
            return WasteSeverity.CRITICAL
    
    def _save_to_db(self, incident: WasteIncident) -> None:
        """Save incident to database"""
        try:
            from app.database.models import WasteLog
            from app.database.database import SessionLocal
            
            db = SessionLocal()
            waste_log = WasteLog(
                date=incident.timestamp,
                meal_type=incident.meal_type,
                dish_name=incident.dish_name,
                waste_quantity=incident.quantity_kg,
                waste_reason=incident.reason,
                co2_footprint=incident.co2_footprint,
                water_footprint=incident.water_footprint,
                cost_loss=incident.cost_loss,
                notes=incident.notes
            )
            db.add(waste_log)
            db.commit()
            db.close()
        except Exception as e:
            logger.error(f"Failed to save waste incident to database: {e}")
    
    def _check_alerts(self, incident: WasteIncident) -> None:
        """Check if incident triggers any alerts"""
        alerts = []
        
        # Check per-incident threshold
        if incident.quantity_kg > self.alert_thresholds['per_incident']:
            alerts.append(f"Large waste incident: {incident.quantity_kg}kg of {incident.dish_name}")
        
        # Check daily totals
        daily_total = self.get_daily_total(incident.timestamp.date())
        if daily_total > self.alert_thresholds['daily_waste']:
            alerts.append(f"Daily waste threshold exceeded: {daily_total:.1f}kg")
        
        # Check repetition
        week_ago = incident.timestamp - timedelta(days=7)
        same_dish_count = sum(
            1 for i in self.incidents
            if i.dish_name == incident.dish_name
            and i.timestamp >= week_ago
        )
        if same_dish_count >= self.alert_thresholds['repetition']:
            alerts.append(f"Frequent waste: {incident.dish_name} wasted {same_dish_count} times in past week")
        
        if alerts:
            self._send_alerts(alerts, incident)
    
    def _send_alerts(self, alerts: List[str], incident: WasteIncident) -> None:
        """Send alerts to relevant channels"""
        logger.warning(f"WASTE ALERTS: {', '.join(alerts)}")
        
        # Here you could integrate with email, Slack, etc.
        # For now, just log and print
        print("\n" + "="*50)
        print("🚨 WASTE ALERT 🚨")
        print("="*50)
        for alert in alerts:
            print(f"• {alert}")
        print(f"Time: {incident.timestamp.strftime('%Y-%m-%d %H:%M')}")
        print(f"Dish: {incident.dish_name}")
        print(f"Quantity: {incident.quantity_kg}kg")
        print(f"Reason: {incident.reason}")
        print("="*50 + "\n")
    
    def get_daily_total(self, target_date: date) -> float:
        """Get total waste for a specific date"""
        start = datetime.combine(target_date, datetime.min.time())
        end = datetime.combine(target_date, datetime.max.time())
        
        return sum(
            i.quantity_kg for i in self.incidents
            if start <= i.timestamp <= end
        )
    
    def get_weekly_summary(self, end_date: Optional[datetime] = None) -> WasteSummary:
        """Get waste summary for the last 7 days"""
        if end_date is None:
            end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=7)
        
        return self.get_summary(start_date, end_date)
    
    def get_monthly_summary(self, year: int, month: int) -> WasteSummary:
        """Get waste summary for a specific month"""
        start_date = datetime(year, month, 1)
        if month == 12:
            end_date = datetime(year + 1, 1, 1) - timedelta(microseconds=1)
        else:
            end_date = datetime(year, month + 1, 1) - timedelta(microseconds=1)
        
        return self.get_summary(start_date, end_date)
    
    def get_summary(self, start_date: datetime, end_date: datetime) -> WasteSummary:
        """Get waste summary for a time period"""
        # Filter incidents in date range
        period_incidents = [
            i for i in self.incidents
            if start_date <= i.timestamp <= end_date
        ]
        
        if not period_incidents:
            return WasteSummary(
                period_start=start_date,
                period_end=end_date
            )
        
        # Initialize summary
        summary = WasteSummary(
            period_start=start_date,
            period_end=end_date,
            incidents_count=len(period_incidents)
        )
        
        # Aggregate data
        category_totals = defaultdict(float)
        meal_totals = defaultdict(float)
        dish_totals = defaultdict(float)
        severity_counts = defaultdict(int)
        reasons = []
        
        for incident in period_incidents:
            summary.total_waste_kg += incident.quantity_kg
            summary.total_cost_loss += incident.cost_loss
            summary.total_co2 += incident.co2_footprint
            summary.total_water += incident.water_footprint
            
            category_totals[incident.category] += incident.quantity_kg
            meal_totals[incident.meal_type] += incident.quantity_kg
            dish_totals[incident.dish_name] += incident.quantity_kg
            severity_counts[incident.severity] += 1
            reasons.append((incident.reason, incident.quantity_kg))
        
        summary.by_category = dict(category_totals)
        summary.by_meal = dict(meal_totals)
        summary.by_dish = dict(dish_totals)
        summary.by_severity = dict(severity_counts)
        
        # Get top reasons
        reason_totals = defaultdict(float)
        for reason, qty in reasons:
            reason_totals[reason] += qty
        summary.top_reasons = sorted(
            reason_totals.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        return summary
    
    def get_waste_trends(self, days: int = 30) -> Dict[str, Any]:
        """Analyze waste trends over time"""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        # Daily waste
        daily_waste = {}
        current = start_date
        while current <= end_date:
            day_total = self.get_daily_total(current.date())
            daily_waste[current.date()] = day_total
            current += timedelta(days=1)
        
        # Calculate moving averages
        dates = sorted(daily_waste.keys())
        values = [daily_waste[d] for d in dates]
        
        moving_avg_7 = []
        moving_avg_30 = []
        
        for i in range(len(values)):
            # 7-day moving average
            start_7 = max(0, i - 6)
            avg_7 = sum(values[start_7:i+1]) / (i - start_7 + 1)
            moving_avg_7.append(avg_7)
            
            # 30-day moving average
            start_30 = max(0, i - 29)
            avg_30 = sum(values[start_30:i+1]) / (i - start_30 + 1)
            moving_avg_30.append(avg_30)
        
        # Identify trends
        trend = "stable"
        if len(values) >= 7:
            first_week = sum(values[:7]) / 7
            last_week = sum(values[-7:]) / 7
            change_pct = ((last_week - first_week) / first_week) * 100 if first_week > 0 else 0
            
            if change_pct > 10:
                trend = "increasing"
            elif change_pct < -10:
                trend = "decreasing"
        
        # Find peak waste days
        peak_days = sorted(
            [(d, v) for d, v in daily_waste.items() if v > 0],
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return {
            'period_days': days,
            'daily_waste': {str(k): round(v, 2) for k, v in daily_waste.items()},
            'moving_averages': {
                '7_day': [round(x, 2) for x in moving_avg_7],
                '30_day': [round(x, 2) for x in moving_avg_30]
            },
            'trend': trend,
            'peak_days': [(str(d), round(v, 2)) for d, v in peak_days],
            'statistics': {
                'mean': round(np.mean(values), 2),
                'median': round(float(np.median(values)), 2),
                'std': round(np.std(values), 2),
                'max': round(max(values), 2),
                'min': round(min(values), 2),
                'total': round(sum(values), 2)
            }
        }
    
    def get_waste_by_category_pie(self) -> go.Figure:
        """Create pie chart of waste by category"""
        summary = self.get_weekly_summary()
        
        categories = list(summary.by_category.keys())
        values = list(summary.by_category.values())
        
        # Color mapping
        colors = {
            WasteCategory.PREPARATION: '#FF9999',
            WasteCategory.SERVING: '#66B2FF',
            WasteCategory.SPOILAGE: '#99FF99',
            WasteCategory.OVERPRODUCTION: '#FFCC99',
            WasteCategory.QUALITY_ISSUE: '#FF99CC',
            WasteCategory.STORAGE_LOSS: '#99CCFF',
            WasteCategory.TRANSPORT_DAMAGE: '#FFB266',
            WasteCategory.EXPIRY: '#FF6666'
        }
        
        fig = go.Figure(data=[go.Pie(
            labels=[c.value.replace('_', ' ').title() for c in categories],
            values=values,
            marker=dict(colors=[colors.get(c, '#808080') for c in categories]),
            textinfo='label+percent',
            hoverinfo='label+value+percent',
            textposition='auto'
        )])
        
        fig.update_layout(
            title="Waste Distribution by Category",
            showlegend=True,
            height=400
        )
        
        return fig
    
    def get_waste_timeline_chart(self, days: int = 30) -> go.Figure:
        """Create timeline chart of waste over time"""
        trends = self.get_waste_trends(days)
        
        dates = list(trends['daily_waste'].keys())
        values = list(trends['daily_waste'].values())
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Daily Waste', 'Moving Averages'),
            vertical_spacing=0.15
        )
        
        # Daily waste bars
        fig.add_trace(
            go.Bar(
                x=dates,
                y=values,
                name='Daily Waste',
                marker_color='#FF6B6B',
                opacity=0.7
            ),
            row=1, col=1
        )
        
        # Moving averages
        fig.add_trace(
            go.Scatter(
                x=dates[6:],
                y=trends['moving_averages']['7_day'][6:],
                name='7-Day Average',
                line=dict(color='#4ECDC4', width=3)
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=dates[29:],
                y=trends['moving_averages']['30_day'][29:],
                name='30-Day Average',
                line=dict(color='#45B7D1', width=3, dash='dash')
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title=f"Waste Trends - Last {days} Days",
            height=600,
            showlegend=True,
            hovermode='x unified'
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Waste (kg)", row=1, col=1)
        fig.update_yaxes(title_text="Average Waste (kg)", row=2, col=1)
        
        return fig
    
    def identify_problem_areas(self) -> List[Dict[str, Any]]:
        """Identify problem areas and root causes"""
        problems = []
        
        # Get last 30 days summary
        summary = self.get_summary(
            datetime.utcnow() - timedelta(days=30),
            datetime.utcnow()
        )
        
        # Check by category
        for category, amount in summary.by_category.items():
            percentage = (amount / summary.total_waste_kg * 100) if summary.total_waste_kg > 0 else 0
            
            if percentage > 30:
                problems.append({
                    'area': 'category',
                    'name': category.value,
                    'issue': f"High waste in {category.value} category ({percentage:.1f}%)",
                    'amount': amount,
                    'percentage': percentage,
                    'severity': 'high' if percentage > 50 else 'medium',
                    'recommendations': self._get_category_recommendations(category)
                })
        
        # Check by dish
        for dish, amount in summary.by_dish.items():
            percentage = (amount / summary.total_waste_kg * 100) if summary.total_waste_kg > 0 else 0
            
            if percentage > 15:
                problems.append({
                    'area': 'dish',
                    'name': dish,
                    'issue': f"High waste for dish: {dish} ({percentage:.1f}%)",
                    'amount': amount,
                    'percentage': percentage,
                    'severity': 'high' if percentage > 25 else 'medium',
                    'recommendations': [
                        f"Review recipe for {dish}",
                        "Consider reducing portion size",
                        "Survey student preferences for this dish"
                    ]
                })
        
        # Check by meal type
        for meal, amount in summary.by_meal.items():
            percentage = (amount / summary.total_waste_kg * 100) if summary.total_waste_kg > 0 else 0
            
            if percentage > 40:
                problems.append({
                    'area': 'meal',
                    'name': meal,
                    'issue': f"High waste during {meal} ({percentage:.1f}%)",
                    'amount': amount,
                    'percentage': percentage,
                    'severity': 'high' if percentage > 60 else 'medium',
                    'recommendations': [
                        f"Analyze {meal} menu items",
                        "Check attendance patterns for this meal",
                        "Review portion sizes for {meal}"
                    ]
                })
        
        # Check top reasons
        for reason, amount in summary.top_reasons[:3]:
            percentage = (amount / summary.total_waste_kg * 100) if summary.total_waste_kg > 0 else 0
            
            if percentage > 20:
                problems.append({
                    'area': 'reason',
                    'name': reason,
                    'issue': f"Common waste reason: {reason} ({percentage:.1f}%)",
                    'amount': amount,
                    'percentage': percentage,
                    'severity': 'high' if percentage > 35 else 'medium',
                    'recommendations': self._get_reason_recommendations(reason)
                })
        
        return sorted(problems, key=lambda x: x['percentage'], reverse=True)
    
    def _get_category_recommendations(self, category: WasteCategory) -> List[str]:
        """Get recommendations for waste category"""
        recommendations = {
            WasteCategory.PREPARATION: [
                "Train staff on efficient preparation techniques",
                "Use vegetable peels for stock/broth",
                "Implement precise measuring tools",
                "Consider pre-cut vegetables from supplier"
            ],
            WasteCategory.SERVING: [
                "Offer smaller portion options",
                "Implement trayless dining",
                "Conduct student surveys on preferences",
                "Adjust recipes based on feedback"
            ],
            WasteCategory.SPOILAGE: [
                "Improve inventory rotation (FIFO)",
                "Check storage temperatures regularly",
                "Reduce order quantities for perishables",
                "Implement first-expiry-first-out system"
            ],
            WasteCategory.OVERPRODUCTION: [
                "Improve demand forecasting",
                "Adjust preparation quantities",
                "Consider cook-to-order for some items",
                "Donate excess food to charity"
            ],
            WasteCategory.QUALITY_ISSUE: [
                "Review supplier quality standards",
                "Train staff on quality checks",
                "Adjust recipes for consistency",
                "Implement quality control checklist"
            ],
            WasteCategory.STORAGE_LOSS: [
                "Organize storage areas",
                "Monitor temperature/humidity",
                "Use airtight containers",
                "Implement pest control measures"
            ],
            WasteCategory.TRANSPORT_DAMAGE: [
                "Improve packaging",
                "Train handling staff",
                "Review transport procedures",
                "Use better vehicles/containers"
            ],
            WasteCategory.EXPIRY: [
                "Improve inventory tracking",
                "Use near-expiry items first",
                "Adjust ordering frequency",
                "Donate near-expiry items"
            ]
        }
        
        return recommendations.get(category, ["Review waste reduction strategies"])
    
    def _get_reason_recommendations(self, reason: str) -> List[str]:
        """Get recommendations for specific waste reason"""
        reason_lower = reason.lower()
        
        if "taste" in reason_lower or "flavor" in reason_lower:
            return [
                "Survey students for feedback",
                "Adjust seasoning/recipes",
                "Taste-test before serving",
                "Consider alternative recipes"
            ]
        elif "portion" in reason_lower or "size" in reason_lower:
            return [
                "Offer multiple portion sizes",
                "Reduce standard portion size",
                "Allow self-serving",
                "Monitor portion consumption"
            ]
        elif "quality" in reason_lower:
            return [
                "Review supplier quality",
                "Check storage conditions",
                "Train staff on quality standards",
                "Implement quality checks"
            ]
        elif "forecast" in reason_lower or "prediction" in reason_lower:
            return [
                "Improve demand prediction model",
                "Use historical data better",
                "Consider external factors (weather, events)",
                "Adjust for day of week patterns"
            ]
        else:
            return [
                "Investigate root cause",
                "Track this reason more closely",
                "Discuss with kitchen staff",
                "Implement corrective actions"
            ]
    
    def generate_waste_report(self, days: int = 30) -> str:
        """Generate comprehensive waste report"""
        summary = self.get_summary(
            datetime.utcnow() - timedelta(days=days),
            datetime.utcnow()
        )
        
        trends = self.get_waste_trends(days)
        problems = self.identify_problem_areas()
        
        lines = []
        lines.append("=" * 60)
        lines.append(f"🍽️  FOOD WASTE REPORT - Last {days} Days")
        lines.append("=" * 60)
        lines.append(f"Period: {summary.period_start.date()} to {summary.period_end.date()}")
        lines.append("")
        
        # Summary statistics
        lines.append("📊 SUMMARY STATISTICS")
        lines.append("-" * 40)
        lines.append(f"Total Waste: {summary.total_waste_kg:.1f} kg")
        lines.append(f"Daily Average: {summary.total_waste_kg/days:.1f} kg/day")
        lines.append(f"Total Cost Loss: ₹{summary.total_cost_loss:,.0f}")
        lines.append(f"CO₂ Footprint: {summary.total_co2:.1f} kg")
        lines.append(f"Water Footprint: {summary.total_water:,.0f} liters")
        lines.append(f"Total Incidents: {summary.incidents_count}")
        lines.append("")
        
        # Breakdown by category
        lines.append("📦 WASTE BY CATEGORY")
        lines.append("-" * 40)
        for category, amount in sorted(summary.by_category.items(), key=lambda x: x[1], reverse=True):
            percentage = (amount / summary.total_waste_kg * 100) if summary.total_waste_kg > 0 else 0
            lines.append(f"{category.value.replace('_', ' ').title():20s}: {amount:6.1f} kg ({percentage:5.1f}%)")
        lines.append("")
        
        # Top wasted dishes
        lines.append("🍲 TOP WASTED DISHES")
        lines.append("-" * 40)
        for dish, amount in sorted(summary.by_dish.items(), key=lambda x: x[1], reverse=True)[:5]:
            percentage = (amount / summary.total_waste_kg * 100) if summary.total_waste_kg > 0 else 0
            lines.append(f"{dish:25s}: {amount:6.1f} kg ({percentage:5.1f}%)")
        lines.append("")
        
        # Top reasons
        lines.append("⚠️  TOP WASTE REASONS")
        lines.append("-" * 40)
        for reason, amount in summary.top_reasons[:5]:
            percentage = (amount / summary.total_waste_kg * 100) if summary.total_waste_kg > 0 else 0
            lines.append(f"{reason:30s}: {amount:6.1f} kg ({percentage:5.1f}%)")
        lines.append("")
        
        # Problem areas
        if problems:
            lines.append("🚨 IDENTIFIED PROBLEM AREAS")
            lines.append("-" * 40)
            for problem in problems[:5]:
                lines.append(f"• {problem['issue']}")
                for rec in problem['recommendations'][:2]:
                    lines.append(f"  - {rec}")
            lines.append("")
        
        # Trend analysis
        lines.append("📈 TREND ANALYSIS")
        lines.append("-" * 40)
        lines.append(f"Trend Direction: {trends['trend'].title()}")
        lines.append(f"Mean Daily Waste: {trends['statistics']['mean']:.1f} kg")
        lines.append(f"Peak Waste Day: {trends['peak_days'][0][0] if trends['peak_days'] else 'N/A'} ({trends['peak_days'][0][1] if trends['peak_days'] else 0:.1f} kg)")
        lines.append("")
        
        # Recommendations
        lines.append("💡 RECOMMENDATIONS")
        lines.append("-" * 40)
        all_recs = []
        for problem in problems[:3]:
            all_recs.extend(problem['recommendations'][:2])
        
        for i, rec in enumerate(set(all_recs), 1):
            lines.append(f"{i}. {rec}")
        
        lines.append("")
        lines.append("=" * 60)
        lines.append("End of Report")
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def export_waste_data(self, format: str = 'json') -> Any:
        """Export waste data for analysis"""
        if format == 'json':
            data = {
                'incidents': [
                    {
                        'id': i.id,
                        'timestamp': i.timestamp.isoformat(),
                        'category': i.category.value,
                        'meal_type': i.meal_type,
                        'dish_name': i.dish_name,
                        'quantity_kg': i.quantity_kg,
                        'reason': i.reason,
                        'severity': i.severity.value,
                        'cost_loss': i.cost_loss,
                        'co2_footprint': i.co2_footprint,
                        'water_footprint': i.water_footprint,
                        'notes': i.notes
                    }
                    for i in self.incidents
                ],
                'summary': self.get_weekly_summary().to_dict(),
                'exported_at': datetime.utcnow().isoformat()
            }
            return json.dumps(data, indent=2)
        
        elif format == 'csv':
            import io
            output = io.StringIO()
            output.write("ID,Timestamp,Category,Meal Type,Dish Name,Quantity (kg),Reason,Severity,Cost Loss (₹),CO2 (kg),Water (L)\n")
            
            for i in self.incidents:
                output.write(f"{i.id},{i.timestamp.isoformat()},{i.category.value},{i.meal_type},{i.dish_name},{i.quantity_kg},{i.reason},{i.severity.value},{i.cost_loss},{i.co2_footprint},{i.water_footprint}\n")
            
            return output.getvalue()
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def clear_old_incidents(self, days: int = 90) -> int:
        """Clear incidents older than specified days"""
        cutoff = datetime.utcnow() - timedelta(days=days)
        old_count = len([i for i in self.incidents if i.timestamp < cutoff])
        
        self.incidents = [i for i in self.incidents if i.timestamp >= cutoff]
        
        logger.info(f"Cleared {old_count} old incidents")
        return old_count