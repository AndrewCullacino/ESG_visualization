"""
Strategy D ML Simulation - ESG Supply Chain Emission Reduction
================================================================
This standalone simulation uses machine learning to model and optimize
emission reduction strategies for suppliers in Zone IV (Observation Zone).

Strategy D Focus: Low-emission, low-cooperation suppliers
- Target: 5-10% emission reduction over 3 years
- Approach: Basic management + automated monitoring + minimal intervention
- ML Method: Multi-output regression + cost-benefit optimization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
import os

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Chinese font configuration
rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'PingFang SC', 'STHeiti']
rcParams['axes.unicode_minus'] = False

# ============================================================================
# PART 1: Synthetic Data Generation - Zone IV Supplier Profiles
# ============================================================================

class SupplierDataGeneratorD:
    """Generate realistic synthetic supplier data for Zone IV suppliers"""
    
    def __init__(self, n_suppliers=500, seed=42):
        self.n_suppliers = n_suppliers
        self.rng = np.random.default_rng(seed)
        
    def generate_suppliers(self):
        """Generate Zone IV supplier characteristics"""
        
        # Zone IV criteria: Low emission + Low cooperation
        # Emission: 1,000 - 8,000 tons CO2 (lower than other zones)
        # Cooperation score: 2-6 out of 10 (low cooperation)
        
        baseline_emission = self.rng.uniform(1000, 8000, self.n_suppliers)
        cooperation_score = self.rng.integers(2, 7, self.n_suppliers)
        
        # Additional supplier characteristics
        annual_revenue = baseline_emission * self.rng.uniform(50, 160, self.n_suppliers)
        employee_count = self.rng.integers(30, 400, self.n_suppliers)
        years_partnership = self.rng.integers(1, 8, self.n_suppliers)  # Shorter partnerships
        tech_adoption_level = self.rng.uniform(0.1, 0.4, self.n_suppliers)  # Low tech adoption
        financial_capacity = self.rng.uniform(0.2, 0.9, self.n_suppliers)
        
        # Zone IV specific attributes
        monitoring_readiness = self.rng.uniform(0.2, 0.6, self.n_suppliers)  # Limited monitoring capability
        cost_sensitivity = self.rng.uniform(0.6, 0.95, self.n_suppliers)  # Very cost-sensitive
        self_motivation = self.rng.uniform(0.1, 0.4, self.n_suppliers)  # Low self-initiative
        replacement_risk = self.rng.uniform(0.3, 0.7, self.n_suppliers)  # Moderate replacement risk
        
        # Industry type (weighted by likelihood for lower emissions)
        industry_types = self.rng.choice(
            ['Accessories', 'Finishing', 'Packaging', 'Small Parts'], 
            size=self.n_suppliers,
            p=[0.40, 0.30, 0.20, 0.10]  # More accessories and finishing
        )
        
        df = pd.DataFrame({
            'supplier_id': [f'SUP_D_{i:03d}' for i in range(self.n_suppliers)],
            'baseline_emission': baseline_emission,
            'cooperation_score': cooperation_score,
            'annual_revenue': annual_revenue,
            'employee_count': employee_count,
            'years_partnership': years_partnership,
            'tech_adoption_level': tech_adoption_level,
            'financial_capacity': financial_capacity,
            'monitoring_readiness': monitoring_readiness,
            'cost_sensitivity': cost_sensitivity,
            'self_motivation': self_motivation,
            'replacement_risk': replacement_risk,
            'industry_type': industry_types
        })
        
        return df


# ============================================================================
# PART 2: Technology & Basic Management Database
# ============================================================================

class TechnologyDatabaseD:
    """Database of low-cost, basic emission reduction technologies suitable for Zone IV"""
    
    @staticmethod
    def get_technologies():
        """Return basic, low-cost technologies for Zone IV suppliers"""
        
        technologies = [
            {
                'tech_id': 'D_TECH_001',
                'name': 'LED照明改造 (LED Lighting Retrofit)',
                'category': 'Energy Efficiency',
                'reduction_rate': 0.03,  # 3% reduction
                'cost_per_unit': 5000,
                'implementation_months': 1,
                'annual_savings': 1500,
                'maintenance_cost': 200
            },
            {
                'tech_id': 'D_TECH_002',
                'name': '定时器自动控制 (Timer Control)',
                'category': 'Automation',
                'reduction_rate': 0.02,
                'cost_per_unit': 3000,
                'implementation_months': 1,
                'annual_savings': 1000,
                'maintenance_cost': 150
            },
            {
                'tech_id': 'D_TECH_003',
                'name': '设备维护优化 (Equipment Maintenance)',
                'category': 'Operations',
                'reduction_rate': 0.025,
                'cost_per_unit': 4000,
                'implementation_months': 2,
                'annual_savings': 1200,
                'maintenance_cost': 300
            },
            {
                'tech_id': 'D_TECH_004',
                'name': '能源计量表安装 (Energy Metering)',
                'category': 'Monitoring',
                'reduction_rate': 0.015,
                'cost_per_unit': 2500,
                'implementation_months': 1,
                'annual_savings': 800,
                'maintenance_cost': 100
            },
            {
                'tech_id': 'D_TECH_005',
                'name': '简易保温改造 (Basic Insulation)',
                'category': 'Energy Efficiency',
                'reduction_rate': 0.02,
                'cost_per_unit': 3500,
                'implementation_months': 2,
                'annual_savings': 900,
                'maintenance_cost': 150
            },
            {
                'tech_id': 'D_TECH_006',
                'name': '空调温控优化 (HVAC Optimization)',
                'category': 'Energy Efficiency',
                'reduction_rate': 0.018,
                'cost_per_unit': 2800,
                'implementation_months': 1,
                'annual_savings': 750,
                'maintenance_cost': 120
            },
            {
                'tech_id': 'D_TECH_007',
                'name': '节能风机安装 (Energy-Efficient Fans)',
                'category': 'Equipment',
                'reduction_rate': 0.022,
                'cost_per_unit': 4200,
                'implementation_months': 2,
                'annual_savings': 1100,
                'maintenance_cost': 180
            },
            {
                'tech_id': 'D_TECH_008',
                'name': '用水优化管理 (Water Usage Optimization)',
                'category': 'Operations',
                'reduction_rate': 0.012,
                'cost_per_unit': 2000,
                'implementation_months': 1,
                'annual_savings': 600,
                'maintenance_cost': 80
            },
            {
                'tech_id': 'D_TECH_009',
                'name': '压缩空气泄漏检测 (Air Leak Detection)',
                'category': 'Maintenance',
                'reduction_rate': 0.015,
                'cost_per_unit': 2200,
                'implementation_months': 1,
                'annual_savings': 700,
                'maintenance_cost': 100
            },
            {
                'tech_id': 'D_TECH_010',
                'name': '操作培训基础版 (Basic Operation Training)',
                'category': 'Training',
                'reduction_rate': 0.01,
                'cost_per_unit': 1500,
                'implementation_months': 1,
                'annual_savings': 500,
                'maintenance_cost': 50
            }
        ]
        
        return pd.DataFrame(technologies)


class BasicManagementDatabase:
    """Database of basic management and monitoring initiatives for Zone IV"""
    
    @staticmethod
    def get_initiatives():
        """Return low-cost management initiatives"""
        
        initiatives = [
            {
                'initiative_id': 'D_MGT_001',
                'name': '季度数据报告 (Quarterly Reporting)',
                'category': 'Monitoring',
                'annual_cost': 1000,
                'effectiveness': 0.15,  # Improves cooperation by 0.15 points
                'automation_level': 0.3
            },
            {
                'initiative_id': 'D_MGT_002',
                'name': '在线培训课程 (Online Training)',
                'category': 'Capacity Building',
                'annual_cost': 1500,
                'effectiveness': 0.20,
                'automation_level': 0.8
            },
            {
                'initiative_id': 'D_MGT_003',
                'name': '自动监测系统 (Automated Monitoring)',
                'category': 'Technology',
                'annual_cost': 3000,
                'effectiveness': 0.25,
                'automation_level': 0.9
            },
            {
                'initiative_id': 'D_MGT_004',
                'name': '简化版ESG手册 (Simplified ESG Guide)',
                'category': 'Documentation',
                'annual_cost': 800,
                'effectiveness': 0.10,
                'automation_level': 0.5
            },
            {
                'initiative_id': 'D_MGT_005',
                'name': '基础技术支持热线 (Basic Tech Support)',
                'category': 'Support',
                'annual_cost': 2000,
                'effectiveness': 0.18,
                'automation_level': 0.4
            }
        ]
        
        return pd.DataFrame(initiatives)


# ============================================================================
# PART 3: Basic Management ML Model
# ============================================================================

class BasicManagementML:
    """Machine Learning model to predict optimal basic management strategies"""
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def generate_training_data(self, suppliers_df, n_scenarios=1000):
        """Generate synthetic training data for basic management optimization"""
        
        training_data = []
        
        for _ in range(n_scenarios):
            # Sample supplier characteristics
            supplier = suppliers_df.sample(n=1).iloc[0]
            
            # Simulate different management strategies
            monitoring_frequency = np.random.choice([1, 2, 4, 12])  # times per year
            training_investment = np.random.uniform(0, 3000)
            automation_level = np.random.uniform(0, 0.8)
            technical_support = np.random.uniform(0, 2000)
            
            # Calculate outcomes based on supplier characteristics
            baseline_cooperation = supplier['cooperation_score']
            cost_sensitivity = supplier['cost_sensitivity']
            monitoring_readiness = supplier['monitoring_readiness']
            self_motivation = supplier['self_motivation']
            
            # Predict cooperation improvement
            cooperation_improvement = (
                monitoring_frequency * 0.05 * monitoring_readiness +
                training_investment / 1000 * 0.1 * (1 - cost_sensitivity) +
                automation_level * 0.3 * monitoring_readiness +
                technical_support / 1000 * 0.15 * (1 - self_motivation) +
                np.random.normal(0, 0.1)
            )
            cooperation_improvement = np.clip(cooperation_improvement, 0, 1.5)
            
            # Predict cost-effectiveness
            total_cost = (
                monitoring_frequency * 200 +
                training_investment +
                automation_level * 3000 +
                technical_support
            )
            
            # Predict emission reduction potential
            reduction_potential = (
                cooperation_improvement * 0.04 +
                automation_level * 0.02 +
                np.random.uniform(0.01, 0.03)
            )
            reduction_potential = np.clip(reduction_potential, 0.03, 0.12)
            
            training_data.append({
                'baseline_emission': supplier['baseline_emission'],
                'cooperation_score': baseline_cooperation,
                'monitoring_readiness': monitoring_readiness,
                'cost_sensitivity': cost_sensitivity,
                'self_motivation': self_motivation,
                'financial_capacity': supplier['financial_capacity'],
                # Outputs
                'optimal_monitoring_freq': monitoring_frequency,
                'optimal_training_budget': training_investment,
                'optimal_automation_level': automation_level,
                'optimal_support_budget': technical_support,
                'expected_cooperation_improvement': cooperation_improvement,
                'expected_reduction_potential': reduction_potential,
                'total_cost': total_cost
            })
        
        return pd.DataFrame(training_data)
    
    def train(self, suppliers_df):
        """Train ML models to predict optimal basic management strategies"""
        
        print("📊 Generating training data for basic management ML models...")
        training_data = self.generate_training_data(suppliers_df, n_scenarios=2000)
        
        # Features and targets
        feature_cols = ['baseline_emission', 'cooperation_score', 'monitoring_readiness',
                       'cost_sensitivity', 'self_motivation', 'financial_capacity']
        target_cols = ['optimal_monitoring_freq', 'optimal_training_budget', 
                      'optimal_automation_level', 'optimal_support_budget']
        
        self.feature_names = feature_cols
        X = training_data[feature_cols]
        
        # Train separate model for each output
        print("\n🤖 Training ML models for each management parameter...")
        
        for target in target_cols:
            print(f"  Training model for: {target}")
            y = training_data[target]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            if target == target_cols[0]:  # Only fit scaler once
                X_train_scaled = self.scaler.fit_transform(X_train)
            else:
                X_train_scaled = self.scaler.transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train ensemble model
            rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
            gb_model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
            
            rf_model.fit(X_train_scaled, y_train)
            gb_model.fit(X_train_scaled, y_train)
            
            # Ensemble prediction
            rf_pred = rf_model.predict(X_test_scaled)
            gb_pred = gb_model.predict(X_test_scaled)
            ensemble_pred = (rf_pred + gb_pred) / 2
            
            # Evaluate
            mse = mean_squared_error(y_test, ensemble_pred)
            r2 = r2_score(y_test, ensemble_pred)
            
            print(f"    ✓ MSE: {mse:.4f}, R²: {r2:.4f}")
            
            # Store models
            self.models[target] = {'rf': rf_model, 'gb': gb_model}
        
        print("\n✅ All basic management models trained successfully!\n")
    
    def predict_strategy(self, supplier_features):
        """Predict optimal basic management strategy for a supplier"""
        
        X = pd.DataFrame([supplier_features])[self.feature_names]
        X_scaled = self.scaler.transform(X)
        
        strategy = {}
        for target, models in self.models.items():
            rf_pred = models['rf'].predict(X_scaled)[0]
            gb_pred = models['gb'].predict(X_scaled)[0]
            strategy[target] = (rf_pred + gb_pred) / 2
        
        return strategy


# ============================================================================
# PART 4: Emission Reduction Simulator for Zone IV
# ============================================================================

class EmissionReductionSimulatorD:
    """Simulate 3-year emission reduction pathway for Zone IV suppliers"""
    
    def __init__(self, tech_db, management_db):
        self.tech_db = tech_db
        self.management_db = management_db
    
    def simulate_reduction_pathway(self, supplier, management_strategy, budget=30000, mode='minimal'):
        """
        Simulate 3-year emission reduction pathway
        
        Args:
            supplier: Supplier dataframe row
            management_strategy: Dict of predicted management strategy
            budget: Available investment budget (USD) - lower for Zone IV
            mode: 'minimal' (5-7%), 'balanced' (7-10%), or 'aggressive' (10-12%)
        """
        
        # Set reduction targets based on mode
        target_map = {
            'minimal': [0.02, 0.04, 0.05],      # Very conservative
            'balanced': [0.03, 0.06, 0.08],     # Moderate
            'aggressive': [0.04, 0.08, 0.10]    # Higher end for Zone IV
        }
        
        targets = target_map.get(mode, target_map['minimal'])
        
        # Extract management parameters
        monitoring_freq = management_strategy['optimal_monitoring_freq']
        training_budget = management_strategy['optimal_training_budget']
        automation_level = management_strategy['optimal_automation_level']
        support_budget = management_strategy['optimal_support_budget']
        
        # Calculate management costs
        annual_management_cost = (
            monitoring_freq * 200 +
            training_budget / 3 +  # Amortized over 3 years
            automation_level * 1000 +  # Annual automation cost
            support_budget
        )
        
        # Select cost-effective technologies
        tech_budget = budget - (annual_management_cost * 3)
        tech_budget = max(tech_budget, budget * 0.6)  # At least 60% for tech
        
        selected_techs = []
        remaining_budget = tech_budget
        
        # Sort by cost-effectiveness (reduction per dollar)
        self.tech_db['cost_effectiveness'] = (
            self.tech_db['reduction_rate'] / self.tech_db['cost_per_unit']
        )
        sorted_techs = self.tech_db.sort_values('cost_effectiveness', ascending=False)
        
        for _, tech in sorted_techs.iterrows():
            if remaining_budget >= tech['cost_per_unit']:
                selected_techs.append(tech)
                remaining_budget -= tech['cost_per_unit']
                if len(selected_techs) >= 4:  # Limit to 4 technologies
                    break
        
        # Simulate year-by-year reduction
        baseline = supplier['baseline_emission']
        yearly_emissions = [baseline]
        yearly_reductions = [0]
        cumulative_reduction = [0]
        
        cooperation_factor = 1.0 + (supplier['cooperation_score'] - 2) / 10  # 0.8-1.4
        self_motivation_factor = 1.0 + supplier['self_motivation'] * 0.5
        
        for year in range(1, 4):
            # Technology-based reduction
            tech_reduction = sum([t['reduction_rate'] for t in selected_techs[:year+1]])
            
            # Management effectiveness (gradually improves cooperation)
            management_effect = (automation_level * 0.02 + 
                               monitoring_freq / 12 * 0.015) * year
            
            # Calculate actual reduction (limited by cooperation)
            total_reduction_rate = (tech_reduction + management_effect) * cooperation_factor
            
            # Add some randomness to create variability
            random_factor = np.random.uniform(0.9, 1.1)
            total_reduction_rate *= random_factor
            
            total_reduction_rate = min(total_reduction_rate, targets[year-1] * 1.2)
            
            actual_reduction = baseline * total_reduction_rate
            new_emission = baseline - actual_reduction
            
            yearly_emissions.append(new_emission)
            yearly_reductions.append(actual_reduction - cumulative_reduction[-1])
            cumulative_reduction.append(actual_reduction)
        
        # Calculate investment breakdown
        tech_investment = sum([t['cost_per_unit'] for t in selected_techs])
        total_investment = tech_investment + (annual_management_cost * 3)
        
        # Calculate ROI
        annual_savings = sum([t['annual_savings'] for t in selected_techs])
        maintenance_cost = sum([t['maintenance_cost'] for t in selected_techs])
        net_annual_benefit = annual_savings - maintenance_cost - annual_management_cost
        
        payback_period = total_investment / net_annual_benefit if net_annual_benefit > 0 else 999
        
        # Check if target achieved
        final_reduction_rate = cumulative_reduction[-1] / baseline
        target_rate = targets[-1]
        target_achieved = final_reduction_rate >= target_rate * 0.9
        
        return {
            'supplier_id': supplier['supplier_id'],
            'baseline_emission': baseline,
            'yearly_emissions': yearly_emissions,
            'yearly_reductions': yearly_reductions,
            'cumulative_reduction': cumulative_reduction,
            'selected_technologies': [t['name'] for t in selected_techs],
            'tech_investment': tech_investment,
            'management_cost': annual_management_cost * 3,
            'total_investment': total_investment,
            'annual_savings': annual_savings,
            'payback_period': payback_period,
            'final_reduction_rate': final_reduction_rate,
            'target_rate': target_rate,
            'target_achieved': target_achieved,
            'monitoring_frequency': monitoring_freq,
            'automation_level': automation_level,
            'cooperation_factor': cooperation_factor
        }


# ============================================================================
# PART 5: Visualization
# ============================================================================

class StrategyVisualizerD:
    """Visualize Strategy D simulation results"""
    
    @staticmethod
    def plot_results(simulation_results, suppliers_df):
        """Create comprehensive visualization of Zone IV simulation results"""
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Color scheme for Zone IV
        zone_color = '#95A5A6'  # Gray
        accent_color = '#34495E'  # Dark gray
        
        # 1. Emission Reduction Pathways (Top Left)
        ax1 = fig.add_subplot(gs[0, 0])
        for result in simulation_results[:15]:  # Show 15 suppliers
            years = [0, 1, 2, 3]
            emissions = result['yearly_emissions']
            ax1.plot(years, emissions, marker='o', alpha=0.6, linewidth=2)
        
        ax1.set_xlabel('年份 (Year)', fontsize=11, weight='bold')
        ax1.set_ylabel('碳排放量 (吨 CO₂e)', fontsize=11, weight='bold')
        ax1.set_title('D区供应商三年减排路径\nZone IV 3-Year Reduction Pathway', 
                     fontsize=12, weight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(['样本供应商'], fontsize=9)
        
        # 2. Reduction Rate Distribution (Top Middle)
        ax2 = fig.add_subplot(gs[0, 1])
        reduction_rates = [r['final_reduction_rate'] * 100 for r in simulation_results]
        # Use bar chart instead of histogram to avoid bin issues
        from collections import Counter
        rate_counts = Counter([round(r, 1) for r in reduction_rates])
        rates_sorted = sorted(rate_counts.keys())
        counts = [rate_counts[r] for r in rates_sorted]
        ax2.bar(range(len(rates_sorted)), counts, color=zone_color, edgecolor='black', alpha=0.7)
        ax2.set_xticks(range(0, len(rates_sorted), max(1, len(rates_sorted)//10)))
        ax2.set_xticklabels([f'{rates_sorted[i]:.1f}' for i in range(0, len(rates_sorted), max(1, len(rates_sorted)//10))], rotation=45)
        ax2.axvline(np.mean(reduction_rates), color='red', linestyle='--', 
                   linewidth=2, label=f'平均: {np.mean(reduction_rates):.1f}%')
        ax2.set_xlabel('减排率 (%)', fontsize=11, weight='bold')
        ax2.set_ylabel('供应商数量', fontsize=11, weight='bold')
        ax2.set_title('减排率分布\nReduction Rate Distribution', fontsize=12, weight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Investment vs Reduction Scatter (Top Right)
        ax3 = fig.add_subplot(gs[0, 2])
        investments = [r['total_investment'] for r in simulation_results]
        reductions = [r['cumulative_reduction'][-1] for r in simulation_results]
        scatter = ax3.scatter(investments, reductions, c=reduction_rates, 
                            cmap='RdYlGn', s=100, alpha=0.6, edgecolors='black')
        ax3.set_xlabel('总投资 (USD)', fontsize=11, weight='bold')
        ax3.set_ylabel('总减排量 (吨 CO₂e)', fontsize=11, weight='bold')
        ax3.set_title('投资回报分析\nInvestment vs Reduction', fontsize=12, weight='bold')
        plt.colorbar(scatter, ax=ax3, label='减排率 (%)')
        ax3.grid(True, alpha=0.3)
        
        # 4. Technology Adoption (Middle Left)
        ax4 = fig.add_subplot(gs[1, 0])
        tech_counter = {}
        for result in simulation_results:
            for tech in result['selected_technologies']:
                tech_short = tech.split('(')[0].strip()
                tech_counter[tech_short] = tech_counter.get(tech_short, 0) + 1
        
        top_techs = sorted(tech_counter.items(), key=lambda x: x[1], reverse=True)[:8]
        tech_names = [t[0] for t in top_techs]
        tech_counts = [t[1] for t in top_techs]
        
        ax4.barh(tech_names, tech_counts, color=zone_color, edgecolor='black')
        ax4.set_xlabel('采用次数', fontsize=11, weight='bold')
        ax4.set_title('热门技术采用排名\nTop Technologies Adopted', fontsize=12, weight='bold')
        ax4.grid(True, alpha=0.3, axis='x')
        
        # 5. Payback Period Distribution (Middle Middle)
        ax5 = fig.add_subplot(gs[1, 1])
        payback_periods = [min(r['payback_period'], 15) for r in simulation_results]
        # Use bar chart for payback periods
        from collections import Counter
        payback_counts = Counter([round(p, 1) for p in payback_periods])
        paybacks_sorted = sorted(payback_counts.keys())
        counts = [payback_counts[p] for p in paybacks_sorted]
        ax5.bar(range(len(paybacks_sorted)), counts, color=accent_color, edgecolor='black', alpha=0.7)
        ax5.set_xticks(range(0, len(paybacks_sorted), max(1, len(paybacks_sorted)//10)))
        ax5.set_xticklabels([f'{paybacks_sorted[i]:.1f}' for i in range(0, len(paybacks_sorted), max(1, len(paybacks_sorted)//10))], rotation=45)
        ax5.axvline(np.mean(payback_periods), color='red', linestyle='--', 
                   linewidth=2, label=f'平均: {np.mean(payback_periods):.1f}年')
        ax5.set_xlabel('投资回收期 (年)', fontsize=11, weight='bold')
        ax5.set_ylabel('供应商数量', fontsize=11, weight='bold')
        ax5.set_title('投资回收期分布\nPayback Period Distribution', fontsize=12, weight='bold')
        ax5.legend(fontsize=9)
        ax5.grid(True, alpha=0.3, axis='y')
        
        # 6. Cooperation Factor Impact (Middle Right)
        ax6 = fig.add_subplot(gs[1, 2])
        coop_factors = [r['cooperation_factor'] for r in simulation_results]
        final_rates = [r['final_reduction_rate'] * 100 for r in simulation_results]
        ax6.scatter(coop_factors, final_rates, s=80, alpha=0.6, 
                   color=zone_color, edgecolors='black')
        z = np.polyfit(coop_factors, final_rates, 1)
        p = np.poly1d(z)
        ax6.plot(sorted(coop_factors), p(sorted(coop_factors)), 
                "r--", linewidth=2, label='趋势线')
        ax6.set_xlabel('配合度因子', fontsize=11, weight='bold')
        ax6.set_ylabel('减排率 (%)', fontsize=11, weight='bold')
        ax6.set_title('配合度对减排效果的影响\nCooperation Impact', fontsize=12, weight='bold')
        ax6.legend(fontsize=9)
        ax6.grid(True, alpha=0.3)
        
        # 7. Cost Breakdown (Bottom Left)
        ax7 = fig.add_subplot(gs[2, 0])
        avg_tech = np.mean([r['tech_investment'] for r in simulation_results])
        avg_mgmt = np.mean([r['management_cost'] for r in simulation_results])
        
        categories = ['技术投资\nTechnology', '管理成本\nManagement']
        costs = [avg_tech, avg_mgmt]
        colors_bar = [zone_color, accent_color]
        
        bars = ax7.bar(categories, costs, color=colors_bar, edgecolor='black', width=0.6)
        ax7.set_ylabel('平均成本 (USD)', fontsize=11, weight='bold')
        ax7.set_title('成本结构分析\nCost Breakdown', fontsize=12, weight='bold')
        ax7.grid(True, alpha=0.3, axis='y')
        
        for bar in bars:
            height = bar.get_height()
            ax7.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height:.0f}', ha='center', va='bottom', fontsize=10, weight='bold')
        
        # 8. Success Rate & Efficiency (Bottom Middle)
        ax8 = fig.add_subplot(gs[2, 1])
        success_count = sum([r['target_achieved'] for r in simulation_results])
        fail_count = len(simulation_results) - success_count
        
        labels = [f'达标\nAchieved\n({success_count})', f'未达标\nNot Achieved\n({fail_count})']
        sizes = [success_count, fail_count]
        colors_pie = ['#27AE60', '#E74C3C']
        explode = (0.05, 0)
        
        ax8.pie(sizes, explode=explode, labels=labels, colors=colors_pie,
               autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10, 'weight': 'bold'})
        ax8.set_title('目标达成率\nTarget Achievement Rate', fontsize=12, weight='bold')
        
        # 9. Key Metrics Summary (Bottom Right)
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')
        
        total_baseline = sum([r['baseline_emission'] for r in simulation_results])
        total_reduction = sum([r['cumulative_reduction'][-1] for r in simulation_results])
        total_investment = sum([r['total_investment'] for r in simulation_results])
        avg_efficiency = total_reduction / total_investment if total_investment > 0 else 0
        
        summary_text = f"""
        策略D 关键指标
        STRATEGY D KEY METRICS
        
        ━━━━━━━━━━━━━━━━━━━━━━
        
        📊 供应商样本数
           {len(simulation_results)} 家
        
        🏭 基线总排放
           {total_baseline:,.0f} 吨 CO₂e
        
        ✅ 总减排量
           {total_reduction:,.0f} 吨 CO₂e
           ({total_reduction/total_baseline*100:.1f}%)
        
        💰 总投资
           ${total_investment:,.0f} USD
        
        📈 平均效率
           {avg_efficiency:.4f} 吨/美元
        
        🎯 达标率
           {success_count/len(simulation_results)*100:.1f}%
        
        ⏱️  平均回收期
           {np.mean(payback_periods):.1f} 年
        
        ━━━━━━━━━━━━━━━━━━━━━━
        """
        
        ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes,
                fontsize=11, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # Main title
        fig.suptitle('策略D机器学习模拟 - IV区观察区 (低排放×低配合)\nStrategy D ML Simulation - Zone IV Observation Zone',
                    fontsize=16, weight='bold', y=0.98)
        
        return fig


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "="*80)
    print("STRATEGY D ML SIMULATION")
    print("ESG Supply Chain Emission Reduction - Zone IV (Observation Zone)")
    print("="*80 + "\n")
    
    # Step 1: Generate synthetic supplier data
    print("📋 Step 1: Generating synthetic supplier data for Zone IV...")
    data_gen = SupplierDataGeneratorD(n_suppliers=2000, seed=42)
    suppliers_df = data_gen.generate_suppliers()
    print(f"✓ Generated {len(suppliers_df)} Zone IV suppliers")
    print(f"  - Average baseline emission: {suppliers_df['baseline_emission'].mean():.0f} tons CO₂")
    print(f"  - Average cooperation score: {suppliers_df['cooperation_score'].mean():.1f}/10")
    print(f"  - Average cost sensitivity: {suppliers_df['cost_sensitivity'].mean():.2f}")
    print(f"  - Average self-motivation: {suppliers_df['self_motivation'].mean():.2f}")
    
    # Step 2: Load technology and management databases
    print("\n🔧 Step 2: Loading technology and basic management databases...")
    tech_db = TechnologyDatabaseD.get_technologies()
    management_db = BasicManagementDatabase.get_initiatives()
    print(f"✓ Loaded {len(tech_db)} low-cost technologies")
    print(f"✓ Loaded {len(management_db)} basic management initiatives")
    
    # Step 3: Train ML models for basic management strategy
    print("\n🤖 Step 3: Training ML models for basic management optimization...")
    management_ml = BasicManagementML()
    management_ml.train(suppliers_df)
    
    # Step 4: Select sample suppliers for simulation
    print("\n🎯 Step 4: Selecting sample suppliers for detailed simulation...")
    sample_suppliers = suppliers_df.sample(n=100, random_state=42)
    print(f"✓ Selected {len(sample_suppliers)} suppliers for detailed analysis")
    
    # Step 5: Run emission reduction simulations
    print("\n⚡ Step 5: Running emission reduction simulations...")
    simulator = EmissionReductionSimulatorD(tech_db, management_db)
    simulation_results = []
    
    for idx, supplier in sample_suppliers.iterrows():
        # Predict optimal management strategy
        supplier_features = {
            'baseline_emission': supplier['baseline_emission'],
            'cooperation_score': supplier['cooperation_score'],
            'monitoring_readiness': supplier['monitoring_readiness'],
            'cost_sensitivity': supplier['cost_sensitivity'],
            'self_motivation': supplier['self_motivation'],
            'financial_capacity': supplier['financial_capacity']
        }
        
        management_strategy = management_ml.predict_strategy(supplier_features)
        
        # Run simulation
        result = simulator.simulate_reduction_pathway(
            supplier, management_strategy, 
            budget=30000,  # Lower budget for Zone IV
            mode='minimal'
        )
        simulation_results.append(result)
    
    print(f"✓ Completed {len(simulation_results)} simulations")
    
    # Step 6: Visualize results
    print("\n📊 Step 6: Generating visualization...")
    visualizer = StrategyVisualizerD()
    fig = visualizer.plot_results(simulation_results, sample_suppliers)
    
    output_path = 'Strategy_D_ML_Simulation.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Visualization saved: {output_path}")
    
    # Step 7: Generate summary report
    print("\n📈 Step 7: Generating summary report...")
    total_baseline = sum([r['baseline_emission'] for r in simulation_results])
    total_reduction = sum([r['cumulative_reduction'][-1] for r in simulation_results])
    total_investment = sum([r['total_investment'] for r in simulation_results])
    avg_efficiency = total_reduction / total_investment if total_investment > 0 else 0
    success_rate = sum([r['target_achieved'] for r in simulation_results]) / len(simulation_results) * 100
    avg_payback = np.mean([min(r['payback_period'], 15) for r in simulation_results])
    avg_automation = np.mean([r['automation_level'] for r in simulation_results])
    
    print("\n" + "="*80)
    print("SIMULATION SUMMARY REPORT - STRATEGY D")
    print("="*80)
    print(f"Total Suppliers Analyzed: {len(sample_suppliers)}")
    print(f"Total Baseline Emissions: {total_baseline:,.0f} tons CO₂e")
    print(f"Total Reduction Achieved: {total_reduction:,.0f} tons CO₂e ({total_reduction/total_baseline*100:.1f}%)")
    print(f"Total Investment Required: ${total_investment:,.0f} USD")
    print(f"Average Efficiency: {avg_efficiency:.4f} tons CO₂e per USD")
    print(f"Target Achievement Rate: {success_rate:.1f}%")
    print(f"Average Payback Period: {avg_payback:.1f} years")
    print(f"Average Automation Level: {avg_automation:.2f}")
    print("="*80 + "\n")
    
    # Step 8: Export data to CSV and Excel files
    print("\n💾 Step 8: Exporting data to CSV/Excel files...")
    export_data_for_charts(sample_suppliers, simulation_results, suppliers_df, tech_db, management_db)
    
    plt.show()
    
    return suppliers_df, simulation_results, sample_suppliers


def export_data_for_charts(sample_suppliers, simulation_results, all_suppliers, tech_db, management_db):
    """Export simulation data in formats compatible with visualizer"""
    
    # 1. Three-year reduction pathway data
    pathway_data = []
    for idx, (supplier_row, result) in enumerate(zip(sample_suppliers.iterrows(), simulation_results)):
        _, supplier = supplier_row
        for year in range(4):
            pathway_data.append({
                '供应商ID': result['supplier_id'],
                '年份': f'Y{year}',
                '碳排放量': result['yearly_emissions'][year],
                '年度减排量': result['yearly_reductions'][year] if year > 0 else 0,
                '累计减排量': result['cumulative_reduction'][year],
                '减排率': result['cumulative_reduction'][year] / result['baseline_emission'] * 100 if result['baseline_emission'] > 0 else 0
            })
    
    pathway_df = pd.DataFrame(pathway_data)
    
    # 2. Supplier classification data
    classification_data = []
    for idx, supplier_row in sample_suppliers.iterrows():
        classification_data.append({
            '供应商ID': supplier_row['supplier_id'],
            '基线排放': supplier_row['baseline_emission'],
            '配合度得分': supplier_row['cooperation_score'],
            '行业类型': supplier_row['industry_type'],
            '财务能力': supplier_row['financial_capacity'],
            '技术采用水平': supplier_row['tech_adoption_level'],
            '监测就绪度': supplier_row['monitoring_readiness'],
            '成本敏感度': supplier_row['cost_sensitivity'],
            '自我激励度': supplier_row['self_motivation'],
            '象限分类': 'IV-观察区',
            '优先级': '低',
            '建议策略': '基础管理-自动监测-简化要求'
        })
    
    classification_df = pd.DataFrame(classification_data)
    
    # 3. Investment budget allocation
    budget_data = []
    for idx, (supplier_row, result) in enumerate(zip(sample_suppliers.iterrows(), simulation_results)):
        _, supplier = supplier_row
        budget_data.append({
            '供应商ID': result['supplier_id'],
            '技术投资': result['tech_investment'],
            '管理成本': result['management_cost'],
            '总投资': result['total_investment'],
            '年度节省': result['annual_savings'],
            '投资回收期': min(result['payback_period'], 15),
            '监测频率': result['monitoring_frequency'],
            '自动化水平': result['automation_level'],
            '已选技术': ', '.join(result['selected_technologies'][:3]),
            '目标达成': '是' if result['target_achieved'] else '否'
        })
    
    budget_df = pd.DataFrame(budget_data)
    
    # 4. Technology database
    tech_db_export = tech_db.copy()
    tech_db_export.columns = ['技术ID', '技术名称', '类别', '减排率', '单位成本', 
                              '实施周期(月)', '年度节省', '维护成本', '成本效益']
    
    # 5. Management database
    management_db_export = management_db.copy()
    management_db_export.columns = ['倡议ID', '倡议名称', '类别', '年度成本', 
                                    '有效性', '自动化水平']
    
    # 6. Detailed supplier profiles
    detailed_suppliers = sample_suppliers.copy()
    detailed_suppliers['baseline_emission'] = detailed_suppliers['baseline_emission'].round(2)
    detailed_suppliers['annual_revenue'] = detailed_suppliers['annual_revenue'].round(2)
    detailed_suppliers['tech_adoption_level'] = detailed_suppliers['tech_adoption_level'].round(3)
    detailed_suppliers['financial_capacity'] = detailed_suppliers['financial_capacity'].round(3)
    detailed_suppliers['monitoring_readiness'] = detailed_suppliers['monitoring_readiness'].round(3)
    detailed_suppliers['cost_sensitivity'] = detailed_suppliers['cost_sensitivity'].round(3)
    detailed_suppliers['self_motivation'] = detailed_suppliers['self_motivation'].round(3)
    detailed_suppliers['replacement_risk'] = detailed_suppliers['replacement_risk'].round(3)
    
    # 7. Strategy summary
    strategy_summary = []
    for idx, result in enumerate(simulation_results):
        strategy_summary.append({
            '供应商ID': result['supplier_id'],
            '基线排放': result['baseline_emission'],
            '最终减排量': result['cumulative_reduction'][-1],
            '减排率': result['final_reduction_rate'] * 100,
            '总投资': result['total_investment'],
            '效率(吨/美元)': result['cumulative_reduction'][-1] / result['total_investment'] if result['total_investment'] > 0 else 0,
            '投资回收期': min(result['payback_period'], 15),
            '目标达成': result['target_achieved'],
            '配合度因子': result['cooperation_factor'],
            '自动化水平': result['automation_level']
        })
    
    strategy_summary_df = pd.DataFrame(strategy_summary)
    
    # Save to CSV files
    csv_dir = os.getcwd()
    
    pathway_df.to_csv(os.path.join(csv_dir, 'ML_simulation_D_三年减排路径.csv'), 
                     index=False, encoding='utf-8-sig')
    classification_df.to_csv(os.path.join(csv_dir, 'ML_simulation_D_四象限分类.csv'), 
                            index=False, encoding='utf-8-sig')
    budget_df.to_csv(os.path.join(csv_dir, 'ML_simulation_D_投资预算分配.csv'), 
                    index=False, encoding='utf-8-sig')
    tech_db_export.to_csv(os.path.join(csv_dir, 'ML_simulation_D_技术数据库.csv'), 
                         index=False, encoding='utf-8-sig')
    management_db_export.to_csv(os.path.join(csv_dir, 'ML_simulation_D_基础管理数据库.csv'),
                                index=False, encoding='utf-8-sig')
    detailed_suppliers.to_csv(os.path.join(csv_dir, 'ML_simulation_D_供应商详细信息.csv'), 
                             index=False, encoding='utf-8-sig')
    strategy_summary_df.to_csv(os.path.join(csv_dir, 'ML_simulation_D_strategy_summary.csv'),
                              index=False, encoding='utf-8-sig')
    
    print(f"✓ CSV files saved:")
    print(f"  - ML_simulation_D_三年减排路径.csv ({len(pathway_df)} rows)")
    print(f"  - ML_simulation_D_四象限分类.csv ({len(classification_df)} rows)")
    print(f"  - ML_simulation_D_投资预算分配.csv ({len(budget_df)} rows)")
    print(f"  - ML_simulation_D_技术数据库.csv ({len(tech_db_export)} rows)")
    print(f"  - ML_simulation_D_基础管理数据库.csv ({len(management_db_export)} rows)")
    print(f"  - ML_simulation_D_供应商详细信息.csv ({len(detailed_suppliers)} rows)")
    print(f"  - ML_simulation_D_strategy_summary.csv ({len(strategy_summary_df)} rows)")
    
    # Save to Excel file with multiple sheets
    try:
        excel_path = os.path.join(csv_dir, 'ML_simulation_D_ESG供应商数据.xlsx')
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            pathway_df.to_excel(writer, sheet_name='三年减排路径', index=False)
            classification_df.to_excel(writer, sheet_name='四象限分类', index=False)
            budget_df.to_excel(writer, sheet_name='投资预算分配', index=False)
            tech_db_export.to_excel(writer, sheet_name='技术数据库', index=False)
            management_db_export.to_excel(writer, sheet_name='基础管理数据库', index=False)
            detailed_suppliers.to_excel(writer, sheet_name='供应商详细信息', index=False)
            strategy_summary_df.to_excel(writer, sheet_name='策略汇总', index=False)
        print(f"✓ Excel file saved: {excel_path}")
    except ImportError:
        print("⚠ openpyxl not installed. Excel file not generated.")
        print("  Install with: pip install openpyxl")
    
    return


if __name__ == "__main__":
    suppliers_df, simulation_results, sample_suppliers = main()
    
    print("✅ Strategy D ML Simulation Complete!")
    print("\n💡 Key Findings:")
    print("  1. Zone IV suppliers require minimal investment with automated monitoring")
    print("  2. Low-cost technologies provide adequate 5-10% reduction targets")
    print("  3. Automation and simplified management reduce overhead costs")
    print("  4. Focus on cost-effectiveness and quick payback periods")
    print("\n🎯 Next Steps:")
    print("  - Implement automated monitoring systems to reduce manual effort")
    print("  - Consider supplier replacement if cost-benefit remains unfavorable")
    print("  - Maintain basic compliance without heavy investment")
    print("  - Integrate with A, B, C strategies for complete portfolio optimization")
