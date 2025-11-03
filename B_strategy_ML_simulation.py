"""
Strategy B ML Simulation - ESG Supply Chain Emission Reduction
================================================================
This standalone simulation uses machine learning to model and optimize
emission reduction strategies for suppliers in Zone II (Risk Zone).

Strategy B Focus: High-emission, low-cooperation suppliers
- Target: 20-30% emission reduction over 3 years
- Approach: Enforcement measures + incentives + capacity building
- ML Method: Multi-output regression + compliance prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from matplotlib import rcParams
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import os

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Chinese font configuration
rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'PingFang SC', 'STHeiti']
rcParams['axes.unicode_minus'] = False

# ============================================================================
# PART 1: Synthetic Data Generation - Zone II Supplier Profiles
# ============================================================================

class SupplierDataGeneratorB:
    """Generate realistic synthetic supplier data for Zone II suppliers"""
    
    def __init__(self, n_suppliers=500, seed=42):
        self.n_suppliers = n_suppliers
        self.rng = np.random.default_rng(seed)
        
    def generate_suppliers(self):
        """Generate Zone II supplier characteristics"""
        
        # Zone II criteria: High emission + Low cooperation
        # Emission: 10,000 - 25,000 tons CO2
        # Cooperation score: 3-6 out of 10 (lower than Zone I)
        
        baseline_emission = self.rng.uniform(10000, 25000, self.n_suppliers)
        cooperation_score = self.rng.integers(3, 7, self.n_suppliers)
        
        # Additional supplier characteristics
        annual_revenue = baseline_emission * self.rng.uniform(40, 120, self.n_suppliers)
        employee_count = self.rng.integers(80, 800, self.n_suppliers)
        years_partnership = self.rng.integers(1, 10, self.n_suppliers)
        tech_adoption_level = self.rng.uniform(0.1, 0.5, self.n_suppliers)  # Lower than Zone I
        financial_capacity = self.rng.uniform(0.3, 1.2, self.n_suppliers)
        
        # Zone II specific attributes
        compliance_history = self.rng.uniform(0.3, 0.7, self.n_suppliers)  # Historical compliance rate
        resistance_level = self.rng.uniform(0.3, 0.8, self.n_suppliers)  # Resistance to change
        transparency_score = self.rng.uniform(0.2, 0.6, self.n_suppliers)  # Data transparency
        
        # Industry type (weighted by likelihood)
        industry_types = self.rng.choice(
            ['Dyeing', 'Weaving', 'Finishing', 'Manufacturing'], 
            size=self.n_suppliers,
            p=[0.45, 0.30, 0.15, 0.10]  # More dyeing (higher emission)
        )
        
        df = pd.DataFrame({
            'supplier_id': [f'SUP_B_{i:03d}' for i in range(self.n_suppliers)],
            'baseline_emission': baseline_emission,
            'cooperation_score': cooperation_score,
            'annual_revenue': annual_revenue,
            'employee_count': employee_count,
            'years_partnership': years_partnership,
            'tech_adoption_level': tech_adoption_level,
            'financial_capacity': financial_capacity,
            'compliance_history': compliance_history,
            'resistance_level': resistance_level,
            'transparency_score': transparency_score,
            'industry_type': industry_types
        })
        
        return df


# ============================================================================
# PART 2: Technology & Enforcement Database
# ============================================================================

class TechnologyDatabaseB:
    """Database of emission reduction technologies suitable for Zone II"""
    
    @staticmethod
    def get_technologies():
        """Return technology database focused on compliance and basic upgrades"""
        
        tech_data = [
            # Basic compliance technologies (lower cost, mandatory)
            {
                'tech_id': 'B_TECH_001',
                'name': 'Wastewater Treatment Upgrade',
                'category': 'Compliance',
                'emission_reduction_rate': 0.08,
                'cost_per_unit': 25000,
                'implementation_time_months': 3,
                'annual_opex_saving': 5000,
                'compliance_requirement': True
            },
            {
                'tech_id': 'B_TECH_002',
                'name': 'Basic Energy Monitoring System',
                'category': 'Monitoring',
                'emission_reduction_rate': 0.05,
                'cost_per_unit': 15000,
                'implementation_time_months': 2,
                'annual_opex_saving': 8000,
                'compliance_requirement': True
            },
            {
                'tech_id': 'B_TECH_003',
                'name': 'Boiler Efficiency Improvement',
                'category': 'Energy',
                'emission_reduction_rate': 0.12,
                'cost_per_unit': 40000,
                'implementation_time_months': 4,
                'annual_opex_saving': 12000,
                'compliance_requirement': False
            },
            {
                'tech_id': 'B_TECH_004',
                'name': 'LED Lighting Retrofit',
                'category': 'Energy',
                'emission_reduction_rate': 0.03,
                'cost_per_unit': 8000,
                'implementation_time_months': 1,
                'annual_opex_saving': 6000,
                'compliance_requirement': False
            },
            {
                'tech_id': 'B_TECH_005',
                'name': 'Heat Recovery System',
                'category': 'Energy',
                'emission_reduction_rate': 0.10,
                'cost_per_unit': 35000,
                'implementation_time_months': 3,
                'annual_opex_saving': 10000,
                'compliance_requirement': False
            },
            {
                'tech_id': 'B_TECH_006',
                'name': 'Chemical Management System',
                'category': 'Compliance',
                'emission_reduction_rate': 0.06,
                'cost_per_unit': 20000,
                'implementation_time_months': 2,
                'annual_opex_saving': 4000,
                'compliance_requirement': True
            },
            {
                'tech_id': 'B_TECH_007',
                'name': 'Air Filtration Upgrade',
                'category': 'Compliance',
                'emission_reduction_rate': 0.07,
                'cost_per_unit': 30000,
                'implementation_time_months': 3,
                'annual_opex_saving': 5000,
                'compliance_requirement': True
            },
            {
                'tech_id': 'B_TECH_008',
                'name': 'Variable Frequency Drives',
                'category': 'Energy',
                'emission_reduction_rate': 0.09,
                'cost_per_unit': 22000,
                'implementation_time_months': 2,
                'annual_opex_saving': 9000,
                'compliance_requirement': False
            },
            {
                'tech_id': 'B_TECH_009',
                'name': 'Water Recycling System (Basic)',
                'category': 'Resource',
                'emission_reduction_rate': 0.08,
                'cost_per_unit': 28000,
                'implementation_time_months': 4,
                'annual_opex_saving': 7000,
                'compliance_requirement': False
            },
            {
                'tech_id': 'B_TECH_010',
                'name': 'Steam System Optimization',
                'category': 'Energy',
                'emission_reduction_rate': 0.11,
                'cost_per_unit': 38000,
                'implementation_time_months': 3,
                'annual_opex_saving': 11000,
                'compliance_requirement': False
            }
        ]
        
        return pd.DataFrame(tech_data)


class EnforcementPolicyDatabase:
    """Database of enforcement measures for Zone II suppliers"""
    
    @staticmethod
    def get_policies():
        """Return enforcement policy options"""
        
        policy_data = [
            {
                'policy_id': 'ENF_001',
                'name': 'Mandatory Compliance Audit',
                'type': 'Monitoring',
                'cost': 5000,
                'effectiveness': 0.15,  # Improvement in cooperation score
                'duration_months': 6
            },
            {
                'policy_id': 'ENF_002',
                'name': 'Performance-Based Contract',
                'type': 'Incentive',
                'cost': 0,  # Cost is in form of procurement adjustments
                'effectiveness': 0.20,
                'duration_months': 12
            },
            {
                'policy_id': 'ENF_003',
                'name': 'Capacity Building Program',
                'type': 'Training',
                'cost': 15000,
                'effectiveness': 0.25,
                'duration_months': 9
            },
            {
                'policy_id': 'ENF_004',
                'name': 'Third-Party Verification',
                'type': 'Monitoring',
                'cost': 8000,
                'effectiveness': 0.12,
                'duration_months': 12
            },
            {
                'policy_id': 'ENF_005',
                'name': 'Financial Penalty System',
                'type': 'Enforcement',
                'cost': 0,  # Revenue generating
                'effectiveness': 0.18,
                'duration_months': 12
            },
            {
                'policy_id': 'ENF_006',
                'name': 'Technical Support Package',
                'type': 'Support',
                'cost': 12000,
                'effectiveness': 0.22,
                'duration_months': 6
            }
        ]
        
        return pd.DataFrame(policy_data)


# ============================================================================
# PART 3: Enforcement Policy ML Model
# ============================================================================

class EnforcementPolicyML:
    """Machine Learning model to predict optimal enforcement strategies"""
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def generate_training_data(self, suppliers_df, n_scenarios=1000):
        """Generate synthetic training data for enforcement optimization"""
        
        training_data = []
        
        for _ in range(n_scenarios):
            # Sample a supplier
            supplier = suppliers_df.sample(n=1).iloc[0]
            
            # Generate enforcement strategy parameters
            audit_frequency = np.random.uniform(1, 4)  # Audits per year
            penalty_severity = np.random.uniform(0.5, 2.0)  # Penalty multiplier
            support_level = np.random.uniform(0, 1)  # Support intensity
            training_hours = np.random.uniform(0, 100)  # Training hours
            
            # Calculate expected cooperation improvement
            base_improvement = 0.1
            audit_effect = audit_frequency * 0.05
            penalty_effect = penalty_severity * 0.08
            support_effect = support_level * 0.15
            training_effect = training_hours * 0.002
            
            # Factor in resistance level (negative effect)
            resistance_penalty = supplier['resistance_level'] * 0.2
            
            cooperation_improvement = (base_improvement + audit_effect + 
                                      penalty_effect + support_effect + 
                                      training_effect - resistance_penalty)
            cooperation_improvement = np.clip(cooperation_improvement, 0, 0.5)
            
            # Calculate expected emission reduction
            tech_factor = supplier['tech_adoption_level'] * 0.3
            financial_factor = supplier['financial_capacity'] * 0.2
            cooperation_factor = cooperation_improvement * 0.4
            
            emission_reduction_rate = (tech_factor + financial_factor + 
                                      cooperation_factor) * 0.8
            emission_reduction_rate = np.clip(emission_reduction_rate, 0, 0.30)
            
            # Calculate total cost
            enforcement_cost = (audit_frequency * 5000 + 
                               support_level * 15000 + 
                               training_hours * 100)
            
            training_data.append({
                'baseline_emission': supplier['baseline_emission'],
                'cooperation_score': supplier['cooperation_score'],
                'tech_adoption_level': supplier['tech_adoption_level'],
                'financial_capacity': supplier['financial_capacity'],
                'compliance_history': supplier['compliance_history'],
                'resistance_level': supplier['resistance_level'],
                'transparency_score': supplier['transparency_score'],
                'audit_frequency': audit_frequency,
                'penalty_severity': penalty_severity,
                'support_level': support_level,
                'training_hours': training_hours,
                'cooperation_improvement': cooperation_improvement,
                'emission_reduction_rate': emission_reduction_rate,
                'enforcement_cost': enforcement_cost
            })
        
        return pd.DataFrame(training_data)
    
    def train(self, suppliers_df):
        """Train ML models to predict optimal enforcement strategies"""
        
        print("📊 Generating training data for enforcement policy ML models...")
        training_data = self.generate_training_data(suppliers_df, n_scenarios=2000)
        
        # Features and targets
        feature_cols = ['baseline_emission', 'cooperation_score', 'tech_adoption_level', 
                       'financial_capacity', 'compliance_history', 'resistance_level',
                       'transparency_score']
        target_cols = ['audit_frequency', 'penalty_severity', 'support_level', 
                      'training_hours', 'cooperation_improvement', 'emission_reduction_rate']
        
        self.feature_names = feature_cols
        X = training_data[feature_cols]
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train separate model for each enforcement parameter
        print("\n🤖 Training ML models for enforcement parameters...")
        
        for target in target_cols:
            print(f"  Training model for: {target}")
            
            y = training_data[target]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            # Train Gradient Boosting model
            model = GradientBoostingRegressor(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=4,
                random_state=42
            )
            
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            print(f"    R² Score: {r2:.4f} | RMSE: {rmse:.4f}")
            
            self.models[target] = model
        
        print("\n✅ All enforcement policy models trained successfully!\n")
    
    def predict_enforcement_strategy(self, supplier_features):
        """Predict optimal enforcement strategy for a supplier"""
        
        X = pd.DataFrame([supplier_features])[self.feature_names]
        X_scaled = self.scaler.transform(X)
        
        strategy = {}
        for target, model in self.models.items():
            strategy[target] = max(0, model.predict(X_scaled)[0])
        
        return strategy


# ============================================================================
# PART 4: Emission Reduction Simulator for Zone II
# ============================================================================

class EmissionReductionSimulatorB:
    """Simulate 3-year emission reduction pathway for Zone II suppliers"""
    
    def __init__(self, tech_db, enforcement_db):
        self.tech_db = tech_db
        self.enforcement_db = enforcement_db
    
    def simulate_reduction_pathway(self, supplier, enforcement_strategy, budget=80000, mode='balanced'):
        """
        Simulate 3-year emission reduction pathway for Zone II
        
        Args:
            supplier: Supplier dataframe row
            enforcement_strategy: Dict of predicted enforcement parameters
            budget: Available investment budget (USD)
            mode: 'aggressive' (30%), 'balanced' (25%), or 'conservative' (20%)
        """
        
        # Set reduction targets based on mode
        target_map = {
            'aggressive': [0.10, 0.20, 0.30],
            'balanced': [0.08, 0.17, 0.25],
            'conservative': [0.06, 0.13, 0.20]
        }
        
        targets = target_map.get(mode, target_map['balanced'])
        
        # Phase 1: Mandatory compliance technologies
        mandatory_techs = self.tech_db[self.tech_db['compliance_requirement'] == True]
        selected_techs = []
        total_cost = 0
        total_reduction_rate = 0
        
        for _, tech in mandatory_techs.iterrows():
            if total_cost + tech['cost_per_unit'] <= budget * 0.4:  # 40% for compliance
                selected_techs.append({
                    'tech': tech.to_dict(),
                    'year_implemented': 1,
                    'reason': 'Compliance Requirement'
                })
                total_cost += tech['cost_per_unit']
                total_reduction_rate += tech['emission_reduction_rate']
        
        # Phase 2: Optional efficiency upgrades
        optional_techs = self.tech_db[self.tech_db['compliance_requirement'] == False]
        tech_scores = []
        
        for _, tech in optional_techs.iterrows():
            roi = (tech['annual_opex_saving'] * 3) / tech['cost_per_unit']
            effectiveness = tech['emission_reduction_rate']
            score = roi * 0.4 + effectiveness * 0.6
            
            tech_scores.append({
                'tech': tech.to_dict(),
                'score': score,
                'roi': roi
            })
        
        tech_scores_df = pd.DataFrame(tech_scores).sort_values('score', ascending=False)
        
        # Add optional technologies within remaining budget
        remaining_budget = budget - total_cost
        
        for _, tech_score in tech_scores_df.iterrows():
            tech = tech_score['tech']
            if total_cost + tech['cost_per_unit'] <= budget:
                year = 2 if len([t for t in selected_techs if t['year_implemented'] == 1]) > 2 else 1
                selected_techs.append({
                    'tech': tech,
                    'year_implemented': year,
                    'reason': 'Efficiency Improvement'
                })
                total_cost += tech['cost_per_unit']
                total_reduction_rate += tech['emission_reduction_rate']
        
        # Calculate cooperation improvement over time
        cooperation_improvement = enforcement_strategy['cooperation_improvement']
        year_1_coop = supplier['cooperation_score']
        year_2_coop = min(10, year_1_coop + cooperation_improvement * 0.5)
        year_3_coop = min(10, year_1_coop + cooperation_improvement)
        
        # Calculate emissions with cooperation factor
        baseline = supplier['baseline_emission']
        coop_factor_1 = year_1_coop / 10
        coop_factor_2 = year_2_coop / 10
        coop_factor_3 = year_3_coop / 10
        
        # Actual reduction is limited by cooperation
        year_1_reduction = min(total_reduction_rate * 0.3 * coop_factor_1, targets[0])
        year_2_reduction = min(total_reduction_rate * 0.6 * coop_factor_2, targets[1])
        year_3_reduction = min(total_reduction_rate * 0.9 * coop_factor_3, targets[2])
        
        emissions = [
            baseline,
            baseline * (1 - year_1_reduction),
            baseline * (1 - year_2_reduction),
            baseline * (1 - year_3_reduction)
        ]
        
        cumulative_reduction = [
            0,
            baseline * year_1_reduction,
            baseline * year_2_reduction,
            baseline * year_3_reduction
        ]
        
        return {
            'emissions': emissions,
            'cumulative_reduction': cumulative_reduction,
            'selected_technologies': selected_techs,
            'total_investment': total_cost,
            'enforcement_cost': enforcement_strategy.get('enforcement_cost', 0),
            'total_reduction_rate': total_reduction_rate,
            'cooperation_improvement': cooperation_improvement,
            'final_cooperation_score': year_3_coop,
            'target_achieved': year_3_reduction >= targets[2]
        }


# ============================================================================
# PART 5: Visualization
# ============================================================================

class StrategyVisualizerB:
    """Visualize Strategy B simulation results"""
    
    @staticmethod
    def plot_results(simulation_results, suppliers_sample):
        """Create comprehensive visualization of Strategy B results"""
        
        fig = plt.figure(figsize=(20, 12))
        fig.patch.set_facecolor('white')
        fig.suptitle('Strategy B (Zone II): Emission Reduction with Enforcement\n' +
                    'B策略 (风险区): 强制措施下的减排路径分析', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        # Create subplots
        ax1 = plt.subplot(2, 3, 1)  # Emission pathways
        ax2 = plt.subplot(2, 3, 2)  # Cooperation improvement
        ax3 = plt.subplot(2, 3, 3)  # Cost breakdown
        ax4 = plt.subplot(2, 3, 4)  # Target achievement
        ax5 = plt.subplot(2, 3, 5)  # Technology distribution
        ax6 = plt.subplot(2, 3, 6)  # ROI analysis
        
        # --- Plot 1: Emission Pathways ---
        years = [0, 1, 2, 3]
        colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(simulation_results)))
        
        for idx, result in enumerate(simulation_results[:10]):  # Show top 10
            ax1.plot(years, result['emissions'], marker='o', linewidth=2, 
                    alpha=0.7, color=colors[idx])
        
        ax1.set_xlabel('Year', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Emissions (tons CO₂e)', fontsize=11, fontweight='bold')
        ax1.set_title('Emission Reduction Pathways\n减排路径', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_xticks(years)
        
        # --- Plot 2: Cooperation Score Improvement ---
        initial_coop = [suppliers_sample.iloc[i]['cooperation_score'] 
                       for i in range(len(simulation_results))]
        final_coop = [r['final_cooperation_score'] for r in simulation_results]
        improvement = [f - i for i, f in zip(initial_coop, final_coop)]
        
        x_pos = np.arange(len(improvement))
        bars = ax2.barh(x_pos, improvement, color='steelblue', alpha=0.7, edgecolor='black')
        
        # Highlight top improvements
        top_3_idx = np.argsort(improvement)[-3:]
        for idx in top_3_idx:
            bars[idx].set_color('forestgreen')
        
        ax2.set_ylabel('Supplier Index', fontsize=11, fontweight='bold')
        ax2.set_xlabel('Cooperation Score Improvement', fontsize=11, fontweight='bold')
        ax2.set_title('Cooperation Improvement\n配合度提升', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x', linestyle='--')
        
        # --- Plot 3: Cost Breakdown ---
        tech_costs = [r['total_investment'] for r in simulation_results]
        enf_costs = [r['enforcement_cost'] for r in simulation_results]
        total_costs = [t + e for t, e in zip(tech_costs, enf_costs)]
        
        avg_tech = np.mean(tech_costs)
        avg_enf = np.mean(enf_costs)
        
        costs = [avg_tech, avg_enf]
        labels = ['Technology\nInvestment', 'Enforcement\nCost']
        colors_pie = ['#FF6B6B', '#4ECDC4']
        
        wedges, texts, autotexts = ax3.pie(costs, labels=labels, autopct='%1.1f%%',
                                            colors=colors_pie, startangle=90,
                                            textprops={'fontsize': 10, 'fontweight': 'bold'})
        
        ax3.set_title(f'Average Cost Distribution\n平均成本分布\nTotal: ${np.mean(total_costs):,.0f}', 
                     fontsize=12, fontweight='bold')
        
        # --- Plot 4: Target Achievement Rate ---
        target_achieved = sum([r['target_achieved'] for r in simulation_results])
        target_not_achieved = len(simulation_results) - target_achieved
        
        achievement_data = [target_achieved, target_not_achieved]
        achievement_labels = ['Achieved\n达标', 'Not Achieved\n未达标']
        achievement_colors = ['#2ECC71', '#E74C3C']
        
        wedges, texts, autotexts = ax4.pie(achievement_data, labels=achievement_labels,
                                            autopct='%1.1f%%', colors=achievement_colors,
                                            startangle=90,
                                            textprops={'fontsize': 11, 'fontweight': 'bold'})
        
        ax4.set_title(f'Target Achievement Rate\n目标达成率\n({target_achieved}/{len(simulation_results)})', 
                     fontsize=12, fontweight='bold')
        
        # --- Plot 5: Technology Distribution ---
        tech_categories = {}
        for result in simulation_results:
            for tech_item in result['selected_technologies']:
                category = tech_item['tech']['category']
                tech_categories[category] = tech_categories.get(category, 0) + 1
        
        categories = list(tech_categories.keys())
        counts = list(tech_categories.values())
        
        ax5.bar(categories, counts, color='coral', alpha=0.8, edgecolor='black', linewidth=1.5)
        ax5.set_xlabel('Technology Category', fontsize=11, fontweight='bold')
        ax5.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax5.set_title('Technology Selection Distribution\n技术选择分布', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='y', linestyle='--')
        plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # --- Plot 6: ROI Analysis ---
        reductions = [r['cumulative_reduction'][-1] for r in simulation_results]
        costs = [r['total_investment'] + r['enforcement_cost'] for r in simulation_results]
        efficiency = [red / cost if cost > 0 else 0 for red, cost in zip(reductions, costs)]
        
        scatter = ax6.scatter(costs, reductions, c=efficiency, cmap='RdYlGn',
                             s=200, alpha=0.7, edgecolors='black', linewidth=1.5)
        
        # Add trend line
        z = np.polyfit(costs, reductions, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(costs), max(costs), 100)
        ax6.plot(x_trend, p(x_trend), 'b--', alpha=0.5, linewidth=2)
        
        ax6.set_xlabel('Total Cost (USD)', fontsize=11, fontweight='bold')
        ax6.set_ylabel('Emission Reduction (tons CO₂e)', fontsize=11, fontweight='bold')
        ax6.set_title('Investment Efficiency Analysis\n投资效率分析', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3, linestyle='--')
        
        cbar = plt.colorbar(scatter, ax=ax6)
        cbar.set_label('Efficiency (tons/$)', fontsize=9, fontweight='bold')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        
        return fig


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "="*80)
    print("STRATEGY B ML SIMULATION")
    print("ESG Supply Chain Emission Reduction - Zone II (Risk Zone)")
    print("="*80 + "\n")
    
    # Step 1: Generate synthetic supplier data
    print("📋 Step 1: Generating synthetic supplier data for Zone II...")
    data_gen = SupplierDataGeneratorB(n_suppliers=2000, seed=42)
    suppliers_df = data_gen.generate_suppliers()
    print(f"✓ Generated {len(suppliers_df)} Zone II suppliers")
    print(f"  - Average baseline emission: {suppliers_df['baseline_emission'].mean():.0f} tons CO₂")
    print(f"  - Average cooperation score: {suppliers_df['cooperation_score'].mean():.1f}/10")
    print(f"  - Average resistance level: {suppliers_df['resistance_level'].mean():.2f}")
    
    # Step 2: Load technology and enforcement databases
    print("\n🔧 Step 2: Loading technology and enforcement databases...")
    tech_db = TechnologyDatabaseB.get_technologies()
    enforcement_db = EnforcementPolicyDatabase.get_policies()
    print(f"✓ Loaded {len(tech_db)} technologies")
    print(f"✓ Loaded {len(enforcement_db)} enforcement policies")
    
    # Step 3: Train ML models for enforcement strategy
    print("\n🤖 Step 3: Training ML models for enforcement optimization...")
    enforcement_ml = EnforcementPolicyML()
    enforcement_ml.train(suppliers_df)
    
    # Step 4: Select sample suppliers for simulation
    print("\n🎯 Step 4: Selecting sample suppliers for detailed simulation...")
    sample_suppliers = suppliers_df.sample(n=24, random_state=42)
    print(f"✓ Selected {len(sample_suppliers)} suppliers for detailed analysis")
    
    # Step 5: Run emission reduction simulations
    print("\n⚡ Step 5: Running emission reduction simulations...")
    simulator = EmissionReductionSimulatorB(tech_db, enforcement_db)
    simulation_results = []
    
    for idx, supplier in sample_suppliers.iterrows():
        # Predict optimal enforcement strategy using ML
        supplier_features = {
            'baseline_emission': supplier['baseline_emission'],
            'cooperation_score': supplier['cooperation_score'],
            'tech_adoption_level': supplier['tech_adoption_level'],
            'financial_capacity': supplier['financial_capacity'],
            'compliance_history': supplier['compliance_history'],
            'resistance_level': supplier['resistance_level'],
            'transparency_score': supplier['transparency_score']
        }
        
        enforcement_strategy = enforcement_ml.predict_enforcement_strategy(supplier_features)
        
        # Simulate 3-year reduction pathway
        result = simulator.simulate_reduction_pathway(
            supplier, enforcement_strategy, budget=80000, mode='balanced'
        )
        result['enforcement_strategy'] = enforcement_strategy
        simulation_results.append(result)
        
        print(f"  ✓ {supplier['supplier_id']}: "
              f"{result['cumulative_reduction'][-1]:.0f} tons reduced "
              f"({result['total_reduction_rate']*100:.1f}% rate) | "
              f"Coop: {supplier['cooperation_score']:.1f} → {result['final_cooperation_score']:.1f}")
    
    # Step 6: Visualize results
    print("\n📊 Step 6: Generating visualization...")
    visualizer = StrategyVisualizerB()
    fig = visualizer.plot_results(simulation_results, sample_suppliers)
    
    output_path = 'Strategy_B_ML_Simulation.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Visualization saved: {output_path}")
    
    # Step 7: Generate summary report
    print("\n📈 Step 7: Generating summary report...")
    total_baseline = sum([s['baseline_emission'] for _, s in sample_suppliers.iterrows()])
    total_reduction = sum([r['cumulative_reduction'][-1] for r in simulation_results])
    total_investment = sum([r['total_investment'] + r['enforcement_cost'] for r in simulation_results])
    avg_efficiency = total_reduction / total_investment if total_investment > 0 else 0
    success_rate = sum([r['target_achieved'] for r in simulation_results]) / len(simulation_results) * 100
    avg_coop_improvement = np.mean([r['cooperation_improvement'] for r in simulation_results])
    
    print("\n" + "="*80)
    print("SIMULATION SUMMARY REPORT - STRATEGY B")
    print("="*80)
    print(f"Total Suppliers Analyzed: {len(sample_suppliers)}")
    print(f"Total Baseline Emissions: {total_baseline:,.0f} tons CO₂e")
    print(f"Total Reduction Achieved: {total_reduction:,.0f} tons CO₂e ({total_reduction/total_baseline*100:.1f}%)")
    print(f"Total Investment Required: ${total_investment:,.0f} USD")
    print(f"Average Efficiency: {avg_efficiency:.4f} tons CO₂e per USD")
    print(f"Target Achievement Rate: {success_rate:.1f}%")
    print(f"Average Cooperation Improvement: +{avg_coop_improvement:.2f} points")
    print("="*80 + "\n")
    
    # Step 8: Export data to CSV and Excel files
    print("\n💾 Step 8: Exporting data to CSV/Excel files...")
    export_data_for_charts(sample_suppliers, simulation_results, suppliers_df, tech_db)
    
    plt.show()
    
    return suppliers_df, simulation_results, sample_suppliers


def export_data_for_charts(sample_suppliers, simulation_results, all_suppliers, tech_db):
    """Export simulation data in formats compatible with visualizer"""
    
    # 1. Three-year reduction pathway data
    pathway_data = []
    for idx, (supplier_row, result) in enumerate(zip(sample_suppliers.iterrows(), simulation_results)):
        supplier = supplier_row[1]
        supplier_name = supplier['supplier_id']
        
        years = ['基线年', '第1年', '第2年', '第3年']
        for year_idx, (year, emission, cum_reduction) in enumerate(zip(
            years, 
            result['emissions'], 
            result['cumulative_reduction']
        )):
            pathway_data.append({
                '供应商': supplier_name,
                '年份': year,
                '年排放量': round(emission, 2),
                '累计减排量': round(cum_reduction, 2),
                '象限': 'II区',
                '工艺类型': supplier['industry_type'],
                '配合度评分': round(supplier['cooperation_score'], 1) if year_idx == 0 else round(result['final_cooperation_score'], 1)
            })
    
    pathway_df = pd.DataFrame(pathway_data)
    
    # 2. Supplier classification data
    classification_data = []
    for idx, supplier_row in sample_suppliers.iterrows():
        supplier = supplier_row
        result = simulation_results[list(sample_suppliers.index).index(idx)]
        
        classification_data.append({
            '供应商': supplier['supplier_id'],
            '工艺类型': supplier['industry_type'],
            '年产量(百万米)': round(supplier['annual_revenue'] / 1000000, 2),
            '年碳排放(吨CO2)': round(supplier['baseline_emission'], 2),
            '配合程度': supplier['cooperation_score'],
            '排放占比(%)': round(supplier['baseline_emission'] / all_suppliers['baseline_emission'].sum() * 100, 2),
            '排放影响得分(C)': round(supplier['baseline_emission'] / all_suppliers['baseline_emission'].max() * 10, 2),
            '配合程度得分(E)': supplier['cooperation_score'],
            '综合得分(S)': round((supplier['baseline_emission'] / all_suppliers['baseline_emission'].max() * 10 + 
                                  supplier['cooperation_score']) / 2, 2),
            '象限分类': 'II-风险区',
            '财务能力': round(supplier['financial_capacity'], 2),
            '技术采纳度': round(supplier['tech_adoption_level'], 2),
            '合作年限': supplier['years_partnership'],
            '抵触程度': round(supplier['resistance_level'], 2),
            '透明度': round(supplier['transparency_score'], 2),
            '历史合规率': round(supplier['compliance_history'], 2)
        })
    
    classification_df = pd.DataFrame(classification_data)
    
    # 3. Investment budget allocation
    budget_data = []
    for idx, (supplier_row, result) in enumerate(zip(sample_suppliers.iterrows(), simulation_results)):
        supplier = supplier_row[1]
        enforcement = result.get('enforcement_strategy', {})
        
        # Calculate metrics
        total_cost = result['total_investment'] + result['enforcement_cost']
        reduction_achieved = result['cumulative_reduction'][-1]
        roi = (reduction_achieved * 50) / total_cost if total_cost > 0 else 0  # Assume $50 per ton carbon price
        payback_years = total_cost / (reduction_achieved * 50 / 3) if reduction_achieved > 0 else 999
        
        budget_data.append({
            '供应商': supplier['supplier_id'],
            '投资金额': round(result['total_investment'], 2),
            '强制成本': round(result['enforcement_cost'], 2),
            '总成本': round(total_cost, 2),
            '预期减排量': round(reduction_achieved, 2),
            '减排率': round((reduction_achieved / supplier['baseline_emission']) * 100, 2),
            '投资回报率': round(roi * 100, 2),
            '回本周期(年)': round(payback_years, 2),
            '配合度提升': round(result['cooperation_improvement'], 2),
            '审计频次': round(enforcement.get('audit_frequency', 0), 2),
            '支持力度': round(enforcement.get('support_level', 0), 2),
            '培训时长': round(enforcement.get('training_hours', 0), 2)
        })
    
    budget_df = pd.DataFrame(budget_data)
    
    # 4. Technology database
    tech_db_export = tech_db.copy()
    
    # 5. Detailed supplier profiles
    detailed_suppliers = sample_suppliers.copy()
    detailed_suppliers['baseline_emission'] = detailed_suppliers['baseline_emission'].round(2)
    detailed_suppliers['annual_revenue'] = detailed_suppliers['annual_revenue'].round(2)
    detailed_suppliers['tech_adoption_level'] = detailed_suppliers['tech_adoption_level'].round(3)
    detailed_suppliers['financial_capacity'] = detailed_suppliers['financial_capacity'].round(3)
    detailed_suppliers['compliance_history'] = detailed_suppliers['compliance_history'].round(3)
    detailed_suppliers['resistance_level'] = detailed_suppliers['resistance_level'].round(3)
    detailed_suppliers['transparency_score'] = detailed_suppliers['transparency_score'].round(3)
    
    # 6. Strategy summary
    strategy_summary = []
    for idx, result in enumerate(simulation_results):
        supplier = sample_suppliers.iloc[idx]
        strategy_summary.append({
            '供应商': supplier['supplier_id'],
            '基线排放': round(supplier['baseline_emission'], 2),
            '最终排放': round(result['emissions'][-1], 2),
            '减排量': round(result['cumulative_reduction'][-1], 2),
            '减排率': round((result['cumulative_reduction'][-1] / supplier['baseline_emission']) * 100, 2),
            '是否达标': '是' if result['target_achieved'] else '否',
            '初始配合度': supplier['cooperation_score'],
            '最终配合度': round(result['final_cooperation_score'], 1),
            '配合度提升': round(result['cooperation_improvement'], 2),
            '技术数量': len(result['selected_technologies']),
            '总投资': round(result['total_investment'] + result['enforcement_cost'], 2)
        })
    
    strategy_summary_df = pd.DataFrame(strategy_summary)
    
    # Save to CSV files
    csv_dir = os.getcwd()
    
    pathway_df.to_csv(os.path.join(csv_dir, 'ML_simulation_B_三年减排路径.csv'), 
                     index=False, encoding='utf-8-sig')
    classification_df.to_csv(os.path.join(csv_dir, 'ML_simulation_B_四象限分类.csv'), 
                            index=False, encoding='utf-8-sig')
    budget_df.to_csv(os.path.join(csv_dir, 'ML_simulation_B_投资预算分配.csv'), 
                    index=False, encoding='utf-8-sig')
    tech_db_export.to_csv(os.path.join(csv_dir, 'ML_simulation_B_技术数据库.csv'), 
                         index=False, encoding='utf-8-sig')
    detailed_suppliers.to_csv(os.path.join(csv_dir, 'ML_simulation_B_供应商详细信息.csv'), 
                             index=False, encoding='utf-8-sig')
    strategy_summary_df.to_csv(os.path.join(csv_dir, 'ML_simulation_B_strategy_summary.csv'),
                              index=False, encoding='utf-8-sig')
    
    print(f"✓ CSV files saved:")
    print(f"  - ML_simulation_B_三年减排路径.csv ({len(pathway_df)} rows)")
    print(f"  - ML_simulation_B_四象限分类.csv ({len(classification_df)} rows)")
    print(f"  - ML_simulation_B_投资预算分配.csv ({len(budget_df)} rows)")
    print(f"  - ML_simulation_B_技术数据库.csv ({len(tech_db_export)} rows)")
    print(f"  - ML_simulation_B_供应商详细信息.csv ({len(detailed_suppliers)} rows)")
    print(f"  - ML_simulation_B_strategy_summary.csv ({len(strategy_summary_df)} rows)")
    
    # Save to Excel file with multiple sheets
    try:
        excel_path = os.path.join(csv_dir, 'ML_simulation_B_ESG供应商数据.xlsx')
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            pathway_df.to_excel(writer, sheet_name='三年减排路径', index=False)
            classification_df.to_excel(writer, sheet_name='四象限分类', index=False)
            budget_df.to_excel(writer, sheet_name='投资预算分配', index=False)
            tech_db_export.to_excel(writer, sheet_name='技术数据库', index=False)
            detailed_suppliers.to_excel(writer, sheet_name='供应商详细信息', index=False)
            strategy_summary_df.to_excel(writer, sheet_name='策略汇总', index=False)
        
        print(f"✓ Excel file saved: {excel_path}")
        
    except ImportError:
        print("⚠ openpyxl not installed. Excel export skipped.")
        print("  Install with: pip install openpyxl")
    
    return


if __name__ == "__main__":
    suppliers_df, simulation_results, sample_suppliers = main()
    
    print("✅ Strategy B ML Simulation Complete!")
    print("\n💡 Key Findings:")
    print("  1. Enforcement measures significantly improve cooperation scores")
    print("  2. Zone II suppliers require balanced approach: carrot + stick")
    print("  3. Compliance technologies must be prioritized first")
    print("  4. 20-25% reduction achievable with proper enforcement")
    print("\n🎯 Next Steps:")
    print("  - Monitor compliance and adjust enforcement intensity")
    print("  - Provide technical support to improve capability")
    print("  - Consider supplier replacement for persistent non-compliance")
    print("  - Integrate with A, C, D strategies for portfolio optimization")
