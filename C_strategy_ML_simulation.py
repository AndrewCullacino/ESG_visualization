"""
Strategy C ML Simulation - ESG Supply Chain Emission Reduction
================================================================
This standalone simulation uses machine learning to model and optimize
emission reduction strategies for suppliers in Zone III (Learning Zone).

Strategy C Focus: Low-emission, high-cooperation suppliers
- Target: 10-15% emission reduction over 3 years
- Approach: Knowledge sharing + best practice documentation + innovation
- ML Method: Multi-output regression + knowledge transfer optimization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
# PART 1: Synthetic Data Generation - Zone III Supplier Profiles
# ============================================================================

class SupplierDataGeneratorC:
    """Generate realistic synthetic supplier data for Zone III suppliers"""
    
    def __init__(self, n_suppliers=500, seed=42):
        self.n_suppliers = n_suppliers
        self.rng = np.random.default_rng(seed)
        
    def generate_suppliers(self):
        """Generate Zone III supplier characteristics"""
        
        # Zone III criteria: Low emission + High cooperation
        # Emission: 2,000 - 10,000 tons CO2 (lower than Zone I/II)
        # Cooperation score: 7-10 out of 10 (high cooperation)
        
        baseline_emission = self.rng.uniform(2000, 10000, self.n_suppliers)
        cooperation_score = self.rng.integers(7, 11, self.n_suppliers)
        
        # Additional supplier characteristics
        annual_revenue = baseline_emission * self.rng.uniform(60, 180, self.n_suppliers)
        employee_count = self.rng.integers(50, 500, self.n_suppliers)
        years_partnership = self.rng.integers(3, 20, self.n_suppliers)  # Longer partnerships
        tech_adoption_level = self.rng.uniform(0.5, 0.95, self.n_suppliers)  # High tech adoption
        financial_capacity = self.rng.uniform(0.7, 2.5, self.n_suppliers)
        
        # Zone III specific attributes
        innovation_capacity = self.rng.uniform(0.6, 0.95, self.n_suppliers)  # High innovation
        knowledge_sharing_score = self.rng.uniform(0.7, 0.98, self.n_suppliers)  # Willing to share
        best_practice_adoption = self.rng.uniform(0.6, 0.95, self.n_suppliers)  # Quick adopters
        certification_level = self.rng.integers(2, 6, self.n_suppliers)  # Number of ESG certs
        
        # Industry type (weighted by likelihood for lower emissions)
        industry_types = self.rng.choice(
            ['Finishing', 'Manufacturing', 'Weaving', 'Accessories'], 
            size=self.n_suppliers,
            p=[0.35, 0.30, 0.25, 0.10]  # Less dyeing (lower emission)
        )
        
        df = pd.DataFrame({
            'supplier_id': [f'SUP_C_{i:03d}' for i in range(self.n_suppliers)],
            'baseline_emission': baseline_emission,
            'cooperation_score': cooperation_score,
            'annual_revenue': annual_revenue,
            'employee_count': employee_count,
            'years_partnership': years_partnership,
            'tech_adoption_level': tech_adoption_level,
            'financial_capacity': financial_capacity,
            'innovation_capacity': innovation_capacity,
            'knowledge_sharing_score': knowledge_sharing_score,
            'best_practice_adoption': best_practice_adoption,
            'certification_level': certification_level,
            'industry_type': industry_types
        })
        
        return df


# ============================================================================
# PART 2: Technology & Knowledge Sharing Database
# ============================================================================

class TechnologyDatabaseC:
    """Database of emission reduction technologies suitable for Zone III"""
    
    @staticmethod
    def get_technologies():
        """Return technology database focused on innovation and optimization"""
        
        tech_data = [
            # Advanced optimization technologies
            {
                'tech_id': 'C_TECH_001',
                'name': 'AI-Powered Energy Management',
                'category': 'Innovation',
                'emission_reduction_rate': 0.06,
                'cost_per_unit': 35000,
                'implementation_time_months': 4,
                'annual_opex_saving': 15000,
                'knowledge_transfer_value': 0.90
            },
            {
                'tech_id': 'C_TECH_002',
                'name': 'IoT Process Monitoring',
                'category': 'Monitoring',
                'emission_reduction_rate': 0.04,
                'cost_per_unit': 28000,
                'implementation_time_months': 3,
                'annual_opex_saving': 12000,
                'knowledge_transfer_value': 0.85
            },
            {
                'tech_id': 'C_TECH_003',
                'name': 'Advanced Water Recycling',
                'category': 'Resource',
                'emission_reduction_rate': 0.05,
                'cost_per_unit': 32000,
                'implementation_time_months': 5,
                'annual_opex_saving': 10000,
                'knowledge_transfer_value': 0.80
            },
            {
                'tech_id': 'C_TECH_004',
                'name': 'Carbon Footprint Tracking System',
                'category': 'Monitoring',
                'emission_reduction_rate': 0.03,
                'cost_per_unit': 18000,
                'implementation_time_months': 2,
                'annual_opex_saving': 8000,
                'knowledge_transfer_value': 0.95
            },
            {
                'tech_id': 'C_TECH_005',
                'name': 'Renewable Energy Integration',
                'category': 'Energy',
                'emission_reduction_rate': 0.08,
                'cost_per_unit': 55000,
                'implementation_time_months': 6,
                'annual_opex_saving': 18000,
                'knowledge_transfer_value': 0.75
            },
            {
                'tech_id': 'C_TECH_006',
                'name': 'Process Automation & Optimization',
                'category': 'Efficiency',
                'emission_reduction_rate': 0.05,
                'cost_per_unit': 42000,
                'implementation_time_months': 4,
                'annual_opex_saving': 14000,
                'knowledge_transfer_value': 0.88
            },
            {
                'tech_id': 'C_TECH_007',
                'name': 'Circular Economy Model',
                'category': 'Resource',
                'emission_reduction_rate': 0.06,
                'cost_per_unit': 38000,
                'implementation_time_months': 5,
                'annual_opex_saving': 11000,
                'knowledge_transfer_value': 0.92
            },
            {
                'tech_id': 'C_TECH_008',
                'name': 'Smart Grid Integration',
                'category': 'Energy',
                'emission_reduction_rate': 0.04,
                'cost_per_unit': 30000,
                'implementation_time_months': 3,
                'annual_opex_saving': 13000,
                'knowledge_transfer_value': 0.82
            },
            {
                'tech_id': 'C_TECH_009',
                'name': 'Green Chemistry Adoption',
                'category': 'Innovation',
                'emission_reduction_rate': 0.05,
                'cost_per_unit': 45000,
                'implementation_time_months': 6,
                'annual_opex_saving': 9000,
                'knowledge_transfer_value': 0.87
            },
            {
                'tech_id': 'C_TECH_010',
                'name': 'Predictive Maintenance System',
                'category': 'Efficiency',
                'emission_reduction_rate': 0.03,
                'cost_per_unit': 25000,
                'implementation_time_months': 3,
                'annual_opex_saving': 10000,
                'knowledge_transfer_value': 0.84
            }
        ]
        
        return pd.DataFrame(tech_data)


class KnowledgeSharingDatabase:
    """Database of knowledge sharing and capacity building initiatives"""
    
    @staticmethod
    def get_initiatives():
        """Return knowledge sharing initiative options"""
        
        initiatives = [
            {
                'initiative_id': 'KS_001',
                'name': 'Best Practice Documentation',
                'category': 'Documentation',
                'cost': 8000,
                'duration_months': 2,
                'knowledge_multiplier': 1.15,
                'network_effect': 0.20
            },
            {
                'initiative_id': 'KS_002',
                'name': 'Peer-to-Peer Learning Network',
                'category': 'Networking',
                'cost': 12000,
                'duration_months': 3,
                'knowledge_multiplier': 1.25,
                'network_effect': 0.35
            },
            {
                'initiative_id': 'KS_003',
                'name': 'Innovation Workshop Series',
                'category': 'Training',
                'cost': 15000,
                'duration_months': 4,
                'knowledge_multiplier': 1.20,
                'network_effect': 0.28
            },
            {
                'initiative_id': 'KS_004',
                'name': 'Case Study Publication',
                'category': 'Documentation',
                'cost': 10000,
                'duration_months': 3,
                'knowledge_multiplier': 1.18,
                'network_effect': 0.25
            },
            {
                'initiative_id': 'KS_005',
                'name': 'Technology Transfer Program',
                'category': 'Transfer',
                'cost': 20000,
                'duration_months': 6,
                'knowledge_multiplier': 1.30,
                'network_effect': 0.40
            }
        ]
        
        return pd.DataFrame(initiatives)


# ============================================================================
# PART 3: Knowledge Transfer ML Model
# ============================================================================

class KnowledgeTransferML:
    """Machine Learning model to optimize knowledge sharing strategies"""
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def generate_training_data(self, suppliers_df, n_scenarios=1000):
        """Generate synthetic training data for knowledge transfer optimization"""
        
        training_data = []
        rng = np.random.default_rng(42)
        
        for _ in range(n_scenarios):
            # Randomly sample supplier characteristics
            idx = rng.integers(0, len(suppliers_df))
            supplier = suppliers_df.iloc[idx]
            
            # Simulate knowledge sharing initiatives
            num_initiatives = rng.integers(1, 4)
            documentation_effort = rng.uniform(0.3, 0.95)
            networking_intensity = rng.uniform(0.4, 0.98)
            innovation_investment = rng.uniform(5000, 30000)
            
            # Calculate outcomes based on supplier characteristics
            base_knowledge_gain = (
                supplier['innovation_capacity'] * 0.3 +
                supplier['knowledge_sharing_score'] * 0.3 +
                supplier['tech_adoption_level'] * 0.2 +
                supplier['best_practice_adoption'] * 0.2
            )
            
            # Knowledge transfer effectiveness
            knowledge_transfer_rate = base_knowledge_gain * (
                1 + documentation_effort * 0.3 +
                networking_intensity * 0.4 +
                (innovation_investment / 50000) * 0.3
            ) * rng.uniform(0.85, 1.15)
            
            # Emission reduction through knowledge application
            emission_reduction_rate = (
                0.03 + knowledge_transfer_rate * 0.15 +
                supplier['tech_adoption_level'] * 0.05
            ) * rng.uniform(0.9, 1.1)
            
            # Network effects
            network_value = (
                supplier['knowledge_sharing_score'] * networking_intensity * 
                supplier['cooperation_score'] / 10
            ) * rng.uniform(0.8, 1.2)
            
            # Innovation potential
            innovation_score = (
                supplier['innovation_capacity'] * 0.5 +
                knowledge_transfer_rate * 0.3 +
                (innovation_investment / 50000) * 0.2
            ) * rng.uniform(0.85, 1.15)
            
            training_data.append({
                'baseline_emission': supplier['baseline_emission'],
                'cooperation_score': supplier['cooperation_score'],
                'innovation_capacity': supplier['innovation_capacity'],
                'knowledge_sharing_score': supplier['knowledge_sharing_score'],
                'tech_adoption_level': supplier['tech_adoption_level'],
                'best_practice_adoption': supplier['best_practice_adoption'],
                'certification_level': supplier['certification_level'],
                'documentation_effort': documentation_effort,
                'networking_intensity': networking_intensity,
                'innovation_investment': innovation_investment,
                'num_initiatives': num_initiatives,
                'knowledge_transfer_rate': np.clip(knowledge_transfer_rate, 0, 1),
                'emission_reduction_rate': np.clip(emission_reduction_rate, 0.02, 0.15),
                'network_value': np.clip(network_value, 0, 1),
                'innovation_score': np.clip(innovation_score, 0, 1)
            })
        
        return pd.DataFrame(training_data)
    
    def train(self, suppliers_df):
        """Train ML models to predict optimal knowledge sharing strategies"""
        
        print("📊 Generating training data for knowledge transfer ML models...")
        training_data = self.generate_training_data(suppliers_df, n_scenarios=2000)
        
        # Features and targets
        feature_cols = ['baseline_emission', 'cooperation_score', 'innovation_capacity',
                       'knowledge_sharing_score', 'tech_adoption_level', 'best_practice_adoption',
                       'certification_level']
        target_cols = ['documentation_effort', 'networking_intensity', 'innovation_investment',
                      'knowledge_transfer_rate', 'emission_reduction_rate', 'network_value',
                      'innovation_score']
        
        self.feature_names = feature_cols
        X = training_data[feature_cols]
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train separate model for each knowledge transfer parameter
        print("\n🤖 Training ML models for knowledge transfer parameters...")
        
        for target in target_cols:
            y = training_data[target]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            # Train Random Forest model
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=15)
            rf_model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = rf_model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            print(f"  ✓ {target}: R² = {r2:.4f}, RMSE = {rmse:.4f}")
            
            self.models[target] = rf_model
        
        print("\n✅ All knowledge transfer models trained successfully!\n")
    
    def predict_knowledge_strategy(self, supplier_features):
        """Predict optimal knowledge sharing strategy for a supplier"""
        
        X = pd.DataFrame([supplier_features])[self.feature_names]
        X_scaled = self.scaler.transform(X)
        
        strategy = {}
        for target, model in self.models.items():
            strategy[target] = model.predict(X_scaled)[0]
        
        return strategy


# ============================================================================
# PART 4: Emission Reduction Simulator for Zone III
# ============================================================================

class EmissionReductionSimulatorC:
    """Simulate 3-year emission reduction pathway for Zone III suppliers"""
    
    def __init__(self, tech_db, knowledge_db):
        self.tech_db = tech_db
        self.knowledge_db = knowledge_db
    
    def simulate_reduction_pathway(self, supplier, knowledge_strategy, budget=50000, mode='innovation'):
        """
        Simulate 3-year emission reduction pathway for Zone III
        
        Args:
            supplier: Supplier dataframe row
            knowledge_strategy: Dict of predicted knowledge transfer parameters
            budget: Available investment budget (USD)
            mode: 'innovation' (15%), 'balanced' (12%), or 'maintenance' (10%)
        """
        
        # Set reduction targets based on mode
        target_map = {
            'innovation': [0.05, 0.10, 0.15],
            'balanced': [0.04, 0.08, 0.12],
            'maintenance': [0.03, 0.07, 0.10]
        }
        
        targets = target_map.get(mode, target_map['balanced'])
        
        # Phase 1: Technology selection based on knowledge transfer value
        selected_techs = []
        tech_scores = []
        
        for _, tech in self.tech_db.iterrows():
            # Score based on ROI + knowledge transfer value
            roi_score = (tech['annual_opex_saving'] / tech['cost_per_unit']) if tech['cost_per_unit'] > 0 else 0
            knowledge_score = tech['knowledge_transfer_value']
            reduction_score = tech['emission_reduction_rate']
            
            combined_score = (
                roi_score * 0.3 +
                knowledge_score * 0.4 +  # Higher weight for knowledge value
                reduction_score * 0.3
            )
            
            tech_scores.append({
                'tech_id': tech['tech_id'],
                'name': tech['name'],
                'cost': tech['cost_per_unit'],
                'reduction_rate': tech['emission_reduction_rate'],
                'knowledge_value': tech['knowledge_transfer_value'],
                'score': combined_score
            })
        
        tech_scores_df = pd.DataFrame(tech_scores).sort_values('score', ascending=False)
        
        # Allocate 70% of budget to technology
        tech_budget = budget * 0.7
        total_cost = 0
        total_reduction_rate = 0
        total_knowledge_value = 0
        
        for _, tech_score in tech_scores_df.iterrows():
            if total_cost + tech_score['cost'] <= tech_budget:
                selected_techs.append({
                    'tech_id': tech_score['tech_id'],
                    'name': tech_score['name'],
                    'cost': tech_score['cost'],
                    'reduction_rate': tech_score['reduction_rate'],
                    'knowledge_value': tech_score['knowledge_value']
                })
                total_cost += tech_score['cost']
                total_reduction_rate += tech_score['reduction_rate']
                total_knowledge_value += tech_score['knowledge_value']
        
        # Phase 2: Knowledge sharing initiatives
        knowledge_budget = budget * 0.3
        selected_initiatives = []
        
        for _, initiative in self.knowledge_db.iterrows():
            if knowledge_budget >= initiative['cost']:
                selected_initiatives.append({
                    'name': initiative['name'],
                    'cost': initiative['cost'],
                    'multiplier': initiative['knowledge_multiplier'],
                    'network_effect': initiative['network_effect']
                })
                knowledge_budget -= initiative['cost']
        
        # Calculate knowledge multiplier
        knowledge_multiplier = 1.0
        network_effect = 0.0
        for init in selected_initiatives:
            knowledge_multiplier *= init['multiplier']
            network_effect += init['network_effect']
        
        # Apply knowledge multiplier to reduction effectiveness
        effective_reduction_rate = total_reduction_rate * knowledge_multiplier
        
        # Calculate emissions with knowledge transfer effect
        baseline = supplier['baseline_emission']
        
        # Progressive improvement over 3 years
        year_1_reduction = min(effective_reduction_rate * 0.4, targets[0])
        year_2_reduction = min(effective_reduction_rate * 0.7, targets[1])
        year_3_reduction = min(effective_reduction_rate * 1.0, targets[2])
        
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
        
        # Calculate knowledge transfer metrics
        knowledge_transfer_score = (
            total_knowledge_value * knowledge_multiplier * 
            supplier['knowledge_sharing_score']
        )
        
        return {
            'emissions': emissions,
            'cumulative_reduction': cumulative_reduction,
            'selected_technologies': selected_techs,
            'selected_initiatives': selected_initiatives,
            'total_investment': total_cost + (budget * 0.3 - knowledge_budget),
            'tech_investment': total_cost,
            'knowledge_investment': budget * 0.3 - knowledge_budget,
            'total_reduction_rate': effective_reduction_rate,
            'knowledge_multiplier': knowledge_multiplier,
            'network_effect': network_effect,
            'knowledge_transfer_score': knowledge_transfer_score,
            'target_achieved': year_3_reduction >= targets[2],
            'innovation_potential': knowledge_strategy.get('innovation_score', 0)
        }


# ============================================================================
# PART 5: Visualization
# ============================================================================

class StrategyVisualizerC:
    """Visualize Strategy C simulation results"""
    
    @staticmethod
    def plot_results(simulation_results, suppliers_sample):
        """Create comprehensive visualization of Strategy C results"""
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Plot 1: Emission Reduction Pathways (Top 6)
        ax1 = fig.add_subplot(gs[0, :2])
        top_6_indices = sorted(range(len(simulation_results)), 
                              key=lambda i: simulation_results[i]['cumulative_reduction'][-1],
                              reverse=True)[:6]
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, 6))
        years = [0, 1, 2, 3]
        
        for plot_idx, idx in enumerate(top_6_indices):
            result = simulation_results[idx]
            supplier = suppliers_sample.iloc[idx]
            ax1.plot(years, result['emissions'], marker='o', linewidth=2.5, 
                    color=colors[plot_idx], label=f"{supplier['supplier_id']}")
        
        ax1.set_xlabel('Year', fontsize=12, weight='bold')
        ax1.set_ylabel('Emissions (tons CO₂e)', fontsize=12, weight='bold')
        ax1.set_title('Zone III: Top 6 Emission Reduction Pathways', fontsize=14, weight='bold')
        ax1.legend(loc='upper right', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Knowledge Transfer Effectiveness
        ax2 = fig.add_subplot(gs[0, 2])
        knowledge_scores = [r['knowledge_transfer_score'] for r in simulation_results]
        cooperation_scores = [s['cooperation_score'] for _, s in suppliers_sample.iterrows()]
        
        scatter = ax2.scatter(cooperation_scores, knowledge_scores, 
                            s=100, alpha=0.6, c=knowledge_scores, cmap='YlGn', edgecolors='black')
        ax2.set_xlabel('Cooperation Score', fontsize=10, weight='bold')
        ax2.set_ylabel('Knowledge Transfer Score', fontsize=10, weight='bold')
        ax2.set_title('Knowledge Transfer\nEffectiveness', fontsize=12, weight='bold')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax2, label='Transfer Score')
        
        # Plot 3: Investment Breakdown
        ax3 = fig.add_subplot(gs[1, 0])
        tech_investments = [r['tech_investment'] for r in simulation_results]
        knowledge_investments = [r['knowledge_investment'] for r in simulation_results]
        
        x_pos = np.arange(min(len(simulation_results), 10))
        width = 0.35
        
        ax3.bar(x_pos - width/2, tech_investments[:10], width, label='Technology', color='steelblue', alpha=0.8)
        ax3.bar(x_pos + width/2, knowledge_investments[:10], width, label='Knowledge', color='orange', alpha=0.8)
        
        ax3.set_xlabel('Supplier Index', fontsize=10, weight='bold')
        ax3.set_ylabel('Investment (USD)', fontsize=10, weight='bold')
        ax3.set_title('Investment Distribution', fontsize=12, weight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Innovation Potential vs. Reduction
        ax4 = fig.add_subplot(gs[1, 1])
        innovation_scores = [r['innovation_potential'] for r in simulation_results]
        reduction_rates = [r['cumulative_reduction'][-1] / suppliers_sample.iloc[i]['baseline_emission'] * 100 
                          for i, r in enumerate(simulation_results)]
        
        ax4.scatter(innovation_scores, reduction_rates, s=100, alpha=0.6, 
                   c=innovation_scores, cmap='plasma', edgecolors='black')
        ax4.set_xlabel('Innovation Potential', fontsize=10, weight='bold')
        ax4.set_ylabel('Reduction Rate (%)', fontsize=10, weight='bold')
        ax4.set_title('Innovation vs. Reduction', fontsize=12, weight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Knowledge Multiplier Effect
        ax5 = fig.add_subplot(gs[1, 2])
        knowledge_multipliers = [r['knowledge_multiplier'] for r in simulation_results]
        network_effects = [r['network_effect'] for r in simulation_results]
        
        ax5.scatter(knowledge_multipliers, network_effects, s=100, alpha=0.6,
                   c='green', edgecolors='black')
        ax5.set_xlabel('Knowledge Multiplier', fontsize=10, weight='bold')
        ax5.set_ylabel('Network Effect', fontsize=10, weight='bold')
        ax5.set_title('Knowledge Leverage', fontsize=12, weight='bold')
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Total Emission Reduction
        ax6 = fig.add_subplot(gs[2, 0])
        total_baseline = sum([s['baseline_emission'] for _, s in suppliers_sample.iterrows()])
        total_final = sum([r['emissions'][-1] for r in simulation_results])
        total_reduction = total_baseline - total_final
        
        categories = ['Baseline\nEmissions', 'Final\nEmissions', 'Total\nReduction']
        values = [total_baseline, total_final, total_reduction]
        colors_bar = ['#d62728', '#2ca02c', '#1f77b4']
        
        bars = ax6.bar(categories, values, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=2)
        ax6.set_ylabel('Emissions (tons CO₂e)', fontsize=10, weight='bold')
        ax6.set_title('Total Emission Impact', fontsize=12, weight='bold')
        
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:,.0f}',
                    ha='center', va='bottom', fontsize=10, weight='bold')
        
        # Plot 7: ROI Analysis
        ax7 = fig.add_subplot(gs[2, 1])
        roi_values = []
        for i, result in enumerate(simulation_results):
            reduction = result['cumulative_reduction'][-1]
            investment = result['total_investment']
            roi = (reduction * 50) / investment if investment > 0 else 0  # $50 per ton CO2
            roi_values.append(roi)
        
        ax7.hist(roi_values, bins=15, color='teal', alpha=0.7, edgecolor='black')
        ax7.axvline(np.mean(roi_values), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(roi_values):.2f}')
        ax7.set_xlabel('ROI Ratio', fontsize=10, weight='bold')
        ax7.set_ylabel('Frequency', fontsize=10, weight='bold')
        ax7.set_title('Investment ROI Distribution', fontsize=12, weight='bold')
        ax7.legend()
        ax7.grid(True, alpha=0.3, axis='y')
        
        # Plot 8: Summary Statistics
        ax8 = fig.add_subplot(gs[2, 2])
        ax8.axis('off')
        
        total_suppliers = len(simulation_results)
        avg_reduction = np.mean([r['cumulative_reduction'][-1] for r in simulation_results])
        avg_investment = np.mean([r['total_investment'] for r in simulation_results])
        success_rate = sum([r['target_achieved'] for r in simulation_results]) / total_suppliers * 100
        avg_knowledge_multiplier = np.mean([r['knowledge_multiplier'] for r in simulation_results])
        avg_network_effect = np.mean([r['network_effect'] for r in simulation_results])
        
        summary_text = f"""
╔══════════════════════════════╗
║   STRATEGY C SUMMARY        ║
║   (Zone III - Learning)     ║
╚══════════════════════════════╝

📊 Suppliers: {total_suppliers}

🎯 Performance:
  • Avg Reduction: {avg_reduction:,.0f} tons
  • Success Rate: {success_rate:.1f}%
  • Target: 10-15% reduction

💰 Investment:
  • Avg Investment: ${avg_investment:,.0f}
  • ROI: {np.mean(roi_values):.2f}x

🧠 Knowledge Transfer:
  • Avg Multiplier: {avg_knowledge_multiplier:.2f}x
  • Network Effect: {avg_network_effect:.2f}

✅ Zone III suppliers excel in
   knowledge sharing and 
   innovation adoption
        """
        
        ax8.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.suptitle('Strategy C: Zone III (Learning Zone) - ML Simulation Results', 
                    fontsize=18, weight='bold', y=0.98)
        
        return fig


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "="*80)
    print("STRATEGY C ML SIMULATION")
    print("ESG Supply Chain Emission Reduction - Zone III (Learning Zone)")
    print("="*80 + "\n")
    
    # Step 1: Generate synthetic supplier data
    print("📋 Step 1: Generating synthetic supplier data for Zone III...")
    data_gen = SupplierDataGeneratorC(n_suppliers=2000, seed=42)
    suppliers_df = data_gen.generate_suppliers()
    print(f"✓ Generated {len(suppliers_df)} Zone III suppliers")
    print(f"  - Average baseline emission: {suppliers_df['baseline_emission'].mean():.0f} tons CO₂")
    print(f"  - Average cooperation score: {suppliers_df['cooperation_score'].mean():.1f}/10")
    print(f"  - Average innovation capacity: {suppliers_df['innovation_capacity'].mean():.2f}")
    print(f"  - Average knowledge sharing: {suppliers_df['knowledge_sharing_score'].mean():.2f}")
    
    # Step 2: Load technology and knowledge sharing databases
    print("\n🔧 Step 2: Loading technology and knowledge sharing databases...")
    tech_db = TechnologyDatabaseC.get_technologies()
    knowledge_db = KnowledgeSharingDatabase.get_initiatives()
    print(f"✓ Loaded {len(tech_db)} technologies")
    print(f"✓ Loaded {len(knowledge_db)} knowledge sharing initiatives")
    
    # Step 3: Train ML models for knowledge transfer strategy
    print("\n🤖 Step 3: Training ML models for knowledge transfer optimization...")
    knowledge_ml = KnowledgeTransferML()
    knowledge_ml.train(suppliers_df)
    
    # Step 4: Select sample suppliers for simulation
    print("\n🎯 Step 4: Selecting sample suppliers for detailed simulation...")
    sample_suppliers = suppliers_df.sample(n=100, random_state=42)
    print(f"✓ Selected {len(sample_suppliers)} suppliers for detailed analysis")
    
    # Step 5: Run emission reduction simulations
    print("\n⚡ Step 5: Running emission reduction simulations...")
    simulator = EmissionReductionSimulatorC(tech_db, knowledge_db)
    simulation_results = []
    
    for idx, supplier in sample_suppliers.iterrows():
        # Predict optimal knowledge transfer strategy using ML
        supplier_features = {
            'baseline_emission': supplier['baseline_emission'],
            'cooperation_score': supplier['cooperation_score'],
            'innovation_capacity': supplier['innovation_capacity'],
            'knowledge_sharing_score': supplier['knowledge_sharing_score'],
            'tech_adoption_level': supplier['tech_adoption_level'],
            'best_practice_adoption': supplier['best_practice_adoption'],
            'certification_level': supplier['certification_level']
        }
        
        knowledge_strategy = knowledge_ml.predict_knowledge_strategy(supplier_features)
        
        # Simulate 3-year reduction pathway
        result = simulator.simulate_reduction_pathway(
            supplier, knowledge_strategy, budget=50000, mode='balanced'
        )
        result['knowledge_strategy'] = knowledge_strategy
        simulation_results.append(result)
        
        print(f"  ✓ {supplier['supplier_id']}: "
              f"{result['cumulative_reduction'][-1]:.0f} tons reduced "
              f"({result['total_reduction_rate']*100:.1f}% rate) | "
              f"Knowledge: {result['knowledge_multiplier']:.2f}x")
    
    # Step 6: Visualize results
    print("\n📊 Step 6: Generating visualization...")
    visualizer = StrategyVisualizerC()
    fig = visualizer.plot_results(simulation_results, sample_suppliers)
    
    output_path = 'Strategy_C_ML_Simulation.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Visualization saved: {output_path}")
    
    # Step 7: Generate summary report
    print("\n📈 Step 7: Generating summary report...")
    total_baseline = sum([s['baseline_emission'] for _, s in sample_suppliers.iterrows()])
    total_reduction = sum([r['cumulative_reduction'][-1] for r in simulation_results])
    total_investment = sum([r['total_investment'] for r in simulation_results])
    avg_efficiency = total_reduction / total_investment if total_investment > 0 else 0
    success_rate = sum([r['target_achieved'] for r in simulation_results]) / len(simulation_results) * 100
    avg_knowledge_multiplier = np.mean([r['knowledge_multiplier'] for r in simulation_results])
    avg_innovation = np.mean([r['innovation_potential'] for r in simulation_results])
    
    print("\n" + "="*80)
    print("SIMULATION SUMMARY REPORT - STRATEGY C")
    print("="*80)
    print(f"Total Suppliers Analyzed: {len(sample_suppliers)}")
    print(f"Total Baseline Emissions: {total_baseline:,.0f} tons CO₂e")
    print(f"Total Reduction Achieved: {total_reduction:,.0f} tons CO₂e ({total_reduction/total_baseline*100:.1f}%)")
    print(f"Total Investment Required: ${total_investment:,.0f} USD")
    print(f"Average Efficiency: {avg_efficiency:.4f} tons CO₂e per USD")
    print(f"Target Achievement Rate: {success_rate:.1f}%")
    print(f"Average Knowledge Multiplier: {avg_knowledge_multiplier:.2f}x")
    print(f"Average Innovation Potential: {avg_innovation:.2f}")
    print("="*80 + "\n")
    
    # Step 8: Export data to CSV and Excel files
    print("\n💾 Step 8: Exporting data to CSV/Excel files...")
    export_data_for_charts(sample_suppliers, simulation_results, suppliers_df, tech_db, knowledge_db)
    
    plt.show()
    
    return suppliers_df, simulation_results, sample_suppliers


def export_data_for_charts(sample_suppliers, simulation_results, all_suppliers, tech_db, knowledge_db):
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
                '碳排放(吨CO2)': round(emission, 2),
                '累计减排量(吨CO2)': round(cum_reduction, 2),
                '减排率(%)': round((cum_reduction / supplier['baseline_emission']) * 100, 2) if supplier['baseline_emission'] > 0 else 0,
                '知识倍增器': round(result['knowledge_multiplier'], 2),
                '网络效应': round(result['network_effect'], 2)
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
            '象限分类': 'III-学习区',
            '财务能力': round(supplier['financial_capacity'], 2),
            '技术采纳度': round(supplier['tech_adoption_level'], 2),
            '创新能力': round(supplier['innovation_capacity'], 2),
            '知识共享得分': round(supplier['knowledge_sharing_score'], 2),
            '最佳实践采纳': round(supplier['best_practice_adoption'], 2),
            '认证等级': supplier['certification_level'],
            '合作年限': supplier['years_partnership']
        })
    
    classification_df = pd.DataFrame(classification_data)
    
    # 3. Investment budget allocation
    budget_data = []
    for idx, (supplier_row, result) in enumerate(zip(sample_suppliers.iterrows(), simulation_results)):
        supplier = supplier_row[1]
        knowledge = result.get('knowledge_strategy', {})
        
        # Calculate metrics
        total_cost = result['total_investment']
        reduction_achieved = result['cumulative_reduction'][-1]
        roi = (reduction_achieved * 50) / total_cost if total_cost > 0 else 0  # Assume $50 per ton carbon price
        payback_years = total_cost / (reduction_achieved * 50 / 3) if reduction_achieved > 0 else 999
        
        budget_data.append({
            '供应商': supplier['supplier_id'],
            '技术投资': round(result['tech_investment'], 2),
            '知识投资': round(result['knowledge_investment'], 2),
            '总投资': round(total_cost, 2),
            '预期减排量': round(reduction_achieved, 2),
            '减排率': round((reduction_achieved / supplier['baseline_emission']) * 100, 2),
            '投资回报率': round(roi * 100, 2),
            '回本周期(年)': round(payback_years, 2),
            '知识倍增器': round(result['knowledge_multiplier'], 2),
            '网络效应': round(result['network_effect'], 2),
            '创新潜力': round(result['innovation_potential'], 2),
            '知识转移得分': round(result['knowledge_transfer_score'], 2)
        })
    
    budget_df = pd.DataFrame(budget_data)
    
    # 4. Technology database
    tech_db_export = tech_db.copy()
    tech_db_export.columns = ['技术ID', '技术名称', '类别', '减排率', '单位成本', 
                              '实施周期(月)', '年度运营节省', '知识转移价值']
    
    # 5. Knowledge sharing database
    knowledge_db_export = knowledge_db.copy()
    knowledge_db_export.columns = ['倡议ID', '倡议名称', '类别', '成本', 
                                   '持续时间(月)', '知识倍增器', '网络效应']
    
    # 6. Detailed supplier profiles
    detailed_suppliers = sample_suppliers.copy()
    detailed_suppliers['baseline_emission'] = detailed_suppliers['baseline_emission'].round(2)
    detailed_suppliers['annual_revenue'] = detailed_suppliers['annual_revenue'].round(2)
    detailed_suppliers['tech_adoption_level'] = detailed_suppliers['tech_adoption_level'].round(3)
    detailed_suppliers['financial_capacity'] = detailed_suppliers['financial_capacity'].round(3)
    detailed_suppliers['innovation_capacity'] = detailed_suppliers['innovation_capacity'].round(3)
    detailed_suppliers['knowledge_sharing_score'] = detailed_suppliers['knowledge_sharing_score'].round(3)
    detailed_suppliers['best_practice_adoption'] = detailed_suppliers['best_practice_adoption'].round(3)
    
    # 7. Strategy summary
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
            '技术数量': len(result['selected_technologies']),
            '知识倡议数': len(result['selected_initiatives']),
            '知识倍增器': round(result['knowledge_multiplier'], 2),
            '网络效应': round(result['network_effect'], 2),
            '创新潜力': round(result['innovation_potential'], 2),
            '总投资': round(result['total_investment'], 2)
        })
    
    strategy_summary_df = pd.DataFrame(strategy_summary)
    
    # Save to CSV files
    csv_dir = os.getcwd()
    
    pathway_df.to_csv(os.path.join(csv_dir, 'ML_simulation_C_三年减排路径.csv'), 
                     index=False, encoding='utf-8-sig')
    classification_df.to_csv(os.path.join(csv_dir, 'ML_simulation_C_四象限分类.csv'), 
                            index=False, encoding='utf-8-sig')
    budget_df.to_csv(os.path.join(csv_dir, 'ML_simulation_C_投资预算分配.csv'), 
                    index=False, encoding='utf-8-sig')
    tech_db_export.to_csv(os.path.join(csv_dir, 'ML_simulation_C_技术数据库.csv'), 
                         index=False, encoding='utf-8-sig')
    knowledge_db_export.to_csv(os.path.join(csv_dir, 'ML_simulation_C_知识共享数据库.csv'),
                               index=False, encoding='utf-8-sig')
    detailed_suppliers.to_csv(os.path.join(csv_dir, 'ML_simulation_C_供应商详细信息.csv'), 
                             index=False, encoding='utf-8-sig')
    strategy_summary_df.to_csv(os.path.join(csv_dir, 'ML_simulation_C_strategy_summary.csv'),
                              index=False, encoding='utf-8-sig')
    
    print(f"✓ CSV files saved:")
    print(f"  - ML_simulation_C_三年减排路径.csv ({len(pathway_df)} rows)")
    print(f"  - ML_simulation_C_四象限分类.csv ({len(classification_df)} rows)")
    print(f"  - ML_simulation_C_投资预算分配.csv ({len(budget_df)} rows)")
    print(f"  - ML_simulation_C_技术数据库.csv ({len(tech_db_export)} rows)")
    print(f"  - ML_simulation_C_知识共享数据库.csv ({len(knowledge_db_export)} rows)")
    print(f"  - ML_simulation_C_供应商详细信息.csv ({len(detailed_suppliers)} rows)")
    print(f"  - ML_simulation_C_strategy_summary.csv ({len(strategy_summary_df)} rows)")
    
    # Save to Excel file with multiple sheets
    try:
        excel_path = os.path.join(csv_dir, 'ML_simulation_C_ESG供应商数据.xlsx')
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            pathway_df.to_excel(writer, sheet_name='三年减排路径', index=False)
            classification_df.to_excel(writer, sheet_name='四象限分类', index=False)
            budget_df.to_excel(writer, sheet_name='投资预算分配', index=False)
            tech_db_export.to_excel(writer, sheet_name='技术数据库', index=False)
            knowledge_db_export.to_excel(writer, sheet_name='知识共享数据库', index=False)
            detailed_suppliers.to_excel(writer, sheet_name='供应商详细信息', index=False)
            strategy_summary_df.to_excel(writer, sheet_name='策略摘要', index=False)
        
        print(f"✓ Excel file saved: {excel_path}")
        
    except ImportError:
        print("⚠ openpyxl not installed. Excel export skipped.")
        print("  Install with: pip install openpyxl")
    
    return


if __name__ == "__main__":
    suppliers_df, simulation_results, sample_suppliers = main()
    
    print("✅ Strategy C ML Simulation Complete!")
    print("\n💡 Key Findings:")
    print("  1. Zone III suppliers excel at knowledge sharing and innovation")
    print("  2. Knowledge multiplier effects significantly enhance reduction effectiveness")
    print("  3. Lower absolute emissions but high cooperation enables best practice dissemination")
    print("  4. 10-15% reduction achievable with focus on innovation and knowledge transfer")
    print("\n🎯 Next Steps:")
    print("  - Document best practices for transfer to Zone II and IV")
    print("  - Establish peer-to-peer learning networks")
    print("  - Leverage as innovation testbed for new technologies")
    print("  - Create knowledge sharing partnerships with other zones")
    print("  - Integrate with A, B, D strategies for portfolio optimization")
