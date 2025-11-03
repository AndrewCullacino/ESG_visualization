# ESG Supply Chain Emission Reduction Strategies

A comprehensive machine learning-based simulation and analysis framework for optimizing carbon emission reduction strategies across supply chain partners.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Strategy Details](#strategy-details)
- [Usage](#usage)
- [Output Files](#output-files)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

---

## 🎯 Overview

This project implements four distinct ESG (Environmental, Social, Governance) strategies for supply chain emission reduction, using machine learning models to simulate, optimize, and analyze investment decisions across different supplier zones.

### Four Strategic Zones

| Zone | Strategy | Target Suppliers | Emission Target | Approach |
|------|----------|------------------|----------------|----------|
| **Zone I (A)** | Core Optimization (核心优化) | High emission + High cooperation | 40% reduction | Aggressive tech investment |
| **Zone II (B)** | Risk Management (风险管理) | High emission + Low cooperation | 20-25% reduction | Mandatory compliance |
| **Zone III (C)** | Learning Zone (学习区) | Low emission + High cooperation | 3-5% improvement | Knowledge sharing |
| **Zone IV (D)** | Observation Zone (观察区) | Low emission + Low cooperation | 5-10% reduction | Automated monitoring |

### Key Features

- 🤖 **Machine Learning Models**: Random Forest and Gradient Boosting regressors for emission prediction
- 📊 **Comprehensive Visualizations**: Multi-dimensional charts including radar charts, heatmaps, and investment analysis
- 💰 **ROI Analysis**: Detailed cost-effectiveness and payback period calculations
- 📈 **Multi-Year Projections**: 3-year emission reduction pathways
- 🎨 **Professional Charts**: High-resolution outputs with Chinese and English labels
- 📑 **Detailed Reports**: Automated CSV exports for supplier data, budget allocations, and performance metrics

---

## 📁 Project Structure

```
ESG/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .gitignore                        # Git ignore rules
│
├── Strategy Simulation Files
│   ├── A_strategy_ML_simulation.py   # Zone I: Core optimization strategy
│   ├── B_strategy_ML_simulation.py   # Zone II: Risk management strategy
│   ├── C_strategy_ML_simulation.py   # Zone III: Learning zone strategy
│   └── D_strategy_ML_simulation.py   # Zone IV: Observation zone strategy
│
├── Visualization Files
│   ├── A_visualiser.py               # Zone A specific visualizations
│   ├── B_visualiser.py               # Zone B specific visualizations
│   ├── C_visualiser.py               # Zone C specific visualizations
│   ├── C_visualiser_focused.py       # Zone C focused analysis
│   ├── C_visualiser_tech_knowledge_combined.py  # Combined tech/knowledge viz
│   ├── D_visualiser.py               # Zone D specific visualizations
│   ├── D_visualiser_focused.py       # Zone D focused analysis
│   └── 2dimension_matrix.py          # 2D matrix visualization
│
├── Comparative Analysis Files
│   ├── radar_chart_visualizer.py     # All-zone radar chart comparison
│   ├── cost_return_comparative_analysis.py     # Cross-strategy cost analysis
│   ├── cost_return_advanced_visualizer.py      # Advanced financial viz
│   ├── cost_return_professional_visualizer.py  # Professional ROI charts
│   ├── pestel_professional_visualizer.py       # PESTEL analysis
│   └── C_tech_cost_combined.py       # Technology cost analysis
│
├── Strategy Documents (PDF)
│   ├── A_strategy.pdf                # Strategy A documentation
│   ├── B_strategy.pdf                # Strategy B documentation
│   ├── C_strategy.pdf                # Strategy C documentation
│   └── D_strategy.pdf                # Strategy D documentation
│
├── Generated Data Files (CSV/XLSX)
│   ├── ML_simulation_A_*.csv         # Zone A simulation results
│   ├── ML_simulation_B_*.csv         # Zone B simulation results
│   ├── ML_simulation_C_*.csv         # Zone C simulation results
│   ├── ML_simulation_D_*.csv         # Zone D simulation results
│   └── ML_simulation_*_ESG供应商数据.xlsx  # Supplier data exports
│
├── Summary Reports (Markdown)
│   ├── COST_RETURN_ANALYSIS_SUMMARY.md  # Cross-strategy financial analysis
│   └── RADAR_CHARTS_SUMMARY.md          # Radar chart visualization guide
│
├── Generated Charts (PNG)
│   ├── all_zones_radar_chart_analysis.png    # 4-zone radar comparison
│   ├── cross_strategy_*.png                  # Cross-strategy analysis charts
│   ├── professional_*.png                    # Professional visualization outputs
│   ├── A减排量.png, A投资回报.png            # Zone A specific charts
│   ├── B区*.png                              # Zone B specific charts
│   ├── C区*.png                              # Zone C specific charts
│   ├── D区*.png                              # Zone D specific charts
│   └── 图一四象限.png                        # 4-quadrant classification
│
└── articles/                         # Reference PDFs (excluded from git)
    ├── BSR_Apparel_Supply_Chain_Carbon_Report.pdf
    ├── ESG_SupplyChain.pdf
    └── energies-12-02783.pdf
```

---

## 🔧 Prerequisites

### System Requirements

- **Operating System**: macOS, Linux, or Windows
- **Python**: Version 3.8 or higher
- **Memory**: Minimum 4GB RAM (8GB recommended for large simulations)
- **Storage**: ~500MB for dependencies + ~100MB for generated outputs

### Font Requirements (for Chinese Characters)

The visualizations include Chinese labels. Ensure one of these fonts is installed:
- **macOS**: SimHei, PingFang SC, or STHeiti (usually pre-installed)
- **Windows**: SimHei or Microsoft YaHei (usually pre-installed)
- **Linux**: Install with `sudo apt-get install fonts-wqy-zenhei` or similar

---

## 📥 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/AndrewCullacino/ESG_visualization.git
cd ESG_visualization
```

### Step 2: Create Virtual Environment

We strongly recommend using a virtual environment to avoid dependency conflicts.

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import pandas, numpy, matplotlib, sklearn, seaborn; print('✅ All packages installed successfully!')"
```

If you see the success message, you're ready to go! 🎉

---

## 🚀 Quick Start

### Run a Single Strategy Simulation

Each strategy can be run independently. Here's how to simulate Zone A (Core Optimization):

```bash
# Run Zone A simulation (generates CSV files and visualizations)
python A_strategy_ML_simulation.py

# Generate Zone A visualizations from the CSV data
python A_visualiser.py
```

### Run All Strategies

To run all four strategies sequentially:

```bash
# Strategy A (Zone I - Core Optimization)
python A_strategy_ML_simulation.py

# Strategy B (Zone II - Risk Management)
python B_strategy_ML_simulation.py

# Strategy C (Zone III - Learning Zone)
python C_strategy_ML_simulation.py

# Strategy D (Zone IV - Observation Zone)
python D_strategy_ML_simulation.py
```

### Generate Comparative Analysis

After running all strategies, generate cross-strategy comparisons:

```bash
# Comprehensive radar chart comparing all zones
python radar_chart_visualizer.py

# Cost and return analysis across strategies
python cost_return_comparative_analysis.py

# Advanced financial visualizations
python cost_return_professional_visualizer.py

# PESTEL strategic analysis
python pestel_professional_visualizer.py
```

---

## 📊 Strategy Details

### Strategy A: Core Optimization (核心优化区)

**Target**: High-emission, high-cooperation suppliers

**Key Features**:
- Aggressive 40% emission reduction target over 3 years
- Heavy investment in technology upgrades
- Partnership-based approach
- Expected ROI: ~200%+
- Payback period: 1-2 years

**Outputs**:
- `ML_simulation_A_strategy_summary.csv` - Overall strategy metrics
- `ML_simulation_A_供应商详细信息.csv` - Detailed supplier information
- `ML_simulation_A_技术数据库.csv` - Technology upgrade database
- `ML_simulation_A_投资预算分配.csv` - Investment budget allocation
- `ML_simulation_A_四象限分类.csv` - Supplier quadrant classification
- `ML_simulation_A_三年减排路径.csv` - 3-year emission reduction pathway
- `ML_simulation_A_ESG供应商数据.xlsx` - Comprehensive Excel export

### Strategy B: Risk Management (风险管理区)

**Target**: High-emission, low-cooperation suppliers

**Key Features**:
- Mandatory compliance approach
- 20-25% emission reduction target
- Strong oversight and monitoring
- Investment: ~$98,000 per supplier
- Cost: $21-25 per ton CO2e

**Outputs**:
- Similar CSV structure to Strategy A
- Additional compliance tracking metrics
- Risk assessment scores

### Strategy C: Learning Zone (学习区)

**Target**: Low-emission, high-cooperation suppliers

**Key Features**:
- Knowledge sharing and best practices
- 3-5% incremental improvement
- Innovation and capability building
- Investment: ~$26,000 per supplier
- Long-term payback (7+ years)

**Outputs**:
- Knowledge sharing database
- Innovation metrics
- Technology transfer tracking

### Strategy D: Observation Zone (观察区)

**Target**: Low-emission, low-cooperation suppliers

**Key Features**:
- Automated monitoring system
- 5-10% emission reduction
- Minimal manual oversight
- Investment: ~$17,500 per supplier
- Cost: ~$58 per ton CO2e

**Outputs**:
- Automated monitoring logs
- Basic management metrics
- Cost-effective technology implementations

---

## 💻 Usage

### Basic Workflow

1. **Run Simulation**: Execute strategy simulation file
2. **Review CSV Outputs**: Check generated data files
3. **Generate Visualizations**: Run visualizer scripts
4. **Analyze Results**: Review PNG charts and summary reports
5. **Compare Strategies**: Run comparative analysis scripts

### Example: Complete Analysis Pipeline

```bash
# Step 1: Run all strategy simulations
for strategy in A B C D; do
    python ${strategy}_strategy_ML_simulation.py
done

# Step 2: Generate individual visualizations
python A_visualiser.py
python B_visualiser.py
python C_visualiser.py
python D_visualiser.py

# Step 3: Create comprehensive comparisons
python radar_chart_visualizer.py
python cost_return_comparative_analysis.py
python pestel_professional_visualizer.py

# Step 4: Review outputs
echo "✅ Analysis complete! Check the PNG and CSV files."
```

### Customization Options

#### Modify Simulation Parameters

Edit the top of any `*_strategy_ML_simulation.py` file:

```python
# Example: Change number of suppliers in Zone A
class SupplierDataGenerator:
    def __init__(self, n_suppliers=500, seed=42):  # Change 500 to your desired number
        self.n_suppliers = n_suppliers
        self.rng = np.random.default_rng(seed)
```

#### Adjust Visualization Settings

Edit chart generation parameters:

```python
# Example: Change chart size and DPI
plt.figure(figsize=(24, 24), dpi=300)  # Adjust dimensions and resolution
```

#### Change Output Paths

Modify file paths in the scripts:

```python
# Example: Save to different directory
output_dir = 'custom_output/'
os.makedirs(output_dir, exist_ok=True)
df.to_csv(f'{output_dir}/custom_filename.csv', index=False)
```

---

## 📈 Output Files

### CSV Files Generated

Each strategy generates 7+ CSV files:

1. **`*_strategy_summary.csv`**: Overall performance metrics
2. **`*_供应商详细信息.csv`**: Detailed supplier profiles
3. **`*_技术数据库.csv`**: Technology upgrade options
4. **`*_投资预算分配.csv`**: Budget allocation by supplier
5. **`*_四象限分类.csv`**: Quadrant classification data
6. **`*_三年减排路径.csv`**: Year-by-year emission projections
7. **`*_ESG供应商数据.xlsx`**: Comprehensive Excel export

### Visualization Outputs

High-resolution PNG files (300 DPI, suitable for presentations):

- **Radar Charts**: Multi-dimensional strategy comparisons
- **Heatmaps**: Cost-effectiveness matrices
- **Bar Charts**: Investment and reduction comparisons
- **Scatter Plots**: ROI and efficiency analysis
- **Line Charts**: Multi-year emission pathways
- **Box Plots**: Distribution analysis

### Summary Reports

- **COST_RETURN_ANALYSIS_SUMMARY.md**: Financial analysis across all strategies
- **RADAR_CHARTS_SUMMARY.md**: Visualization methodology and insights

---

## 🐛 Troubleshooting

### Common Issues

#### 1. Chinese Characters Not Displaying

**Problem**: Boxes or question marks instead of Chinese text in charts.

**Solution**:
```python
# Add at the top of the script
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['Arial Unicode MS']  # For macOS
# or
rcParams['font.sans-serif'] = ['Microsoft YaHei']   # For Windows
# or
rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei'] # For Linux
```

#### 2. Module Not Found Error

**Problem**: `ModuleNotFoundError: No module named 'sklearn'`

**Solution**:
```bash
# Make sure virtual environment is activated
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows

# Reinstall requirements
pip install -r requirements.txt
```

#### 3. Memory Error During Simulation

**Problem**: `MemoryError` or system slowdown.

**Solution**:
```python
# Reduce number of suppliers in the simulation
# Edit the simulation file and change:
n_suppliers = 100  # Instead of 500
```

#### 4. File Not Found Error

**Problem**: `FileNotFoundError` when running visualizers.

**Solution**:
```bash
# Make sure to run the simulation first
python A_strategy_ML_simulation.py

# Then run the visualizer
python A_visualiser.py
```

#### 5. Git Push Failed

**Problem**: Push rejected due to large files.

**Solution**:
The repository is configured to exclude large files (`.gitignore`). Generated PNG and PDF files should not be pushed. If you need to share them:

```bash
# Use Git LFS for large files
git lfs install
git lfs track "*.png"
git lfs track "*.pdf"
git add .gitattributes
git commit -m "Add LFS tracking"
```

---

## 📖 Additional Resources

### Documentation

- **Strategy PDFs**: Detailed methodology in `A_strategy.pdf`, `B_strategy.pdf`, etc.
- **Summary Reports**: Analysis insights in Markdown files
- **Code Comments**: Extensive inline documentation in Python files

### Example Use Cases

1. **Executive Presentation**: Use `all_zones_radar_chart_analysis.png` for strategic overview
2. **Financial Review**: Reference `COST_RETURN_ANALYSIS_SUMMARY.md` for ROI details
3. **Supplier Selection**: Use quadrant classification CSVs to identify priority targets
4. **Budget Planning**: Reference investment allocation CSVs for resource planning

---

## 🤝 Contributing

### Reporting Issues

If you encounter any problems:

1. Check the [Troubleshooting](#troubleshooting) section
2. Review the code comments in the relevant Python file
3. Create an issue with:
   - Python version: `python --version`
   - Operating system
   - Error message and stack trace
   - Steps to reproduce

### Development Guidelines

When contributing code:

1. Follow PEP 8 style guidelines
2. Add docstrings to new functions
3. Include Chinese translations for chart labels
4. Test with multiple scenarios
5. Update README if adding new features

---

## 📄 License

This project is intended for educational and research purposes. Please review the license terms before commercial use.

---

## 👥 Authors

**Andrew Cullacino** - Project Lead

---

## 🙏 Acknowledgments

- Machine learning models based on scikit-learn
- Visualization framework using matplotlib and seaborn
- Data processing with pandas and numpy
- Reference papers in `articles/` directory

---

## 📮 Contact

For questions or collaboration opportunities:

- **GitHub**: https://github.com/AndrewCullacino/ESG_visualization
- **Issues**: https://github.com/AndrewCullacino/ESG_visualization/issues

---

## 🔄 Version History

### v1.0.0 (Current)
- ✅ Initial release with all 4 strategies
- ✅ Complete ML simulation framework
- ✅ Comprehensive visualization suite
- ✅ Cross-strategy analysis tools
- ✅ Professional reporting outputs

---

**Last Updated**: November 3, 2025  
**Project Status**: ✅ Active Development  
**Python Version**: 3.8+  
**Tested On**: macOS, Windows 10/11

---

Made with ❤️ for sustainable supply chain management
