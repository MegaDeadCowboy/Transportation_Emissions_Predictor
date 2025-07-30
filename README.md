# Transportation Emissions Predictor

**A machine learning tool for predicting state-level transportation greenhouse gas emissions using economic, demographic, and policy indicators.**

## Overview

This project creates an interactive dashboard and predictive model for state-level transportation emissions, enabling policymakers to compare state efficiency, predict emissions, simulate policy scenarios, and identify best practices. The analysis demonstrates that Vehicle Miles Traveled (VMT) explains 75.9% of state emission variation, providing a clear policy lever for emission reduction.

**Key Finding**: A 10% reduction in VMT could decrease transportation emissions by 7.3% nationally.

## Features

### Machine Learning Models
- **Single Predictor Model**: R² = 0.759 using Vehicle Miles Traveled
- **Multi-Linear Model**: R² = 0.774 combining VMT, population, and GDP
- **Random Forest Model**: R² = 0.830 with feature importance analysis
- **Policy Simulator**: Quantifies emission reduction through VMT policies

### Interactive Dashboard
- **National Overview**: Choropleth map showing emissions by state with key metrics
- **State Deep Dive**: Individual state analysis with performance comparisons
- **Policy Simulator**: Interactive VMT reduction scenarios with impact modeling
- **State Comparison**: Side-by-side benchmarking and efficiency rankings

### Data Integration
- **EPA/EIA**: Transportation energy consumption and CO2 emissions
- **Census Bureau**: State population and demographic data
- **BEA**: State-level GDP and economic indicators
- **FHWA**: Vehicle miles traveled statistics
- **Feature Engineering**: 57 features engineered, 5 core features selected for modeling to avoid overfitting
## Quick Start

### Setup
```bash
git clone https://github.com/MegaDeadCowboy/Transportation_Emissions_Predictor.git
cd transportation-emissions-predictor
pip install -r requirements.txt
```

### Run Dashboard
```bash
streamlit run src/app.py
```
Opens at: http://localhost:8501

### Data Pipeline
```bash
# 1. Collect data
python src/data_collector.py

# 2. Engineer features  
python src/feature_engineer.py

# 3. Train models
python src/model_dev.py

# 4. Launch dashboard
streamlit run src/app.py
```

## Results

### Statistical Findings
- **VMT Correlation**: r = 0.871 with transportation emissions
- **State Efficiency Range**: 2x variation between most and least efficient states
- **Economic Relationship**: GDP correlation r = 0.802, but efficiency varies by state
- **Policy Impact**: Linear relationship between VMT reduction and emission cuts

### State Performance
- **Most Efficient**: California (3.70 tons CO2 per capita)
- **Highest Total Emissions**: Texas (dark red on choropleth map)
- **Regional Patterns**: West Coast states generally more efficient
- **Policy Opportunities**: 4 distinct state clusters identified through analysis

## Technical Implementation

### Project Structure
```
transportation-emissions-predictor/
├── src/
│   ├── app.py                    # Streamlit dashboard
│   ├── data_collector.py         # Data pipeline
│   ├── feature_engineer.py       # Feature engineering
│   └── model_dev.py              # Model development
├── data/
│   ├── processed/
│   │   ├── master_dataset.csv    # Core dataset (10 states, 11 variables)
│   │   └── enhanced_dataset.csv  # Extended dataset (57 features)
├── models/
│   └── simple_*.pkl              # Trained models
└── requirements.txt
```

### Dependencies
- Python 3.8+
- Streamlit for dashboard
- Pandas/NumPy for data processing
- Plotly for visualizations
- Scikit-learn for machine learning
- Joblib for model persistence

## Methodology

### Data Sources
The analysis integrates multiple federal datasets:
- Transportation emissions derived from EIA energy consumption data
- State population from Census Bureau APIs
- Economic indicators from Bureau of Economic Analysis
- Vehicle miles traveled from Federal Highway Administration

### Statistical Approach
- Small sample handling (n=10 states) with Leave-One-Out cross-validation
- Feature selection based on statistical significance
- Model interpretability prioritized over complexity
- Policy scenario modeling using validated correlations

### Model Validation
- R² scores ranging from 0.759 (single predictor) to 0.830 (ensemble)
- Cross-validation confirms model robustness
- Feature importance analysis identifies VMT as primary driver
- Residual analysis shows good model fit

## Policy Applications

### Scenario Analysis
The interactive simulator demonstrates emission reduction potential:
- 10% VMT reduction: 7.3% emission decrease
- Combined with efficiency improvements: up to 15% reduction possible
- State-specific targeting based on current performance gaps

### Best Practices
- California model: High efficiency despite large economy
- Urban planning strategies that reduce travel demand
- Public transportation and alternative mobility investments
- Regional coordination for transportation policy

## Contact

[Carver Rasmussen] - [carver.rasmussen@gmail.com]

Project Repository: [https://github.com/MegaDeadCowboy/Transportation_Emissions_Predictor]

Live Dashboard: [http://localhost:8501]

## License

MIT License - see LICENSE file for details.