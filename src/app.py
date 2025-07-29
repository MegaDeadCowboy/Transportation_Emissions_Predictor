# Transportation Emissions Predictor Dashboard
# Hours 7-8: Interactive Dashboard Creation
# EcoDataLab Application Project - Final Phase

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Transportation Emissions Predictor",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: rgba(0, 0, 0, 0.7);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #2E8B57;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        color: #ffffff;
    }
    .metric-container h3 {
        color: #4CAF50;
        font-size: 1rem;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    .metric-container h2 {
        color: #ffffff;
        font-size: 2rem;
        margin: 0.5rem 0;
        font-weight: 700;
    }
    .metric-container p {
        color: #cccccc;
        font-size: 0.9rem;
        margin: 0;
    }
    .insight-box {
        background-color: rgba(0, 0, 0, 0.7);
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        margin: 1rem 0;
        color: #ffffff;
    }
    .insight-box h4 {
        color: #64B5F6;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    .insight-box p {
        color: #ffffff;
        font-size: 1rem;
        margin: 0;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load the enhanced dataset and models."""
    try:
        df = pd.read_csv('data/processed/enhanced_dataset.csv')
        return df
    except FileNotFoundError:
        # Create sample data if files not found
        st.warning("⚠️ Enhanced dataset not found. Using sample data for demonstration.")
        return create_sample_data()

@st.cache_data
def load_models():
    """Load the trained models."""
    try:
        results = joblib.load('models/simple_results.pkl')
        features = joblib.load('models/simple_features.pkl')
        return results, features
    except FileNotFoundError:
        st.warning("⚠️ Models not found. Using demo results.")
        return create_demo_results()

def create_sample_data():
    """Create sample data for demonstration."""
    states = ['CA', 'TX', 'NY', 'FL', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI']
    data = []
    
    # Realistic data based on actual patterns
    state_data = {
        'CA': {'emissions': 150.2, 'population': 39538223, 'gdp': 3598103, 'vmt': 340000},
        'TX': {'emissions': 200.1, 'population': 30029572, 'gdp': 2417542, 'vmt': 280000},
        'NY': {'emissions': 120.3, 'population': 20201249, 'gdp': 2006800, 'vmt': 120000},
        'FL': {'emissions': 140.5, 'population': 22610726, 'gdp': 1102484, 'vmt': 220000},
        'IL': {'emissions': 90.2, 'population': 12812508, 'gdp': 864777, 'vmt': 110000},
        'PA': {'emissions': 95.1, 'population': 13002700, 'gdp': 895845, 'vmt': 100000},
        'OH': {'emissions': 85.3, 'population': 11799448, 'gdp': 787376, 'vmt': 110000},
        'GA': {'emissions': 75.2, 'population': 10912876, 'gdp': 635922, 'vmt': 120000},
        'NC': {'emissions': 70.8, 'population': 10698973, 'gdp': 656924, 'vmt': 110000},
        'MI': {'emissions': 80.1, 'population': 10037261, 'gdp': 559654, 'vmt': 100000}
    }
    
    for state_code, values in state_data.items():
        data.append({
            'year': 2022,
            'state_code': state_code,
            'transport_co2_mmt': values['emissions'],
            'population': values['population'],
            'real_gdp_millions': values['gdp'],
            'vmt_millions': values['vmt'],
            'emissions_per_capita': values['emissions'] * 1000000 / values['population'],
            'vmt_per_capita': values['vmt'] * 1000000 / values['population'],
            'gdp_per_capita': values['gdp'] * 1000000 / values['population'],
            'data_source': 'demo'
        })
    
    return pd.DataFrame(data)

def create_demo_results():
    """Create demo model results."""
    results = {
        'single_predictor': {
            'feature': 'vmt_millions',
            'r2': 0.759,
            'rmse': 25.4,
            'correlation': 0.871
        },
        'multi_linear': {
            'features': ['vmt_millions', 'population', 'real_gdp_millions'],
            'r2': 0.774,
            'rmse': 24.1,
            'coefficients': {
                'vmt_millions': 0.425,
                'population': 0.000003,
                'real_gdp_millions': 0.012
            }
        }
    }
    features = ['vmt_millions', 'population', 'real_gdp_millions', 'emissions_per_capita', 'vmt_per_capita']
    return results, features

def create_national_overview(df):
    """Create the national overview dashboard."""
    st.markdown("<h1 class='main-header'>🌱 Transportation Emissions Predictor</h1>", unsafe_allow_html=True)
    st.markdown("**A machine learning tool for predicting state-level transportation greenhouse gas emissions**")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_emissions = df['transport_co2_mmt'].sum()
    avg_per_capita = df['emissions_per_capita'].mean()
    total_vmt = df['vmt_millions'].sum()
    efficiency_leader = df.loc[df['emissions_per_capita'].idxmin(), 'state_code']
    
    with col1:
        st.markdown("""
        <div class='metric-container'>
            <h3>🏭 Total Emissions</h3>
            <h2>{:.1f} MMT CO₂</h2>
            <p>Combined transportation emissions</p>
        </div>
        """.format(total_emissions), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='metric-container'>
            <h3>👥 Average Per Capita</h3>
            <h2>{:.1f} tons/person</h2>
            <p>Transportation emissions per person</p>
        </div>
        """.format(avg_per_capita), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='metric-container'>
            <h3>🚗 Total VMT</h3>
            <h2>{:.0f}B miles</h2>
            <p>Vehicle miles traveled</p>
        </div>
        """.format(total_vmt/1000), unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class='metric-container'>
            <h3>🏆 Efficiency Leader</h3>
            <h2>{}</h2>
            <p>Lowest emissions per capita</p>
        </div>
        """.format(efficiency_leader), unsafe_allow_html=True)
    
    # Choropleth map
    st.subheader("🗺️ Transportation Emissions by State")
    
    # Create state mapping for visualization
    state_mapping = {
        'CA': 'California', 'TX': 'Texas', 'NY': 'New York', 'FL': 'Florida',
        'IL': 'Illinois', 'PA': 'Pennsylvania', 'OH': 'Ohio', 'GA': 'Georgia',
        'NC': 'North Carolina', 'MI': 'Michigan'
    }
    
    df_map = df.copy()
    df_map['state_name'] = df_map['state_code'].map(state_mapping)
    
    fig_map = px.choropleth(
        df_map,
        locations='state_code',
        color='transport_co2_mmt',
        locationmode='USA-states',
        color_continuous_scale='Reds',
        title='Transportation CO₂ Emissions by State (MMT)',
        labels={'transport_co2_mmt': 'Emissions (MMT CO₂)'}
    )
    
    fig_map.update_layout(
        geo_scope='usa',
        height=500,
        title_x=0.5
    )
    
    st.plotly_chart(fig_map, use_container_width=True)
    
    # Top insights
    st.subheader("💡 Key Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top emitters
        top_emitters = df.nlargest(5, 'transport_co2_mmt')
        
        fig_bar1 = px.bar(
            top_emitters,
            x='state_code',
            y='transport_co2_mmt',
            title='Top 5 Total Emitters',
            labels={'transport_co2_mmt': 'Emissions (MMT CO₂)', 'state_code': 'State'},
            color='transport_co2_mmt',
            color_continuous_scale='Reds'
        )
        fig_bar1.update_layout(showlegend=False)
        st.plotly_chart(fig_bar1, use_container_width=True)
    
    with col2:
        # Most efficient
        most_efficient = df.nsmallest(5, 'emissions_per_capita')
        
        fig_bar2 = px.bar(
            most_efficient,
            x='state_code',
            y='emissions_per_capita',
            title='Top 5 Most Efficient (Per Capita)',
            labels={'emissions_per_capita': 'Emissions (tons/person)', 'state_code': 'State'},
            color='emissions_per_capita',
            color_continuous_scale='Greens_r'
        )
        fig_bar2.update_layout(showlegend=False)
        st.plotly_chart(fig_bar2, use_container_width=True)

def create_state_deep_dive(df):
    """Create state-specific analysis."""
    st.header("🔍 State Deep Dive Analysis")
    
    # State selector
    selected_state = st.selectbox(
        "Select a state for detailed analysis:",
        options=sorted(df['state_code'].unique()),
        index=0
    )
    
    state_data = df[df['state_code'] == selected_state].iloc[0]
    
    # State metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Total Emissions",
            f"{state_data['transport_co2_mmt']:.1f} MMT CO₂",
            delta=None
        )
    
    with col2:
        st.metric(
            "Emissions per Capita",
            f"{state_data['emissions_per_capita']:.2f} tons/person",
            delta=None
        )
    
    with col3:
        national_avg = df['emissions_per_capita'].mean()
        delta_pct = ((state_data['emissions_per_capita'] - national_avg) / national_avg) * 100
        st.metric(
            "vs National Average",
            f"{delta_pct:+.1f}%",
            delta=f"{delta_pct:+.1f}% vs avg"
        )
    
    # Multi-variable analysis
    st.subheader(f"📊 {selected_state} Performance Analysis")
    
    # Create comparison with national averages
    metrics = ['transport_co2_mmt', 'emissions_per_capita', 'vmt_per_capita', 'gdp_per_capita']
    metric_labels = ['Total Emissions (MMT)', 'Emissions/Capita (tons)', 'VMT/Capita (miles)', 'GDP/Capita ($)']
    
    fig_radar = go.Figure()
    
    # Normalize values for radar chart
    state_values = []
    national_values = []
    
    for metric in metrics:
        state_val = state_data[metric]
        national_avg = df[metric].mean()
        national_max = df[metric].max()
        
        # Normalize to 0-1 scale
        state_normalized = state_val / national_max
        national_normalized = national_avg / national_max
        
        state_values.append(state_normalized)
        national_values.append(national_normalized)
    
    fig_radar.add_trace(go.Scatterpolar(
        r=state_values,
        theta=metric_labels,
        fill='toself',
        name=selected_state,
        line_color='red'
    ))
    
    fig_radar.add_trace(go.Scatterpolar(
        r=national_values,
        theta=metric_labels,
        fill='toself',
        name='National Average',
        line_color='blue',
        opacity=0.6
    ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title=f"{selected_state} vs National Average Performance"
    )
    
    st.plotly_chart(fig_radar, use_container_width=True)
    
    # Key relationships
    st.subheader("🔗 Key Relationships")
    
    # Scatter plot: VMT vs Emissions
    fig_scatter = px.scatter(
        df,
        x='vmt_millions',
        y='transport_co2_mmt',
        size='population',
        color='emissions_per_capita',
        hover_name='state_code',
        title='VMT vs Emissions (bubble size = population)',
        labels={
            'vmt_millions': 'Vehicle Miles Traveled (millions)',
            'transport_co2_mmt': 'Transportation Emissions (MMT CO₂)',
            'emissions_per_capita': 'Emissions per Capita'
        },
        color_continuous_scale='Viridis'
    )
    
    # Highlight selected state
    selected_data = df[df['state_code'] == selected_state]
    fig_scatter.add_trace(
        go.Scatter(
            x=selected_data['vmt_millions'],
            y=selected_data['transport_co2_mmt'],
            mode='markers',
            marker=dict(size=15, color='red', symbol='star'),
            name=f'{selected_state} (Selected)',
            showlegend=True
        )
    )
    
    st.plotly_chart(fig_scatter, use_container_width=True)

def create_policy_simulator(df, results):
    """Create policy scenario simulator."""
    st.header("🎯 Policy Scenario Simulator")
    
    st.markdown("""
    <div class='insight-box'>
        <h4>💡 Model Insights</h4>
        <p>Our analysis shows that <strong>Vehicle Miles Traveled (VMT)</strong> explains 75.9% of emission variation across states. 
        Use the simulator below to explore policy scenarios.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display model performance
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Model Accuracy (R²)",
            f"{results['single_predictor']['r2']:.3f}",
            delta="Strong predictive power"
        )
    
    with col2:
        st.metric(
            "Key Predictor",
            "Vehicle Miles Traveled",
            delta="Primary policy lever"
        )
    
    with col3:
        st.metric(
            "VMT Correlation",
            f"{results['single_predictor']['correlation']:.3f}",
            delta="Very strong relationship"
        )
    
    # Policy scenario sliders
    st.subheader("🎛️ Policy Scenario Controls")
    
    col1, col2 = st.columns(2)
    
    with col1:
        vmt_change = st.slider(
            "VMT Change (%)",
            min_value=-30,
            max_value=30,
            value=0,
            step=1,
            help="Simulate the impact of changes in Vehicle Miles Traveled"
        )
        
        population_change = st.slider(
            "Population Change (%)",
            min_value=-10,
            max_value=20,
            value=0,
            step=1,
            help="Account for population growth/decline"
        )
    
    with col2:
        gdp_change = st.slider(
            "GDP Change (%)",
            min_value=-20,
            max_value=30,
            value=0,
            step=1,
            help="Consider economic growth impacts"
        )
        
        efficiency_improvement = st.slider(
            "Efficiency Improvement (%)",
            min_value=0,
            max_value=50,
            value=0,
            step=1,
            help="Technology and policy efficiency gains"
        )
    
    # Calculate scenario impacts
    baseline_emissions = df['transport_co2_mmt'].sum()
    
    # Simple impact calculation based on correlations
    vmt_impact = (vmt_change / 100) * 0.759  # Using R² from VMT model
    pop_impact = (population_change / 100) * 0.3  # Moderate correlation
    gdp_impact = (gdp_change / 100) * 0.2  # Economic activity factor
    efficiency_impact = -(efficiency_improvement / 100) * 0.5  # Direct reduction
    
    total_impact = vmt_impact + pop_impact + gdp_impact + efficiency_impact
    projected_emissions = baseline_emissions * (1 + total_impact)
    emission_change = ((projected_emissions - baseline_emissions) / baseline_emissions) * 100
    
    # Results display
    st.subheader("📊 Scenario Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Baseline Emissions",
            f"{baseline_emissions:.1f} MMT CO₂",
            delta=None
        )
    
    with col2:
        delta_color = "normal" if emission_change == 0 else ("inverse" if emission_change < 0 else "normal")
        st.metric(
            "Projected Emissions",
            f"{projected_emissions:.1f} MMT CO₂",
            delta=f"{emission_change:+.1f}%"
        )
    
    with col3:
        st.metric(
            "Net Impact",
            f"{emission_change:+.1f}%",
            delta="Target: -50% by 2030"
        )
    
    # Scenario breakdown
    st.subheader("🔍 Impact Breakdown")
    
    impact_data = pd.DataFrame({
        'Factor': ['VMT Change', 'Population', 'Economic Growth', 'Efficiency Gains'],
        'Impact (%)': [vmt_impact * 100, pop_impact * 100, gdp_impact * 100, efficiency_impact * 100],
        'Input Change (%)': [vmt_change, population_change, gdp_change, -efficiency_improvement]
    })
    
    fig_waterfall = go.Figure()
    
    # Create waterfall chart
    x_vals = impact_data['Factor'].tolist() + ['Net Effect']
    y_vals = impact_data['Impact (%)'].tolist() + [sum(impact_data['Impact (%)'])]
    colors = ['red' if y > 0 else 'green' for y in y_vals[:-1]] + ['blue']
    
    fig_waterfall.add_trace(
        go.Bar(
            x=x_vals,
            y=y_vals,
            marker_color=colors,
            text=[f"{y:+.1f}%" for y in y_vals],
            textposition='outside'
        )
    )
    
    fig_waterfall.update_layout(
        title="Policy Impact Breakdown",
        xaxis_title="Policy Factor",
        yaxis_title="Emission Change (%)",
        showlegend=False
    )
    
    st.plotly_chart(fig_waterfall, use_container_width=True)
    
    # Policy recommendations
    st.subheader("💼 Policy Recommendations")
    
    if vmt_change < -10:
        st.success("🎯 **Strong VMT Reduction**: This scenario represents ambitious transportation demand management. Consider policies like congestion pricing, improved public transit, and remote work incentives.")
    elif vmt_change < 0:
        st.info("📈 **Moderate VMT Reduction**: Good progress on transportation efficiency. Focus on urban planning and alternative transportation modes.")
    elif vmt_change > 10:
        st.warning("⚠️ **VMT Increase**: Rising travel demand. Consider this scenario for economic growth periods and plan accordingly.")
    
    if efficiency_improvement > 20:
        st.success("🔧 **High Efficiency Gains**: Ambitious technology adoption. This could include rapid EV deployment, fuel efficiency standards, and smart transportation systems.")

def create_state_comparison(df):
    """Create state comparison interface."""
    st.header("⚖️ State Comparison Tool")
    
    # Multi-select for states
    selected_states = st.multiselect(
        "Select states to compare (2-5 recommended):",
        options=sorted(df['state_code'].unique()),
        default=['CA', 'TX', 'NY', 'FL']
    )
    
    if len(selected_states) < 2:
        st.warning("Please select at least 2 states for comparison.")
        return
    
    comparison_df = df[df['state_code'].isin(selected_states)].copy()
    
    # Comparison metrics
    st.subheader("📊 Side-by-Side Comparison")
    
    metrics_to_compare = {
        'transport_co2_mmt': 'Total Emissions (MMT)',
        'emissions_per_capita': 'Emissions per Capita (tons)',
        'vmt_millions': 'Vehicle Miles Traveled (millions)',
        'vmt_per_capita': 'VMT per Capita (miles)',
        'gdp_per_capita': 'GDP per Capita ($)',
        'population': 'Population'
    }
    
    # Create comparison table
    comparison_table = comparison_df[['state_code'] + list(metrics_to_compare.keys())].copy()
    comparison_table = comparison_table.set_index('state_code')
    
    # Format numbers
    comparison_table['transport_co2_mmt'] = comparison_table['transport_co2_mmt'].round(1)
    comparison_table['emissions_per_capita'] = comparison_table['emissions_per_capita'].round(2)
    comparison_table['vmt_per_capita'] = comparison_table['vmt_per_capita'].round(0)
    comparison_table['gdp_per_capita'] = comparison_table['gdp_per_capita'].round(0)
    comparison_table['population'] = comparison_table['population'].astype(int)
    
    # Rename columns
    comparison_table.columns = [metrics_to_compare[col] for col in comparison_table.columns]
    
    st.dataframe(comparison_table, use_container_width=True)
    
    # Visual comparisons
    col1, col2 = st.columns(2)
    
    with col1:
        # Emissions comparison
        fig_comp1 = px.bar(
            comparison_df,
            x='state_code',
            y='transport_co2_mmt',
            title='Total Transportation Emissions',
            labels={'transport_co2_mmt': 'Emissions (MMT CO₂)', 'state_code': 'State'},
            color='transport_co2_mmt',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig_comp1, use_container_width=True)
    
    with col2:
        # Efficiency comparison
        fig_comp2 = px.bar(
            comparison_df,
            x='state_code',
            y='emissions_per_capita',
            title='Emissions per Capita (Efficiency)',
            labels={'emissions_per_capita': 'Emissions per Capita (tons)', 'state_code': 'State'},
            color='emissions_per_capita',
            color_continuous_scale='RdYlGn_r'
        )
        st.plotly_chart(fig_comp2, use_container_width=True)
    
    # Performance ranking
    st.subheader("🏆 Performance Rankings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Efficiency Ranking** (Lower is Better)")
        efficiency_rank = comparison_df.sort_values('emissions_per_capita')[['state_code', 'emissions_per_capita']]
        for i, (_, row) in enumerate(efficiency_rank.iterrows(), 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            st.write(f"{medal} {row['state_code']}: {row['emissions_per_capita']:.2f} tons/person")
    
    with col2:
        st.write("**Total Impact Ranking** (Absolute Emissions)")
        total_rank = comparison_df.sort_values('transport_co2_mmt', ascending=False)[['state_code', 'transport_co2_mmt']]
        for i, (_, row) in enumerate(total_rank.iterrows(), 1):
            icon = "🔴" if i == 1 else "🟡" if i == 2 else "🟢"
            st.write(f"{icon} {row['state_code']}: {row['transport_co2_mmt']:.1f} MMT CO₂")

def main():
    """Main dashboard application."""
    
    # Load data
    df = load_data()
    results, features = load_models()
    
    # Sidebar navigation
    st.sidebar.title("🌱 Navigation")
    st.sidebar.markdown("---")
    
    page = st.sidebar.selectbox(
        "Choose Analysis:",
        ["🏠 National Overview", "🔍 State Deep Dive", "🎯 Policy Simulator", "⚖️ State Comparison"]
    )
    
    # Model info sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 Model Performance")
    st.sidebar.metric("VMT Model R²", f"{results['single_predictor']['r2']:.3f}")
    st.sidebar.metric("Key Predictor", "Vehicle Miles Traveled")
    st.sidebar.metric("VMT Correlation", f"{results['single_predictor']['correlation']:.3f}")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### 📊 Data Sources
    - **EPA/EIA**: Transportation emissions
    - **Census**: Population data
    - **BEA**: Economic indicators  
    - **FHWA**: Vehicle miles traveled
    
    ### 🎯 Built for EcoDataLab
    Demonstrating climate policy analysis capabilities
    """)
    
    # Main content based on selection
    if page == "🏠 National Overview":
        create_national_overview(df)
    elif page == "🔍 State Deep Dive":
        create_state_deep_dive(df)
    elif page == "🎯 Policy Simulator":
        create_policy_simulator(df, results)
    elif page == "⚖️ State Comparison":
        create_state_comparison(df)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🌱 Transportation Emissions Predictor | Built with Streamlit | EcoDataLab Application Project</p>
        <p>Predictive Model: R² = 0.83 | Key Finding: VMT explains 75.9% of emission variation</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()