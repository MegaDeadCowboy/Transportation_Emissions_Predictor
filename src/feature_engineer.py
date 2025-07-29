# Feature Engineering & Advanced EDA - Hours 3-4
# Transportation Emissions Predictor - EcoDataLab Application Project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

class TransportationFeatureEngineer:
    """
    Advanced feature engineering for transportation emissions prediction.
    Creates policy-relevant features and performs statistical analysis.
    """
    
    def __init__(self, df):
        self.df = df.copy()
        self.original_features = df.columns.tolist()
        self.engineered_features = []
        self.feature_descriptions = {}
        
    def create_efficiency_features(self):
        """
        Create transportation efficiency metrics.
        """
        print("🔧 Creating Efficiency Features...")
        
        # 1. Emissions Intensity (tons CO2 per million GDP)
        self.df['emissions_intensity'] = (self.df['transport_co2_mmt'] / 
                                         self.df['real_gdp_millions']) * 1000
        
        # 2. Transport Efficiency (VMT per ton CO2)
        self.df['transport_efficiency'] = (self.df['vmt_millions'] / 
                                          self.df['transport_co2_mmt'])
        
        # 3. Economic Transport Intensity (VMT per GDP)
        self.df['economic_transport_intensity'] = (self.df['vmt_millions'] / 
                                                  self.df['real_gdp_millions'])
        
        # 4. Population Efficiency (emissions per 1000 people)
        self.df['emissions_per_1000_pop'] = (self.df['transport_co2_mmt'] * 1000000 / 
                                            self.df['population'])
        
        new_features = ['emissions_intensity', 'transport_efficiency', 
                       'economic_transport_intensity', 'emissions_per_1000_pop']
        
        self.engineered_features.extend(new_features)
        
        # Feature descriptions
        self.feature_descriptions.update({
            'emissions_intensity': 'Transportation CO2 tons per $M GDP',
            'transport_efficiency': 'Vehicle miles traveled per ton CO2',
            'economic_transport_intensity': 'VMT per million GDP',
            'emissions_per_1000_pop': 'CO2 tons per 1000 people'
        })
        
        print(f"✅ Created {len(new_features)} efficiency features")
        return new_features
    
    def create_temporal_features(self):
        """
        Create time-based features and trends.
        """
        print("📅 Creating Temporal Features...")
        
        # Ensure data is sorted
        self.df = self.df.sort_values(['state_code', 'year'])
        
        new_features = []
        
        # 1. Year-over-year change rates
        for col in ['transport_co2_mmt', 'vmt_millions', 'real_gdp_millions', 'population']:
            yoy_col = f'{col}_yoy_change'
            self.df[yoy_col] = self.df.groupby('state_code')[col].pct_change()
            new_features.append(yoy_col)
            
            self.feature_descriptions[yoy_col] = f'Year-over-year % change in {col}'
        
        # 2. 3-year rolling averages
        for col in ['transport_co2_mmt', 'emissions_per_capita']:
            rolling_col = f'{col}_3yr_avg'
            self.df[rolling_col] = (self.df.groupby('state_code')[col]
                                   .rolling(window=3, min_periods=1)
                                   .mean().reset_index(0, drop=True))
            new_features.append(rolling_col)
            
            self.feature_descriptions[rolling_col] = f'3-year rolling average of {col}'
        
        # 3. Linear trend (slope) for each state
        def calculate_trend(group):
            if len(group) < 3:
                return 0
            x = np.arange(len(group))
            slope, _, _, _, _ = stats.linregress(x, group)
            return slope
        
        trend_col = 'emissions_trend_slope'
        self.df[trend_col] = (self.df.groupby('state_code')['transport_co2_mmt']
                             .transform(lambda x: calculate_trend(x)))
        new_features.append(trend_col)
        
        self.feature_descriptions[trend_col] = 'Linear trend slope of emissions over time'
        
        # 4. Detrended emissions (actual - trend)
        def detrend_emissions(group):
            if len(group) < 3:
                return group
            x = np.arange(len(group))
            slope, intercept, _, _, _ = stats.linregress(x, group)
            trend = slope * x + intercept
            return group - trend
        
        detrend_col = 'emissions_detrended'
        self.df[detrend_col] = (self.df.groupby('state_code')['transport_co2_mmt']
                               .transform(lambda x: detrend_emissions(x)))
        new_features.append(detrend_col)
        
        self.feature_descriptions[detrend_col] = 'Emissions with linear trend removed'
        
        self.engineered_features.extend(new_features)
        print(f"✅ Created {len(new_features)} temporal features")
        return new_features
    
    def create_regional_features(self):
        """
        Create regional and geographic features.
        """
        print("🗺️ Creating Regional Features...")
        
        # US Census Regions
        census_regions = {
            'Northeast': ['CT', 'ME', 'MA', 'NH', 'RI', 'VT', 'NJ', 'NY', 'PA'],
            'Midwest': ['IL', 'IN', 'MI', 'OH', 'WI', 'IA', 'KS', 'MN', 'MO', 'NE', 'ND', 'SD'],
            'South': ['DE', 'FL', 'GA', 'MD', 'NC', 'SC', 'VA', 'WV', 'AL', 'KY', 'MS', 'TN', 'AR', 'LA', 'OK', 'TX'],
            'West': ['AZ', 'CO', 'ID', 'MT', 'NV', 'NM', 'UT', 'WY', 'AK', 'CA', 'HI', 'OR', 'WA']
        }
        
        # Map states to regions
        state_to_region = {}
        for region, states in census_regions.items():
            for state in states:
                state_to_region[state] = region
        
        self.df['census_region'] = self.df['state_code'].map(state_to_region)
        
        # Create regional dummy variables
        region_dummies = pd.get_dummies(self.df['census_region'], prefix='region')
        self.df = pd.concat([self.df, region_dummies], axis=1)
        
        new_features = ['census_region'] + region_dummies.columns.tolist()
        
        # Economic size categories
        gdp_quartiles = self.df['real_gdp_millions'].quantile([0.25, 0.5, 0.75])
        
        def categorize_economy(gdp):
            if gdp <= gdp_quartiles[0.25]:
                return 'Small'
            elif gdp <= gdp_quartiles[0.5]:
                return 'Medium'
            elif gdp <= gdp_quartiles[0.75]:
                return 'Large'
            else:
                return 'Very Large'
        
        self.df['economy_size'] = self.df['real_gdp_millions'].apply(categorize_economy)
        
        # Create economy size dummies
        economy_dummies = pd.get_dummies(self.df['economy_size'], prefix='economy')
        self.df = pd.concat([self.df, economy_dummies], axis=1)
        
        new_features.extend(['economy_size'] + economy_dummies.columns.tolist())
        
        self.engineered_features.extend(new_features)
        
        # Update descriptions
        for region in census_regions.keys():
            self.feature_descriptions[f'region_{region}'] = f'Binary: State in {region} region'
        
        for size in ['Small', 'Medium', 'Large', 'Very Large']:
            self.feature_descriptions[f'economy_{size}'] = f'Binary: {size} economy by GDP'
        
        print(f"✅ Created {len(new_features)} regional features")
        return new_features
    
    def create_comparative_features(self):
        """
        Create comparative features relative to national averages.
        """
        print("📊 Creating Comparative Features...")
        
        new_features = []
        
        # Calculate national averages by year
        national_stats = self.df.groupby('year').agg({
            'transport_co2_mmt': 'mean',
            'emissions_per_capita': 'mean',
            'vmt_per_capita': 'mean',
            'real_gdp_millions': 'mean'
        }).add_suffix('_national_avg')
        
        # Merge with original data
        self.df = self.df.merge(national_stats, left_on='year', right_index=True)
        
        # Create relative features
        comparison_features = {
            'emissions_vs_national': ('transport_co2_mmt', 'transport_co2_mmt_national_avg'),
            'per_capita_vs_national': ('emissions_per_capita', 'emissions_per_capita_national_avg'),
            'vmt_vs_national': ('vmt_per_capita', 'vmt_per_capita_national_avg'),
            'gdp_vs_national': ('real_gdp_millions', 'real_gdp_millions_national_avg')
        }
        
        for feature_name, (actual_col, avg_col) in comparison_features.items():
            # Ratio to national average
            ratio_col = f'{feature_name}_ratio'
            self.df[ratio_col] = self.df[actual_col] / self.df[avg_col]
            new_features.append(ratio_col)
            
            # Deviation from national average
            dev_col = f'{feature_name}_deviation'
            self.df[dev_col] = self.df[actual_col] - self.df[avg_col]
            new_features.append(dev_col)
            
            self.feature_descriptions[ratio_col] = f'Ratio of state to national average {actual_col}'
            self.feature_descriptions[dev_col] = f'Deviation from national average {actual_col}'
        
        self.engineered_features.extend(new_features)
        print(f"✅ Created {len(new_features)} comparative features")
        return new_features
    
    def create_interaction_features(self):
        """
        Create interaction features between key variables.
        """
        print("🔗 Creating Interaction Features...")
        
        new_features = []
        
        # Key interactions for transportation emissions
        interactions = [
            ('population', 'vmt_per_capita', 'pop_vmt_interaction'),
            ('real_gdp_millions', 'transport_efficiency', 'gdp_efficiency_interaction'),
            ('emissions_per_capita', 'vmt_per_capita', 'per_capita_interaction'),
            ('population', 'real_gdp_millions', 'pop_gdp_interaction')
        ]
        
        for var1, var2, interaction_name in interactions:
            if var1 in self.df.columns and var2 in self.df.columns:
                self.df[interaction_name] = self.df[var1] * self.df[var2]
                new_features.append(interaction_name)
                
                self.feature_descriptions[interaction_name] = f'Interaction between {var1} and {var2}'
        
        # Ratio features
        ratios = [
            ('vmt_millions', 'population', 'vmt_per_person_exact'),
            ('real_gdp_millions', 'transport_co2_mmt', 'gdp_per_emission_unit'),
            ('transport_co2_mmt', 'vmt_millions', 'emissions_per_vmt_unit')
        ]
        
        for numerator, denominator, ratio_name in ratios:
            if numerator in self.df.columns and denominator in self.df.columns:
                self.df[ratio_name] = self.df[numerator] / (self.df[denominator] + 1e-8)  # Avoid division by zero
                new_features.append(ratio_name)
                
                self.feature_descriptions[ratio_name] = f'Ratio of {numerator} to {denominator}'
        
        self.engineered_features.extend(new_features)
        print(f"✅ Created {len(new_features)} interaction features")
        return new_features
    
    def perform_clustering_analysis(self):
        """
        Perform clustering analysis to identify state groups with similar patterns.
        """
        print("🎯 Performing Clustering Analysis...")
        
        # Select features for clustering
        clustering_features = [
            'emissions_per_capita', 'vmt_per_capita', 'gdp_per_capita',
            'transport_efficiency', 'emissions_intensity'
        ]
        
        # Get the most recent year data for clustering
        recent_data = self.df[self.df['year'] == self.df['year'].max()].copy()
        
        # Prepare clustering data
        cluster_data = recent_data[clustering_features].fillna(recent_data[clustering_features].mean())
        
        # Standardize features
        scaler = StandardScaler()
        cluster_data_scaled = scaler.fit_transform(cluster_data)
        
        # Perform K-means clustering
        n_clusters = 4  # Northeast, South, Midwest, West-like grouping
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(cluster_data_scaled)
        
        # Add cluster labels to recent data
        recent_data['emission_cluster'] = cluster_labels
        
        # Map clusters back to all years for each state
        state_clusters = recent_data[['state_code', 'emission_cluster']].set_index('state_code')['emission_cluster']
        self.df['emission_cluster'] = self.df['state_code'].map(state_clusters)
        
        # Create cluster dummy variables
        cluster_dummies = pd.get_dummies(self.df['emission_cluster'], prefix='cluster')
        self.df = pd.concat([self.df, cluster_dummies], axis=1)
        
        new_features = ['emission_cluster'] + cluster_dummies.columns.tolist()
        self.engineered_features.extend(new_features)
        
        # Analyze clusters
        print("\n🔍 Cluster Analysis Results:")
        cluster_analysis = recent_data.groupby('emission_cluster')[clustering_features].mean()
        
        for cluster_id in range(n_clusters):
            states_in_cluster = recent_data[recent_data['emission_cluster'] == cluster_id]['state_code'].tolist()
            print(f"\nCluster {cluster_id}: {', '.join(states_in_cluster)}")
            print(f"  Avg Emissions/Capita: {cluster_analysis.loc[cluster_id, 'emissions_per_capita']:.2f}")
            print(f"  Avg Transport Efficiency: {cluster_analysis.loc[cluster_id, 'transport_efficiency']:.1f}")
            print(f"  Avg GDP/Capita: ${cluster_analysis.loc[cluster_id, 'gdp_per_capita']:,.0f}")
        
        # Update descriptions
        for i in range(n_clusters):
            self.feature_descriptions[f'cluster_{i}'] = f'Binary: State in emissions cluster {i}'
        
        print(f"✅ Created {len(new_features)} clustering features")
        return new_features, cluster_analysis
    
    def perform_statistical_tests(self):
        """
        Perform statistical tests for feature relationships.
        """
        print("📊 Performing Statistical Tests...")
        
        # Key variables for testing
        key_vars = ['transport_co2_mmt', 'emissions_per_capita', 'vmt_per_capita', 
                   'real_gdp_millions', 'population', 'transport_efficiency']
        
        # Correlation matrix with p-values
        correlation_results = {}
        
        for var1 in key_vars:
            for var2 in key_vars:
                if var1 != var2 and var1 in self.df.columns and var2 in self.df.columns:
                    # Remove NaN values
                    data1 = self.df[var1].dropna()
                    data2 = self.df[var2].dropna()
                    
                    # Align the data
                    common_idx = data1.index.intersection(data2.index)
                    if len(common_idx) > 3:
                        corr, p_value = pearsonr(data1[common_idx], data2[common_idx])
                        correlation_results[f"{var1}_vs_{var2}"] = {
                            'correlation': corr,
                            'p_value': p_value,
                            'significant': p_value < 0.05
                        }
        
        # Print significant correlations
        print("\n🔗 Significant Correlations (p < 0.05):")
        significant_corrs = {k: v for k, v in correlation_results.items() 
                           if v['significant'] and abs(v['correlation']) > 0.3}
        
        for relationship, stats in sorted(significant_corrs.items(), 
                                        key=lambda x: abs(x[1]['correlation']), reverse=True):
            print(f"  {relationship}: r={stats['correlation']:.3f}, p={stats['p_value']:.3f}")
        
        return correlation_results
    
    def create_feature_summary(self):
        """
        Create a comprehensive summary of all features.
        """
        print("\n📋 FEATURE ENGINEERING SUMMARY")
        print("=" * 40)
        
        # Count features by category
        efficiency_features = [f for f in self.engineered_features if 'efficiency' in f or 'intensity' in f]
        temporal_features = [f for f in self.engineered_features if 'yoy' in f or 'trend' in f or 'avg' in f]
        regional_features = [f for f in self.engineered_features if 'region' in f or 'economy' in f]
        comparative_features = [f for f in self.engineered_features if 'national' in f or 'ratio' in f or 'deviation' in f]
        interaction_features = [f for f in self.engineered_features if 'interaction' in f or 'per_' in f]
        clustering_features = [f for f in self.engineered_features if 'cluster' in f]
        
        print(f"Original Features: {len(self.original_features)}")
        print(f"Efficiency Features: {len(efficiency_features)}")
        print(f"Temporal Features: {len(temporal_features)}")
        print(f"Regional Features: {len(regional_features)}")
        print(f"Comparative Features: {len(comparative_features)}")
        print(f"Interaction Features: {len(interaction_features)}")
        print(f"Clustering Features: {len(clustering_features)}")
        print(f"Total Features: {len(self.df.columns)}")
        print(f"New Features Created: {len(self.engineered_features)}")
        
        # Feature importance preview (using correlation with target)
        if 'transport_co2_mmt' in self.df.columns:
            target_corrs = {}
            for feature in self.df.select_dtypes(include=[np.number]).columns:
                if feature != 'transport_co2_mmt':
                    corr = self.df[feature].corr(self.df['transport_co2_mmt'])
                    if not np.isnan(corr):
                        target_corrs[feature] = abs(corr)
            
            print(f"\n🎯 Top 10 Features Correlated with Emissions:")
            top_features = sorted(target_corrs.items(), key=lambda x: x[1], reverse=True)[:10]
            for feature, corr in top_features:
                print(f"  {feature}: {corr:.3f}")
        
        return {
            'total_features': len(self.df.columns),
            'new_features': len(self.engineered_features),
            'feature_categories': {
                'efficiency': efficiency_features,
                'temporal': temporal_features,
                'regional': regional_features,
                'comparative': comparative_features,
                'interaction': interaction_features,
                'clustering': clustering_features
            }
        }
    
    def run_full_feature_engineering(self):
        """
        Run the complete feature engineering pipeline.
        """
        print("🚀 TRANSPORTATION EMISSIONS FEATURE ENGINEERING")
        print("=" * 50)
        
        # Step 1: Create efficiency features
        self.create_efficiency_features()
        
        # Step 2: Create temporal features
        self.create_temporal_features()
        
        # Step 3: Create regional features
        self.create_regional_features()
        
        # Step 4: Create comparative features
        self.create_comparative_features()
        
        # Step 5: Create interaction features
        self.create_interaction_features()
        
        # Step 6: Perform clustering analysis
        cluster_features, cluster_analysis = self.perform_clustering_analysis()
        
        # Step 7: Statistical testing
        correlation_results = self.perform_statistical_tests()
        
        # Step 8: Create summary
        summary = self.create_feature_summary()
        
        # Save enhanced dataset
        output_path = 'data/processed/enhanced_dataset.csv'
        self.df.to_csv(output_path, index=False)
        print(f"\n💾 Enhanced dataset saved to: {output_path}")
        
        print(f"\n✅ FEATURE ENGINEERING COMPLETE!")
        print(f"Ready for Hours 5-6: Model Development")
        
        return self.df, summary, correlation_results, cluster_analysis


# Example usage
if __name__ == "__main__":
    # Load the dataset from Hours 1-2
    try:
        df = pd.read_csv('data/processed/master_dataset.csv')
        print(f"📊 Loaded dataset: {df.shape}")
        
        # Initialize feature engineer
        engineer = TransportationFeatureEngineer(df)
        
        # Run full feature engineering
        enhanced_df, summary, correlations, clusters = engineer.run_full_feature_engineering()
        
        print(f"\n🎯 HOURS 3-4 COMPLETE!")
        print(f"Dataset enhanced from {len(df.columns)} to {len(enhanced_df.columns)} features")
        print(f"Ready for machine learning model development!")
        
    except FileNotFoundError:
        print("❌ Error: master_dataset.csv not found.")
        print("Please run the data collection script first (Hours 1-2)")
        print("Run: python src/data_collector.py")