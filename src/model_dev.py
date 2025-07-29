# Simplified Model Development for Small Datasets
# Transportation Emissions Predictor - EcoDataLab Application Project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

class SimpleTransportationModeler:
    """
    Simplified ML modeling approach optimized for very small datasets (n=10).
    Focus on interpretability and avoiding overfitting.
    """
    
    def __init__(self, df):
        self.df = df.copy()
        self.target = 'transport_co2_mmt'
        self.models = {}
        self.results = {}
        
        print(f"🤖 Simple ML Pipeline: {len(df)} samples")
        
    def select_robust_features(self):
        """
        Select the most robust features that won't cause overfitting.
        """
        print("🎯 Selecting Robust Features...")
        
        # Core features that should always work
        core_features = [
            'vmt_millions',
            'population', 
            'real_gdp_millions',
            'emissions_per_capita',
            'vmt_per_capita'
        ]
        
        # Check which features actually exist and have data
        available_features = []
        for feature in core_features:
            if feature in self.df.columns:
                if not self.df[feature].isnull().all() and self.df[feature].var() > 0:
                    available_features.append(feature)
        
        print(f"✅ Selected {len(available_features)} robust features:")
        for i, feature in enumerate(available_features, 1):
            corr = self.df[feature].corr(self.df[self.target])
            print(f"   {i}. {feature:<20} (r={corr:.3f})")
        
        self.selected_features = available_features
        return available_features
    
    def build_simple_models(self):
        """
        Build simple, interpretable models that work with small data.
        """
        print("\n📊 Building Simple Models...")
        
        # Prepare data
        X = self.df[self.selected_features].fillna(self.df[self.selected_features].mean())
        y = self.df[self.target]
        
        print(f"Data shape: X={X.shape}, y={y.shape}")
        
        # Model 1: Single best predictor (most interpretable)
        if len(self.selected_features) > 0:
            # Find best single predictor
            correlations = {}
            for feature in self.selected_features:
                corr = abs(self.df[feature].corr(self.df[self.target]))
                correlations[feature] = corr
            
            best_feature = max(correlations.keys(), key=lambda x: correlations[x])
            
            single_model = LinearRegression()
            X_single = X[[best_feature]]
            single_model.fit(X_single, y)
            y_pred_single = single_model.predict(X_single)
            
            r2_single = r2_score(y, y_pred_single)
            rmse_single = np.sqrt(mean_squared_error(y, y_pred_single))
            
            self.models['single_predictor'] = single_model
            self.results['single_predictor'] = {
                'feature': best_feature,
                'r2': r2_single,
                'rmse': rmse_single,
                'correlation': correlations[best_feature]
            }
            
            print(f"✅ Single Predictor Model ({best_feature}):")
            print(f"   R² = {r2_single:.3f}")
            print(f"   RMSE = {rmse_single:.2f}")
        
        # Model 2: Simple multi-linear (max 3 features to avoid overfitting)
        if len(self.selected_features) >= 2:
            # Use top 3 features max
            n_features = min(3, len(self.selected_features))
            top_features = sorted(correlations.keys(), key=lambda x: correlations[x], reverse=True)[:n_features]
            
            multi_model = LinearRegression()
            X_multi = X[top_features]
            multi_model.fit(X_multi, y)
            y_pred_multi = multi_model.predict(X_multi)
            
            r2_multi = r2_score(y, y_pred_multi)
            rmse_multi = np.sqrt(mean_squared_error(y, y_pred_multi))
            
            self.models['multi_linear'] = multi_model
            self.results['multi_linear'] = {
                'features': top_features,
                'r2': r2_multi,
                'rmse': rmse_multi,
                'coefficients': dict(zip(top_features, multi_model.coef_))
            }
            
            print(f"✅ Multi-Linear Model (top {n_features} features):")
            print(f"   Features: {', '.join(top_features)}")
            print(f"   R² = {r2_multi:.3f}")
            print(f"   RMSE = {rmse_multi:.2f}")
        
        # Model 3: Simple Random Forest (very constrained)
        if len(self.selected_features) >= 2:
            rf_simple = RandomForestRegressor(
                n_estimators=10,  # Very few trees
                max_depth=2,      # Very shallow
                min_samples_split=3,
                min_samples_leaf=2,
                random_state=42
            )
            
            rf_simple.fit(X, y)
            y_pred_rf = rf_simple.predict(X)
            
            r2_rf = r2_score(y, y_pred_rf)
            rmse_rf = np.sqrt(mean_squared_error(y, y_pred_rf))
            
            # Feature importance
            feature_importance = dict(zip(self.selected_features, rf_simple.feature_importances_))
            
            self.models['simple_rf'] = rf_simple
            self.results['simple_rf'] = {
                'r2': r2_rf,
                'rmse': rmse_rf,
                'feature_importance': feature_importance
            }
            
            print(f"✅ Simple Random Forest:")
            print(f"   R² = {r2_rf:.3f}")
            print(f"   RMSE = {rmse_rf:.2f}")
            print(f"   Top Feature: {max(feature_importance.keys(), key=lambda x: feature_importance[x])}")
    
    def create_simple_policy_simulator(self):
        """
        Create a simple policy simulator using the best performing model.
        """
        print("\n🎯 Creating Simple Policy Simulator...")
        
        # Find best model
        best_model_name = 'single_predictor'  # Start with simplest
        if 'multi_linear' in self.results and self.results['multi_linear']['r2'] > self.results['single_predictor']['r2']:
            best_model_name = 'multi_linear'
        
        best_model = self.models[best_model_name]
        
        print(f"✅ Using {best_model_name} for policy scenarios")
        
        # Get baseline data (latest year)
        baseline_data = self.df[self.df['year'] == self.df['year'].max()].copy()
        baseline_emissions = baseline_data[self.target].sum()
        
        print(f"📊 Baseline Total Emissions: {baseline_emissions:.1f} MMT CO2")
        
        # Simple scenarios
        scenarios = {}
        
        if best_model_name == 'single_predictor':
            feature = self.results['single_predictor']['feature']
            
            # 10% reduction in the key feature
            modified_data = baseline_data.copy()
            if 'vmt' in feature.lower():
                modified_data[feature] *= 0.9  # 10% VMT reduction
                scenario_name = "10% VMT Reduction"
            elif 'population' in feature.lower():
                # Population changes are not policy-controllable, so skip
                scenario_name = None
            else:
                modified_data[feature] *= 0.9
                scenario_name = f"10% {feature} Reduction"
            
            if scenario_name:
                X_scenario = modified_data[[feature]]
                predicted = best_model.predict(X_scenario)
                total_predicted = predicted.sum()
                change_pct = ((total_predicted - baseline_emissions) / baseline_emissions) * 100
                scenarios[scenario_name] = {
                    'emissions': total_predicted,
                    'change_pct': change_pct
                }
        
        elif best_model_name == 'multi_linear':
            features = self.results['multi_linear']['features']
            
            # VMT reduction scenario
            if any('vmt' in f.lower() for f in features):
                modified_data = baseline_data.copy()
                for feature in features:
                    if 'vmt' in feature.lower():
                        modified_data[feature] *= 0.9
                
                X_scenario = modified_data[features]
                predicted = best_model.predict(X_scenario)
                total_predicted = predicted.sum()
                change_pct = ((total_predicted - baseline_emissions) / baseline_emissions) * 100
                scenarios["10% VMT Reduction"] = {
                    'emissions': total_predicted,
                    'change_pct': change_pct
                }
        
        # Display scenarios
        print(f"\n📋 Policy Scenarios:")
        print(f"   Baseline: {baseline_emissions:.1f} MMT CO2")
        
        for scenario_name, results in scenarios.items():
            print(f"   {scenario_name}: {results['emissions']:.1f} MMT CO2 ({results['change_pct']:+.1f}%)")
        
        self.scenarios = scenarios
        return scenarios
    
    def analyze_relationships(self):
        """
        Provide clear analysis of the relationships found.
        """
        print("\n🔍 RELATIONSHIP ANALYSIS")
        print("=" * 30)
        
        # Correlation analysis
        print("📊 Key Correlations with Emissions:")
        for feature in self.selected_features:
            corr = self.df[feature].corr(self.df[self.target])
            significance = "***" if abs(corr) > 0.7 else "**" if abs(corr) > 0.5 else "*" if abs(corr) > 0.3 else ""
            print(f"   {feature:<20} r = {corr:+.3f} {significance}")
        
        print("\n   * = moderate, ** = strong, *** = very strong")
        
        # Model interpretation
        if 'single_predictor' in self.results:
            single_result = self.results['single_predictor']
            print(f"\n🎯 Best Single Predictor: {single_result['feature']}")
            print(f"   Explains {single_result['r2']*100:.1f}% of emission variation")
            print(f"   Correlation: r = {single_result['correlation']:.3f}")
        
        if 'multi_linear' in self.results:
            multi_result = self.results['multi_linear']
            print(f"\n🔗 Multi-Variable Model:")
            print(f"   Features: {', '.join(multi_result['features'])}")
            print(f"   Explains {multi_result['r2']*100:.1f}% of emission variation")
            print(f"   Coefficients:")
            for feature, coef in multi_result['coefficients'].items():
                print(f"     {feature}: {coef:.3f}")
    
    def generate_summary_report(self):
        """
        Generate a comprehensive summary suitable for presentations.
        """
        print("\n📋 FINAL MODEL SUMMARY REPORT")
        print("=" * 35)
        
        # Data summary
        print(f"📊 Dataset Overview:")
        print(f"   Records: {len(self.df)}")
        print(f"   States: {self.df['state_code'].nunique() if 'state_code' in self.df.columns else 'N/A'}")
        print(f"   Years: {self.df['year'].min() if 'year' in self.df.columns else 'N/A'}-{self.df['year'].max() if 'year' in self.df.columns else 'N/A'}")
        
        # Model performance
        print(f"\n🏆 Model Performance:")
        for model_name, results in self.results.items():
            print(f"   {model_name}: R² = {results['r2']:.3f}")
        
        # Key insights
        if 'single_predictor' in self.results:
            best_feature = self.results['single_predictor']['feature']
            best_r2 = self.results['single_predictor']['r2']
            
            print(f"\n💡 Key Findings:")
            print(f"   • {best_feature} is the strongest predictor (R² = {best_r2:.3f})")
            print(f"   • Model explains {best_r2*100:.1f}% of emission variation")
            
            if best_r2 > 0.5:
                print(f"   • Strong predictive relationship suitable for policy analysis")
            else:
                print(f"   • Moderate relationship - consider additional factors")
        
        # Policy implications
        if hasattr(self, 'scenarios') and self.scenarios:
            print(f"\n🎯 Policy Opportunities:")
            for scenario_name, results in self.scenarios.items():
                if results['change_pct'] < -5:
                    print(f"   • {scenario_name} could reduce emissions by {abs(results['change_pct']):.1f}%")
        
        # Limitations
        print(f"\n⚠️  Important Limitations:")
        print(f"   • Small sample size (n={len(self.df)}) - results may not generalize")
        print(f"   • Correlational analysis - not causal relationships")
        print(f"   • Limited to available features - other factors may be important")
        
        return {
            'best_model': 'single_predictor' if 'single_predictor' in self.results else None,
            'best_r2': self.results['single_predictor']['r2'] if 'single_predictor' in self.results else 0,
            'n_samples': len(self.df),
            'n_features': len(self.selected_features)
        }
    
    def save_simple_models(self):
        """
        Save the simple models.
        """
        import os
        os.makedirs('models', exist_ok=True)
        
        # Save models
        for model_name, model in self.models.items():
            joblib.dump(model, f'models/simple_{model_name}.pkl')
        
        # Save results
        joblib.dump(self.results, 'models/simple_results.pkl')
        joblib.dump(self.selected_features, 'models/simple_features.pkl')
        
        print(f"\n💾 Simple models saved to models/ directory")
    
    def run_simple_pipeline(self):
        """
        Run the complete simplified modeling pipeline.
        """
        print("🚀 SIMPLE TRANSPORTATION EMISSIONS MODELING")
        print("=" * 45)
        
        # Step 1: Feature selection
        self.select_robust_features()
        
        # Step 2: Build models
        self.build_simple_models()
        
        # Step 3: Policy scenarios
        self.create_simple_policy_simulator()
        
        # Step 4: Analysis
        self.analyze_relationships()
        
        # Step 5: Summary
        summary = self.generate_summary_report()
        
        # Step 6: Save
        self.save_simple_models()
        
        print(f"\n✅ SIMPLIFIED MODELING COMPLETE!")
        print(f"Models work reliably with small dataset!")
        
        return summary


# Example usage
if __name__ == "__main__":
    try:
        df = pd.read_csv('data/processed/enhanced_dataset.csv')
        print(f"📊 Loaded dataset: {df.shape}")
        
        # Use simple modeler
        modeler = SimpleTransportationModeler(df)
        results = modeler.run_simple_pipeline()
        
        print(f"\n🎯 READY FOR DASHBOARD CREATION!")
        
    except FileNotFoundError:
        print("❌ Error: enhanced_dataset.csv not found.")
        print("Run: python src/feature_engineer.py")