import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap
from matplotlib.ticker import FuncFormatter

# Prevent graph cropping
def save_show_close(filename):
    plt.tight_layout()
    plt.savefig(filename, dpi=300) # Increased DPI for professional sharpness
    plt.close()
    print(f"  Saved → {filename}")

# Add labels on the bars
def add_bar_labels(ax, orientation='h', decimals=2):
    for container in ax.containers:
        ax.bar_label(container, fmt=f'%.{decimals}f', padding=5, fontweight='bold', fontsize=9)

# 1. DATA LOADING
def load_audit_data():
    try:
        res_df = pd.read_csv('test_predictions_residuals.csv')
        shap_val_df = pd.read_csv('xgb3_shap_values.csv')
        shap_imp_df = pd.read_csv('xgb3_shap_importance.csv')
        call_shap = pd.read_csv('xgb3_shap_call.csv')
        put_shap = pd.read_csv('xgb3_shap_put.csv')
        feature_names = shap_val_df.columns.tolist()
        return res_df, shap_val_df, shap_imp_df, feature_names, call_shap, put_shap
    except Exception as e:
        print(f"Error loading files: {e}")
        return None, None, None, None, None, None

# 2. PLOT GENERATION
def generate_all_plots(res_df, shap_val_df, shap_imp_df, feature_names, call_shap, put_shap):
    print("Starting Plot Generation...")
    
    # --- A. Global Importance ---
    plt.figure(figsize=(15, 10))
    top_20 = shap_imp_df.head(20)
    ax_a = sns.barplot(x='mean_abs_shap', y='feature', data=top_20, hue='feature', palette='magma', legend=False)
    add_bar_labels(ax_a, orientation='h')
    plt.title('Top 20 Variables Influencing Option Price', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Mean |SHAP Value| ($)', fontsize=12)
    save_show_close('shap_global_importance.png')

    # --- B. Comparison: Call vs Put ---
    # Merge and filter for top 20 global features
    top_20_list = shap_imp_df.head(20)['feature'].tolist()
    comp_df = pd.merge(
        call_shap.rename(columns={'mean_abs_shap': 'Call'}),
        put_shap.rename(columns={'mean_abs_shap': 'Put'}),
        on='feature'
    )
    comp_df = comp_df[comp_df['feature'].isin(top_20_list)].sort_values('Call', ascending=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), sharey=True)
    
    # Left Panel: Call (Midnight Blue)
    sns.barplot(x='Call', y='feature', data=comp_df, color='#2980B9', ax=ax1, edgecolor='black')
    add_bar_labels(ax1)
    ax1.invert_xaxis() # Makes them face inward for comparison
    
    # Right Panel: Put (Crimson Red)
    sns.barplot(x='Put', y='feature', data=comp_df, color='#d64545', ax=ax2, edgecolor='black')
    add_bar_labels(ax2)
    ax2.set_ylabel('') # Hide y-label as it's shared

    plt.suptitle('Variable Importance Comparison Between Instruments', fontsize=18, fontweight='bold', y=0.98)
    save_show_close('shap_comparison_call_put.png')

    # --- C. SHAP Dependence Plot  ---
    try:
        top_feat = 'Strike' if 'Strike' in res_df.columns else res_df.columns
        x_data, y_data = res_df[top_feat].values.ravel(), shap_val_df[top_feat].values.ravel()
        min_len = min(len(x_data), len(y_data))
        
        plt.figure(figsize=(10, 6))
        # scatter color is dark slate, line is a sharp crimson
        sns.regplot(x=x_data[:min_len], y=y_data[:min_len], 
                    scatter_kws={'alpha':0.25, 'color':'#2980B9', 's':25}, 
                    line_kws={'color':'#d64545', 'linewidth': 3}, 
                    lowess=True)

        plt.title(f'Individual Influence: Impact of {top_feat} on Price', fontsize=14, fontweight='bold', pad=15)
        plt.xlabel(f'Actual Value of {top_feat}', fontsize=12)
        plt.ylabel('SHAP Value (Price Impact in $)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.3)
        sns.despine()
        save_show_close('shap_dependence.png')
    except Exception as e:
        print(f"  Dependence Plot failed: {e}")

    # --- D. Waterfall Plot  ---
    try:
        row_idx = 0 
        single_shap = shap_val_df.iloc[row_idx]
        single_feat_vals = res_df.loc[row_idx, feature_names]
        base_value = shap_val_df.values.mean() 
        
        waterfall_df = pd.DataFrame({
            'feature': feature_names,
            'val': [single_shap[f] for f in feature_names],
            'actual_val': [single_feat_vals[f] for f in feature_names]
        }).sort_values(by='val', key=abs, ascending=False).head(10)

        waterfall_df['step_end'] = waterfall_df['val'].cumsum() + base_value
        waterfall_df['step_start'] = waterfall_df['step_end'] - waterfall_df['val']
        waterfall_df['label'] = waterfall_df.apply(lambda x: f"{x['feature']} ({x['actual_val']:.2f})", axis=1)

        plt.figure(figsize=(12, 8))
        # Deep blue for positive, Deep red for negative
        colors = ['#d64545' if x < 0 else '#2980B9' for x in waterfall_df['val']]
        
        plt.barh(waterfall_df['label'], waterfall_df['val'], left=waterfall_df['step_start'], color=colors, edgecolor='black')
        
        for i, row in enumerate(waterfall_df.itertuples()):
            plt.text(row.step_end, i, f" {'+' if row.val > 0 else ''}{row.val:.2f}", 
                     va='center', fontweight='bold', fontsize=10)

        plt.axvline(base_value, color='black', linestyle='--', label=f'Dataset Baseline (${base_value:.2f})')
        plt.title(f'Price Breakdown for PANW', fontsize=14, fontweight='bold', pad=15)
        plt.gca().invert_yaxis()
        plt.legend()
        save_show_close('shap_waterfall_PANW.png')
    except Exception as e:
        print(f"  Error in Waterfall Plot: {e}")

if __name__ == "__main__":
    res_df, shap_val_df, shap_imp_df, feature_names, call_shap, put_shap = load_audit_data()
    if res_df is not None:
        generate_all_plots(res_df, shap_val_df, shap_imp_df, feature_names, call_shap, put_shap)
        print("\n✅ SUCCESS! Images generated with data labels and professional themes.")