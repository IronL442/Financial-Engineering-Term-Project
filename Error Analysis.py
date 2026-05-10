import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set professional theme and palette
sns.set_theme(style="whitegrid")
BLUE_PALETTE = ["#D4E6F1", "#7FB3D5", "#2980B9"] # Light to Dark Blue

def add_data_labels(ax, decimals=2, orientation='v'):
    """Adds numeric values on top or side of bars for direct comparison."""
    for container in ax.containers:
        if orientation == 'v':
            ax.bar_label(container, fmt=f'%.{decimals}f', padding=3, fontsize=10, fontweight='bold')
        else:
            ax.bar_label(container, fmt=f'%.{decimals}f', padding=5, fontsize=9, fontweight='bold')

def generate_enhanced_audit():
    print("Loading data for enhanced error audit...")
    moneyness_df = pd.read_csv('residual_by_moneyness.csv')
    tau_df = pd.read_csv('residual_by_tau.csv')
    test_res_df = pd.read_csv('test_predictions_residuals.csv')

    # --- 1. MONEYNESS AUDIT ---
    plt.figure(figsize=(10, 7))
    money_melt = moneyness_df.melt(id_vars=['moneyness_cat'], value_vars=['MAE_BS', 'MAE_ANN3', 'MAE_XGB3'], 
                                   var_name='Model', value_name='MAE')
    money_melt['Model'] = money_melt['Model'].str.replace('MAE_', '')
    money_melt['moneyness_cat'] = pd.Categorical(money_melt['moneyness_cat'], categories=['OTM', 'ATM', 'ITM'], ordered=True)
    
    ax1 = sns.barplot(data=money_melt, x='moneyness_cat', y='MAE', hue='Model', palette=BLUE_PALETTE)
    add_data_labels(ax1, orientation='v')
    
    plt.title('MAE by Moneyness Category', fontsize=15, fontweight='bold', pad=20)
    plt.ylabel('Mean Absolute Error ($)', fontsize=12)
    plt.ylim(0, money_melt['MAE'].max() * 1.15) 
    plt.legend(title='Model Architecture', loc='upper right')
    plt.tight_layout()
    plt.savefig('error_moneyness.png')
    plt.close()

    # --- 2. TIME TO EXPIRY (TAU) ---
    plt.figure(figsize=(10, 7))
    tau_melt = tau_df.melt(id_vars=['tau_cat'], value_vars=['MAE_BS', 'MAE_ANN3', 'MAE_XGB3'], 
                           var_name='Model', value_name='MAE')
    tau_melt['Model'] = tau_melt['Model'].str.replace('MAE_', '')
    tau_melt['tau_cat'] = pd.Categorical(tau_melt['tau_cat'], categories=['Short(<2mo)', 'Mid(2mo-1yr)', 'Long(>1yr)'], ordered=True)
    
    ax2 = sns.barplot(data=tau_melt, x='tau_cat', y='MAE', hue='Model', palette=BLUE_PALETTE)
    add_data_labels(ax2, orientation='v')
    
    plt.title('MAE by Time Horizon (Tau)', fontsize=15, fontweight='bold', pad=20)
    plt.ylabel('Mean Absolute Error ($)', fontsize=12)
    plt.ylim(0, tau_melt['MAE'].max() * 1.15)
    plt.tight_layout()
    plt.savefig('error_tau.png')
    plt.close()

    # --- 3. SECTOR ANALYSIS (Horizontal for better readability) ---
    sector_cols = [col for col in test_res_df.columns if col.startswith('sector_')]
    sector_data = []
    for col in sector_cols:
        sector_name = col.replace('sector_', '')
        subset = test_res_df[test_res_df[col] == 1]
        if not subset.empty:
            sector_data.append({'Sector': sector_name, 'BS': subset['resid_BS'].abs().mean(),
                               'ANN3': subset['resid_ANN3'].abs().mean(), 'XGB3': subset['resid_XGB3'].abs().mean()})
    
    sector_melt = pd.DataFrame(sector_data).sort_values('XGB3').melt(id_vars=['Sector'], var_name='Model', value_name='MAE')
    
    plt.figure(figsize=(12, 8))
    ax3 = sns.barplot(data=sector_melt, y='Sector', x='MAE', hue='Model', palette=BLUE_PALETTE)
    add_data_labels(ax3, orientation='h')
    
    plt.title('Performance Across Market Sectors', fontsize=15, fontweight='bold', pad=20)
    plt.xlabel('Mean Absolute Error ($)', fontsize=12)
    plt.xlim(0, sector_melt['MAE'].max() * 1.2)
    plt.tight_layout()
    plt.savefig('error_sector.png')
    plt.close()

if __name__ == "__main__":
    generate_enhanced_audit()
    print("✅ SUCCESS! Check your folder for: enhanced_error_moneyness.png, enhanced_error_tau.png, enhanced_error_sector.png")