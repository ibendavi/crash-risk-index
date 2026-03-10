"""
Generate dashboard_data.json for the crash risk dashboard.
Reads from crash_index_dataset.csv and outputs a JSON file
that the HTML dashboard consumes.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime


def generate_dashboard_data(data_dir=None):
    if data_dir is None:
        data_dir = Path(__file__).parent
    data_dir = Path(data_dir)

    # Load dataset
    csv_path = data_dir / 'crash_index_dataset.csv'
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    latest = df.iloc[-1]
    latest_date = df.index[-1].strftime('%Y-%m-%d')

    # --- Indicator metadata ---
    # Categories for grouping in the dashboard
    CATEGORIES = {
        'Volatility': ['VIX', 'VVIX', 'SKEW', 'REALIZED_VOL', 'VRP_INV', 'VIX_RV_SPREAD_INV'],
        'Credit': ['HY_OAS', 'IG_OAS', 'BBB_OAS', 'CP_SPREAD',
                    'HY_OAS_MOMENTUM', 'EBP'],
        'Funding Stress': ['NFCI', 'KCFSI'],
        'Yield Curve': ['YC_10Y2Y_INV', 'YC_10Y3M_INV', 'FFR_10Y_INV',
                        'NYFED_RECESS_PROB'],
        'Labor Market': ['INIT_CLAIMS', 'SAHM', 'UNRATE', 'NFP_MOM_INV'],
        'Macro Activity': ['DGORDER_YOY_INV', 'INDPRO_YOY_INV', 'PERMIT_YOY_INV',
                           'PHILLY_MFG_INV', 'UMICH_INV', 'M2_GROWTH_INV'],
        'Market Technicals': ['SP500_VS_200DMA_INV', 'DEATH_CROSS', 'RSI_14',
                              'MOMENTUM_12_1_INV', 'DRAWDOWN_1Y'],
        'Valuation / Positioning': ['BUFFETT_IND', 'HH_EQUITY_ALLOC', 'MARGIN_DEBT',
                                     'MARGIN_DEBT_YOY', 'COT_LEV_NET_LONG',
                                     'COT_AM_NET_LONG'],
        'Cross-Asset': ['GOLD_SP_RATIO', 'CU_AU_RATIO_INV', 'DXY'],
        'Monetary Policy': ['FED_FUNDS', 'RRP_YOY_INV', 'SLOOS'],
    }

    # Display names
    DISPLAY_NAMES = {
        'VIX': 'VIX (Fear Index)',
        'VVIX': 'VVIX (Vol of VIX)',
        'SKEW': 'CBOE SKEW Index',
        'REALIZED_VOL': 'Realized Volatility (21d)',
        'VRP_INV': 'Variance Risk Premium',
        'VIX_RV_SPREAD_INV': 'VIX-Realized Vol Spread',
        'HY_OAS': 'High-Yield Credit Spread',
        'IG_OAS': 'Investment-Grade Spread',
        'CCC_OAS': 'CCC Credit Spread',
        'BBB_OAS': 'BBB Credit Spread',
        'CP_SPREAD': 'Commercial Paper Spread',
        'HY_OAS_MOMENTUM': 'HY Spread Momentum (1M)',
        'EBP': 'Excess Bond Premium',
        'TED_SPREAD': 'TED Spread',
        'NFCI': 'Chicago Fed NFCI',
        'STLFSI': 'St. Louis Fed Stress Index',
        'KCFSI': 'KC Fed Stress Index',
        'YC_10Y2Y_INV': 'Yield Curve (10Y-2Y)',
        'YC_10Y3M_INV': 'Yield Curve (10Y-3M)',
        'FFR_10Y_INV': 'Fed Funds - 10Y Spread',
        'NYFED_RECESS_PROB': 'NY Fed Recession Prob',
        'INIT_CLAIMS': 'Initial Jobless Claims',
        'SAHM': 'Sahm Rule',
        'UNRATE': 'Unemployment Rate',
        'NFP_MOM_INV': 'Nonfarm Payrolls Momentum',
        'DGORDER_YOY_INV': 'Durable Goods Orders YoY',
        'INDPRO_YOY_INV': 'Industrial Production YoY',
        'PERMIT_YOY_INV': 'Housing Permits YoY',
        'PHILLY_MFG_INV': 'Philly Fed Manufacturing',
        'UMICH_INV': 'UMich Consumer Sentiment',
        'M2_GROWTH_INV': 'M2 Money Supply Growth',
        'SP500_VS_200DMA_INV': 'S&P 500 vs 200-DMA',
        'DEATH_CROSS': 'Death Cross (50/200 MA)',
        'RSI_14': 'RSI (14-day)',
        'MOMENTUM_12_1_INV': '12-1 Month Momentum',
        'DRAWDOWN_1Y': 'Trailing 1Y Drawdown',
        'BUFFETT_IND': 'Buffett Indicator',
        'HH_EQUITY_ALLOC': 'Household Equity Allocation',
        'MARGIN_DEBT': 'FINRA Margin Debt',
        'MARGIN_DEBT_YOY': 'Margin Debt YoY Growth',
        'COT_LEV_NET_LONG': 'CFTC Leveraged Net Short',
        'COT_AM_NET_LONG': 'CFTC Asset Mgr Net Long',
        'GOLD_SP_RATIO': 'Gold / S&P 500 Ratio',
        'CU_AU_RATIO_INV': 'Copper / Gold Ratio',
        'OIL_SHOCK': 'Oil Price Shock',
        'DXY': 'US Dollar Index',
        'FED_FUNDS': 'Fed Funds Rate',
        'RRP_YOY_INV': 'Fed Reverse Repo YoY',
        'SLOOS': 'Bank Lending Standards',
    }

    # Publication frequency
    FREQUENCY = {
        'VIX': 'Daily', 'VVIX': 'Daily', 'SKEW': 'Daily',
        'REALIZED_VOL': 'Daily', 'VRP_INV': 'Daily', 'VIX_RV_SPREAD_INV': 'Daily',
        'HY_OAS': 'Daily', 'IG_OAS': 'Daily', 'CCC_OAS': 'Daily',
        'BBB_OAS': 'Daily', 'CP_SPREAD': 'Daily', 'HY_OAS_MOMENTUM': 'Daily',
        'EBP': 'Monthly', 'TED_SPREAD': 'Daily',
        'NFCI': 'Weekly', 'STLFSI': 'Weekly', 'KCFSI': 'Monthly',
        'YC_10Y2Y_INV': 'Daily', 'YC_10Y3M_INV': 'Daily',
        'FFR_10Y_INV': 'Daily', 'NYFED_RECESS_PROB': 'Daily',
        'INIT_CLAIMS': 'Weekly', 'SAHM': 'Monthly', 'UNRATE': 'Monthly',
        'NFP_MOM_INV': 'Monthly',
        'DGORDER_YOY_INV': 'Monthly', 'INDPRO_YOY_INV': 'Monthly',
        'PERMIT_YOY_INV': 'Monthly', 'PHILLY_MFG_INV': 'Monthly',
        'UMICH_INV': 'Monthly', 'M2_GROWTH_INV': 'Monthly',
        'SP500_VS_200DMA_INV': 'Daily', 'DEATH_CROSS': 'Daily',
        'RSI_14': 'Daily', 'MOMENTUM_12_1_INV': 'Daily', 'DRAWDOWN_1Y': 'Daily',
        'BUFFETT_IND': 'Quarterly', 'HH_EQUITY_ALLOC': 'Quarterly',
        'MARGIN_DEBT': 'Monthly', 'MARGIN_DEBT_YOY': 'Monthly',
        'COT_LEV_NET_LONG': 'Weekly', 'COT_AM_NET_LONG': 'Weekly',
        'GOLD_SP_RATIO': 'Daily', 'CU_AU_RATIO_INV': 'Daily',
        'OIL_SHOCK': 'Daily', 'DXY': 'Daily',
        'FED_FUNDS': 'Monthly', 'RRP_YOY_INV': 'Daily', 'SLOOS': 'Quarterly',
    }

    # --- Build indicator data ---
    ds_cols = [c for c in df.columns if c.startswith('DS_')]
    indicators = []

    for cat_name, cat_indicators in CATEGORIES.items():
        for ind_name in cat_indicators:
            ds_col = f'DS_{ind_name}'
            if ds_col not in df.columns:
                continue

            danger = latest.get(ds_col, np.nan)
            raw = latest.get(ind_name, np.nan)

            if pd.isna(danger):
                continue

            # Compute staleness
            raw_series = df[ind_name].dropna()
            if len(raw_series) > 0:
                # Find last date where value actually changed (not ffilled)
                changes = raw_series.diff().ne(0)
                last_change_dates = raw_series.index[changes]
                if len(last_change_dates) > 0:
                    last_update = last_change_dates[-1].strftime('%Y-%m-%d')
                else:
                    last_update = raw_series.index[-1].strftime('%Y-%m-%d')
            else:
                last_update = 'N/A'

            indicators.append({
                'id': ind_name,
                'name': DISPLAY_NAMES.get(ind_name, ind_name),
                'category': cat_name,
                'danger_score': round(float(danger), 1),
                'raw_value': round(float(raw), 4) if not pd.isna(raw) else None,
                'frequency': FREQUENCY.get(ind_name, 'Unknown'),
                'last_update': last_update,
            })

    # --- Composite history (last 5 years daily, then monthly before that) ---
    composite = df['COMPOSITE'].dropna()

    # Last 5 years: daily
    five_years_ago = composite.index[-1] - pd.DateOffset(years=5)
    recent = composite[composite.index >= five_years_ago]

    # Before that: monthly (first business day of month)
    older = composite[composite.index < five_years_ago]
    if len(older) > 0:
        older_monthly = older.resample('MS').first().dropna()
    else:
        older_monthly = pd.Series(dtype=float)

    # Combine
    history_index = list(older_monthly.index) + list(recent.index)
    history_values = list(older_monthly.values) + list(recent.values)

    composite_history = [
        {'date': d.strftime('%Y-%m-%d'), 'value': round(float(v), 2)}
        for d, v in zip(history_index, history_values)
    ]

    # --- Category averages ---
    category_scores = {}
    for cat_name, cat_indicators in CATEGORIES.items():
        scores = []
        for ind_name in cat_indicators:
            ds_col = f'DS_{ind_name}'
            if ds_col in df.columns and not pd.isna(latest.get(ds_col)):
                scores.append(float(latest[ds_col]))
        if scores:
            category_scores[cat_name] = round(np.mean(scores), 1)

    # --- Assemble output ---
    dashboard = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'latest_date': latest_date,
        'composite_score': round(float(latest.get('COMPOSITE', 0)), 1),
        'n_indicators': int(latest.get('N_INDICATORS', 0)),
        'category_scores': category_scores,
        'indicators': indicators,
        'composite_history': composite_history,
    }

    # Save
    out_path = data_dir / 'dashboard_data.json'
    with open(out_path, 'w') as f:
        json.dump(dashboard, f, indent=2)
    print(f"Dashboard data saved to {out_path}")
    print(f"  Composite: {dashboard['composite_score']}/100")
    print(f"  Indicators: {len(indicators)}")
    print(f"  History points: {len(composite_history)}")

    return dashboard


if __name__ == '__main__':
    generate_dashboard_data()
