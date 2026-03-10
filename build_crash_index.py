"""
Market Crash Risk Index Builder
================================
Downloads ~50 free indicators, computes expanding-window percentile ranks,
runs per-indicator univariate logistic regressions to predict crash
probability, and aggregates predictions across indicators.

Usage:
    python build_crash_index.py

Output:
    crash_index_dataset.csv   - daily dataset with indicators + crash probs
    crash_index_backtest.png  - backtest chart
"""

import argparse
import warnings
import sys
import io
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
START_DATE = '1988-01-01'  # extra history for lookback windows
ANALYSIS_START = '1990-01-01'  # actual analysis starts here
END_DATE = datetime.today().strftime('%Y-%m-%d')
OUTPUT_DIR = Path(__file__).parent
MIN_HISTORY = 252  # minimum days before computing percentile rank

# Crash probability model parameters
CRASH_THRESHOLDS = [5, 10, 15, 20]  # max drawdown thresholds (%)
HORIZONS = {'3M': 63, '6M': 126, '12M': 252}  # in business days
REFIT_EVERY = 21  # refit logistic regression monthly
MIN_TRAIN_OBS = 100  # minimum training observations before first prediction
PRIMARY_THRESHOLD = 10  # primary crash definition: DD > 10%
PRIMARY_HORIZON = '6M'  # primary crash definition: within 6 months


# ============================================================================
# 1. DATA DOWNLOAD
# ============================================================================

def download_fred_csv(series_id, start_date=START_DATE):
    """Download a single FRED series via direct CSV (no API key needed)."""
    url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start_date}'
    try:
        r = requests.get(url, timeout=30,
                         headers={'User-Agent': 'Mozilla/5.0 (research)'})
        r.raise_for_status()
        if '<html' in r.text[:200].lower():
            return pd.Series(dtype=float)  # got error page, not CSV
        df = pd.read_csv(io.StringIO(r.text), index_col=0, parse_dates=True, na_values='.')
        s = df.iloc[:, 0]
        s = pd.to_numeric(s, errors='coerce')
        s.index = pd.to_datetime(s.index)
        s.name = series_id
        return s
    except Exception:
        return pd.Series(dtype=float)


def download_fred_series(series_dict):
    """Download multiple FRED series into a DataFrame (no API key)."""
    data = {}
    for name, series_id in series_dict.items():
        try:
            s = download_fred_csv(series_id)
            if len(s) > 0:
                data[name] = s
                print(f"  [OK] {name} ({series_id}): {len(s)} obs")
            else:
                print(f"  [EMPTY] {name} ({series_id})")
        except Exception as e:
            print(f"  [FAIL] {name} ({series_id}): {e}")
    if data:
        df = pd.DataFrame(data)
        df.index = pd.to_datetime(df.index)
        return df
    return pd.DataFrame()


def download_yfinance_series(tickers_dict):
    """Download multiple yfinance tickers into a DataFrame."""
    data = {}
    for name, ticker in tickers_dict.items():
        try:
            df = yf.download(ticker, start=START_DATE, end=END_DATE,
                             progress=False, auto_adjust=True)
            if df is not None and len(df) > 0:
                data[name] = df['Close'].squeeze()
                print(f"  [OK] {name} ({ticker}): {len(df)} obs")
            else:
                print(f"  [EMPTY] {name} ({ticker})")
        except Exception as e:
            print(f"  [FAIL] {name} ({ticker}): {e}")
    if data:
        df = pd.DataFrame(data)
        df.index = pd.to_datetime(df.index)
        return df
    return pd.DataFrame()


def download_ebp():
    """Download Excess Bond Premium from Fed website."""
    try:
        url = 'https://www.federalreserve.gov/econres/notes/feds-notes/ebp_csv.csv'
        r = requests.get(url, timeout=30,
                         headers={'User-Agent': 'Mozilla/5.0 (research)'})
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), parse_dates=['date'], index_col='date')
        print(f"  [OK] EBP: {len(df)} obs")
        return df
    except Exception as e:
        print(f"  [FAIL] EBP: {e}")
        return pd.DataFrame()


def download_finra_margin():
    """Download FINRA margin debt statistics (monthly, from 1997)."""
    try:
        url = 'https://www.finra.org/sites/default/files/2021-03/margin-statistics.xlsx'
        r = requests.get(url, timeout=30,
                         headers={'User-Agent': 'Mozilla/5.0 (research)'})
        r.raise_for_status()
        df = pd.read_excel(io.BytesIO(r.content), sheet_name=0)
        # Parse: 'Year-Month' col and 'Debit Balances' col = margin debt
        df['date'] = pd.to_datetime(df['Year-Month'], format='%Y-%m')
        debit_col = [c for c in df.columns if 'Debit' in c][0]
        margin = df.set_index('date')[debit_col].sort_index()
        margin = pd.to_numeric(margin, errors='coerce')
        margin.name = 'MARGIN_DEBT'
        print(f"  [OK] FINRA Margin Debt: {len(margin.dropna())} obs, "
              f"last={margin.dropna().index[-1].date()}")
        return margin
    except Exception as e:
        print(f"  [FAIL] FINRA Margin: {e}")
        return pd.Series(dtype=float, name='MARGIN_DEBT')


def download_cboe_put_call():
    """Download CBOE total put/call ratio (daily, from Nov 2006)."""
    try:
        url = 'https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/totalpc.csv'
        r = requests.get(url, timeout=30,
                         headers={'User-Agent': 'Mozilla/5.0 (research)'})
        r.raise_for_status()
        lines = r.text.split('\n')
        # Find header line (skip disclaimer)
        header_idx = None
        for i, line in enumerate(lines):
            if 'DATE' in line and 'P/C' in line:
                header_idx = i
                break
        if header_idx is None:
            print("  [FAIL] CBOE Put/Call: header not found")
            return pd.Series(dtype=float, name='PUT_CALL_RATIO')
        df = pd.read_csv(io.StringIO('\n'.join(lines[header_idx:])))
        df.columns = [c.strip() for c in df.columns]
        df['DATE'] = pd.to_datetime(df['DATE'], format='mixed')
        pc_col = [c for c in df.columns if 'P/C' in c or 'Ratio' in c][0]
        pc = df.set_index('DATE')[pc_col].sort_index()
        pc = pd.to_numeric(pc, errors='coerce')
        pc.name = 'PUT_CALL_RATIO'
        print(f"  [OK] CBOE Put/Call Ratio: {len(pc.dropna())} obs, "
              f"last={pc.dropna().index[-1].date()}")
        return pc
    except Exception as e:
        print(f"  [FAIL] CBOE Put/Call: {e}")
        return pd.Series(dtype=float, name='PUT_CALL_RATIO')


def download_cftc_cot():
    """Download CFTC COT financial futures — S&P 500 net speculative positioning (weekly, from 2010)."""
    import zipfile
    try:
        all_data = []
        for year in range(2010, datetime.today().year + 1):
            url = f'https://www.cftc.gov/files/dea/history/fut_fin_txt_{year}.zip'
            r = requests.get(url, timeout=60,
                             headers={'User-Agent': 'Mozilla/5.0 (research)'})
            if r.status_code != 200:
                continue
            z = zipfile.ZipFile(io.BytesIO(r.content))
            fname = z.namelist()[0]
            df = pd.read_csv(z.open(fname))
            # Filter S&P 500 Consolidated
            sp = df[df['Market_and_Exchange_Names'].str.contains(
                'S&P 500 Consolidated', case=False, na=False)]
            if len(sp) > 0:
                all_data.append(sp)
        if not all_data:
            print("  [FAIL] CFTC COT: no S&P 500 data found")
            return pd.DataFrame()
        cot = pd.concat(all_data)
        cot['date'] = pd.to_datetime(cot['Report_Date_as_YYYY-MM-DD'])
        cot = cot.drop_duplicates(subset='date').set_index('date').sort_index()
        cot = cot[cot.index.notna()]
        # Net speculative = Leveraged Money (Long - Short)
        # High net long = crowded speculative bet = vulnerability
        # Use % of open interest (already scaled, no need for raw contract counts)
        cot['LEV_NET_LONG'] = cot['Pct_of_OI_Lev_Money_Long_All'] - cot['Pct_of_OI_Lev_Money_Short_All']
        cot['AM_NET_LONG'] = cot['Pct_of_OI_Asset_Mgr_Long_All'] - cot['Pct_of_OI_Asset_Mgr_Short_All']
        print(f"  [OK] CFTC COT S&P 500: {len(cot)} obs, "
              f"last={cot.index[-1].date()}")
        return cot[['LEV_NET_LONG', 'AM_NET_LONG']]
    except Exception as e:
        print(f"  [FAIL] CFTC COT: {e}")
        return pd.DataFrame()


def download_all_data():
    """Master download function."""
    print("\n=== Downloading FRED data ===")
    fred_series = {
        # Volatility
        'VIX': 'VIXCLS',
        # Yield curves
        'YC_10Y2Y': 'T10Y2Y',
        'YC_10Y3M': 'T10Y3M',
        # Credit spreads
        'HY_OAS': 'BAMLH0A0HYM2',
        'IG_OAS': 'BAMLC0A0CM',
        'CCC_OAS': 'BAMLH0A3HYC',
        'BAA_AAA': 'BAMLC0A4CBBB',  # BBB OAS (proxy for quality spread)
        # Funding stress
        # TED_SPREAD (TEDRATE) removed: LIBOR discontinued Jan 2022, series stale
        'CP_RATE': 'DCPF3M',  # 3M financial CP rate (daily)
        'TBILL_3M': 'DTB3',
        # Financial conditions / stress
        'NFCI': 'NFCI',
        # STLFSI (STLFSI2) removed: discontinued Jan 2022, series stale
        'KCFSI': 'KCFSI',
        # Macro - labor
        'INIT_CLAIMS': 'ICSA',
        'CLAIMS_4WK': 'IC4WSA',
        'SAHM': 'SAHMREALTIME',
        'UNRATE': 'UNRATE',
        'PAYEMS': 'PAYEMS',
        'CLF16OV': 'CLF16OV',  # Civilian labor force (for scaling claims)
        # Macro - activity
        # NAPM (ISM PMI) discontinued on FRED; use durable goods orders as proxy
        'DGORDER': 'DGORDER',
        'INDPRO': 'INDPRO',
        'PERMIT': 'PERMIT',
        'UMICH': 'UMCSENT',
        # Monetary
        'M2': 'M2SL',
        'FEDFUNDS': 'FEDFUNDS',
        # Rates
        'DGS10': 'DGS10',
        'DGS2': 'DGS2',
        'DGS3MO': 'DGS3MO',
        # Valuation / household
        'HH_EQUITY': 'BOGZ1FL153064486Q',
        # Lending
        'SLOOS_CI': 'DRTSCILM',
        # Philly Fed Manufacturing Survey
        'PHILLY_MFG': 'GACDFSA066MSFRBPHI',
        # Commodities
        'WTI_OIL': 'DCOILWTICO',
        'GOLD': 'GOLDAMGBD228NLBM',
        # Dollar
        'TWD_DOLLAR': 'DTWEXBGS',
        # S&P 500
        'SP500_FRED': 'SP500',
        # GDP (for Buffett indicator)
        'GDP': 'GDP',
        # Fed Reverse Repo (ON RRP) - liquidity plumbing
        'REVERSE_REPO': 'RRPONTSYD',
    }
    fred_df = download_fred_series(fred_series)

    print("\n=== Downloading yfinance data ===")
    yf_tickers = {
        'SP500': '^GSPC',
        'VVIX': '^VVIX',
        'SKEW': '^SKEW',
        'VIX_YF': '^VIX',
        'GOLD_FUT': 'GC=F',
        'COPPER': 'HG=F',
        'OIL_FUT': 'CL=F',
        'DXY': 'DX-Y.NYB',
        'WILSHIRE': '^W5000',
        'IMPL_CORR': '^COR1M',
    }
    yf_df = download_yfinance_series(yf_tickers)

    print("\n=== Downloading EBP from Fed ===")
    ebp_df = download_ebp()

    print("\n=== Downloading FINRA Margin Debt ===")
    margin_debt = download_finra_margin()

    # CBOE Put/Call Ratio: discontinued archives (stopped 2019), skipped

    print("\n=== Downloading CFTC COT (S&P 500 spec positioning) ===")
    cot_df = download_cftc_cot()

    return fred_df, yf_df, ebp_df, margin_debt, None, cot_df


# ============================================================================
# 2. COMPUTE DERIVED INDICATORS
# ============================================================================

# Publication lags in business days — shift each indicator forward to avoid
# look-ahead bias.  A lag of 0 means the indicator is available same-day
# (market-derived or released intraday before the close).
PUBLICATION_LAG = {
    # Same-day (0 biz days) — market-derived or real-time
    'VIX': 0, 'VVIX': 0, 'SKEW': 0, 'IMPL_CORR': 0,
    'HY_OAS': 0, 'IG_OAS': 0, 'CCC_OAS': 0, 'BBB_OAS': 0,
    'CP_SPREAD': 0, 'DXY': 0,
    'REALIZED_VOL': 0, 'VRP_INV': 0, 'VIX_RV_SPREAD_INV': 0,
    'SP500_VS_200DMA_INV': 0, 'DEATH_CROSS': 0, 'RSI_14': 0,
    'MOMENTUM_12_1_INV': 0, 'DRAWDOWN_1Y': 0,
    'GOLD_SP_RATIO': 0, 'CU_AU_RATIO_INV': 0,
    'RRP_YOY_INV': 0, 'HY_OAS_MOMENTUM': 0,
    'PUT_CALL_INV': 0,
    'NYFED_RECESS_PROB': 0,
    # 1 biz day — Treasury H.15 released 2:30 PM, use next close
    'YC_10Y2Y_INV': 1, 'YC_10Y3M_INV': 1, 'FFR_10Y_INV': 1,
    # 4 biz days — weekly releases (Thursday / Friday)
    'INIT_CLAIMS': 4, 'NFCI': 4,
    # 25 biz days — KCFSI is monthly (not weekly)
    'KCFSI': 25,
    # 5 biz days — COT: Tuesday data, Friday release
    'COT_LEV_NET_LONG': 5, 'COT_AM_NET_LONG': 5,
    # 18 biz days — durable goods orders
    'DGORDER_YOY_INV': 18,
    # 20 biz days — industrial production, M2, permits
    'INDPRO_YOY_INV': 20, 'M2_GROWTH_INV': 20, 'PERMIT_YOY_INV': 20,
    # 25 biz days — monthly macro with ~1 month lag
    'UNRATE': 25, 'SAHM': 25, 'NFP_MOM_INV': 25,
    'FED_FUNDS': 25, 'PHILLY_MFG_INV': 25, 'EBP': 25,
    'MARGIN_DEBT': 45, 'MARGIN_DEBT_YOY': 45,  # FINRA releases ~2 months late
    'UMICH_INV': 25,
    'SLOOS': 65,  # quarterly survey, ~2.5 month publication lag
    # 65 biz days — quarterly data with ~2.5 month lag
    'HH_EQUITY_ALLOC': 65,
    'BUFFETT_IND': 65,
}


def apply_publication_lags(indicators):
    """Shift each indicator by its publication lag (in business days) to
    prevent look-ahead bias.  Only non-zero lags produce actual shifts."""
    shifted_count = 0
    skipped = []
    print("\n=== Applying publication-lag shifts ===")
    for col in indicators.columns:
        lag = PUBLICATION_LAG.get(col)
        if lag is None:
            skipped.append(col)
            continue
        if lag > 0:
            indicators[col] = indicators[col].shift(lag, freq='B')
            shifted_count += 1
            print(f"  {col}: shifted {lag} business days")
    if skipped:
        print(f"  [WARN] No lag defined for: {', '.join(skipped)} — left unshifted")
    print(f"  Total shifted: {shifted_count} indicators "
          f"({len(indicators.columns) - shifted_count - len(skipped)} same-day, "
          f"{len(skipped)} undefined)")
    # After shifting with freq='B' the index may have new dates; realign to
    # the original business-day index so downstream code is unaffected.
    orig_idx = indicators.index
    indicators = indicators.reindex(orig_idx)
    return indicators


def compute_indicators(fred_df, yf_df, ebp_df, margin_debt=None,
                       put_call=None, cot_df=None):
    """Compute all derived indicators and return a unified daily DataFrame."""
    print("\n=== Computing derived indicators ===")

    # Create a daily date index
    idx = pd.date_range(START_DATE, END_DATE, freq='B')
    indicators = pd.DataFrame(index=idx)

    # Helper: resample lower-freq to daily (forward-fill)
    def to_daily(series, name=''):
        if series is None or len(series) == 0:
            return pd.Series(dtype=float, index=idx)
        # dropna() first: when mixed-freq series are in one DataFrame,
        # weekly/monthly data has NaN on business days — drop those
        # so ffill propagates from the actual observation dates
        s = series.dropna().reindex(idx, method='ffill')
        return s

    # ------------------------------------------------------------------
    # A. DIRECT INDICATORS (higher = more danger)
    # ------------------------------------------------------------------

    # VIX
    if 'VIX' in fred_df.columns:
        indicators['VIX'] = to_daily(fred_df['VIX'])
    elif 'VIX_YF' in yf_df.columns:
        indicators['VIX'] = to_daily(yf_df['VIX_YF'])

    # VVIX
    if 'VVIX' in yf_df.columns:
        indicators['VVIX'] = to_daily(yf_df['VVIX'])

    # SKEW (higher = more tail risk)
    if 'SKEW' in yf_df.columns:
        indicators['SKEW'] = to_daily(yf_df['SKEW'])

    # Implied Correlation (higher = more systemic)
    if 'IMPL_CORR' in yf_df.columns:
        indicators['IMPL_CORR'] = to_daily(yf_df['IMPL_CORR'])

    # HY OAS (higher = more stress)
    if 'HY_OAS' in fred_df.columns:
        indicators['HY_OAS'] = to_daily(fred_df['HY_OAS'])

    # IG OAS
    if 'IG_OAS' in fred_df.columns:
        indicators['IG_OAS'] = to_daily(fred_df['IG_OAS'])

    # CCC OAS
    if 'CCC_OAS' in fred_df.columns:
        indicators['CCC_OAS'] = to_daily(fred_df['CCC_OAS'])

    # BBB OAS
    if 'BAA_AAA' in fred_df.columns:
        indicators['BBB_OAS'] = to_daily(fred_df['BAA_AAA'])

    # TED Spread: REMOVED (LIBOR discontinued Jan 2022)

    # CP Spread = CP rate - T-bill
    if 'CP_RATE' in fred_df.columns and 'TBILL_3M' in fred_df.columns:
        cp_spread = fred_df['CP_RATE'] - fred_df['TBILL_3M']
        indicators['CP_SPREAD'] = to_daily(cp_spread)

    # NFCI (higher = tighter conditions = more stress)
    if 'NFCI' in fred_df.columns:
        indicators['NFCI'] = to_daily(fred_df['NFCI'])

    # St. Louis FSI: REMOVED (discontinued Jan 2022)

    # KC FSI
    if 'KCFSI' in fred_df.columns:
        indicators['KCFSI'] = to_daily(fred_df['KCFSI'])

    # Initial Claims scaled by civilian labor force (higher = more stress)
    claims_raw = None
    if 'CLAIMS_4WK' in fred_df.columns:
        claims_raw = to_daily(fred_df['CLAIMS_4WK'])
    elif 'INIT_CLAIMS' in fred_df.columns:
        claims_raw = to_daily(fred_df['INIT_CLAIMS'])
    if claims_raw is not None:
        if 'CLF16OV' in fred_df.columns:
            clf = to_daily(fred_df['CLF16OV'])
            # Claims in raw numbers, CLF16OV in thousands → convert CLF to raw
            indicators['INIT_CLAIMS'] = (claims_raw / (clf * 1000)) * 100
            print("  [OK] Initial Claims (scaled by labor force)")
        else:
            indicators['INIT_CLAIMS'] = claims_raw
            print("  [OK] Initial Claims (raw, no labor force data)")

    # Sahm Rule (higher = closer to recession)
    if 'SAHM' in fred_df.columns:
        indicators['SAHM'] = to_daily(fred_df['SAHM'])

    # SLOOS (higher = tighter lending)
    if 'SLOOS_CI' in fred_df.columns:
        indicators['SLOOS'] = to_daily(fred_df['SLOOS_CI'])

    # Unemployment rate (higher = worse)
    if 'UNRATE' in fred_df.columns:
        indicators['UNRATE'] = to_daily(fred_df['UNRATE'])

    # Household equity allocation (higher = more euphoria = more danger)
    if 'HH_EQUITY' in fred_df.columns:
        indicators['HH_EQUITY_ALLOC'] = to_daily(fred_df['HH_EQUITY'])

    # DXY (higher = stronger dollar = EM stress)
    if 'DXY' in yf_df.columns:
        indicators['DXY'] = to_daily(yf_df['DXY'])
    elif 'TWD_DOLLAR' in fred_df.columns:
        indicators['DXY'] = to_daily(fred_df['TWD_DOLLAR'])

    # EBP
    if len(ebp_df) > 0 and 'ebp' in ebp_df.columns:
        indicators['EBP'] = to_daily(ebp_df['ebp'])

    # ------------------------------------------------------------------
    # B. INVERTED INDICATORS (lower original = more danger, so we invert)
    # ------------------------------------------------------------------

    # Yield Curve 10Y-2Y (INVERTED: more negative = more danger)
    if 'YC_10Y2Y' in fred_df.columns:
        indicators['YC_10Y2Y_INV'] = to_daily(fred_df['YC_10Y2Y'])

    # Yield Curve 10Y-3M (INVERTED)
    if 'YC_10Y3M' in fred_df.columns:
        indicators['YC_10Y3M_INV'] = to_daily(fred_df['YC_10Y3M'])

    # Durable Goods Orders YoY (INVERTED: lower = more danger)
    if 'DGORDER' in fred_df.columns:
        dgo_raw = fred_df['DGORDER'].dropna()
        dgo_yoy = dgo_raw.pct_change(12) * 100  # 12 monthly obs = 1 year
        indicators['DGORDER_YOY_INV'] = to_daily(dgo_yoy)

    # UMich Sentiment (INVERTED: lower = more danger)
    if 'UMICH' in fred_df.columns:
        indicators['UMICH_INV'] = to_daily(fred_df['UMICH'])

    # Philly Fed Manufacturing (INVERTED: lower = more danger)
    if 'PHILLY_MFG' in fred_df.columns:
        indicators['PHILLY_MFG_INV'] = to_daily(fred_df['PHILLY_MFG'])

    # ------------------------------------------------------------------
    # C. COMPUTED INDICATORS FROM PRICE DATA
    # ------------------------------------------------------------------

    # Get S&P 500 daily prices
    sp500 = None
    if 'SP500' in yf_df.columns:
        sp500 = to_daily(yf_df['SP500'])
    elif 'SP500_FRED' in fred_df.columns:
        sp500 = to_daily(fred_df['SP500_FRED'])

    if sp500 is not None and sp500.notna().sum() > 252:
        sp_ret = np.log(sp500 / sp500.shift(1))

        # --- Realized Volatility (22-day annualized) ---
        rv_22 = sp_ret.rolling(22).std() * np.sqrt(252) * 100
        indicators['REALIZED_VOL'] = rv_22
        print("  [OK] Realized Volatility")

        # --- Variance Risk Premium (VIX^2 - RV^2) ---
        # Note: VRP < 0 is danger (realized > implied), so we invert
        if 'VIX' in indicators.columns:
            vix = indicators['VIX']
            vrp = vix**2 - rv_22**2
            indicators['VRP_INV'] = vrp  # auto-oriented later
            print("  [OK] Variance Risk Premium")

        # --- VIX - Realized Vol Spread ---
        if 'VIX' in indicators.columns:
            vix_rv_spread = indicators['VIX'] - rv_22
            indicators['VIX_RV_SPREAD_INV'] = vix_rv_spread  # auto-oriented later
            print("  [OK] VIX-RV Spread")

        # --- S&P 500 vs 200-DMA (inverted: below = danger) ---
        sma_200 = sp500.rolling(200).mean()
        pct_from_200dma = (sp500 / sma_200 - 1) * 100
        indicators['SP500_VS_200DMA_INV'] = pct_from_200dma  # auto-oriented later
        print("  [OK] S&P 500 vs 200-DMA")

        # --- Death Cross signal (1 = 50-DMA below 200-DMA) ---
        sma_50 = sp500.rolling(50).mean()
        death_cross = (sma_50 < sma_200).astype(float) * 100
        indicators['DEATH_CROSS'] = death_cross
        print("  [OK] Death Cross")

        # --- RSI(14) of S&P 500 ---
        delta = sp500.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/14, min_periods=14).mean()
        avg_loss = loss.ewm(alpha=1/14, min_periods=14).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - 100 / (1 + rs)
        # Very high RSI = overbought = danger of pullback
        indicators['RSI_14'] = rsi
        print("  [OK] RSI(14)")

        # --- 12-1 Month Momentum (inverted: negative = danger) ---
        mom_12_1 = (sp500.shift(21) / sp500.shift(252) - 1) * 100
        indicators['MOMENTUM_12_1_INV'] = mom_12_1  # auto-oriented later
        print("  [OK] 12-1 Month Momentum")

        # --- Max Drawdown (trailing 252-day) ---
        rolling_max = sp500.rolling(252, min_periods=1).max()
        drawdown = (sp500 / rolling_max - 1) * 100
        indicators['DRAWDOWN_1Y'] = drawdown  # auto-oriented later
        print("  [OK] Trailing 1Y Drawdown")

        # --- Amihud Illiquidity (aggregate, using S&P 500 ETF volume as proxy) ---
        # We'll use S&P 500 returns / approx dollar volume
        # This is a rough proxy; ideally we'd compute across individual stocks

    # --- Gold / S&P Ratio (higher = more fear) ---
    gold = None
    if 'GOLD_FUT' in yf_df.columns:
        gold = to_daily(yf_df['GOLD_FUT'])
    elif 'GOLD' in fred_df.columns:
        gold = to_daily(fred_df['GOLD'])
    if gold is not None and sp500 is not None:
        gold_sp_ratio = gold / sp500
        indicators['GOLD_SP_RATIO'] = gold_sp_ratio
        print("  [OK] Gold/S&P Ratio")

    # --- Copper / Gold Ratio (inverted: lower = more danger) ---
    copper = yf_df.get('COPPER')
    if copper is not None and gold is not None:
        copper_d = to_daily(copper)
        cu_au_ratio = copper_d / gold
        indicators['CU_AU_RATIO_INV'] = cu_au_ratio  # auto-oriented later
        print("  [OK] Copper/Gold Ratio")

    # Oil Shock removed: Hamilton-style indicator is mostly zeros (only fires
    # when oil exceeds 12-month high), poor percentile behavior, and near-zero
    # predictive power for equity crashes (+0.03 correlation).

    # --- M2 Money Supply YoY Growth (inverted: lower/negative = danger) ---
    if 'M2' in fred_df.columns:
        m2_raw = fred_df['M2'].dropna()
        m2_yoy = m2_raw.pct_change(12) * 100  # 12 monthly obs = 1 year
        indicators['M2_GROWTH_INV'] = to_daily(m2_yoy)  # auto-oriented later
        print("  [OK] M2 YoY Growth")

    # --- Nonfarm Payrolls Momentum (3-mo avg change, inverted) ---
    if 'PAYEMS' in fred_df.columns:
        # Compute change on raw monthly data BEFORE forward-fill to avoid blocky artifacts
        nfp_raw = fred_df['PAYEMS'].dropna()
        nfp_change = nfp_raw.diff()  # monthly change
        nfp_3mo = nfp_change.rolling(3).mean()  # 3-month average change
        indicators['NFP_MOM_INV'] = to_daily(nfp_3mo)  # auto-oriented later
        print("  [OK] NFP Momentum")

    # --- Housing Permits YoY (inverted) ---
    if 'PERMIT' in fred_df.columns:
        permit_raw = fred_df['PERMIT'].dropna()
        permit_yoy = permit_raw.pct_change(12) * 100
        indicators['PERMIT_YOY_INV'] = to_daily(permit_yoy)  # auto-oriented later
        print("  [OK] Housing Permits YoY")

    # --- Industrial Production YoY (inverted) ---
    if 'INDPRO' in fred_df.columns:
        indpro_raw = fred_df['INDPRO'].dropna()
        indpro_yoy = indpro_raw.pct_change(12) * 100
        indicators['INDPRO_YOY_INV'] = to_daily(indpro_yoy)  # auto-oriented later
        print("  [OK] Industrial Production YoY")

    # --- NY Fed Recession Probability (from 10Y-3M spread) ---
    if 'YC_10Y3M' in fred_df.columns:
        spread = to_daily(fred_df['YC_10Y3M'])
        from scipy.stats import norm
        recession_prob = norm.cdf(-0.6045 - 0.7374 * spread) * 100
        indicators['NYFED_RECESS_PROB'] = recession_prob
        print("  [OK] NY Fed Recession Probability")

    # --- Buffett Indicator (Market Cap / GDP, using Wilshire 5000) ---
    if 'WILSHIRE' in yf_df.columns and 'GDP' in fred_df.columns:
        wilshire = to_daily(yf_df['WILSHIRE'])
        gdp = to_daily(fred_df['GDP'])
        # Wilshire 5000: at inception (1980), 1 point ≈ $1B market cap.
        # As of 2024: ~48K index vs ~$50T actual market cap → proxy still reasonable.
        # GDP is in $B annual rate.
        buffett = (wilshire / gdp * 100) if gdp is not None else None
        if buffett is not None:
            indicators['BUFFETT_IND'] = buffett
            print("  [OK] Buffett Indicator (proxy)")

    # --- FINRA Margin Debt scaled by GDP (higher = more speculation = more danger) ---
    if margin_debt is not None and len(margin_debt.dropna()) > 0:
        md = to_daily(margin_debt)
        # Scale by GDP to remove nominal economic growth
        if 'GDP' in fred_df.columns:
            gdp = to_daily(fred_df['GDP'])
            # margin_debt in $M, GDP in $B → convert to same units ($B)
            md_scaled = (md / 1000) / gdp * 100  # margin debt as % of GDP
            indicators['MARGIN_DEBT'] = md_scaled
            print("  [OK] Margin Debt / GDP")
        else:
            indicators['MARGIN_DEBT'] = md
            print("  [OK] Margin Debt (raw, no GDP data)")
        # YoY growth of GDP-scaled margin debt (captures leverage growth vs economy)
        # Compute on raw monthly data before forward-fill
        if 'GDP' in fred_df.columns:
            gdp_raw = fred_df['GDP'].dropna()
            md_raw = margin_debt.dropna()
            # Align monthly data, compute ratio, then YoY
            md_gdp_raw = (md_raw / 1000) / gdp_raw.reindex(md_raw.index, method='ffill') * 100
            md_yoy_raw = md_gdp_raw.pct_change(12) * 100  # 12 monthly obs
            indicators['MARGIN_DEBT_YOY'] = to_daily(md_yoy_raw)
        else:
            md_yoy_raw = margin_debt.dropna().pct_change(12) * 100
            indicators['MARGIN_DEBT_YOY'] = to_daily(md_yoy_raw)
        print("  [OK] Margin Debt YoY Growth (GDP-scaled)")

    # --- CBOE Put/Call Ratio (higher = more puts = more fear = contrarian bullish) ---
    # But extreme LOW values = complacency = danger. Invert: low P/C = high danger
    if put_call is not None and len(put_call.dropna()) > 0:
        pc = to_daily(put_call)
        # Invert: low put/call = complacency = danger
        indicators['PUT_CALL_INV'] = pc  # auto-oriented later
        print("  [OK] Put/Call Ratio (inverted: low P/C = danger)")

    # --- CFTC COT Positioning ---
    # Two groups with OPPOSITE logic:
    #   Leveraged money (hedge funds, CTAs): smart/fast money.
    #     Net short = informed bearish bet = DANGER. → INVERT.
    #   Asset managers (pensions, mutual funds): slow/herding money.
    #     Net long = crowding/complacency = DANGER. → DO NOT INVERT.
    if cot_df is not None and len(cot_df) > 0:
        if 'LEV_NET_LONG' in cot_df.columns:
            lev_net = to_daily(cot_df['LEV_NET_LONG'])
            indicators['COT_LEV_NET_LONG'] = lev_net  # auto-oriented later
            print("  [OK] CFTC COT Leveraged Money (inverted: net short = danger)")
        if 'AM_NET_LONG' in cot_df.columns:
            am_net = to_daily(cot_df['AM_NET_LONG'])
            indicators['COT_AM_NET_LONG'] = am_net  # NOT inverted: net long = crowding = danger
            print("  [OK] CFTC COT Asset Manager (net long = crowding = danger)")

    # --- Fed Reverse Repo (ON RRP) ---
    # High usage = excess liquidity parked at Fed; sudden drops = liquidity withdrawal = danger
    if 'REVERSE_REPO' in fred_df.columns:
        rrp = to_daily(fred_df['REVERSE_REPO'])
        # YoY change: declining RRP = liquidity draining = tightening
        rrp_yoy = rrp.pct_change(252) * 100
        # Replace inf/-inf (from division by zero when RRP was 0 a year ago)
        rrp_yoy = rrp_yoy.replace([np.inf, -np.inf], np.nan)
        indicators['RRP_YOY_INV'] = rrp_yoy  # auto-oriented later
        print("  [OK] Fed Reverse Repo YoY Change (inverted)")

    # --- Credit Spread Momentum (1-month change in HY OAS) ---
    if 'HY_OAS' in indicators.columns:
        hy_mom = indicators['HY_OAS'].diff(21)  # 1-month change
        indicators['HY_OAS_MOMENTUM'] = hy_mom
        print("  [OK] HY OAS Momentum")

    # --- Fed Funds Rate level (higher = tighter) ---
    if 'FEDFUNDS' in fred_df.columns:
        indicators['FED_FUNDS'] = to_daily(fred_df['FEDFUNDS'])
        print("  [OK] Fed Funds Rate")

    # --- Fed Funds - 10Y (inverted curve from funds rate perspective) ---
    if 'FEDFUNDS' in fred_df.columns and 'DGS10' in fred_df.columns:
        ffr = to_daily(fred_df['FEDFUNDS'])
        gs10 = to_daily(fred_df['DGS10'])
        ffr_10y = ffr - gs10  # positive = inverted
        indicators['FFR_10Y_INV'] = ffr_10y
        print("  [OK] Fed Funds - 10Y Spread")

    # ---- Apply publication-lag shifts to avoid look-ahead bias ----
    indicators = apply_publication_lags(indicators)

    # Trim to analysis period
    indicators = indicators.loc[ANALYSIS_START:]

    # --- Data freshness tracking ---
    # For each underlying raw series, find the last actual (non-ffilled) observation date
    today = pd.Timestamp(END_DATE)
    freshness = {}

    # Track freshness from FRED raw data
    for name in fred_df.columns:
        raw = fred_df[name].dropna()
        if len(raw) > 0:
            last_obs = raw.index[-1]
            staleness = (today - last_obs).days
            freshness[name] = {'last_obs': last_obs, 'stale_days': staleness}

    # Track freshness from yfinance raw data
    for name in yf_df.columns:
        raw = yf_df[name].dropna()
        if len(raw) > 0:
            last_obs = raw.index[-1]
            staleness = (today - last_obs).days
            freshness[name] = {'last_obs': last_obs, 'stale_days': staleness}

    # EBP freshness
    if len(ebp_df) > 0:
        last_ebp = ebp_df.index[-1]
        freshness['EBP'] = {'last_obs': last_ebp, 'stale_days': (today - last_ebp).days}

    # Put/Call freshness
    if put_call is not None and len(put_call.dropna()) > 0:
        last_pc = put_call.dropna().index[-1]
        freshness['PUT_CALL'] = {'last_obs': last_pc, 'stale_days': (today - last_pc).days}

    # COT freshness
    if cot_df is not None and len(cot_df) > 0:
        last_cot = cot_df.index[-1]
        freshness['COT'] = {'last_obs': last_cot, 'stale_days': (today - last_cot).days}

    # Map derived indicators back to their source series for freshness
    indicator_source = {
        'VIX': 'VIX', 'VVIX': 'VVIX', 'SKEW': 'SKEW',
        'HY_OAS': 'HY_OAS', 'IG_OAS': 'IG_OAS', 'CCC_OAS': 'CCC_OAS',
        'BBB_OAS': 'BAA_AAA',
        'CP_SPREAD': 'CP_RATE', 'NFCI': 'NFCI',
        'KCFSI': 'KCFSI', 'INIT_CLAIMS': 'CLAIMS_4WK',
        'SAHM': 'SAHM', 'SLOOS': 'SLOOS_CI', 'UNRATE': 'UNRATE',
        'HH_EQUITY_ALLOC': 'HH_EQUITY', 'DXY': 'DXY',
        'EBP': 'EBP', 'FED_FUNDS': 'FEDFUNDS',
        'YC_10Y2Y_INV': 'YC_10Y2Y', 'YC_10Y3M_INV': 'YC_10Y3M',
        'DGORDER_YOY_INV': 'DGORDER', 'PHILLY_MFG_INV': 'PHILLY_MFG',
        'UMICH_INV': 'UMICH',
        'REALIZED_VOL': 'SP500', 'VRP_INV': 'VIX',
        'VIX_RV_SPREAD_INV': 'VIX',
        'SP500_VS_200DMA_INV': 'SP500', 'DEATH_CROSS': 'SP500',
        'RSI_14': 'SP500', 'MOMENTUM_12_1_INV': 'SP500',
        'DRAWDOWN_1Y': 'SP500',
        'GOLD_SP_RATIO': 'GOLD_FUT', 'CU_AU_RATIO_INV': 'COPPER',
        'M2_GROWTH_INV': 'M2', 'NFP_MOM_INV': 'PAYEMS',
        'PERMIT_YOY_INV': 'PERMIT', 'INDPRO_YOY_INV': 'INDPRO',
        'NYFED_RECESS_PROB': 'YC_10Y3M', 'BUFFETT_IND': 'WILSHIRE',
        'HY_OAS_MOMENTUM': 'HY_OAS', 'FFR_10Y_INV': 'FEDFUNDS',
        'MARGIN_DEBT': 'MARGIN_DEBT', 'MARGIN_DEBT_YOY': 'MARGIN_DEBT',
        'PUT_CALL_INV': 'PUT_CALL', 'COT_LEV_NET_LONG': 'COT',
        'COT_AM_NET_LONG': 'COT', 'RRP_YOY_INV': 'REVERSE_REPO',
    }

    print(f"\n=== Total indicators: {len(indicators.columns)} ===")
    print(f"=== Date range: {indicators.index[0].date()} to {indicators.index[-1].date()} ===")
    print(f"\n  {'Indicator':30s}  {'Obs':>6s}  {'From':>12s}  {'Source':>15s}  {'Last Obs':>12s}  {'Stale':>6s}")
    print(f"  {'-'*30}  {'-'*6}  {'-'*12}  {'-'*15}  {'-'*12}  {'-'*6}")
    for col in sorted(indicators.columns):
        n_valid = indicators[col].notna().sum()
        first_valid = indicators[col].first_valid_index()
        fv_str = first_valid.date() if first_valid is not None else 'N/A'
        src = indicator_source.get(col, '?')
        if src in freshness:
            lo = freshness[src]['last_obs']
            sd = freshness[src]['stale_days']
            lo_str = lo.strftime('%Y-%m-%d')
            sd_str = f'{sd}d'
        else:
            lo_str = '?'
            sd_str = '?'
        print(f"  {col:30s}  {n_valid:>6d}  {fv_str!s:>12s}  {src:>15s}  {lo_str:>12s}  {sd_str:>6s}")

    # Store freshness for export
    indicators.attrs['freshness'] = freshness
    indicators.attrs['indicator_source'] = indicator_source

    return indicators


# ============================================================================
# 3. NORMALIZE TO DANGER SCORES (0-100 percentile rank, expanding window)
# ============================================================================

def normalize_indicators(indicators):
    """
    Convert each indicator to a 0-100 danger score using expanding-window
    percentile ranks. This avoids look-ahead bias.

    Returns:
        danger_scores: DataFrame with NaN where indicator is unavailable
                       (used by composite — NaN-aware averaging)
        danger_scores_filled: DataFrame with NaN filled (first valid value
                              or 0) — for ML model features
        miss_flags: DataFrame of MISS_{col} flags (1 = originally missing)
    """
    print("\n=== Normalizing indicators to 0-100 danger scores ===")

    danger_scores = pd.DataFrame(index=indicators.index)
    danger_scores_filled = pd.DataFrame(index=indicators.index)
    miss_flags = pd.DataFrame(index=indicators.index)

    for col in indicators.columns:
        # Track which rows were originally missing (before any fill)
        is_missing = indicators[col].isna().astype(int)
        miss_flags[f'MISS_{col}'] = is_missing

        series = indicators[col].dropna()
        if len(series) < MIN_HISTORY:
            print(f"  [SKIP] {col}: only {len(series)} obs (need {MIN_HISTORY})")
            continue

        # Expanding percentile rank (no look-ahead)
        rank = series.expanding(min_periods=MIN_HISTORY).rank(pct=True) * 100
        ds = rank.reindex(indicators.index)

        # Keep NaN version for composite (NaN-aware averaging)
        danger_scores[col] = ds

        # Filled version for ML: fill with first valid value (or 0)
        ds_filled = ds.copy()
        first_valid = ds.first_valid_index()
        if first_valid is not None:
            first_val = ds.loc[first_valid]
            ds_filled = ds_filled.fillna(first_val)
        else:
            ds_filled = ds_filled.fillna(0)
        danger_scores_filled[col] = ds_filled

        n_miss = is_missing.sum()
        pct_miss = n_miss / len(is_missing) * 100
        print(f"  [OK] {col} (missing: {n_miss} = {pct_miss:.1f}%)")

    print(f"\n=== Normalized {len(danger_scores.columns)} indicators ===")
    return danger_scores, danger_scores_filled, miss_flags


# ============================================================================
# 4. COMPUTE CRASH PROBABILITIES (Univariate Logistic Regressions)
# ============================================================================

def compute_forward_max_drawdown(sp500_prices, horizon_days, index):
    """Compute forward max drawdown over a given horizon for each date."""
    sp = sp500_prices.reindex(index, method='ffill')
    sp_arr = sp.values
    n = len(sp_arr)
    fwd_dd = np.full(n, np.nan)
    for i in range(n - 1):
        end = min(i + 1 + horizon_days, n)
        future = sp_arr[i+1:end]
        if np.all(np.isnan(future)):
            continue
        fwd_dd[i] = (np.nanmin(future) / sp_arr[i] - 1) * 100
    return pd.Series(fwd_dd, index=index)


def compute_crash_probabilities(percentiles, sp500_prices):
    """
    For each indicator, run expanding-window logistic regression of
    binary crash outcome on (percentile, percentile^2).

    Aggregates across indicators via median, 75th percentile, and mean.
    """
    print("\n=== Computing crash probabilities (univariate logistic regressions) ===")

    index = percentiles.index
    cols = percentiles.columns.tolist()
    n_dates = len(index)

    # Pre-compute forward max drawdowns for all horizons
    fwd_dds = {}
    for h_name, h_days in HORIZONS.items():
        print(f"  Computing forward max drawdown ({h_name}, {h_days}d)...")
        fwd_dds[h_name] = compute_forward_max_drawdown(sp500_prices, h_days, index)

    # Storage for all per-indicator predictions and aggregated probabilities
    all_indicator_probs = {}  # key: (threshold, horizon) -> DataFrame of per-indicator probs
    aggregate_probs = pd.DataFrame(index=index)

    total_models = len(CRASH_THRESHOLDS) * len(HORIZONS) * len(cols)
    model_count = 0

    for h_name, h_days in HORIZONS.items():
        fwd_dd = fwd_dds[h_name]

        for threshold in CRASH_THRESHOLDS:
            crash_binary = (fwd_dd < -threshold).astype(float)
            # NaN where fwd_dd is NaN (future not available)
            crash_binary = crash_binary.where(fwd_dd.notna())

            key = (threshold, h_name)
            indicator_preds = pd.DataFrame(index=index)

            for col in cols:
                model_count += 1
                pct = percentiles[col]
                pct_sq = pct ** 2

                # Build feature matrix
                X_all = pd.DataFrame({'pct': pct, 'pct_sq': pct_sq})
                predicted = np.full(n_dates, np.nan)

                model = None
                last_fit = -REFIT_EVERY  # force fit on first eligible date

                for t in range(MIN_TRAIN_OBS, n_dates):
                    # Refit periodically
                    if t - last_fit >= REFIT_EVERY or model is None:
                        X_train = X_all.iloc[:t]
                        y_train = crash_binary.iloc[:t]
                        # Valid = both features and target non-NaN
                        valid = X_train.notna().all(axis=1) & y_train.notna()
                        n_valid = valid.sum()
                        if n_valid >= MIN_TRAIN_OBS:
                            y_v = y_train[valid]
                            if y_v.nunique() == 2:
                                model = LogisticRegression(
                                    solver='lbfgs', max_iter=200, C=1.0)
                                model.fit(X_train[valid].values, y_v.values)
                                last_fit = t

                    # Predict at t
                    if model is not None:
                        x_t = X_all.iloc[t]
                        if x_t.notna().all():
                            predicted[t] = model.predict_proba(
                                x_t.values.reshape(1, -1))[0, 1]

                indicator_preds[col] = predicted

            all_indicator_probs[key] = indicator_preds

            # Aggregate across indicators
            tag = f'{threshold}pct_{h_name}'
            n_avail = indicator_preds.notna().sum(axis=1)
            aggregate_probs[f'CRASH_PROB_MEDIAN_{tag}'] = indicator_preds.median(axis=1)
            aggregate_probs[f'CRASH_PROB_P75_{tag}'] = indicator_preds.quantile(0.75, axis=1)
            aggregate_probs[f'CRASH_PROB_MEAN_{tag}'] = indicator_preds.mean(axis=1)
            aggregate_probs[f'N_MODELS_{tag}'] = n_avail

            # Progress
            med_val = indicator_preds.median(axis=1).dropna()
            if len(med_val) > 0:
                print(f"  DD>{threshold}% in {h_name}: median P(crash) latest={med_val.iloc[-1]:.3f}, "
                      f"avg={med_val.mean():.3f}, models={len(cols)}")

    print(f"\n  Total models fitted: {model_count} indicator x crash-definition combinations")
    print(f"  Crash definitions: {len(CRASH_THRESHOLDS)} thresholds x {len(HORIZONS)} horizons "
          f"= {len(CRASH_THRESHOLDS) * len(HORIZONS)}")

    return aggregate_probs, all_indicator_probs


# ============================================================================
# 5. BACKTEST
# ============================================================================

def backtest(aggregate_probs, sp500_prices, tbill_rate=None):
    """
    Backtest crash probability signals against forward S&P 500 returns.
    Uses median and P75 crash probability as signals.
    """
    print("\n=== Running backtest ===")

    primary_tag = f'{PRIMARY_THRESHOLD}pct_{PRIMARY_HORIZON}'
    median_col = f'CRASH_PROB_MEDIAN_{primary_tag}'
    p75_col = f'CRASH_PROB_P75_{primary_tag}'

    if median_col not in aggregate_probs.columns:
        print(f"  [WARN] Primary signal {median_col} not found, skipping backtest")
        return aggregate_probs

    sp = sp500_prices.reindex(aggregate_probs.index, method='ffill')
    sp_daily_ret = sp.pct_change()

    # Forward returns
    bt = aggregate_probs.copy()
    horizons = {'1M': 21, '3M': 63, '6M': 126, '12M': 252}
    for label, days in horizons.items():
        bt[f'FWD_{label}'] = (sp.shift(-days) / sp - 1) * 100

    # Forward 6M max drawdown
    fwd_dd = compute_forward_max_drawdown(sp500_prices, 126, aggregate_probs.index)
    bt['FWD_MAX_DD_6M'] = fwd_dd

    # --- Analysis by quintile of median P(crash) ---
    signal = bt[median_col].dropna()
    if len(signal) > 500:
        bt_valid = bt.loc[signal.index].copy()
        bt_valid['QUINTILE'] = pd.qcut(bt_valid[median_col], 5,
                                        labels=['Q1_Low', 'Q2', 'Q3', 'Q4', 'Q5_High'],
                                        duplicates='drop')

        print(f"\n--- Forward S&P 500 Returns by Median P(crash) Quintile ---")
        print(f"  Signal: {median_col}")
        for col in [c for c in bt_valid.columns if c.startswith('FWD_')]:
            print(f"\n  {col}:")
            summary = bt_valid.groupby('QUINTILE', observed=False)[col].agg(
                ['mean', 'median', 'std', 'count'])
            summary = summary.round(2)
            for q in summary.index:
                row = summary.loc[q]
                print(f"    {q:10s}  mean={row['mean']:+7.2f}%  median={row['median']:+7.2f}%  "
                      f"std={row['std']:6.2f}%  n={int(row['count'])}")

    # --- Realized crash rate by predicted probability bucket ---
    print(f"\n--- Calibration: Predicted vs Realized Crash Rate ---")
    print(f"  (DD > {PRIMARY_THRESHOLD}% within {PRIMARY_HORIZON})")
    actual_crash = (fwd_dd < -PRIMARY_THRESHOLD).astype(float)
    for prob_thresh in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        mask = bt[median_col] >= prob_thresh
        if mask.sum() < 50:
            continue
        realized = actual_crash[mask].mean() * 100
        n = mask.sum()
        print(f"  P(crash) >= {prob_thresh:.0%}:  n={n:>5d}  realized={realized:5.1f}%  "
              f"avg predicted={bt.loc[mask, median_col].mean():.1%}")

    # --- Correlation analysis ---
    print(f"\n--- Correlation: P(crash) vs Forward Returns ---")
    for signal_col in [median_col, p75_col]:
        if signal_col not in bt.columns:
            continue
        print(f"\n  {signal_col}:")
        for col in [c for c in bt.columns if c.startswith('FWD_')]:
            valid = bt[signal_col].notna() & bt[col].notna()
            if valid.sum() < 100:
                continue
            corr = bt.loc[valid, signal_col].corr(bt.loc[valid, col])
            print(f"    {col:20s}  r = {corr:+.4f}")

    # --- Strategy backtest ---
    valid_mask = bt[median_col].notna()
    sp_bh = sp_daily_ret[valid_mask].dropna()

    if tbill_rate is not None:
        rf_daily = tbill_rate.reindex(sp_daily_ret.index, method='ffill') / 100 / 252
    else:
        rf_daily = pd.Series(0.02 / 252, index=sp_daily_ret.index)

    def run_strategy(signal_series, exit_thresh, entry_thresh, label=''):
        """Exit when P(crash) >= exit_thresh, re-enter when < entry_thresh."""
        sig_lag = signal_series.shift(1)
        in_market = pd.Series(True, index=sig_lag.index)
        currently_in = True
        for i in range(len(sig_lag)):
            val = sig_lag.iloc[i]
            if pd.isna(val):
                in_market.iloc[i] = currently_in
                continue
            if currently_in and val >= exit_thresh:
                currently_in = False
            elif not currently_in and val < entry_thresh:
                currently_in = True
            in_market.iloc[i] = currently_in

        sig = in_market.reindex(sp_daily_ret.index)
        sig = sig[valid_mask].dropna()
        common_idx = sig.index.intersection(sp_bh.index)
        sig = sig.loc[common_idx]
        ret = sp_daily_ret.loc[common_idx]

        rf_aligned = rf_daily.reindex(common_idx, method='ffill').fillna(0.02/252)
        strat_ret = ret.where(sig, rf_aligned)
        ann_ret = strat_ret.mean() * 252 * 100
        ann_vol = strat_ret.std() * np.sqrt(252) * 100
        avg_rf = rf_aligned.mean() * 252 * 100
        sharpe = (ann_ret - avg_rf) / ann_vol if ann_vol > 0 else 0
        cum = (1 + strat_ret).cumprod()
        maxdd = ((cum / cum.cummax()) - 1).min() * 100
        pct_in = sig.mean() * 100
        n_trades = (sig != sig.shift(1)).sum()
        total_ret = (cum.iloc[-1] - 1) * 100

        return {
            'label': label, 'ann_ret': ann_ret, 'ann_vol': ann_vol,
            'sharpe': sharpe, 'maxdd': maxdd, 'pct_in': pct_in,
            'n_trades': n_trades, 'total_ret': total_ret, 'cum': cum,
            'signal': sig
        }

    def print_strategy_row(r):
        print(f"  {r['label']:>35s}  {r['ann_ret']:+7.2f}%  {r['ann_vol']:7.2f}%  {r['sharpe']:+6.3f}  "
              f"{r['maxdd']:+7.2f}%  {r['pct_in']:6.1f}%  {r['n_trades']:>4d}  {r['total_ret']:+10.1f}%")

    header = (f"  {'Strategy':>35s}  {'Ann.Ret':>8s}  {'Ann.Vol':>8s}  {'Sharpe':>7s}  "
              f"{'MaxDD':>8s}  {'%InMkt':>7s}  {'#Tr':>4s}  {'TotalRet':>10s}")
    sep = "  " + "-" * 107

    # Buy & Hold baseline
    print(f"\n--- Strategy Backtest: P(DD>{PRIMARY_THRESHOLD}% in {PRIMARY_HORIZON}) ---")
    print(header)
    print(sep)
    bh_cum = (1 + sp_bh).cumprod()
    bh_total = (bh_cum.iloc[-1] - 1) * 100
    bh_ann_ret = sp_bh.mean() * 252 * 100
    bh_ann_vol = sp_bh.std() * np.sqrt(252) * 100
    bh_avg_rf = rf_daily.reindex(sp_bh.index, method='ffill').fillna(0.02/252).mean() * 252 * 100
    bh_sharpe = (bh_ann_ret - bh_avg_rf) / bh_ann_vol if bh_ann_vol > 0 else 0
    print(f"  {'Buy & Hold':>35s}  {bh_ann_ret:+7.2f}%  {bh_ann_vol:7.2f}%  {bh_sharpe:+6.3f}  "
          f"{((bh_cum/bh_cum.cummax())-1).min()*100:+7.2f}%  {'100.0%':>7s}  {'  --':>4s}  {bh_total:+10.1f}%")

    # Grid search over signal source and thresholds
    all_results = []
    for signal_col, signal_name in [(median_col, 'Median'), (p75_col, 'P75')]:
        if signal_col not in bt.columns:
            continue
        signal_series = bt[signal_col]
        for exit_t in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
            for entry_t in [0.05, 0.08, 0.10, 0.15, 0.20]:
                if entry_t >= exit_t:
                    continue
                lbl = f'{signal_name} x>={exit_t:.0%} r<{entry_t:.0%}'
                r = run_strategy(signal_series, exit_t, entry_t, lbl)
                all_results.append(r)
            # Single threshold
            lbl = f'{signal_name} x>={exit_t:.0%}'
            r = run_strategy(signal_series, exit_t, exit_t, lbl)
            all_results.append(r)

    all_results.sort(key=lambda x: x['sharpe'], reverse=True)
    for r in all_results[:20]:
        print_strategy_row(r)

    # Also test across crash definitions
    print(f"\n--- Best Strategy per Crash Definition ---")
    print(header)
    print(sep)
    for h_name in HORIZONS:
        for threshold in CRASH_THRESHOLDS:
            tag = f'{threshold}pct_{h_name}'
            med = f'CRASH_PROB_MEDIAN_{tag}'
            if med not in bt.columns:
                continue
            best_sharpe = -999
            best_r = None
            for exit_t in [0.15, 0.20, 0.25, 0.30]:
                for entry_t in [0.05, 0.10, 0.15]:
                    if entry_t >= exit_t:
                        continue
                    r = run_strategy(bt[med], exit_t, entry_t,
                                     f'DD>{threshold}%/{h_name} x>={exit_t:.0%} r<{entry_t:.0%}')
                    if r['sharpe'] > best_sharpe:
                        best_sharpe = r['sharpe']
                        best_r = r
            if best_r:
                print_strategy_row(best_r)

    return bt


def create_charts(bt, sp500_prices, output_dir):
    """Create backtest visualization charts."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print("\n[WARN] matplotlib not installed, skipping charts")
        return

    print("\n=== Creating charts ===")

    primary_tag = f'{PRIMARY_THRESHOLD}pct_{PRIMARY_HORIZON}'
    median_col = f'CRASH_PROB_MEDIAN_{primary_tag}'
    p75_col = f'CRASH_PROB_P75_{primary_tag}'

    fig, axes = plt.subplots(4, 1, figsize=(16, 20), sharex=True,
                              gridspec_kw={'height_ratios': [2, 1.5, 1, 1]})

    # Panel 1: S&P 500 price
    ax1 = axes[0]
    sp = sp500_prices.reindex(bt.index, method='ffill')
    ax1.semilogy(sp.index, sp.values, 'k-', linewidth=0.8)
    ax1.set_ylabel('S&P 500 (log scale)')
    ax1.set_title(f'Crash Probability Model — P(DD>{PRIMARY_THRESHOLD}% in {PRIMARY_HORIZON})',
                  fontsize=14, fontweight='bold')

    # Shade high-probability periods
    if median_col in bt.columns:
        high_prob = bt[median_col] >= 0.25
        for start, end in _get_periods(high_prob):
            ax1.axvspan(start, end, alpha=0.3, color='red')
        ax1.legend(['S&P 500', 'P(crash) >= 25%'], loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Median crash probability
    ax2 = axes[1]
    if median_col in bt.columns:
        med = bt[median_col] * 100  # convert to percentage
        ax2.fill_between(med.index, 0, med.values, alpha=0.5,
                         color='orangered', label='Median P(crash)')
        if p75_col in bt.columns:
            p75 = bt[p75_col] * 100
            ax2.plot(p75.index, p75.values, color='darkred', linewidth=0.8,
                     alpha=0.7, label='P75 P(crash)')
        ax2.axhline(10, color='orange', linestyle='--', linewidth=0.8, label='10%')
        ax2.axhline(25, color='red', linestyle='--', linewidth=0.8, label='25%')
        ax2.axhline(40, color='darkred', linestyle='--', linewidth=0.8, label='40%')
    ax2.set_ylabel(f'P(DD>{PRIMARY_THRESHOLD}% in {PRIMARY_HORIZON}) %')
    ax2.set_ylim(0, 100)
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Number of models contributing
    n_col = f'N_MODELS_{primary_tag}'
    ax3 = axes[2]
    if n_col in bt.columns:
        ax3.fill_between(bt.index, 0, bt[n_col].values, alpha=0.5, color='steelblue')
    ax3.set_ylabel('# Models')
    ax3.grid(True, alpha=0.3)

    # Panel 4: Forward 6-month max drawdown
    if 'FWD_MAX_DD_6M' in bt.columns:
        ax4 = axes[3]
        dd = bt['FWD_MAX_DD_6M']
        ax4.fill_between(dd.index, 0, dd.values, alpha=0.5, color='purple')
        ax4.set_ylabel('Fwd 6M Max DD (%)')
        ax4.grid(True, alpha=0.3)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    axes[-1].set_xlabel('Date')

    plt.tight_layout()
    chart_path = output_dir / 'crash_index_backtest.png'
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved chart to {chart_path}")


def _get_periods(bool_series):
    """Convert a boolean series to list of (start, end) periods."""
    periods = []
    in_period = False
    start = None
    for date, val in bool_series.items():
        if val and not in_period:
            start = date
            in_period = True
        elif not val and in_period:
            periods.append((start, date))
            in_period = False
    if in_period:
        periods.append((start, bool_series.index[-1]))
    return periods


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Build Market Crash Risk Index')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: script directory)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR

    # 1. Download all data
    fred_df, yf_df, ebp_df, margin_debt, put_call, cot_df = download_all_data()

    # 2. Compute all indicators
    indicators = compute_indicators(fred_df, yf_df, ebp_df, margin_debt,
                                    put_call, cot_df)

    # 3. Percentile rank (expanding window, no look-ahead)
    percentiles, _, miss_flags = normalize_indicators(indicators)

    # 4. Get S&P 500
    sp500 = None
    if 'SP500' in yf_df.columns:
        sp500 = yf_df['SP500']
    elif 'SP500_FRED' in fred_df.columns:
        sp500 = fred_df['SP500_FRED']

    if sp500 is None:
        print("[ERROR] No S&P 500 data — cannot compute crash probabilities")
        return None

    # 5. Compute crash probabilities via per-indicator logistic regressions
    aggregate_probs, indicator_probs = compute_crash_probabilities(percentiles, sp500)

    # 6. Backtest
    tbill = fred_df.get('DGS3MO') if 'DGS3MO' in fred_df.columns else None
    bt = backtest(aggregate_probs, sp500, tbill_rate=tbill)

    # 7. Save dataset
    dataset = indicators.copy()

    # Add percentile ranks
    for col in percentiles.columns:
        dataset[f'PCT_{col}'] = percentiles[col]

    # Add per-indicator crash probs for primary definition
    primary_key = (PRIMARY_THRESHOLD, PRIMARY_HORIZON)
    if primary_key in indicator_probs:
        primary_probs = indicator_probs[primary_key]
        for col in primary_probs.columns:
            dataset[f'PROB_{col}'] = primary_probs[col]

    # Add aggregate crash probabilities
    for col in aggregate_probs.columns:
        dataset[col] = aggregate_probs[col]

    # Add miss flags
    for col in miss_flags.columns:
        dataset[col] = miss_flags[col]

    # Add forward returns from backtest
    if bt is not None:
        for col in bt.columns:
            if col.startswith('FWD_') and col not in dataset.columns:
                dataset[col] = bt[col]

    csv_path = output_dir / 'crash_index_dataset.csv'
    dataset.to_csv(csv_path, float_format='%.4f')
    print(f"\n=== Dataset saved to {csv_path} ===")
    print(f"    Shape: {dataset.shape[0]} rows x {dataset.shape[1]} columns")

    try:
        parquet_path = output_dir / 'crash_index_dataset.parquet'
        dataset.to_parquet(parquet_path)
        print(f"    Also saved as {parquet_path}")
    except Exception:
        pass

    # 8. Create charts
    create_charts(bt, sp500, output_dir)

    # 9. Print current readings
    print("\n" + "="*70)
    print("CURRENT CRASH PROBABILITY READINGS")
    print("="*70)
    latest = dataset.iloc[-1]
    print(f"\nDate: {dataset.index[-1].date()}")

    primary_tag = f'{PRIMARY_THRESHOLD}pct_{PRIMARY_HORIZON}'
    for agg in ['MEDIAN', 'P75', 'MEAN']:
        col = f'CRASH_PROB_{agg}_{primary_tag}'
        if col in dataset.columns:
            val = latest.get(col, np.nan)
            if not pd.isna(val):
                print(f"  P(DD>{PRIMARY_THRESHOLD}% in {PRIMARY_HORIZON}) [{agg}]: {val:.1%}")

    print(f"\nAll crash definitions (median P(crash)):")
    for h_name in HORIZONS:
        for threshold in CRASH_THRESHOLDS:
            tag = f'{threshold}pct_{h_name}'
            col = f'CRASH_PROB_MEDIAN_{tag}'
            if col in dataset.columns:
                val = latest.get(col, np.nan)
                if not pd.isna(val):
                    print(f"  DD>{threshold}% in {h_name}: {val:.1%}")

    # Per-indicator probabilities for primary definition
    prob_cols = [c for c in dataset.columns if c.startswith('PROB_')]
    if prob_cols:
        print(f"\nTop 10 indicators by P(crash) [DD>{PRIMARY_THRESHOLD}% in {PRIMARY_HORIZON}]:")
        current_probs = latest[prob_cols].dropna().sort_values(ascending=False).head(10)
        for col in current_probs.index:
            ind_name = col.replace('PROB_', '')
            print(f"  {ind_name:30s}  P(crash)={current_probs[col]:.1%}")

        print(f"\nBottom 10 indicators:")
        current_low = latest[prob_cols].dropna().sort_values().head(10)
        for col in current_low.index:
            ind_name = col.replace('PROB_', '')
            print(f"  {ind_name:30s}  P(crash)={current_low[col]:.1%}")

    print("\nDone!")
    return dataset


if __name__ == '__main__':
    main()
