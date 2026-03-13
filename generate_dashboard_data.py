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


# Primary crash definition
PRIMARY_THRESHOLD = 10
PRIMARY_HORIZON = '6M'


def generate_dashboard_data(data_dir=None):
    if data_dir is None:
        data_dir = Path(__file__).parent
    data_dir = Path(data_dir)

    # Load dataset
    csv_path = data_dir / 'crash_index_dataset.csv'
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    latest = df.iloc[-1]
    latest_date = df.index[-1].strftime('%Y-%m-%d')

    # Read build_metadata.json early (needed for crash correlations)
    build_metadata = None
    meta_path = data_dir / 'build_metadata.json'
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                build_metadata = json.load(f)
        except Exception:
            pass

    # --- Indicator metadata ---
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
        'Valuation / Positioning': ['CAPE', 'BUFFETT_IND', 'HH_EQUITY_ALLOC',
                                     'MARGIN_DEBT', 'MARGIN_DEBT_YOY',
                                     'COT_LEV_NET_LONG', 'COT_AM_NET_LONG',
                                     'INSIDER_SELL_BUY'],
        'Cross-Asset': ['GOLD_SP_RATIO', 'CU_AU_RATIO_INV', 'DXY'],
        'Monetary Policy': ['FED_FUNDS', 'RRP_YOY_INV', 'SLOOS'],
    }

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
        'NFCI': 'Chicago Fed NFCI',
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
        'BUFFETT_IND': 'Buffett Indicator (Mkt Cap/GDP)',
        'HH_EQUITY_ALLOC': 'Household Equity Allocation',
        'CAPE': 'Shiller CAPE (PE10)',
        'MARGIN_DEBT': 'Margin Debt (% of M2)',
        'MARGIN_DEBT_YOY': 'Margin Debt YoY Growth',
        'COT_LEV_NET_LONG': 'CFTC Leveraged Net Short',
        'COT_AM_NET_LONG': 'CFTC Asset Mgr Net Long',
        'GOLD_SP_RATIO': 'Gold / S&P 500 Ratio',
        'CU_AU_RATIO_INV': 'Copper / Gold Ratio',
        'DXY': 'US Dollar Index',
        'FED_FUNDS': 'Real Fed Funds Rate',
        'RRP_YOY_INV': 'Fed Reverse Repo (ON RRP)',
        'SLOOS': 'Bank Lending Standards',
        'INSIDER_SELL_BUY': 'Insider Sell/Buy Ratio',
    }

    DESCRIPTIONS = {
        'VIX': {
            'what': 'The CBOE Volatility Index measures the market\'s expectation of 30-day S&P 500 volatility, derived from option prices. Often called the "fear gauge."',
            'calc': 'Model-free variance swap formula aggregating prices of OTM S&P 500 puts and calls across two expiration dates bracketing 30 days: VIX = 100 * sqrt((2/T) * sum[(dK/K^2) * e^(rT) * Q(K)] - (1/T)*(F/K0 - 1)^2)',
            'source': 'CBOE (FRED: VIXCLS)',
            'thresholds': {'green': '< 15', 'yellow': '20-30', 'red': '> 30'},
            'direction': 'Higher = more fear = more danger',
        },
        'VVIX': {
            'what': 'The CBOE VVIX measures the expected volatility of the VIX itself -- the "volatility of volatility." Spikes when options markets are uncertain about future fear levels.',
            'calc': 'Same model-free variance swap methodology as VIX, but applied to VIX options instead of S&P 500 options.',
            'source': 'CBOE (yfinance: ^VVIX)',
            'thresholds': {'green': '< 85', 'yellow': '100-120', 'red': '> 120'},
            'direction': 'Higher = more uncertainty about volatility = danger',
        },
        'SKEW': {
            'what': 'The CBOE SKEW Index measures perceived tail risk by quantifying the skewness of S&P 500 returns implied by option prices. Higher values mean the market prices a greater probability of extreme left-tail moves.',
            'calc': 'SKEW = 100 - 10 * S, where S is the implied third moment (skewness) extracted from OTM option prices. Baseline of 100 = log-normal returns (no skew).',
            'source': 'CBOE (yfinance: ^SKEW)',
            'thresholds': {'green': '< 115', 'yellow': '115-135', 'red': '> 135'},
            'direction': 'Higher = more tail risk priced in',
        },
        'REALIZED_VOL': {
            'what': 'The annualized standard deviation of daily S&P 500 log-returns over the past 22 trading days. Measures actual (not implied) market turbulence.',
            'calc': 'RV = std(ln(P_t / P_{t-1}), window=22) * sqrt(252) * 100, where P is the S&P 500 daily close.',
            'source': 'Derived from S&P 500 prices (yfinance: ^GSPC)',
            'thresholds': {'green': '< 12%', 'yellow': '15-25%', 'red': '> 25%'},
            'direction': 'Higher = more turbulence = danger',
        },
        'VRP_INV': {
            'what': 'The Variance Risk Premium is the difference between implied variance (VIX^2) and realized variance. A positive VRP is normal (investors pay a premium for protection). When VRP turns negative, realized volatility has exceeded what was priced in -- the market was surprised.',
            'calc': 'VRP = VIX^2 - RealizedVol(22d)^2. Positive = normal risk premium. Negative = realized > implied = danger.',
            'source': 'Derived from VIX and S&P 500 returns',
            'thresholds': {'green': '100-300', 'yellow': '< 50 or > 400', 'red': '< 0'},
            'direction': 'Lower/negative = more danger (market surprised by vol)',
        },
        'VIX_RV_SPREAD_INV': {
            'what': 'The spread between VIX and realized volatility. A positive spread means implied vol exceeds realized (normal). A negative spread means realized vol has spiked above what was expected.',
            'calc': 'Spread = VIX - RealizedVol(22d). Positive = cushion. Negative = market surprised.',
            'source': 'Derived from VIX and S&P 500 returns',
            'thresholds': {'green': '2-6 pts', 'yellow': '> 10 or < 0', 'red': '< -5'},
            'direction': 'Lower/negative = more danger',
        },
        'HY_OAS': {
            'what': 'The option-adjusted spread on high-yield (junk) corporate bonds over Treasuries. Measures the extra yield investors demand to hold risky corporate debt. Widens sharply when credit risk rises.',
            'calc': 'OAS = HY bond yield - comparable-maturity Treasury yield, adjusted for embedded options using interest rate models. Reported in basis points.',
            'source': 'ICE BofA US High Yield Index (FRED: BAMLH0A0HYM2)',
            'thresholds': {'green': '< 350 bps', 'yellow': '400-600 bps', 'red': '> 600 bps'},
            'direction': 'Higher = more credit risk = danger',
        },
        'IG_OAS': {
            'what': 'The option-adjusted spread on investment-grade corporate bonds. Widens when even high-quality corporate borrowers face stress.',
            'calc': 'Same OAS methodology as HY but applied to investment-grade (BBB- and above) bonds.',
            'source': 'ICE BofA US Corporate Index (FRED: BAMLC0A0CM)',
            'thresholds': {'green': '< 100 bps', 'yellow': '120-200 bps', 'red': '> 200 bps'},
            'direction': 'Higher = more credit risk = danger',
        },
        'BBB_OAS': {
            'what': 'The spread on BBB-rated bonds -- the lowest investment-grade tier. These bonds are one notch above junk, making them sensitive to downgrade risk during downturns.',
            'calc': 'OAS on BBB-rated corporate bonds vs. Treasuries.',
            'source': 'ICE BofA BBB US Corporate Index (FRED: BAMLC0A4CBBB)',
            'thresholds': {'green': '< 130 bps', 'yellow': '150-250 bps', 'red': '> 250 bps'},
            'direction': 'Higher = more credit risk = danger',
        },
        'CP_SPREAD': {
            'what': 'The spread between 3-month financial commercial paper and 3-month Treasury bills. Measures short-term funding stress for financial institutions.',
            'calc': 'CP Spread = 3-Month Financial CP Rate - 3-Month T-Bill Rate (both annualized yields).',
            'source': 'FRED: DCPF3M (CP rate), DTB3 (T-bill rate)',
            'thresholds': {'green': '< 50 bps', 'yellow': '75-150 bps', 'red': '> 150 bps'},
            'direction': 'Higher = more funding stress = danger',
        },
        'HY_OAS_MOMENTUM': {
            'what': 'The 21-day change in high-yield credit spreads. Rapidly widening spreads signal deteriorating credit conditions and often precede broader market stress.',
            'calc': 'HY_OAS_Momentum = HY_OAS(t) - HY_OAS(t-21). Positive = spreads widening.',
            'source': 'Derived from ICE BofA HY OAS',
            'thresholds': {'green': '< 25 bps', 'yellow': '50-100 bps', 'red': '> 100 bps'},
            'direction': 'Higher (widening) = danger',
        },
        'EBP': {
            'what': 'The Excess Bond Premium from Gilchrist & Zakrajsek (2012) decomposes corporate bond spreads into a component explained by firm fundamentals and a residual (the EBP) that captures investor risk appetite. Elevated EBP = risk aversion.',
            'calc': 'Residual from regressing corporate bond spreads on firm-level default risk measures (distance-to-default, leverage, earnings). Requires firm-level bond data.',
            'source': 'Federal Reserve Board (published monthly)',
            'thresholds': {'green': '< 0%', 'yellow': '0-0.5%', 'red': '> 0.5%'},
            'direction': 'Higher = more risk aversion = danger',
        },
        'NFCI': {
            'what': 'The Chicago Fed National Financial Conditions Index aggregates 105 measures of financial activity (money markets, debt, equity, banking, shadow banking). Zero = long-run average conditions.',
            'calc': 'Principal component analysis on 105 weekly financial indicators, normalized so that zero = average and positive = tighter-than-average conditions.',
            'source': 'Chicago Fed (FRED: NFCI)',
            'thresholds': {'green': '< -0.5', 'yellow': '0 to 0.5', 'red': '> 0.5'},
            'direction': 'Higher (positive) = tighter financial conditions = danger',
        },
        'KCFSI': {
            'what': 'The Kansas City Fed Financial Stress Index measures the level of stress in U.S. financial markets using 11 variables including credit spreads, equity vol, and interbank lending.',
            'calc': 'Principal component analysis on 11 financial market variables. Zero = long-run average; positive = above-average stress.',
            'source': 'Kansas City Fed (FRED: KCFSI)',
            'thresholds': {'green': '< 0', 'yellow': '0-1.0', 'red': '> 1.0'},
            'direction': 'Higher = more stress = danger',
        },
        'YC_10Y2Y_INV': {
            'what': 'The spread between 10-year and 2-year Treasury yields. An inverted curve (negative spread) has preceded every U.S. recession since 1970, typically 12-18 months ahead.',
            'calc': '10Y-2Y Spread = 10-Year Treasury Yield - 2-Year Treasury Yield. Negative = inverted.',
            'source': 'FRED: T10Y2Y (or DGS10 minus DGS2)',
            'thresholds': {'green': '> 100 bps', 'yellow': '0-50 bps', 'red': '< 0 bps (inverted)'},
            'direction': 'Lower/negative = more danger (recession signal)',
        },
        'YC_10Y3M_INV': {
            'what': 'The spread between 10-year and 3-month Treasury yields. The NY Fed\'s preferred yield curve measure for recession forecasting. Arguably more reliable than 10Y-2Y.',
            'calc': '10Y-3M Spread = 10-Year Treasury Yield - 3-Month Treasury Yield.',
            'source': 'FRED: T10Y3M (or DGS10 minus DGS3MO)',
            'thresholds': {'green': '> 150 bps', 'yellow': '0-50 bps', 'red': '< 0 bps (inverted)'},
            'direction': 'Lower/negative = more danger',
        },
        'FFR_10Y_INV': {
            'what': 'The spread between the Fed Funds Rate and the 10-year Treasury yield. When the policy rate exceeds the long rate, monetary policy is restrictive -- historically associated with recession.',
            'calc': 'FFR-10Y Spread = Effective Federal Funds Rate - 10-Year Treasury Yield. Positive = policy rate above long rate.',
            'source': 'FRED: FEDFUNDS minus DGS10',
            'thresholds': {'green': '< -100 bps', 'yellow': '-50 to +50 bps', 'red': '> 0 bps'},
            'direction': 'Higher (positive) = restrictive policy = danger',
        },
        'NYFED_RECESS_PROB': {
            'what': 'The NY Fed\'s estimated probability of recession within the next 12 months, based on the Estrella-Mishkin probit model using the yield curve spread.',
            'calc': 'Published by the NY Fed. Based on the Estrella-Mishkin (1998) probit model: P(recession in 12M) = Phi(alpha + beta * Spread_10Y3M). The NY Fed periodically re-estimates the coefficients.',
            'source': 'NY Fed via FRED: RECPROUSM156N (published monthly)',
            'thresholds': {'green': '< 10%', 'yellow': '10-30%', 'red': '> 30%'},
            'direction': 'Higher = more recession risk = danger',
        },
        'INIT_CLAIMS': {
            'what': 'Weekly initial unemployment insurance claims (4-week average), scaled by the civilian labor force. Rising claims signal deteriorating labor market conditions.',
            'calc': '(4-Week Average Initial Claims) / (Civilian Labor Force * 1000) * 100. Expressed as claims per 100 workers.',
            'source': 'DOL via FRED: IC4WSA (4-wk avg), CLF16OV (labor force)',
            'thresholds': {'green': '< 0.15%', 'yellow': '0.15-0.20%', 'red': '> 0.20%'},
            'direction': 'Higher = more layoffs = danger',
        },
        'SAHM': {
            'what': 'Claudia Sahm\'s real-time recession indicator. When the 3-month average unemployment rate rises 0.50 percentage points above its 12-month low, a recession has begun. Every trigger since 1970 has correctly identified recession.',
            'calc': 'Sahm Rule = (3-month avg U-rate) - (min U-rate over prior 12 months). Trigger at >= 0.50 pp.',
            'source': 'FRED: SAHMREALTIME',
            'thresholds': {'green': '< 0.20 pp', 'yellow': '0.30-0.50 pp', 'red': '>= 0.50 pp'},
            'direction': 'Higher = recession signal = danger',
        },
        'UNRATE': {
            'what': 'The U.S. civilian unemployment rate from the Bureau of Labor Statistics. A lagging indicator of recessions but important for confirming economic weakness.',
            'calc': '(Number of unemployed / Civilian labor force) * 100. Based on the Current Population Survey (CPS), ~60,000 households monthly.',
            'source': 'BLS via FRED: UNRATE',
            'thresholds': {'green': '< 4.0%', 'yellow': '4.5-5.5%', 'red': '> 5.5%'},
            'direction': 'Higher = weaker labor market = danger',
        },
        'NFP_MOM_INV': {
            'what': 'The 3-month rolling average of monthly changes in nonfarm payrolls. Smooths noisy monthly jobs reports. Declining momentum signals the labor market is losing steam.',
            'calc': 'NFP_Mom = rolling_mean(diff(PAYEMS), 3 months). PAYEMS is total nonfarm payrolls in thousands.',
            'source': 'BLS via FRED: PAYEMS',
            'thresholds': {'green': '> 200K', 'yellow': '50-100K', 'red': '< 50K'},
            'direction': 'Lower = weaker job growth = danger',
        },
        'DGORDER_YOY_INV': {
            'what': 'Year-over-year growth in new orders for durable goods (goods lasting 3+ years). A leading indicator of business investment and economic health.',
            'calc': 'YoY% = (DGORDER(t) / DGORDER(t-12) - 1) * 100. DGORDER is monthly total durable goods orders ($M, seasonally adjusted).',
            'source': 'Census Bureau via FRED: DGORDER',
            'thresholds': {'green': '> +5%', 'yellow': '-5% to 0%', 'red': '< -5%'},
            'direction': 'Lower/negative = weakening demand = danger',
        },
        'INDPRO_YOY_INV': {
            'what': 'Year-over-year growth in the Federal Reserve\'s Industrial Production Index, measuring output of manufacturing, mining, and utilities. Negative growth has historically accompanied every recession.',
            'calc': 'YoY% = (INDPRO(t) / INDPRO(t-12) - 1) * 100. INDPRO is an index (2017=100).',
            'source': 'Federal Reserve via FRED: INDPRO',
            'thresholds': {'green': '> +2%', 'yellow': '-2% to 0%', 'red': '< -2%'},
            'direction': 'Lower/negative = contraction = danger',
        },
        'PERMIT_YOY_INV': {
            'what': 'Year-over-year growth in new housing building permits. Housing is a leading sector -- permit declines of 20%+ have preceded every recession since 1960.',
            'calc': 'YoY% = (PERMIT(t) / PERMIT(t-12) - 1) * 100. PERMIT is monthly new private housing units authorized (thousands, SAAR).',
            'source': 'Census Bureau via FRED: PERMIT',
            'thresholds': {'green': '> 0%', 'yellow': '-10% to -20%', 'red': '< -20%'},
            'direction': 'Lower/negative = housing weakness = danger',
        },
        'PHILLY_MFG_INV': {
            'what': 'The Philadelphia Fed Manufacturing Business Outlook Survey diffusion index. A reading above 0 indicates expansion. One of the oldest and most-watched regional manufacturing surveys.',
            'calc': 'Diffusion Index = (% reporting increase) - (% reporting decrease) for general business activity. Range roughly -40 to +50.',
            'source': 'Philadelphia Fed via FRED: GACDFSA066MSFRBPHI',
            'thresholds': {'green': '> +10', 'yellow': '-10 to 0', 'red': '< -10'},
            'direction': 'Lower/negative = contraction = danger',
        },
        'UMICH_INV': {
            'what': 'The University of Michigan Consumer Sentiment Index. Based on a monthly survey of ~500 households about their financial conditions and economic outlook. A leading indicator of consumer spending.',
            'calc': 'Composite index from 5 survey questions about personal finances (current and expected) and business conditions (short-run and long-run), plus buying conditions. Base period 1966 = 100.',
            'source': 'University of Michigan via FRED: UMCSENT',
            'thresholds': {'green': '> 85', 'yellow': '70-80', 'red': '< 70'},
            'direction': 'Lower = pessimistic consumers = danger',
        },
        'M2_GROWTH_INV': {
            'what': 'Year-over-year growth in M2 money supply (cash, checking, savings, small CDs, money market funds). Negative M2 growth is extremely rare and signals severe liquidity contraction.',
            'calc': 'YoY% = (M2(t) / M2(t-12) - 1) * 100. M2 is in billions of dollars, seasonally adjusted.',
            'source': 'Federal Reserve via FRED: M2SL',
            'thresholds': {'green': '+3% to +6%', 'yellow': '0-3%', 'red': '< 0% (contraction)'},
            'direction': 'Lower/negative = liquidity contraction = danger',
        },
        'SP500_VS_200DMA_INV': {
            'what': 'The percentage distance of the S&P 500 from its 200-day moving average. A widely used trend indicator -- below the 200-DMA is considered bearish.',
            'calc': '% vs 200-DMA = (S&P500 / SMA_200 - 1) * 100. Positive = above trend. Negative = below trend.',
            'source': 'Derived from S&P 500 daily closes (yfinance: ^GSPC)',
            'thresholds': {'green': '> +5%', 'yellow': '-5% to 0%', 'red': '< -5%'},
            'direction': 'Lower/negative = below trend = danger',
        },
        'DEATH_CROSS': {
            'what': 'A binary technical signal: 1 when the 50-day moving average crosses below the 200-day moving average. Widely cited as a bearish signal in financial media.',
            'calc': 'Death Cross = 1 if SMA_50 < SMA_200, else 0. (Stored as 0 or 100 in the model.)',
            'source': 'Derived from S&P 500 daily closes',
            'thresholds': {'green': 'Not active (0)', 'yellow': 'Converging', 'red': 'Active (1)'},
            'direction': 'Active = bearish trend = danger',
        },
        'RSI_14': {
            'what': 'The Relative Strength Index (14-day), a momentum oscillator measuring the speed and magnitude of recent price changes. Above 70 = overbought (vulnerable to correction), below 30 = oversold.',
            'calc': 'RSI = 100 - 100/(1+RS), where RS = EMA(gains, 14) / EMA(losses, 14). Uses Wilder\'s smoothing (alpha = 1/14).',
            'source': 'Derived from S&P 500 daily closes',
            'thresholds': {'green': '40-60', 'yellow': '> 70 or < 30', 'red': '> 80 or < 20'},
            'direction': 'Extreme values (both high and low) = danger',
        },
        'MOMENTUM_12_1_INV': {
            'what': 'The Jegadeesh-Titman 12-1 month momentum: the S&P 500 return from 12 months ago to 1 month ago, excluding the most recent month. Negative momentum has historically preceded further declines.',
            'calc': 'Mom_12_1 = (P_{t-21} / P_{t-252} - 1) * 100. Skips the last 21 trading days to avoid short-term reversal.',
            'source': 'Derived from S&P 500 daily closes',
            'thresholds': {'green': '> +10%', 'yellow': '-10% to 0%', 'red': '< -10%'},
            'direction': 'Lower/negative = weakening trend = danger',
        },
        'DRAWDOWN_1Y': {
            'what': 'The current S&P 500 drawdown from its trailing 252-day (1-year) high. Measures how far the market has fallen from its recent peak.',
            'calc': 'DD = (P_t / max(P_{t-252:t}) - 1) * 100. Always <= 0. Zero = at the 1-year high.',
            'source': 'Derived from S&P 500 daily closes',
            'thresholds': {'green': '> -5%', 'yellow': '-10% to -15%', 'red': '< -20%'},
            'direction': 'More negative = deeper drawdown = danger',
        },
        'BUFFETT_IND': {
            'what': 'Buffett Indicator deviation from exponential trend, in standard deviations. Per currentmarketvaluation.com and Advisor Perspectives, the raw Market Cap/GDP ratio has an upward exponential trend -- the z-score removes this trend bias.',
            'calc': 'Fit exponential trend to Market Cap/GDP ratio, then z-score = (actual - trend) / std(deviation). Positive = above trend (overvalued), negative = below trend.',
            'source': 'Wilshire 5000 (FRED: WILL5000IND) / GDP (FRED: GDP)',
            'thresholds': {'green': 'z < 0 (below trend)', 'yellow': 'z = 0.5-1.5', 'red': 'z > 1.5 (well above trend)'},
            'direction': 'Higher z-score = more overvalued vs trend = danger',
        },
        'CAPE': {
            'what': 'Excess CAPE Yield (ECY): the Shiller earnings yield minus the real 10-year Treasury yield. Shiller\'s own improvement to raw CAPE -- accounts for the interest rate regime. Low ECY means stocks are expensive relative to bonds.',
            'calc': 'ECY = (1/CAPE)*100 - Real 10Y Yield. Real 10Y from TIPS (DFII10) or nominal 10Y minus CPI YoY. Stored inverted: higher value = more danger (lower ECY = more expensive).',
            'source': 'Shiller CAPE (Yale), TIPS or DGS10-CPI (FRED)',
            'thresholds': {'green': 'ECY > 4%', 'yellow': 'ECY 2-3%', 'red': 'ECY < 2%'},
            'direction': 'Lower ECY (higher stored value) = stocks expensive vs bonds = danger',
        },
        'HH_EQUITY_ALLOC': {
            'what': 'Household equity allocation from the Fed\'s Flow of Funds (Z.1) report. Measures the share of household financial assets allocated to equities. High allocation = late-cycle euphoria.',
            'calc': 'Direct percentage from Fed Z.1 table. Represents corporate equities as a share of total household financial assets.',
            'source': 'Federal Reserve Flow of Funds (FRED: BOGZ1FL153064486Q)',
            'thresholds': {'green': '< 35%', 'yellow': '40-45%', 'red': '> 45%'},
            'direction': 'Higher = more crowded into stocks = danger',
        },
        'MARGIN_DEBT': {
            'what': 'FINRA margin debt as a percentage of M2 money supply. Normalizes for monetary expansion over time, showing how much speculative leverage exists relative to the money in circulation.',
            'calc': 'Margin Debt % = (FINRA Debit Balances in $B) / (M2 Money Supply in $B) * 100.',
            'source': 'FINRA margin statistics (monthly), M2 (FRED: M2SL)',
            'thresholds': {'green': 'Below average', 'yellow': 'Above average', 'red': 'Near highs'},
            'direction': 'Higher = more speculative leverage relative to money supply = danger',
        },
        'MARGIN_DEBT_YOY': {
            'what': 'Year-over-year growth in raw FINRA margin debt (debit balances). Rapid leverage growth signals accelerating speculation. Sharp declines can signal forced deleveraging.',
            'calc': 'YoY% = (MarginDebt(t) / MarginDebt(t-12) - 1) * 100. Computed on raw monthly dollar values.',
            'source': 'FINRA margin statistics (monthly)',
            'thresholds': {'green': '+5% to +15%', 'yellow': '> +25% or < -10%', 'red': '> +30% or < -15%'},
            'direction': 'Extreme growth or decline = danger',
        },
        'COT_LEV_NET_LONG': {
            'what': 'Net positioning of leveraged money (hedge funds, CTAs) in S&P 500 futures from the CFTC Commitments of Traders report. When "smart money" is heavily net short, it can signal informed bearishness.',
            'calc': 'Net Long = %OI Long (Leveraged) - %OI Short (Leveraged). Expressed as percentage of total open interest.',
            'source': 'CFTC Commitments of Traders, Financial Futures (weekly)',
            'thresholds': {'green': 'Moderate net long', 'yellow': 'Z-score > 1 or < -1', 'red': 'Z-score > 2 (extreme net short)'},
            'direction': 'More net short by smart money = danger',
        },
        'COT_AM_NET_LONG': {
            'what': 'Net positioning of asset managers (pensions, mutual funds, insurance) in S&P 500 futures. Extreme net long positioning by "slow money" can signal crowding and vulnerability to unwind.',
            'calc': 'Net Long = %OI Long (Asset Mgr) - %OI Short (Asset Mgr). Expressed as percentage of total open interest.',
            'source': 'CFTC Commitments of Traders, Financial Futures (weekly)',
            'thresholds': {'green': 'Moderate', 'yellow': 'Z-score > 1', 'red': 'Z-score > 2 (extreme crowding)'},
            'direction': 'Higher net long (crowding) = danger',
        },
        'GOLD_SP_RATIO': {
            'what': 'The ratio of gold price to S&P 500. Rising ratio signals flight from risk assets to safe havens. Gold outperforms equities during stress periods.',
            'calc': 'Gold/SPX Ratio = Gold Futures Price / S&P 500 Index Level.',
            'source': 'Gold futures (yfinance: GC=F), S&P 500 (yfinance: ^GSPC)',
            'thresholds': {'green': 'Low (< 0.4)', 'yellow': '0.4-0.6', 'red': '> 0.6 (flight to safety)'},
            'direction': 'Higher = more flight to safety = danger',
        },
        'CU_AU_RATIO_INV': {
            'what': 'The copper-to-gold ratio. Copper ("Dr. Copper") is pro-cyclical (industrial demand), while gold is counter-cyclical (safe haven). A declining ratio signals deteriorating growth expectations.',
            'calc': 'Cu/Au Ratio = Copper Futures Price / Gold Futures Price.',
            'source': 'Copper (yfinance: HG=F), Gold (yfinance: GC=F)',
            'thresholds': {'green': 'Rising', 'yellow': 'Flat', 'red': 'Falling for 3+ months'},
            'direction': 'Lower = risk-off / weak growth = danger',
        },
        'DXY': {
            'what': 'The US Dollar Index measures the value of the dollar against a basket of 6 major currencies (EUR 57.6%, JPY 13.6%, GBP 11.9%, CAD 9.1%, SEK 4.2%, CHF 3.6%). A strong dollar tightens global financial conditions.',
            'calc': 'Geometric weighted average of USD exchange rates vs. 6 currencies: DXY = 50.14348112 * prod(FX_i ^ w_i). Base March 1973 = 100.',
            'source': 'ICE Futures (yfinance: DX-Y.NYB)',
            'thresholds': {'green': '90-100', 'yellow': '> 105 or < 85', 'red': '> 110'},
            'direction': 'Higher = stronger dollar = global tightening = danger',
        },
        'FED_FUNDS': {
            'what': 'The real federal funds rate: nominal fed funds rate minus year-over-year CPI inflation. Positive = restrictive monetary policy (rates above inflation). Negative = accommodative (rates below inflation).',
            'calc': 'Real FF Rate = Effective Federal Funds Rate - CPI YoY%. CPI YoY = (CPI(t)/CPI(t-12) - 1) * 100.',
            'source': 'Federal Reserve via FRED: FEDFUNDS, CPIAUCSL',
            'thresholds': {'green': '< 0% (accommodative)', 'yellow': '0-2% (neutral)', 'red': '> 2% (restrictive)'},
            'direction': 'Higher = tighter real monetary conditions = danger',
        },
        'RRP_YOY_INV': {
            'what': 'Daily outstanding balance in the Fed\'s Overnight Reverse Repo Facility (ON RRP). The ON RRP absorbs excess liquidity — high levels mean ample reserves; low levels mean tighter conditions.',
            'calc': 'Raw ON RRP balance in $B. Percentile rank normalizes across regimes. (YoY% change abandoned because it produces infinities when base was near zero pre-2021.)',
            'source': 'NY Fed via FRED: RRPONTSYD',
            'thresholds': {'green': 'High (ample liquidity)', 'yellow': 'Declining', 'red': 'Near zero (drained)'},
            'direction': 'Lower = less excess liquidity = tighter conditions',
        },
        'SLOOS': {
            'what': 'The Federal Reserve\'s Senior Loan Officer Opinion Survey on bank lending practices. Net percentage of banks tightening standards on C&I loans. Sustained tightening has preceded every recession since 1990.',
            'calc': 'Net % Tightening = (% banks reporting tightened standards) - (% reporting eased standards). Positive = net tightening.',
            'source': 'Federal Reserve Board quarterly survey (FRED: DRTSCILM)',
            'thresholds': {'green': '< 0% (easing)', 'yellow': '20-40% (tightening)', 'red': '> 40% (severe)'},
            'direction': 'Higher = tighter credit = danger',
        },
    }

    FREQUENCY = {
        'VIX': 'Daily', 'VVIX': 'Daily', 'SKEW': 'Daily',
        'REALIZED_VOL': 'Daily', 'VRP_INV': 'Daily', 'VIX_RV_SPREAD_INV': 'Daily',
        'HY_OAS': 'Daily', 'IG_OAS': 'Daily', 'CCC_OAS': 'Daily',
        'BBB_OAS': 'Daily', 'CP_SPREAD': 'Daily', 'HY_OAS_MOMENTUM': 'Daily',
        'EBP': 'Monthly',
        'NFCI': 'Weekly', 'KCFSI': 'Monthly',
        'YC_10Y2Y_INV': 'Daily', 'YC_10Y3M_INV': 'Daily',
        'FFR_10Y_INV': 'Daily', 'NYFED_RECESS_PROB': 'Daily',
        'INIT_CLAIMS': 'Weekly', 'SAHM': 'Monthly', 'UNRATE': 'Monthly',
        'NFP_MOM_INV': 'Monthly',
        'DGORDER_YOY_INV': 'Monthly', 'INDPRO_YOY_INV': 'Monthly',
        'PERMIT_YOY_INV': 'Monthly', 'PHILLY_MFG_INV': 'Monthly',
        'UMICH_INV': 'Monthly', 'M2_GROWTH_INV': 'Monthly',
        'SP500_VS_200DMA_INV': 'Daily', 'DEATH_CROSS': 'Daily',
        'RSI_14': 'Daily', 'MOMENTUM_12_1_INV': 'Daily', 'DRAWDOWN_1Y': 'Daily',
        'CAPE': 'Monthly', 'BUFFETT_IND': 'Quarterly', 'HH_EQUITY_ALLOC': 'Quarterly',
        'MARGIN_DEBT': 'Monthly', 'MARGIN_DEBT_YOY': 'Monthly',
        'COT_LEV_NET_LONG': 'Weekly', 'COT_AM_NET_LONG': 'Weekly',
        'GOLD_SP_RATIO': 'Daily', 'CU_AU_RATIO_INV': 'Daily',
        'DXY': 'Daily',
        'FED_FUNDS': 'Monthly', 'RRP_YOY_INV': 'Daily', 'SLOOS': 'Quarterly',
    }

    # --- Build indicator data (using full-sample percentile ranks) ---
    indicators = []

    # Pre-compute per-indicator correlation with 6M crash binary (DD > 10%)
    crash_binary = (df['FWD_MAX_DD_6M'] < -PRIMARY_THRESHOLD).astype(float) \
                   if 'FWD_MAX_DD_6M' in df.columns else None
    ind_corrs = {}
    if crash_binary is not None:
        for c in df.columns:
            if c.startswith('PCT_'):
                name = c.replace('PCT_', '')
                r = df[c].corr(crash_binary)
                if not np.isnan(r):
                    ind_corrs[name] = round(r, 4)

    for cat_name, cat_indicators in CATEGORIES.items():
        for ind_name in cat_indicators:
            pct_col = f'PCT_{ind_name}'
            if pct_col not in df.columns:
                continue

            pct_rank = latest.get(pct_col, np.nan)
            raw = latest.get(ind_name, np.nan)

            if pd.isna(pct_rank):
                continue

            # Compute staleness
            raw_series = df[ind_name].dropna()
            if len(raw_series) > 0:
                changes = raw_series.diff().ne(0)
                last_change_dates = raw_series.index[changes]
                if len(last_change_dates) > 0:
                    last_update = last_change_dates[-1].strftime('%Y-%m-%d')
                else:
                    last_update = raw_series.index[-1].strftime('%Y-%m-%d')
            else:
                last_update = 'N/A'

            desc = DESCRIPTIONS.get(ind_name, {})
            indicators.append({
                'id': ind_name,
                'name': DISPLAY_NAMES.get(ind_name, ind_name),
                'category': cat_name,
                'crash_prob': round(float(pct_rank), 1),  # percentile rank (0-100)
                'raw_value': round(float(raw), 4) if not pd.isna(raw) else None,
                'crash_corr': ind_corrs.get(ind_name, 0),  # correlation with 6M crash
                'frequency': FREQUENCY.get(ind_name, 'Unknown'),
                'last_update': last_update,
                'what': desc.get('what', ''),
                'calc': desc.get('calc', ''),
                'source': desc.get('source', ''),
                'thresholds': desc.get('thresholds', {}),
                'direction': desc.get('direction', ''),
            })

    # Sort indicators by correlation (highest positive r first = most predictive of crashes)
    indicators.sort(key=lambda x: x.get('crash_corr', 0), reverse=True)

    # --- Aggregate percentile stats across indicators ---
    pct_cols = [c for c in df.columns if c.startswith('PCT_')]

    # Read crash correlations from build metadata (computed by build_crash_index.py)
    crash_corrs = {}
    if build_metadata and 'crash_corrs' in build_metadata:
        crash_corrs = build_metadata['crash_corrs']

    # Correlation-weighted aggregate: sum(|r_i| * pct_i) / sum(|r_i|)
    # Percentiles are already flipped (by build_crash_index.py) so high = danger for all
    current_pcts = {}
    for c in pct_cols:
        val = latest.get(c)
        if not pd.isna(val):
            ind_name = c.replace('PCT_', '')
            current_pcts[ind_name] = float(val)

    if current_pcts and crash_corrs:
        weighted_sum = 0.0
        weight_sum = 0.0
        for ind_name, pct_val in current_pcts.items():
            w = abs(crash_corrs.get(ind_name, 0))
            weighted_sum += w * pct_val
            weight_sum += w
        weighted_agg = round(weighted_sum / weight_sum, 1) if weight_sum > 0 else 50.0
    else:
        weighted_agg = round(float(np.median(list(current_pcts.values()))), 1) if current_pcts else 50.0

    pct_arr = np.array(list(current_pcts.values())) if current_pcts else np.array([])
    if len(pct_arr) > 0:
        crash_prob_median = round(float(np.median(pct_arr)), 1)
        crash_prob_mean = round(float(np.mean(pct_arr)), 1)
        crash_prob_p75 = round(float(np.percentile(pct_arr, 75)), 1)
        crash_prob_p90 = round(float(np.percentile(pct_arr, 90)), 1)
    else:
        crash_prob_median = crash_prob_mean = crash_prob_p75 = crash_prob_p90 = 0
    n_models = len(current_pcts)

    # --- History: correlation-weighted aggregate over time ---
    pct_df = df[pct_cols].dropna(how='all')

    # Build weight vector aligned to pct_cols
    weights = np.array([abs(crash_corrs.get(c.replace('PCT_', ''), 0)) for c in pct_df.columns])
    if weights.sum() > 0:
        # For each date: weighted average of available percentiles (NaN-aware)
        def _weighted_agg_row(row):
            mask = row.notna()
            if mask.sum() == 0:
                return np.nan
            return (row[mask].values * weights[mask.values]).sum() / weights[mask.values].sum()
        weighted_agg_series = pct_df.apply(_weighted_agg_row, axis=1).dropna()
    else:
        weighted_agg_series = pct_df.median(axis=1).dropna()

    def _build_history(series):
        """Downsample: daily for last 5 years, monthly before that."""
        if len(series) == 0:
            return []
        five_years_ago = series.index[-1] - pd.DateOffset(years=5)
        recent = series[series.index >= five_years_ago]
        older = series[series.index < five_years_ago]
        if len(older) > 0:
            older_monthly = older.resample('MS').first().dropna()
        else:
            older_monthly = pd.Series(dtype=float)
        idx = list(older_monthly.index) + list(recent.index)
        vals = list(older_monthly.values) + list(recent.values)
        return [{'date': d.strftime('%Y-%m-%d'), 'value': round(float(v), 2)}
                for d, v in zip(idx, vals)]

    crash_prob_history = _build_history(weighted_agg_series)
    median_pct_series = pct_df.median(axis=1).dropna()
    median_history = _build_history(median_pct_series)

    # --- Category scores (median percentile within each category) ---
    category_scores = {}
    for cat_name, cat_indicators in CATEGORIES.items():
        pcts = []
        for ind_name in cat_indicators:
            pct_col = f'PCT_{ind_name}'
            if pct_col in df.columns and not pd.isna(latest.get(pct_col)):
                pcts.append(float(latest[pct_col]))
        if pcts:
            category_scores[cat_name] = round(float(np.median(pcts)), 1)

    # --- Heatmap data: percentile ranks + forward returns + realized drawdown ---
    # Monthly sampling — use BMS (business month start) to avoid dropping
    # months where the 1st falls on a weekend
    monthly_idx = df.resample('BMS').first().index
    monthly_idx = monthly_idx[monthly_idx.isin(df.index)]

    heatmap_dates = [d.strftime('%Y-%m-%d') for d in monthly_idx]

    # Forward return series (1M, 3M, 6M, 12M)
    fwd_return_series = {}
    for label in ['1M', '3M', '6M', '12M']:
        col = f'FWD_{label}'
        if col in df.columns:
            series = df[col].reindex(monthly_idx)
            fwd_return_series[label] = [round(float(v), 2) if not pd.isna(v) else None
                                        for v in series.values]

    # Forward max drawdown (6M)
    fwd_dd_col = 'FWD_MAX_DD_6M'
    if fwd_dd_col in df.columns:
        fwd_dd_monthly = df[fwd_dd_col].reindex(monthly_idx)
        heatmap_realized = [round(float(v), 2) if not pd.isna(v) else None
                           for v in fwd_dd_monthly.values]
    else:
        heatmap_realized = []

    # Per-indicator percentile rank time series (PCT_ columns)
    heatmap_indicators = []
    for cat_name, cat_inds in CATEGORIES.items():
        for ind_name in cat_inds:
            pct_col = f'PCT_{ind_name}'
            if pct_col not in df.columns:
                continue
            series = df[pct_col].reindex(monthly_idx)
            # Already 0-100 scale
            values = [round(float(v), 1) if not pd.isna(v) else None
                      for v in series.values]
            # Only include if has some data
            if any(v is not None for v in values):
                heatmap_indicators.append({
                    'id': ind_name,
                    'name': DISPLAY_NAMES.get(ind_name, ind_name),
                    'category': cat_name,
                    'values': values,
                })

    # --- Data freshness metadata ---
    market_data_age_days = (datetime.now() - df.index[-1]).days
    data_freshness = {
        'market_data_age_days': market_data_age_days,
        'is_stale': market_data_age_days > 3,
        'latest_market_date': latest_date,
        'generated_at_utc': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        'stale_sources': [],
    }

    # Use build_metadata (read earlier) for stale source info
    if build_metadata:
        data_freshness['stale_sources'] = (
            build_metadata.get('sources_stale_cache', []) +
            build_metadata.get('sources_failed', [])
        )

    # --- Assemble output ---
    dashboard = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'latest_date': latest_date,
        'weighted_agg': weighted_agg,             # headline gauge (corr-weighted)
        'crash_prob_median': crash_prob_median,    # secondary
        'crash_prob_mean': crash_prob_mean,
        'crash_prob_p75': crash_prob_p75,
        'crash_prob_p90': crash_prob_p90,
        'primary_definition': 'Correlation-weighted aggregate percentile',
        'n_models': n_models,
        'category_scores': category_scores,
        'indicators': indicators,
        'crash_prob_history': crash_prob_history,  # pct_above_80 history
        'median_history': median_history,          # median percentile history
        'data_freshness': data_freshness,
        'build_metadata': build_metadata,
        'heatmap': {
            'dates': heatmap_dates,
            'realized_dd': heatmap_realized,
            'fwd_returns': fwd_return_series,
            'indicators': heatmap_indicators,
        },
    }

    # Save
    out_path = data_dir / 'dashboard_data.json'
    with open(out_path, 'w') as f:
        json.dump(dashboard, f, indent=2)
    print(f"Dashboard data saved to {out_path}")
    print(f"  Weighted agg:    {weighted_agg:.1f}%")
    print(f"  Median pctl:     {crash_prob_median:.1f}%")
    print(f"  P75 / P90:       {crash_prob_p75:.1f}% / {crash_prob_p90:.1f}%")
    print(f"  Indicators: {len(indicators)}")
    print(f"  History points: {len(crash_prob_history)}")

    return dashboard


if __name__ == '__main__':
    generate_dashboard_data()
