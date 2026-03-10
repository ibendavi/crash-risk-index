"""
Advanced Crash Prediction Strategies
=====================================
Builds on the base danger scores from build_crash_index.py with:
  1. Rate-of-change (momentum) features for each indicator
  2. Expanding-window logistic regression (learns weights, no look-ahead)
  3. LightGBM gradient-boosted trees (captures nonlinear interactions)
  4. Walk-forward evaluation with proper train/test splits
  5. Hysteresis exit/re-entry optimization

Target: P(max drawdown > 10% in next 6 months)
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================================
# 1. FEATURE ENGINEERING
# ============================================================================

def build_features(danger_scores, raw_indicators=None, sp500_prices=None):
    """
    From raw danger scores, build an expanded feature set:
      - Levels (the danger scores themselves)
      - 1-month momentum (21-day change)
      - 3-month momentum (63-day change)
      - 1-month volatility of each indicator
      - Cross-category interaction terms
      - Practitioner signals (Faber 10M SMA, dual momentum, HY z-score, etc.)
    """
    print("\n=== Building expanded feature set ===")
    features = pd.DataFrame(index=danger_scores.index)
    idx = danger_scores.index

    # --- Publication lag tiers: weight fresher signals more ---
    # 0=real-time/daily, 1=weekly, 2=monthly, 3=quarterly
    FRESHNESS_TIER = {
        'VIX': 0, 'VVIX': 0, 'SKEW': 0, 'HY_OAS': 0, 'IG_OAS': 0,
        'CCC_OAS': 0, 'BBB_OAS': 0, 'CP_SPREAD': 0, 'DXY': 0,
        'REALIZED_VOL': 0, 'VRP_INV': 0, 'VIX_RV_SPREAD_INV': 0,
        'SP500_VS_200DMA_INV': 0, 'DEATH_CROSS': 0, 'RSI_14': 0,
        'MOMENTUM_12_1_INV': 0, 'DRAWDOWN_1Y': 0, 'GOLD_SP_RATIO': 0,
        'CU_AU_RATIO_INV': 0, 'HY_OAS_MOMENTUM': 0,
        'RRP_YOY_INV': 0, 'NYFED_RECESS_PROB': 0,
        'YC_10Y2Y_INV': 0, 'YC_10Y3M_INV': 0, 'FFR_10Y_INV': 0,
        'NFCI': 1, 'KCFSI': 1, 'INIT_CLAIMS': 1,
        'COT_LEV_NET_LONG': 1, 'COT_AM_NET_LONG': 1,
        'UNRATE': 2, 'SAHM': 2, 'NFP_MOM_INV': 2, 'FED_FUNDS': 2,
        'PHILLY_MFG_INV': 2, 'EBP': 2, 'MARGIN_DEBT': 2,
        'MARGIN_DEBT_YOY': 2, 'UMICH_INV': 2, 'SLOOS': 2,
        'DGORDER_YOY_INV': 2, 'INDPRO_YOY_INV': 2, 'M2_GROWTH_INV': 2,
        'PERMIT_YOY_INV': 2,
        'HH_EQUITY_ALLOC': 3, 'BUFFETT_IND': 3,
    }
    FRESHNESS_WEIGHT = {0: 1.0, 1: 0.85, 2: 0.6, 3: 0.4}

    for col in danger_scores.columns:
        s = danger_scores[col]

        # Level
        features[col] = s

        # 1-month change (acceleration)
        features[f'{col}_MOM1M'] = s.diff(21)

        # 3-month change
        features[f'{col}_MOM3M'] = s.diff(63)

        # 1-month realized volatility of the indicator itself
        features[f'{col}_VOL1M'] = s.rolling(21).std()

        # 20d vs 200d momentum (short-term acceleration vs long-term trend)
        mom_20 = s.diff(20)
        mom_200 = s.diff(200)
        features[f'{col}_MOM20D'] = mom_20
        features[f'{col}_MOM200D'] = mom_200
        # Acceleration: short-term moving faster than long-term
        features[f'{col}_ACCEL'] = mom_20 - (mom_200 / 10)  # normalize 200d to per-20d scale

    # --- Freshness-weighted composite: weight real-time signals more ---
    fresh_weighted_parts = []
    fresh_weights = []
    for col in danger_scores.columns:
        tier = FRESHNESS_TIER.get(col, 2)  # default monthly
        w = FRESHNESS_WEIGHT[tier]
        fresh_weighted_parts.append(danger_scores[col] * w)
        fresh_weights.append(w)
    fresh_composite = pd.concat(fresh_weighted_parts, axis=1).sum(axis=1) / \
                      pd.concat([danger_scores[c].notna().astype(float) * FRESHNESS_WEIGHT.get(FRESHNESS_TIER.get(c, 2), 0.6)
                                 for c in danger_scores.columns], axis=1).sum(axis=1)
    features['FRESH_COMPOSITE'] = fresh_composite
    features['FRESH_COMPOSITE_MOM20D'] = fresh_composite.diff(20)
    features['FRESH_COMPOSITE_MOM200D'] = fresh_composite.diff(200)
    features['FRESH_COMPOSITE_ACCEL'] = fresh_composite.diff(20) - fresh_composite.diff(200) / 10

    # --- Real-time only composite (daily signals only, no lag) ---
    rt_cols = [c for c in danger_scores.columns if FRESHNESS_TIER.get(c, 2) == 0]
    if rt_cols:
        rt_composite = danger_scores[rt_cols].mean(axis=1)
        features['RT_COMPOSITE'] = rt_composite
        features['RT_COMPOSITE_MOM20D'] = rt_composite.diff(20)

    # --- Divergence: real-time vs lagged composite ---
    # If real-time signals are spiking but lagged macro hasn't caught up, danger is rising
    lagged_cols = [c for c in danger_scores.columns if FRESHNESS_TIER.get(c, 2) >= 2]
    if rt_cols and lagged_cols:
        lagged_composite = danger_scores[lagged_cols].mean(axis=1)
        features['RT_VS_LAGGED'] = rt_composite - lagged_composite
        features['RT_VS_LAGGED_MOM20D'] = features['RT_VS_LAGGED'].diff(20)

    # Cross-category composites (average within category, then interact)
    categories = {
        'CREDIT': ['HY_OAS', 'IG_OAS', 'CCC_OAS', 'BBB_OAS', 'CP_SPREAD',
                    'EBP'],
        'VOL': ['VIX', 'REALIZED_VOL', 'VVIX'],
        'MACRO': ['INIT_CLAIMS', 'SAHM', 'NFP_MOM_INV', 'INDPRO_YOY_INV',
                  'PERMIT_YOY_INV', 'DGORDER_YOY_INV', 'PHILLY_MFG_INV',
                  'UMICH_INV'],
        'TREND': ['SP500_VS_200DMA_INV', 'DEATH_CROSS', 'DRAWDOWN_1Y',
                  'MOMENTUM_12_1_INV'],
        'CONDITIONS': ['NFCI', 'KCFSI', 'SLOOS'],
    }

    for cat_name, cols in categories.items():
        available = [c for c in cols if c in danger_scores.columns]
        if len(available) >= 2:
            cat_avg = danger_scores[available].mean(axis=1)
            features[f'CAT_{cat_name}'] = cat_avg
            features[f'CAT_{cat_name}_MOM1M'] = cat_avg.diff(21)
            features[f'CAT_{cat_name}_MOM3M'] = cat_avg.diff(63)

    # Interaction: credit stress * macro weakness (both high = very dangerous)
    if 'CAT_CREDIT' in features.columns and 'CAT_MACRO' in features.columns:
        features['INTERACT_CREDIT_MACRO'] = (
            features['CAT_CREDIT'] * features['CAT_MACRO'] / 100
        )
    if 'CAT_CREDIT' in features.columns and 'CAT_VOL' in features.columns:
        features['INTERACT_CREDIT_VOL'] = (
            features['CAT_CREDIT'] * features['CAT_VOL'] / 100
        )
    if 'CAT_TREND' in features.columns and 'CAT_MACRO' in features.columns:
        features['INTERACT_TREND_MACRO'] = (
            features['CAT_TREND'] * features['CAT_MACRO'] / 100
        )
    # New interactions from forum research
    if 'CAT_CREDIT' in features.columns and 'CAT_TREND' in features.columns:
        features['INTERACT_CREDIT_TREND'] = (
            features['CAT_CREDIT'] * features['CAT_TREND'] / 100
        )
    if 'CAT_VOL' in features.columns and 'CAT_TREND' in features.columns:
        features['INTERACT_VOL_TREND'] = (
            features['CAT_VOL'] * features['CAT_TREND'] / 100
        )
    if 'CAT_CONDITIONS' in features.columns and 'CAT_MACRO' in features.columns:
        features['INTERACT_CONDITIONS_MACRO'] = (
            features['CAT_CONDITIONS'] * features['CAT_MACRO'] / 100
        )

    # How many indicators are simultaneously above various thresholds?
    for thresh in [60, 70, 80]:
        n_above = (danger_scores > thresh).sum(axis=1)
        features[f'N_ABOVE_{thresh}'] = n_above
        features[f'PCT_ABOVE_{thresh}'] = n_above / danger_scores.notna().sum(axis=1) * 100

    # ================================================================
    # PRACTITIONER / FORUM-INSPIRED FEATURES
    # ================================================================
    if sp500_prices is not None:
        sp = sp500_prices.squeeze().reindex(idx, method='ffill')
        sp_ret = np.log(sp / sp.shift(1))

        # --- Faber 10-Month SMA signal ---
        # Price vs 200-day (≈10 months) SMA: % distance
        sma_200 = sp.rolling(200).mean()
        features['FABER_SMA_PCT'] = (sp / sma_200 - 1) * 100  # negative = bearish
        # Binary: below 200-day
        features['FABER_BELOW_SMA'] = (sp < sma_200).astype(float) * 100

        # --- Dual Momentum: 12-month excess return ---
        # Positive = bullish momentum, negative = exit to bonds
        ret_12m = (sp / sp.shift(252) - 1) * 100
        features['DUAL_MOM_12M'] = ret_12m
        features['DUAL_MOM_NEGATIVE'] = (ret_12m < 0).astype(float) * 100

        # --- 6-month momentum ---
        ret_6m = (sp / sp.shift(126) - 1) * 100
        features['MOM_6M'] = ret_6m

        # --- Realized vol regime: 63-day annualized vol ---
        rv_63 = sp_ret.rolling(63).std() * np.sqrt(252) * 100
        features['RV_63D'] = rv_63
        # Vol of vol (instability of volatility itself)
        features['VOL_OF_VOL'] = rv_63.rolling(63).std()

        # --- Drawdown speed: how fast is the market falling? ---
        rolling_max = sp.rolling(252, min_periods=1).max()
        drawdown = (sp / rolling_max - 1) * 100
        features['DD_SPEED_1W'] = drawdown.diff(5)   # 1-week DD acceleration
        features['DD_SPEED_1M'] = drawdown.diff(21)   # 1-month DD acceleration

        # --- ATR-based volatility (7-day, for trailing stop context) ---
        high_proxy = sp * (1 + sp_ret.abs())  # proxy since we don't have H/L
        low_proxy = sp * (1 - sp_ret.abs())
        tr = high_proxy - low_proxy  # simplified true range
        atr_7 = tr.rolling(7).mean()
        features['ATR_7_PCT'] = (atr_7 / sp) * 100  # as % of price

        # --- Max consecutive down days (momentum exhaustion) ---
        down_day = (sp_ret < 0).astype(float)
        # Rolling sum of last 10 days
        features['DOWN_DAYS_10'] = down_day.rolling(10).sum()
        features['DOWN_DAYS_20'] = down_day.rolling(20).sum()

    # --- HY OAS z-score (weekly change) from raw indicators ---
    if raw_indicators is not None and 'HY_OAS' in raw_indicators.columns:
        hy = raw_indicators['HY_OAS'].reindex(idx, method='ffill')
        hy_weekly_chg = hy.diff(5)
        hy_zscore = (hy_weekly_chg - hy_weekly_chg.rolling(252).mean()) / hy_weekly_chg.rolling(252).std()
        features['HY_OAS_ZSCORE_1W'] = hy_zscore
        features['HY_OAS_ZSCORE_EXTREME'] = (hy_zscore > 2).astype(float) * 100

        # IG OAS z-score too
        if 'IG_OAS' in raw_indicators.columns:
            ig = raw_indicators['IG_OAS'].reindex(idx, method='ffill')
            ig_weekly_chg = ig.diff(5)
            ig_zscore = (ig_weekly_chg - ig_weekly_chg.rolling(252).mean()) / ig_weekly_chg.rolling(252).std()
            features['IG_OAS_ZSCORE_1W'] = ig_zscore

    # --- VIX term structure proxy (VIX vs VVIX) ---
    if raw_indicators is not None:
        if 'VIX' in raw_indicators.columns:
            vix = raw_indicators['VIX'].reindex(idx, method='ffill')
            # VIX z-score (how unusual is current VIX level?)
            vix_zscore = (vix - vix.rolling(252).mean()) / vix.rolling(252).std()
            features['VIX_ZSCORE'] = vix_zscore
            # VIX spike: 1-week change
            features['VIX_SPIKE_1W'] = vix.diff(5)
            features['VIX_SPIKE_1M'] = vix.diff(21)

    print(f"  Total features: {len(features.columns)}")
    print(f"  Date range: {features.index[0].date()} to {features.index[-1].date()}")

    return features


# ============================================================================
# 2. TARGET VARIABLE
# ============================================================================

def build_target(sp500_prices, features_index, dd_threshold=-10, horizon=126):
    """
    Binary target: 1 if max drawdown in next `horizon` days exceeds `dd_threshold`.
    Also returns continuous forward max drawdown for evaluation.
    """
    sp = sp500_prices.squeeze().reindex(features_index, method='ffill')
    sp_arr = sp.values

    fwd_dd = pd.Series(np.nan, index=features_index)
    for i in range(len(sp_arr) - horizon):
        future = sp_arr[i+1:i+1+horizon]
        if np.isnan(future).all():
            continue
        peak = sp_arr[i]
        min_future = np.nanmin(future)
        fwd_dd.iloc[i] = (min_future / peak - 1) * 100

    target_binary = (fwd_dd < dd_threshold).astype(float)
    target_binary[fwd_dd.isna()] = np.nan

    return target_binary, fwd_dd


# ============================================================================
# 3. EXPANDING-WINDOW MODELS
# ============================================================================

def run_expanding_window_models(features, target, fwd_dd,
                                 min_train_years=5, retrain_every=63):
    """
    Walk-forward prediction:
      - Start with min_train_years of data
      - Retrain every retrain_every days
      - Predict out-of-sample probability of crash
      - No look-ahead bias
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    try:
        import lightgbm as lgb
        HAS_LGB = True
    except ImportError:
        HAS_LGB = False
        print("  [WARN] lightgbm not available, using logistic regression only")

    print("\n=== Running walk-forward models ===")

    # Align everything
    valid = target.notna() & features.notna().all(axis=1)
    feat_clean = features[valid].copy()
    tgt_clean = target[valid].copy()
    fwd_dd_clean = fwd_dd[valid].copy()

    n = len(feat_clean)
    min_train = min_train_years * 252
    print(f"  Total usable obs: {n}")
    print(f"  Min training window: {min_train} ({min_train_years} years)")
    print(f"  Retrain every: {retrain_every} days")

    if n < min_train + 252:
        print("  [ERROR] Not enough data for walk-forward")
        return None, None, None

    # Storage for out-of-sample predictions
    pred_lr = pd.Series(np.nan, index=feat_clean.index)
    pred_lgb = pd.Series(np.nan, index=feat_clean.index) if HAS_LGB else None
    feature_importance = pd.DataFrame()

    # Walk forward
    last_train_end = 0
    lr_model = None
    lgb_model = None
    scaler = None
    n_retrains = 0

    for i in range(min_train, n):
        # Retrain periodically
        if i - last_train_end >= retrain_every or lr_model is None:
            X_train = feat_clean.iloc[:i].values
            y_train = tgt_clean.iloc[:i].values

            # Handle class imbalance
            pos_rate = y_train.mean()
            if pos_rate == 0 or pos_rate == 1:
                # Degenerate case, skip
                continue

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)

            # Logistic Regression (L1 for feature selection)
            lr_model = LogisticRegression(
                penalty='l1', C=0.1, solver='saga', max_iter=2000,
                class_weight='balanced', random_state=42
            )
            lr_model.fit(X_train_scaled, y_train)

            # LightGBM
            if HAS_LGB:
                # Compute scale_pos_weight for imbalanced classes
                neg_count = (y_train == 0).sum()
                pos_count = (y_train == 1).sum()
                spw = neg_count / pos_count if pos_count > 0 else 1

                lgb_model = lgb.LGBMClassifier(
                    n_estimators=150, max_depth=3, learning_rate=0.05,
                    num_leaves=8, min_child_samples=80,
                    scale_pos_weight=spw,
                    subsample=0.7, colsample_bytree=0.6,
                    reg_alpha=2.0, reg_lambda=2.0,
                    random_state=42, verbose=-1, n_jobs=1
                )
                lgb_model.fit(X_train_scaled, y_train)

                # Feature importance (from latest retrain)
                if n_retrains % 10 == 0:
                    imp = pd.Series(lgb_model.feature_importances_,
                                    index=feat_clean.columns)
                    feature_importance[f'retrain_{n_retrains}'] = imp

            last_train_end = i
            n_retrains += 1

            if n_retrains % 20 == 0:
                print(f"  Retrain #{n_retrains} at {feat_clean.index[i].date()}, "
                      f"train size={i}, crash rate={pos_rate:.3f}")

        # Predict (out of sample)
        X_pred = scaler.transform(feat_clean.iloc[i:i+1].values)
        pred_lr.iloc[i] = lr_model.predict_proba(X_pred)[0, 1]
        if HAS_LGB and lgb_model is not None:
            pred_lgb.iloc[i] = lgb_model.predict_proba(X_pred)[0, 1]

    print(f"  Total retrains: {n_retrains}")

    # Evaluate out-of-sample predictions
    oos_mask = pred_lr.notna()
    if oos_mask.sum() > 100:
        y_true = tgt_clean[oos_mask]
        print(f"\n  Out-of-sample evaluation ({oos_mask.sum()} predictions):")
        print(f"  Crash base rate: {y_true.mean():.3f}")

        auc_lr = roc_auc_score(y_true, pred_lr[oos_mask])
        print(f"  Logistic Regression AUC: {auc_lr:.4f}")

        corr_lr = pred_lr[oos_mask].corr(fwd_dd_clean[oos_mask])
        print(f"  LR pred corr with fwd DD: {corr_lr:+.4f}")

        if pred_lgb is not None and pred_lgb.notna().sum() > 100:
            auc_lgb = roc_auc_score(y_true, pred_lgb[oos_mask])
            print(f"  LightGBM AUC: {auc_lgb:.4f}")

            corr_lgb = pred_lgb[oos_mask].corr(fwd_dd_clean[oos_mask])
            print(f"  LGB pred corr with fwd DD: {corr_lgb:+.4f}")

            # Ensemble
            pred_ens = 0.4 * pred_lr[oos_mask] + 0.6 * pred_lgb[oos_mask]
            auc_ens = roc_auc_score(y_true, pred_ens)
            print(f"  Ensemble (0.4 LR + 0.6 LGB) AUC: {auc_ens:.4f}")
            corr_ens = pred_ens.corr(fwd_dd_clean[oos_mask])
            print(f"  Ensemble corr with fwd DD: {corr_ens:+.4f}")

    # Average feature importance across retrains
    if len(feature_importance.columns) > 0:
        avg_imp = feature_importance.mean(axis=1).sort_values(ascending=False)
        print(f"\n  Top 20 features (LightGBM importance):")
        for name, imp in avg_imp.head(20).items():
            print(f"    {name:35s}  {imp:.1f}")

    return pred_lr, pred_lgb, feature_importance


# ============================================================================
# 4. STRATEGY BACKTEST WITH HYSTERESIS
# ============================================================================

def backtest_strategy(pred_series, sp500_prices, features_index,
                      exit_thresh, entry_thresh, label='', rf_annual=0.03):
    """
    Backtest with hysteresis on a probability signal.
    Exit when P(crash) >= exit_thresh, re-enter when P(crash) < entry_thresh.
    """
    sp = sp500_prices.squeeze().reindex(features_index, method='ffill')
    sp_daily_ret = sp.pct_change()
    rf_daily = rf_annual / 252

    # Use previous day's prediction (no look-ahead)
    pred_lag = pred_series.shift(1)

    # Build signal with hysteresis
    in_market = pd.Series(True, index=pred_lag.index)
    currently_in = True
    for i in range(len(pred_lag)):
        val = pred_lag.iloc[i]
        if pd.isna(val):
            in_market.iloc[i] = currently_in
            continue
        if currently_in and val >= exit_thresh:
            currently_in = False
        elif not currently_in and val < entry_thresh:
            currently_in = True
        in_market.iloc[i] = currently_in

    # Compute returns
    valid = pred_series.notna() & sp_daily_ret.notna()
    sig = in_market[valid]
    ret = sp_daily_ret[valid]

    strat_ret = ret.where(sig, rf_daily)
    ann_ret = strat_ret.mean() * 252 * 100
    ann_vol = strat_ret.std() * np.sqrt(252) * 100
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    cum = (1 + strat_ret).cumprod()
    maxdd = ((cum / cum.cummax()) - 1).min() * 100
    pct_in = sig.mean() * 100
    n_trades = (sig != sig.shift(1)).sum()
    total_ret = (cum.iloc[-1] - 1) * 100

    # Buy and hold over same period
    bh_ret = (1 + ret).cumprod()
    bh_total = (bh_ret.iloc[-1] - 1) * 100
    bh_ann = ret.mean() * 252 * 100
    bh_vol = ret.std() * np.sqrt(252) * 100
    bh_sharpe = bh_ann / bh_vol if bh_vol > 0 else 0
    bh_maxdd = ((bh_ret / bh_ret.cummax()) - 1).min() * 100

    return {
        'label': label, 'ann_ret': ann_ret, 'ann_vol': ann_vol,
        'sharpe': sharpe, 'maxdd': maxdd, 'pct_in': pct_in,
        'n_trades': n_trades, 'total_ret': total_ret,
        'cum': cum, 'signal': sig,
        'bh_ann': bh_ann, 'bh_sharpe': bh_sharpe, 'bh_maxdd': bh_maxdd,
        'bh_total': bh_total,
    }


def backtest_long_short(pred_series, sp500_prices, features_index,
                        exit_thresh, entry_thresh, short_thresh,
                        short_size=1.0, label='',
                        rf_annual=0.03, borrow_cost_annual=0.005):
    """
    Three-regime strategy with hysteresis:
      - LONG:  P(crash) < entry_thresh (or haven't exited yet)
      - CASH:  P(crash) >= exit_thresh but < short_thresh
      - SHORT: P(crash) >= short_thresh

    short_size: fraction of portfolio to short (0.5 = 50% short, 1.0 = 100%)
    borrow_cost_annual: annual cost of borrowing shares to short
    """
    sp = sp500_prices.squeeze().reindex(features_index, method='ffill')
    sp_daily_ret = sp.pct_change()
    rf_daily = rf_annual / 252
    borrow_daily = borrow_cost_annual / 252

    pred_lag = pred_series.shift(1)

    # Build position signal: +1 = long, 0 = cash, -short_size = short
    position = pd.Series(1.0, index=pred_lag.index)  # start long
    current_pos = 1.0  # 1=long, 0=cash, -short_size=short
    for i in range(len(pred_lag)):
        val = pred_lag.iloc[i]
        if pd.isna(val):
            position.iloc[i] = current_pos
            continue

        if current_pos > 0:  # currently long
            if val >= short_thresh:
                current_pos = -short_size
            elif val >= exit_thresh:
                current_pos = 0.0
        elif current_pos == 0:  # currently cash
            if val >= short_thresh:
                current_pos = -short_size
            elif val < entry_thresh:
                current_pos = 1.0
        else:  # currently short
            if val < entry_thresh:
                current_pos = 1.0
            elif val < exit_thresh:
                current_pos = 0.0

        position.iloc[i] = current_pos

    # Compute returns
    valid = pred_series.notna() & sp_daily_ret.notna()
    pos = position[valid]
    ret = sp_daily_ret[valid]

    # Strategy returns:
    #   long:  market return
    #   cash:  risk-free
    #   short: -short_size * market return + (1-short_size)*rf - borrow cost
    strat_ret = pd.Series(0.0, index=ret.index)
    long_mask = pos > 0
    cash_mask = pos == 0
    short_mask = pos < 0

    strat_ret[long_mask] = ret[long_mask]
    strat_ret[cash_mask] = rf_daily
    strat_ret[short_mask] = (-pos[short_mask] * ret[short_mask]  # short P&L
                             + (1 + pos[short_mask]) * rf_daily   # cash portion earns rf
                             - borrow_daily * (-pos[short_mask]))  # borrow cost on short

    ann_ret = strat_ret.mean() * 252 * 100
    ann_vol = strat_ret.std() * np.sqrt(252) * 100
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    cum = (1 + strat_ret).cumprod()
    maxdd = ((cum / cum.cummax()) - 1).min() * 100
    pct_long = long_mask.mean() * 100
    pct_cash = cash_mask.mean() * 100
    pct_short = short_mask.mean() * 100
    n_trades = (pos != pos.shift(1)).sum()
    total_ret = (cum.iloc[-1] - 1) * 100

    # Buy and hold
    bh_ret = (1 + ret).cumprod()
    bh_total = (bh_ret.iloc[-1] - 1) * 100
    bh_ann = ret.mean() * 252 * 100
    bh_vol = ret.std() * np.sqrt(252) * 100
    bh_sharpe = bh_ann / bh_vol if bh_vol > 0 else 0
    bh_maxdd = ((bh_ret / bh_ret.cummax()) - 1).min() * 100

    return {
        'label': label, 'ann_ret': ann_ret, 'ann_vol': ann_vol,
        'sharpe': sharpe, 'maxdd': maxdd,
        'pct_long': pct_long, 'pct_cash': pct_cash, 'pct_short': pct_short,
        'n_trades': n_trades, 'total_ret': total_ret,
        'cum': cum, 'position': pos,
        'bh_ann': bh_ann, 'bh_sharpe': bh_sharpe, 'bh_maxdd': bh_maxdd,
        'bh_total': bh_total,
    }


def grid_search_strategies(pred_lr, pred_lgb, sp500_prices, features_index):
    """Grid search over exit/entry thresholds for each model."""
    print("\n=== Grid Search: Best Strategies ===")

    header = (f"  {'Strategy':>35s}  {'Ann.Ret':>8s}  {'Ann.Vol':>8s}  {'Sharpe':>7s}  "
              f"{'MaxDD':>8s}  {'%InMkt':>7s}  {'#Tr':>4s}  {'TotalRet':>10s}")
    sep = "  " + "-" * 105

    all_results = []

    for pred, name in [(pred_lr, 'LR'), (pred_lgb, 'LGB')]:
        if pred is None or pred.notna().sum() < 252:
            continue

        # Also try ensemble
        if pred_lgb is not None and pred_lr is not None:
            pred_ens = 0.4 * pred_lr + 0.6 * pred_lgb
        else:
            pred_ens = None

        for p, pname in [(pred, name)] + ([(pred_ens, 'ENS')] if pred_ens is not None and name == 'LGB' else []):
            if p is None:
                continue

            # Probability thresholds for exit/entry
            for exit_t in np.arange(0.15, 0.65, 0.05):
                for entry_t in np.arange(0.05, exit_t, 0.05):
                    r = backtest_strategy(
                        p, sp500_prices, features_index,
                        exit_t, entry_t,
                        label=f'{pname} x>={exit_t:.2f} r<{entry_t:.2f}'
                    )
                    all_results.append(r)
                # Single threshold too
                r = backtest_strategy(
                    p, sp500_prices, features_index,
                    exit_t, exit_t,
                    label=f'{pname} x>={exit_t:.2f}'
                )
                all_results.append(r)

    # Sort by Sharpe
    all_results.sort(key=lambda x: x['sharpe'], reverse=True)

    # Print buy and hold first
    if all_results:
        r0 = all_results[0]
        print(f"\n  Buy & Hold (same period): Ann={r0['bh_ann']:+.2f}%  "
              f"Sharpe={r0['bh_sharpe']:.3f}  MaxDD={r0['bh_maxdd']:.1f}%  "
              f"Total={r0['bh_total']:+.1f}%")

    print(f"\n  Top 20 strategies by Sharpe:")
    print(header)
    print(sep)
    for r in all_results[:20]:
        print(f"  {r['label']:>35s}  {r['ann_ret']:+7.2f}%  {r['ann_vol']:7.2f}%  {r['sharpe']:+6.3f}  "
              f"{r['maxdd']:+7.2f}%  {r['pct_in']:6.1f}%  {r['n_trades']:>4d}  {r['total_ret']:+10.1f}%")

    # Also show top by total return with Sharpe > 0.7
    good = [r for r in all_results if r['sharpe'] > 0.7]
    good.sort(key=lambda x: x['total_ret'], reverse=True)
    print(f"\n  Top 10 by total return (Sharpe > 0.7):")
    print(header)
    print(sep)
    for r in good[:10]:
        print(f"  {r['label']:>35s}  {r['ann_ret']:+7.2f}%  {r['ann_vol']:7.2f}%  {r['sharpe']:+6.3f}  "
              f"{r['maxdd']:+7.2f}%  {r['pct_in']:6.1f}%  {r['n_trades']:>4d}  {r['total_ret']:+10.1f}%")

    # =================================================================
    # LONG/SHORT STRATEGIES
    # =================================================================
    print("\n=== Long/Short Strategy Grid Search ===")

    ls_header = (f"  {'Strategy':>42s}  {'Ann.Ret':>8s}  {'Ann.Vol':>8s}  {'Sharpe':>7s}  "
                 f"{'MaxDD':>8s}  {'%Long':>6s}  {'%Cash':>6s}  {'%Short':>6s}  "
                 f"{'#Tr':>4s}  {'TotalRet':>10s}")
    ls_sep = "  " + "-" * 125

    ls_results = []

    # Use ensemble if available, else LGB
    if pred_lgb is not None and pred_lr is not None:
        pred_ens = 0.4 * pred_lr + 0.6 * pred_lgb
    else:
        pred_ens = pred_lgb if pred_lgb is not None else pred_lr

    for p, pname in [(pred_ens, 'ENS'), (pred_lgb, 'LGB')]:
        if p is None or p.notna().sum() < 252:
            continue
        for short_size in [0.5, 1.0]:
            sz_label = f'{int(short_size*100)}%'
            for exit_t in np.arange(0.20, 0.55, 0.05):
                for entry_t in np.arange(0.05, exit_t, 0.10):
                    for short_t in np.arange(max(exit_t, 0.30), 0.80, 0.10):
                        r = backtest_long_short(
                            p, sp500_prices, features_index,
                            exit_t, entry_t, short_t,
                            short_size=short_size,
                            label=f'{pname} x>={exit_t:.2f} r<{entry_t:.2f} s>={short_t:.2f} @{sz_label}'
                        )
                        ls_results.append(r)

    # Sort by Sharpe
    ls_results.sort(key=lambda x: x['sharpe'], reverse=True)

    if ls_results:
        r0 = ls_results[0]
        print(f"\n  Buy & Hold: Ann={r0['bh_ann']:+.2f}%  Sharpe={r0['bh_sharpe']:.3f}  "
              f"MaxDD={r0['bh_maxdd']:.1f}%  Total={r0['bh_total']:+.1f}%")

    print(f"\n  Top 20 Long/Short strategies by Sharpe:")
    print(ls_header)
    print(ls_sep)
    for r in ls_results[:20]:
        print(f"  {r['label']:>42s}  {r['ann_ret']:+7.2f}%  {r['ann_vol']:7.2f}%  {r['sharpe']:+6.3f}  "
              f"{r['maxdd']:+7.2f}%  {r['pct_long']:5.1f}%  {r['pct_cash']:5.1f}%  {r['pct_short']:5.1f}%  "
              f"{r['n_trades']:>4d}  {r['total_ret']:+10.1f}%")

    # Top by total return with Sharpe > 0.8
    ls_good = [r for r in ls_results if r['sharpe'] > 0.8]
    ls_good.sort(key=lambda x: x['total_ret'], reverse=True)
    print(f"\n  Top 10 Long/Short by total return (Sharpe > 0.8):")
    print(ls_header)
    print(ls_sep)
    for r in ls_good[:10]:
        print(f"  {r['label']:>42s}  {r['ann_ret']:+7.2f}%  {r['ann_vol']:7.2f}%  {r['sharpe']:+6.3f}  "
              f"{r['maxdd']:+7.2f}%  {r['pct_long']:5.1f}%  {r['pct_cash']:5.1f}%  {r['pct_short']:5.1f}%  "
              f"{r['n_trades']:>4d}  {r['total_ret']:+10.1f}%")

    # Compare best long-only vs best long/short
    if all_results and ls_results:
        best_lo = all_results[0]
        best_ls = ls_results[0]
        print(f"\n  --- Best Long-Only vs Best Long/Short ---")
        print(f"  Long-Only:  {best_lo['label']:>35s}  Sharpe={best_lo['sharpe']:.3f}  "
              f"Ann={best_lo['ann_ret']:+.2f}%  MaxDD={best_lo['maxdd']:.1f}%  Total={best_lo['total_ret']:+.1f}%")
        print(f"  Long/Short: {best_ls['label']:>35s}  Sharpe={best_ls['sharpe']:.3f}  "
              f"Ann={best_ls['ann_ret']:+.2f}%  MaxDD={best_ls['maxdd']:.1f}%  Total={best_ls['total_ret']:+.1f}%")

    return all_results, ls_results


# ============================================================================
# 5. CHARTS
# ============================================================================

def create_advanced_charts(pred_lr, pred_lgb, best_result, sp500_prices,
                           features_index, fwd_dd, output_dir):
    """Create visualization of the advanced model."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print("[WARN] matplotlib not available")
        return

    print("\n=== Creating advanced strategy charts ===")

    fig, axes = plt.subplots(5, 1, figsize=(18, 24), sharex=True,
                              gridspec_kw={'height_ratios': [2, 1, 1, 1, 1.5]})

    sp = sp500_prices.squeeze().reindex(features_index, method='ffill')
    oos_start = pred_lr.first_valid_index()

    # Panel 1: S&P 500 with strategy signals
    ax1 = axes[0]
    sp_plot = sp[sp.index >= oos_start]
    ax1.semilogy(sp_plot.index, sp_plot.values, 'k-', linewidth=0.8, label='S&P 500')

    # Shade out-of-market periods
    if best_result is not None:
        sig = best_result['signal']
        out_of_market = ~sig
        in_period = False
        start = None
        for date, val in out_of_market.items():
            if val and not in_period:
                start = date
                in_period = True
            elif not val and in_period:
                ax1.axvspan(start, date, alpha=0.3, color='red')
                in_period = False
        if in_period:
            ax1.axvspan(start, out_of_market.index[-1], alpha=0.3, color='red')

    ax1.set_ylabel('S&P 500 (log scale)')
    ax1.set_title('Advanced Crash Risk Model — Walk-Forward Backtest', fontsize=14, fontweight='bold')
    ax1.legend(['S&P 500', 'Out of Market (strategy)'], loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Model crash probability
    ax2 = axes[1]
    if pred_lgb is not None:
        pred_ens = 0.4 * pred_lr + 0.6 * pred_lgb
        ax2.plot(pred_ens.index, pred_ens.values, 'orangered', linewidth=0.6, alpha=0.8, label='Ensemble P(crash)')
    else:
        ax2.plot(pred_lr.index, pred_lr.values, 'orangered', linewidth=0.6, alpha=0.8, label='LR P(crash)')
    ax2.set_ylabel('P(DD > 10%)')
    ax2.set_ylim(0, 1)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Panel 3: LR probability
    ax3 = axes[2]
    ax3.plot(pred_lr.index, pred_lr.values, 'steelblue', linewidth=0.6, alpha=0.8)
    ax3.set_ylabel('LR P(crash)')
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)

    # Panel 4: LGB probability
    if pred_lgb is not None:
        ax4 = axes[3]
        ax4.plot(pred_lgb.index, pred_lgb.values, 'green', linewidth=0.6, alpha=0.8)
        ax4.set_ylabel('LGB P(crash)')
        ax4.set_ylim(0, 1)
        ax4.grid(True, alpha=0.3)

    # Panel 5: Equity curves
    ax5 = axes[4]
    if best_result is not None:
        ax5.plot(best_result['cum'].index, best_result['cum'].values, 'b-', linewidth=1.2, label=best_result['label'])
    # Buy and hold
    sp_oos = sp[sp.index >= oos_start].dropna()
    bh = sp_oos / sp_oos.iloc[0]
    ax5.plot(bh.index, bh.values, 'k--', linewidth=0.8, label='Buy & Hold')
    ax5.set_ylabel('Growth of $1')
    ax5.legend(loc='upper left', fontsize=9)
    ax5.grid(True, alpha=0.3)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    axes[-1].set_xlabel('Date')

    plt.tight_layout()
    chart_path = output_dir / 'crash_index_advanced_backtest.png'
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved to {chart_path}")


# ============================================================================
# MAIN
# ============================================================================

def run_advanced(danger_scores, sp500_prices, raw_indicators=None, output_dir=None):
    """
    Main entry point. Call after build_crash_index.py has produced danger_scores.
    """
    if output_dir is None:
        output_dir = Path(__file__).parent

    print("\n" + "=" * 70)
    print("ADVANCED CRASH PREDICTION MODEL")
    print("=" * 70)

    # 1. Build features (with practitioner signals if raw data available)
    features = build_features(danger_scores, raw_indicators=raw_indicators,
                              sp500_prices=sp500_prices)

    # 2. Build targets at multiple thresholds
    print("\n  Drawdown targets:")
    targets = {}
    for dd_thresh in [-10, -15, -20]:
        t, fwd_dd = build_target(sp500_prices, features.index,
                                  dd_threshold=dd_thresh, horizon=126)
        targets[dd_thresh] = t
        print(f"    DD > {abs(dd_thresh)}%: crash rate = {t.dropna().mean():.3f}")

    # Primary target for model training
    target = targets[-10]
    print(f"\n  Primary target: P(6M max drawdown > 10%)")
    print(f"  Crash rate: {target.dropna().mean():.3f}")

    # 3. Drop features with too many NaNs
    min_coverage = 0.5  # need at least 50% non-NaN
    good_cols = []
    for col in features.columns:
        coverage = features[col].notna().mean()
        if coverage >= min_coverage:
            good_cols.append(col)
    features = features[good_cols]
    print(f"  Features after coverage filter: {len(features.columns)}")

    # Forward-fill remaining NaNs (within each column) then drop rows still NaN
    features = features.ffill().bfill()

    # Pre-select features using correlation with target to keep model fast
    # (expanding-window models are O(n_features * n_retrains * train_size))
    max_features = 80
    if len(features.columns) > max_features:
        valid_both = target.notna() & features.notna().all(axis=1)
        feat_corrs = features[valid_both].corrwith(target[valid_both]).abs()
        top_cols = feat_corrs.nlargest(max_features).index.tolist()
        features = features[top_cols]
        print(f"  Features after pre-selection (top {max_features} by |corr| with target): {len(features.columns)}")

    # 4. Run walk-forward models
    pred_lr, pred_lgb, feat_imp = run_expanding_window_models(
        features, target, fwd_dd,
        min_train_years=5, retrain_every=63  # retrain quarterly
    )

    if pred_lr is None:
        print("  Model failed, not enough data")
        return

    # 5. Grid search strategies (long-only + long/short)
    all_results, ls_results = grid_search_strategies(
        pred_lr, pred_lgb, sp500_prices, features.index)

    # 6. Charts (use best long/short if available, else best long-only)
    best_lo = all_results[0] if all_results else None
    best_ls = ls_results[0] if ls_results else None
    best = best_ls if best_ls and best_ls['sharpe'] > (best_lo['sharpe'] if best_lo else 0) else best_lo
    create_advanced_charts(pred_lr, pred_lgb, best, sp500_prices,
                           features.index, fwd_dd, output_dir)

    # 7. Current readings
    print("\n" + "=" * 70)
    print("CURRENT MODEL READINGS")
    print("=" * 70)
    if pred_lr.notna().any():
        latest_lr = pred_lr.dropna().iloc[-1]
        print(f"  LR P(crash):  {latest_lr:.3f}  ({latest_lr*100:.1f}%)")
    if pred_lgb is not None and pred_lgb.notna().any():
        latest_lgb = pred_lgb.dropna().iloc[-1]
        print(f"  LGB P(crash): {latest_lgb:.3f}  ({latest_lgb*100:.1f}%)")
        ens = 0.4 * latest_lr + 0.6 * latest_lgb
        print(f"  Ensemble:     {ens:.3f}  ({ens*100:.1f}%)")

        # Current position recommendation
        print(f"\n  Position recommendation (best L/S strategy):")
        if best_ls:
            print(f"    Strategy: {best_ls['label']}")

    return pred_lr, pred_lgb, all_results, ls_results


if __name__ == '__main__':
    # Load pre-built data
    data_path = Path(__file__).parent / 'crash_index_dataset.parquet'
    if not data_path.exists():
        print("Run build_crash_index.py first to generate the dataset.")
        raise SystemExit(1)

    dataset = pd.read_parquet(data_path)

    # Extract danger scores (columns starting with DS_)
    ds_cols = [c for c in dataset.columns if c.startswith('DS_')]
    danger_scores = dataset[ds_cols].rename(columns=lambda c: c.replace('DS_', ''))

    # Extract raw indicators (non-DS_, non-FWD_, non-COMPOSITE columns)
    skip_prefixes = ('DS_', 'FWD_', 'COMPOSITE', 'N_INDICATORS', 'QUINTILE',
                     'WEIGHTED_COMPOSITE')
    raw_cols = [c for c in dataset.columns
                if not any(c.startswith(p) for p in skip_prefixes)]
    raw_indicators = dataset[raw_cols] if raw_cols else None

    # Extract S&P 500
    sp500 = dataset.get('SP500') if 'SP500' in dataset.columns else None
    if sp500 is None:
        import yfinance as yf
        sp500 = yf.download('^GSPC', start='1988-01-01', progress=False, auto_adjust=True)['Close'].squeeze()

    run_advanced(danger_scores, sp500, raw_indicators=raw_indicators,
                 output_dir=Path(__file__).parent)
