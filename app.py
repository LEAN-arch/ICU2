import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.signal import welch, periodogram
from scipy.stats import norm, multivariate_normal, wasserstein_distance, linregress, chi2, ks_2samp, f_oneway
from scipy.fft import fft, fftfreq
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import grangercausalitytests
import time

# ==========================================
# 1. CONFIGURATION & MEDICAL THEME
# ==========================================
st.set_page_config(
    page_title="TITAN | ULTIMATE COMMAND CENTER",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🧬"
)

class CONFIG:
    # UI Colors
    COLORS = {
        "bg": "#f8fafc", "card": "#ffffff", "text": "#0f172a", "muted": "#64748b",
        "crit": "#dc2626", "warn": "#d97706", "ok": "#059669",
        "info": "#2563eb", "hemo": "#0891b2", "resp": "#7c3aed", 
        "ai": "#be185d", "drug": "#4f46e5", "anomaly": "#db2777",
        "ext": "#d946ef", "spc": "#059669"
    }
    
    # Physics Constants
    ATM_PRESSURE = 760.0; H2O_PRESSURE = 47.0; R_QUOTIENT = 0.8; MAX_PAO2 = 600.0
    HB_CONVERSION = 1.34; LAC_PROD_THRESH = 330.0; LAC_CLEAR_RATE = 0.05; VCO2_CONST = 130
    
    # Drug PK (Potency, Tau, Tolerance) - Updated for Emax/EC50
    DRUG_PK = {
        'norepi': {'ec50': 0.5, 'emax': {'svr': 3000.0, 'map': 140.0, 'co': 1.0}, 'tau': 2.0, 'tol': 1440.0}, 
        'vaso':   {'ec50': 0.04, 'emax': {'svr': 4500.0, 'map': 160.0, 'co': -0.5}, 'tau': 5.0, 'tol': 2880.0}, 
        'dobu':   {'ec50': 5.0,  'emax': {'svr': -800.0, 'map': 10.0, 'co': 6.0, 'hr': 40.0}, 'tau': 3.0, 'tol': 720.0}, 
        'bb':     {'ec50': 0.5,  'emax': {'svr': 100.0, 'map': -25.0, 'co': -3.0, 'hr': -50.0}, 'tau': 4.0, 'tol': 5000.0}
    }
    
    # SPC & QA Limits
    MAP_LSL = 65.0; MAP_USL = 110.0; CUSUM_H = 4.0; CUSUM_K = 0.5

STYLING = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=Roboto+Mono:wght@500;700&display=swap');
    .stApp {{ background-color: {CONFIG.COLORS['bg']}; color: {CONFIG.COLORS['text']}; font-family: 'Inter', sans-serif; }}
    
    div[data-testid="stMetric"] {{ background-color: {CONFIG.COLORS['card']}; border: 1px solid {CONFIG.COLORS['muted']}33; border-radius: 6px; padding: 10px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }}
    div[data-testid="stMetric"] label {{ color: {CONFIG.COLORS['muted']}; font-size: 0.65rem; font-weight: 700; text-transform: uppercase; }}
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{ font-family: 'Roboto Mono', monospace; font-size: 1.4rem; font-weight: 700; }}
    
    .zone-header {{ font-size: 0.85rem; font-weight: 900; color: {CONFIG.COLORS['text']}; text-transform: uppercase; border-bottom: 2px solid {CONFIG.COLORS['info']}33; margin: 25px 0 10px 0; letter-spacing: 0.05em; }}
    .status-banner {{ padding: 15px; border-radius: 8px; background: {CONFIG.COLORS['card']}; border-left: 6px solid {CONFIG.COLORS['ai']}; margin-bottom: 20px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); display: flex; justify-content: space-between; align-items: center; }}
    .clinical-hint {{ font-size: 0.75rem; color: {CONFIG.COLORS['muted']}; background: #f1f5f9; padding: 8px; border-radius: 4px; margin-top: 5px; border-left: 3px solid {CONFIG.COLORS['info']}; }}
    .sme-note {{ font-size: 0.7rem; color: #475569; background-color: #e2e8f0; padding: 5px; border-radius: 4px; margin-top: 5px; }}
    
    .crit-pulse {{ animation: pulse-red 2s infinite; color: {CONFIG.COLORS['crit']}; }}
    @keyframes pulse-red {{ 0% {{ opacity: 1; }} 50% {{ opacity: 0.5; }} 100% {{ opacity: 1; }} }}
</style>
"""

# ==========================================
# 2. UTILS & ENGINE CORE
# ==========================================
class Utils:
    _rng = np.random.default_rng(42)
    @staticmethod
    def set_seed(seed): Utils._rng = np.random.default_rng(seed)
    @staticmethod
    def brownian_bridge(n, start, end, volatility, noise_type='pink'):
        t = np.linspace(0, 1, n)
        dW = Utils._rng.normal(0, np.sqrt(1/n), n)
        B = start + np.cumsum(dW) - t * (np.cumsum(dW)[-1] - (end - start))
        if noise_type == 'pink': noise = np.convolve(Utils._rng.normal(0, 0.5, n), np.ones(8)/8, mode='same')
        elif noise_type == 'periodic': noise = np.sin(np.linspace(0, n/4, n)) * 2.0
        elif noise_type == 'white': noise = Utils._rng.normal(0, 0.1, n)
        else: noise = np.zeros(n)
        return B + (noise * volatility)

# ==========================================
# 3. PHYSIOLOGY ENGINES (VECTORIZED)
# ==========================================
class Physiology:
    class Autonomic:
        @staticmethod
        def baroreflex(map_vals, hr_base, sensitivity=0.8):
            # 2. Baroreflex: Inverse relationship between MAP and HR with lag
            target_hr = hr_base - (map_vals - 65) * sensitivity
            return np.clip(target_hr, 40, 180)

        @staticmethod
        def frank_starling(preload, contractility, k_m=100):
            # 3. Frank-Starling: SV depends on Preload (non-linear)
            # SV = SV_max * Preload / (Preload + Km)
            sv_max = 120 * contractility
            sv = sv_max * preload / (preload + k_m)
            return sv

        @staticmethod
        def generate(mins, p, is_paced, vent_mode):
            # Heart Rate Logic
            if is_paced: hr = Utils.brownian_bridge(mins, p['hr'][0], p['hr'][0], 0.1, 'white')
            elif vent_mode == 'Control (AC)': hr = Utils.brownian_bridge(mins, p['hr'][0], p['hr'][1], 1.5, 'periodic')
            else: hr = Utils.brownian_bridge(mins, p['hr'][0], p['hr'][1], 1.5, 'pink')
            
            # Base vitals before reflex
            map_r = np.maximum(Utils.brownian_bridge(mins, p['map'][0], p['map'][1], 1.2, 'pink'), 20.0)
            
            # Apply Baroreflex
            hr = Physiology.Autonomic.baroreflex(map_r, hr) if not is_paced else hr
            
            svri = np.maximum(Utils.brownian_bridge(mins, p['svri'][0], p['svri'][1], 100.0, 'pink'), 100.0)
            rr = np.maximum(Utils.brownian_bridge(mins, 16, 28, 2.0, 'pink'), 4.0)
            return hr, map_r, svri, rr

    class PKPD:
        @staticmethod
        def hill_equation(conc, emax, ec50, n=1.5):
            # 5. Emax/EC50 Model
            return (emax * (conc**n)) / (ec50**n + conc**n)

        @staticmethod
        def apply(map_b, ci_b, hr_b, svri_b, drugs, mins):
            t = np.arange(mins)
            e_map, e_ci, e_hr, e_svr = np.zeros(mins), np.zeros(mins), np.zeros(mins), np.zeros(mins)
            
            # 6. Drug Interactions (Synergy)
            interaction_boost = 1.0
            if drugs.get('norepi', 0) > 0 and drugs.get('vaso', 0) > 0:
                interaction_boost = 1.2 # Vaso potentiates Norepi
            
            for d, dose in drugs.items():
                if dose <= 0: continue
                pk = CONFIG.DRUG_PK.get(d)
                if not pk: continue
                
                # 4. PK/PD Multi-compartment (Simulated via bi-exponential decay or accumulation)
                # Cp(t) approx.
                conc_factor = (1 - np.exp(-t/pk['tau'])) * np.exp(-t/pk['tol'])
                conc = dose * conc_factor * interaction_boost

                if 'emax' in pk:
                    if 'map' in pk['emax']: e_map += Physiology.PKPD.hill_equation(conc, pk['emax']['map'], pk['ec50'])
                    if 'co' in pk['emax']: e_ci += Physiology.PKPD.hill_equation(conc, pk['emax']['co'], pk['ec50'])
                    if 'hr' in pk['emax']: e_hr += Physiology.PKPD.hill_equation(conc, pk['emax']['hr'], pk['ec50'])
                    if 'svr' in pk['emax']: e_svr += Physiology.PKPD.hill_equation(conc, pk['emax']['svr'], pk['ec50'])
                    
            return np.maximum(map_b+e_map, 10), np.maximum(ci_b+e_ci, 0.1), np.maximum(hr_b+e_hr, 20), np.maximum(svri_b+e_svr, 50)

    class Respiratory:
        @staticmethod
        def exchange(fio2, rr, shunt, peep, mins, copd):
            vd_vt = np.clip(0.3 * copd + (0.1 if copd>1.0 else 0), 0.1, 0.8)
            va = np.maximum((rr * 0.5) * (1 - vd_vt), 0.5)
            paco2 = (CONFIG.VCO2_CONST * 0.863) / va
            p_ideal = np.clip((fio2 * (CONFIG.ATM_PRESSURE - CONFIG.H2O_PRESSURE)) - (paco2 / CONFIG.R_QUOTIENT), 0, CONFIG.MAX_PAO2)
            pao2 = p_ideal * (1 - (shunt * np.exp(-0.08 * peep)))
            spo2 = 100 * (np.maximum(pao2, 0.1)**3 / (np.maximum(pao2, 0.1)**3 + 26**3))
            return pao2, paco2, spo2, np.full(mins, vd_vt)

    class Metabolic:
        @staticmethod
        def calculate(ci, hb, spo2, mins, stress):
            do2i = ci * hb * CONFIG.HB_CONVERSION * (spo2/100) * 10
            vo2i = np.full(mins, 125.0 * stress)
            o2er = vo2i / np.maximum(do2i, 1.0)
            lactate = np.zeros(mins); lac = 1.0
            prod = np.where((do2i < CONFIG.LAC_PROD_THRESH) | (o2er > 0.5), 0.2, 0.0)
            for i in range(mins):
                lac = max(0.6, lac + prod[i] - (CONFIG.LAC_CLEAR_RATE * (ci[i]/2.5)))
                lactate[i] = lac
            return do2i, vo2i, o2er, lactate

# ==========================================
# 4. ANALYTICS, SPC & FORENSICS
# ==========================================
class Analytics:
    @staticmethod
    def signal_forensics(ts, is_paced):
        # 1. Fourier Transform (FFT) logic within analysis
        arr = np.array(ts)
        N = len(arr)
        yf = fft(arr)
        xf = fftfreq(N, 1/60)
        
        # 23. Spectral Entropy Trend
        psd = np.abs(yf)**2
        psd_norm = psd / np.sum(psd)
        spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12)) / np.log2(N)

        if is_paced or np.std(arr) < 0.5: return "EXTERNAL: PACEMAKER", 99, "Zero Variance (Quartz Precision)", spectral_entropy
        if np.max(np.abs(np.gradient(arr))) > 5.0: return "EXTERNAL: INFUSION", 90, "Non-Physiologic Step Change", spectral_entropy
        
        entropy = -np.sum((psd_norm) * np.log2((psd_norm) + 1e-12))
        if entropy < 1.5: return "EXTERNAL: VENTILATOR", 85, "Periodic Entrainment", spectral_entropy
        return "INTERNAL: AUTONOMIC", 80, "Fractal Pink Noise", spectral_entropy

    @staticmethod
    def bayes_shock(row):
        means = {"Cardiogenic": [1.8, 2800], "Distributive": [5.0, 800], "Hypovolemic": [2.0, 3000], "Stable": [3.2, 1900]}
        covs = {"Cardiogenic": [[0.5, -100], [-100, 150000]], "Distributive": [[1.0, -200], [-200, 100000]],
                "Hypovolemic": [[0.4, -50], [-50, 200000]], "Stable": [[0.6, -150], [-150, 150000]]}
        scores = {}; total = 0
        x = [row['CI'], row['SVRI']]
        for k, m in means.items():
            try:
                scores[k] = multivariate_normal.pdf(x, m, covs[k])
                total += scores[k]
            except: scores[k] = 0
        return {k: (v/total)*100 for k, v in scores.items()} if total > 1e-9 else {k:25.0 for k in means}

    @staticmethod
    def rl_advisor(row, drugs):
        if row['MAP'] < 65:
            if row['CI'] < 2.2: return "Titrate Dobutamine", 88
            else: return "Increase Norepi", 90
        return "Maintain", 99

    @staticmethod
    def detect_anomalies(df):
        # 14. Autoencoder (MLPRegressor bottleneck approach)
        scaler = StandardScaler()
        data = df[['MAP','CI','SVRI']].fillna(0)
        X = scaler.fit_transform(data)
        
        # Simple Autoencoder via MLP: Input -> 2 -> Input
        ae = MLPRegressor(hidden_layer_sizes=(2,), activation='tanh', solver='adam', max_iter=200, random_state=42)
        ae.fit(X, X)
        X_pred = ae.predict(X)
        mse = np.mean((X - X_pred)**2, axis=1)
        
        # 9. Drift Detection (Concept Drift via KS Test on recent vs old windows)
        drift = 0
        if len(df) > 60:
            stat, p_val = ks_2samp(df['MAP'].iloc[:30], df['MAP'].iloc[-30:])
            drift = 1 if p_val < 0.05 else 0

        df['anomaly'] = np.where(mse > np.percentile(mse, 95), -1, 1)
        df['drift_flag'] = drift
        return df

    @staticmethod
    def monte_carlo_forecast(df, target='MAP', n_sims=50):
        curr, hist = df[target].iloc[-1], df[target].iloc[-30:]
        vol = max(np.std(hist) if len(hist)>1 else 1.0, 0.5)
        paths = np.array([curr + np.cumsum(np.random.normal(0, vol, 30)) for _ in range(n_sims)])
        return np.percentile(paths, 10, 0), np.percentile(paths, 50, 0), np.percentile(paths, 90, 0)

    @staticmethod
    def spc_multivariate(df):
        # 7. MEWMA (Multivariate EWMA) Implementation
        X = df[['MAP', 'CI', 'SVRI']].to_numpy()
        lambda_ = 0.2
        Z = np.zeros_like(X)
        Z[0] = X[0]
        for i in range(1, len(X)):
            Z[i] = lambda_ * X[i] + (1 - lambda_) * Z[i-1]
        
        try:
            lw = LedoitWolf().fit(X[:60]) 
            diff = X - lw.location_
            t2 = np.sum(np.dot(diff, lw.precision_) * diff, axis=1)
            
            # MEWMA T2
            mewma_diff = Z - lw.location_
            mewma_t2 = np.sum(np.dot(mewma_diff, lw.precision_) * mewma_diff, axis=1)

            pca = PCA(2).fit(StandardScaler().fit_transform(X))
            X_recon = pca.inverse_transform(pca.transform(StandardScaler().fit_transform(X)))
            spe = np.sum((StandardScaler().fit_transform(X) - X_recon)**2, axis=1)
            return t2, spe, mewma_t2
        except: return np.zeros(len(X)), np.zeros(len(X)), np.zeros(len(X))
    
    @staticmethod
    def inverse_centroids(df):
        try:
            if len(df)<10: return ["Data Insufficient"]
            sc = StandardScaler()
            X = sc.fit_transform(df[['CI','SVRI','Lactate']].fillna(0))
            ctrs = sc.inverse_transform(KMeans(3, random_state=42, n_init=10).fit(X).cluster_centers_)
            return [f"C{i+1}: CI={c[0]:.1f}, SVR={c[1]:.0f}" for i,c in enumerate(ctrs)]
        except: return ["Calc Error"]

    @staticmethod
    def kalman_filter(data, Q=1e-5, R=0.1):
        # 17. Kalman Filter (1D)
        n_iter = len(data)
        sz = (n_iter,) 
        xhat = np.zeros(sz)      
        P = np.zeros(sz)         
        xhatminus = np.zeros(sz) 
        Pminus = np.zeros(sz)    
        K = np.zeros(sz)         
        
        xhat[0] = data[0]
        P[0] = 1.0
        
        for k in range(1, n_iter):
            xhatminus[k] = xhat[k-1]
            Pminus[k] = P[k-1] + Q
            K[k] = Pminus[k]/(Pminus[k]+R)
            xhat[k] = xhatminus[k]+K[k]*(data[k]-xhatminus[k])
            P[k] = (1-K[k])*Pminus[k]
            
        return xhat

    @staticmethod
    def check_granger(df):
        # 10. Granger Causality (HR -> MAP)
        try:
            # Check if HR causes MAP changes
            data = df[['MAP', 'HR']].diff().dropna()
            if len(data) > 20:
                res = grangercausalitytests(data, maxlag=2, verbose=False)
                p_val = res[1][0]['ssr_ftest'][1]
                return f"Positive (p={p_val:.3f})" if p_val < 0.05 else "None"
        except: pass
        return "Insuff. Data"

# ==========================================
# 5. QUALITY ASSURANCE ENGINE
# ==========================================
class QualityAssurance:
    @staticmethod
    def calculate_cpk(data, usl, lsl):
        mean = np.mean(data); std = np.std(data)
        if std == 0: return 0
        return min((usl - mean) / (3 * std), (mean - lsl) / (3 * std))

    @staticmethod
    def get_subgroups(data, size=5):
        n = len(data); n_trim = n - (n % size)
        if n_trim == 0: return data.reshape(1, -1)
        return data[:n_trim].reshape(-1, size)

    @staticmethod
    def simulate_noisy_sensor(true_data, bias=5, noise_std=8):
        return true_data + bias + np.random.normal(0, noise_std, len(true_data))

    @staticmethod
    def calc_ewma(data, lam=0.2):
        ewma = np.zeros_like(data); ewma[0] = data[0]
        for i in range(1, len(data)): ewma[i] = lam * data[i] + (1 - lam) * ewma[i-1]
        return ewma

    @staticmethod
    def calc_cusum(data, k=0.5):
        z = (data - np.mean(data)) / (np.std(data) if np.std(data)>0 else 1)
        cp, cm = np.zeros_like(data), np.zeros_like(data)
        for i in range(1, len(data)):
            cp[i] = max(0, z[i] - k + cp[i-1])
            cm[i] = max(0, -k - z[i] + cm[i-1])
        return cp, cm

    @staticmethod
    def calc_mcusum(data, k=0.5):
        # 8. MCUSUM (Multivariate CUSUM) - Simplified Norm approach
        mean_vec = np.mean(data, axis=0)
        inv_cov = np.linalg.inv(np.cov(data.T) + np.eye(data.shape[1])*1e-6)
        s = np.zeros(len(data))
        for i in range(len(data)):
            diff = data[i] - mean_vec
            dist = np.sqrt(diff.T @ inv_cov @ diff)
            if i > 0: s[i] = max(0, s[i-1] + dist - k)
            else: s[i] = max(0, dist - k)
        return s

    @staticmethod
    def check_westgard(data):
        # 19. Full Westgard Rules
        mean, std = np.mean(data), np.std(data)
        if std == 0: return []
        z = (data - mean) / std
        violations = []
        if len(z)>0 and abs(z[-1]) > 3: violations.append("1-3s (Random)") # 1 point outside 3SD
        if len(z)>1 and abs(z[-1]) > 2 and abs(z[-2]) > 2: violations.append("2-2s (Systematic)") # 2 points outside 2SD
        if len(z)>3 and np.ptp(z[-4:]) > 4: violations.append("R-4s (Range)") # Range > 4SD
        if len(z)>4 and all(abs(x) > 1 for x in z[-4:]): violations.append("4-1s (Systematic)") # 4 points outside 1SD
        if len(z)>10 and all(x > 0 for x in z[-10:]): violations.append("10x (Shift)") # 10 points one side of mean
        return violations

    @staticmethod
    def check_nelson(data):
        # 20. Nelson Rules
        violations = []
        if len(data) < 15: return violations
        mean, std = np.mean(data), np.std(data)
        
        # Rule 3: 6 points in a row steadily increasing or decreasing
        diffs = np.diff(data[-7:])
        if all(d > 0 for d in diffs) or all(d < 0 for d in diffs): violations.append("Trend (6 inc/dec)")
        
        # Rule 4: 14 points in a row alternating up and down
        if len(data) >= 14:
            alt = data[-14:]
            diffs = np.diff(alt)
            if all(diffs[i] * diffs[i+1] < 0 for i in range(len(diffs)-1)): violations.append("Oscillation (14 alt)")
            
        return violations

    @staticmethod
    def gauge_rr(df):
        # 21. Gauge R&R (Simplified ANOVA method)
        # Simulating 3 operators measuring the last 10 points
        try:
            part_var = np.var(df['MAP'].iloc[-10:])
            # Simulate operator error
            op_var = 5.0 
            equip_var = 2.0
            total_var = part_var + op_var + equip_var
            rr_pct = (np.sqrt(op_var + equip_var) / np.sqrt(total_var)) * 100
            return rr_pct
        except: return 0.0

class ForecastingEngine:
    @staticmethod
    def fit_predict(data, steps=30):
        try:
            hw = ExponentialSmoothing(data, trend='add').fit().forecast(steps)
        except:
            hw = np.zeros(steps)
        return hw
    
    @staticmethod
    def fit_arima(data, steps=30):
        # 15. ARIMA
        try:
            model = ARIMA(data, order=(1,1,1))
            res = model.fit()
            return res.forecast(steps)
        except: return np.zeros(steps)

    @staticmethod
    def prophet_lite(data, steps=30):
        # 16. Prophet (Lite/Additive Model simulation)
        # Decomposes into Trend + Seasonality
        t = np.arange(len(data))
        # Trend
        poly = np.polyfit(t, data, 1)
        trend = np.polyval(poly, t)
        detrended = data - trend
        # Seasonality (assume ~60 sample cycle)
        season_idx = t % 60
        season_avg = np.zeros(60)
        for i in range(60):
            mask = (season_idx == i)
            if np.any(mask): season_avg[i] = np.mean(detrended[mask])
        
        # Forecast
        future_t = np.arange(len(data), len(data)+steps)
        future_trend = np.polyval(poly, future_t)
        future_season = np.array([season_avg[i % 60] for i in future_t])
        return future_trend + future_season

class MLEngine:
    @staticmethod
    def run_models(df):
        # Prepare Data
        target = 'MAP'
        feats = ['HR', 'CI', 'SVRI', 'Lactate']
        df_clean = df.dropna()
        X = df_clean[feats].iloc[:-1]
        y = df_clean[target].shift(-1).dropna() # Predict next MAP
        X = X.iloc[:len(y)]
        
        last_row = df_clean[feats].iloc[-1].values.reshape(1, -1)
        
        results = {}
        
        if len(X) > 20:
            # 11. Random Forest (Feature Importance)
            rf = RandomForestRegressor(n_estimators=10, max_depth=3)
            rf.fit(X, y)
            results['RF_Imp'] = dict(zip(feats, rf.feature_importances_))
            results['RF_Pred'] = rf.predict(last_row)[0]
            
            # 12. XGB (Gradient Boosting as proxy)
            xgb = GradientBoostingRegressor(n_estimators=10, max_depth=3)
            xgb.fit(X, y)
            results['XGB_Pred'] = xgb.predict(last_row)[0]
            
            # 13. Logistic Regression (Shock Classification)
            # 0 = No Shock (>65), 1 = Shock (<65)
            y_bin = (y < 65).astype(int)
            if len(np.unique(y_bin)) > 1:
                log = LogisticRegression()
                log.fit(X, y_bin)
                results['Shock_Prob'] = log.predict_proba(last_row)[0][1]
            else:
                results['Shock_Prob'] = 0.0
        else:
            results = {'RF_Pred': 0, 'XGB_Pred': 0, 'Shock_Prob': 0, 'RF_Imp': {f:0 for f in feats}}
            
        return results

# ==========================================
# 6. PATIENT SIMULATOR
# ==========================================
class PatientSimulator:
    def __init__(self, mins=360):
        self.mins = mins
        self.t = np.arange(mins)
        
    def run(self, case_id, drugs, fluids, bsa, peep, is_paced, vent_mode):
        cases = {
            "65M Post-CABG": {'ci':(2.4, 1.8), 'map':(75, 55), 'svri':(2000, 1600), 'hr':(85, 95), 'shunt':0.10, 'copd':1.0, 'vo2_stress': 1.0},
            "24F Septic Shock": {'ci':(3.5, 5.5), 'map':(65, 45), 'svri':(1200, 500), 'hr':(110, 140), 'shunt':0.15, 'copd':1.0, 'vo2_stress': 1.4},
            "82M HFpEF Sepsis": {'ci':(2.2, 1.8), 'map':(85, 55), 'svri':(2400, 1800), 'hr':(80, 110), 'shunt':0.20, 'copd':1.5, 'vo2_stress': 1.1},
            "50M Trauma": {'ci':(3.0, 1.5), 'map':(70, 40), 'svri':(2500, 3200), 'hr':(100, 150), 'shunt':0.05, 'copd':1.0, 'vo2_stress': 0.9}
        }
        p = cases[case_id]
        seed = len(case_id)+42
        Utils.set_seed(seed)
        
        # Auto-Regulation
        hr, map_r, svri_r, rr = Physiology.Autonomic.generate(self.mins, p, is_paced, vent_mode)
        
        # 3. Frank-Starling Logic (Preload + Fluids -> CI)
        fluid_status = 10 + (fluids/1000) # Base preload + volume
        contractility = 1.0 if "HFpEF" not in case_id else 0.6
        ci_r = np.array([Physiology.Autonomic.frank_starling(fluid_status, contractility) for _ in range(self.mins)])
        
        ppv = (20 if "Trauma" in case_id else 12) + (np.sin(self.t/8)*4)
        
        map_f, ci_f, hr_f, svri_f = Physiology.PKPD.apply(map_r, ci_r, hr, svri_r, drugs, self.mins)
        pao2, paco2, spo2, vd_vt = Physiology.Respiratory.exchange(drugs['fio2'], rr, p['shunt'], peep, self.mins, p['copd'])
        hb = 8.0 if "Trauma" in case_id else 12.0
        do2i, vo2i, o2er, lactate = Physiology.Metabolic.calculate(ci_f, hb, spo2, self.mins, p['vo2_stress'])
        
        df = pd.DataFrame({
            "Time": self.t, "HR": hr_f, "MAP": map_f, "CI": ci_f, "SVRI": svri_f,
            "CO": ci_f * bsa, "SVR": svri_f / bsa,
            "Lactate": lactate, "SpO2": spo2, "PaO2": pao2, "PaCO2": paco2, "RR": rr,
            "DO2I": do2i, "VO2I": vo2i, "O2ER": o2er, "Vd/Vt": vd_vt,
            "CPO": (map_f * (ci_f * bsa)) / 451
        }).fillna(0)
        return df

# ==========================================
# 7. VISUALIZATION LAYER (FULLY LABELED)
# ==========================================
class Viz:
    @staticmethod
    def spark(data, color, key):
        fig = px.line(x=np.arange(len(data)), y=data)
        fig.update_traces(line_color=color, line_width=2)
        fig.update_layout(showlegend=False, xaxis_visible=False, yaxis_visible=False, margin=dict(l=0,r=0,t=0,b=0), height=35)
        return fig

    @staticmethod
    def bayes(probs, key):
        fig = go.Figure(go.Bar(x=list(probs.values()), y=list(probs.keys()), orientation='h', marker=dict(color=[CONFIG.COLORS['hemo'], CONFIG.COLORS['crit'], CONFIG.COLORS['warn'], CONFIG.COLORS['ok']])))
        fig.update_layout(margin=dict(l=0,r=0,t=25,b=0), height=120, xaxis=dict(title="Probability [%]", range=[0, 100]), title="Bayesian State Estimation")
        return fig

    @staticmethod
    def attractor_3d(df, key):
        r = df.iloc[-60:]
        fig = go.Figure(go.Scatter3d(x=r['CPO'], y=r['SVRI'], z=r['Lactate'], mode='lines+markers', marker=dict(size=3, color=r.index, colorscale='Viridis'), line=dict(width=2)))
        fig.update_layout(scene=dict(xaxis_title='Power [W]', yaxis_title='SVRI [dyn·s]', zaxis_title='Lactate [mM]'), margin=dict(l=0,r=0,b=0,t=30), height=250, title="3D Phase Space Trajectory")
        return fig

    @staticmethod
    def chaos(df, source, key):
        hr = np.maximum(df['HR'].iloc[-120:], 1.0); rr = 60000 / hr
        c = CONFIG.COLORS['ext'] if "EXTERNAL" in source else 'teal'
        fig = go.Figure(go.Scatter(x=rr.iloc[:-1], y=rr.iloc[1:], mode='markers', marker=dict(color=c, size=4, opacity=0.6)))
        fig.update_layout(title=f"Chaos: {source}", height=200, margin=dict(l=20,r=20,t=30,b=20), xaxis_title="RR(n) [ms]", yaxis_title="RR(n+1) [ms]")
        return fig
    
    @staticmethod
    def recurrence_plot(df, key):
        # 22. Phase-space Recurrence (RQA)
        data = df['MAP'].iloc[-60:].values
        n = len(data)
        dist_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dist_mat[i, j] = abs(data[i] - data[j])
        fig = px.imshow(dist_mat, color_continuous_scale='gray_r', origin='lower')
        fig.update_layout(title="Recurrence Plot (Phase Space)", height=200, margin=dict(l=20,r=20,t=30,b=20), xaxis_visible=False, yaxis_visible=False)
        return fig

    @staticmethod
    def spectral(df, key):
        data = df['HR'].iloc[-120:].to_numpy()
        f, Pxx = welch(data, fs=1/60)
        fig = px.line(x=f, y=Pxx)
        fig.add_vline(x=0.04, line_dash="dot", annotation_text="LF"); fig.add_vline(x=0.15, line_dash="dot", annotation_text="HF")
        fig.update_layout(title="Spectral HRV (Fourier)", height=200, margin=dict(l=20,r=20,t=30,b=20), xaxis_title="Frequency [Hz]", yaxis_title="Power Density")
        return fig

    @staticmethod
    def hemodynamic_profile(df, key):
        r = df.iloc[-60:]
        fig = go.Figure()
        fig.add_hline(y=2000, line_dash="dot", annotation_text="Vaso"); fig.add_vline(x=2.2, line_dash="dot", annotation_text="Low Flow")
        fig.add_trace(go.Scatter(x=r['CI'], y=r['SVRI'], mode='markers', marker=dict(color=r.index, colorscale='Viridis'), name="State"))
        fig.update_layout(title="Pump vs Pipes (Forrester)", height=250, margin=dict(l=20,r=20,t=30,b=20), xaxis_title="CI [L/min/m²]", yaxis_title="SVRI [dyn·s·cm⁻⁵·m²]")
        return fig

    @staticmethod
    def phase_space(df, key):
        r = df.iloc[-60:]
        fig = go.Figure()
        fig.add_shape(type="rect", x0=0, x1=0.6, y0=2, y1=15, fillcolor="rgba(255,0,0,0.1)", line_width=0)
        fig.add_trace(go.Scatter(x=r['CPO'], y=r['Lactate'], mode='lines+markers', marker=dict(color=r.index, colorscale='Bluered'), name="Traj"))
        fig.update_layout(title="Hemo-Metabolic Coupling", height=250, margin=dict(l=20,r=20,t=30,b=20), xaxis_title="Power [W]", yaxis_title="Lactate [mM]")
        return fig

    @staticmethod
    def vq_scatter(df, key):
        fig = px.scatter(df.iloc[-60:], x="PaO2", y="SpO2", color="PaCO2", color_continuous_scale="Bluered")
        fig.update_layout(title="V/Q Status", height=250, margin=dict(l=20,r=20,t=30,b=20), xaxis_title="PaO2 [mmHg]", yaxis_title="SpO2 [%]")
        return fig

    @staticmethod
    def spc_charts(df, key):
        data = df['MAP'].to_numpy()
        xbar = np.mean(QualityAssurance.get_subgroups(data), axis=1)
        # FIX: Added specs for 'domain' type trace (Gauge)
        fig = make_subplots(rows=1, cols=3, subplot_titles=("X-Bar", "R-Chart", "Gauge R&R"), 
                            specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "domain"}]])
        fig.add_trace(go.Scatter(y=xbar, mode='lines+markers'), row=1, col=1)
        fig.add_hline(y=np.mean(xbar)+3*np.std(xbar), line_color='red', row=1, col=1)
        fig.add_trace(go.Scatter(y=np.ptp(QualityAssurance.get_subgroups(data), axis=1), mode='lines+markers'), row=1, col=2)
        # Gauge R&R Heatmap
        rr = QualityAssurance.gauge_rr(df)
        fig.add_trace(go.Indicator(mode="number+gauge", value=rr, title="Gauge R&R %", gauge={'axis': {'range': [0, 100]}}), row=1, col=3)
        fig.update_layout(height=250, margin=dict(l=10,r=10,t=30,b=20), title="Statistical Process Control & MSA")
        return fig

    @staticmethod
    def method_comp(df, key):
        true = df['MAP'].iloc[-120:].to_numpy(); noise = QualityAssurance.simulate_noisy_sensor(true)
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Bland-Altman", "Deming Reg"))
        diff = true - noise
        fig.add_trace(go.Scatter(x=(true+noise)/2, y=diff, mode='markers'), row=1, col=1)
        fig.add_hline(y=np.mean(diff)+1.96*np.std(diff), line_dash='dot', line_color='red', row=1, col=1)
        fig.add_trace(go.Scatter(x=true, y=noise, mode='markers'), row=1, col=2)
        fig.update_layout(height=250, margin=dict(l=10,r=10,t=30,b=20), title="Sensor Validation")
        return fig

    @staticmethod
    def cpk_tol(df, key):
        d = df['MAP'].iloc[-120:]; cpk = QualityAssurance.calculate_cpk(d, CONFIG.MAP_USL, CONFIG.MAP_LSL)
        fig = make_subplots(rows=1, cols=2, subplot_titles=(f"Cpk={cpk:.2f}", "Tolerance"))
        fig.add_trace(go.Scatter(x=np.linspace(40,130,100), y=norm.pdf(np.linspace(40,130,100), d.mean(), d.std()), fill='tozeroy'), row=1, col=1)
        fig.add_vline(x=65, line_color='red', row=1, col=1); fig.add_vline(x=110, line_color='red', row=1, col=1)
        fig.add_trace(go.Scatter(y=d, mode='lines'), row=1, col=2)
        fig.add_hrect(y0=65, y1=110, fillcolor='green', opacity=0.1, row=1, col=2)
        fig.update_layout(height=250, margin=dict(l=10,r=10,t=30,b=20), title="Process Capability")
        return fig

    @staticmethod
    def forecast(df, p10, p50, p90, key):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.arange(30), y=df['MAP'].iloc[-30:], name="History", line=dict(color='black')))
        fx = np.arange(30, 60)
        fig.add_trace(go.Scatter(x=np.concatenate([fx, fx[::-1]]), y=np.concatenate([p90, p10[::-1]]), fill='toself', fillcolor='rgba(0,0,255,0.2)', line=dict(width=0), name="CI"))
        fig.add_trace(go.Scatter(x=fx, y=p50, line=dict(dash='dot', color='blue'), name="Median"))
        fig.update_layout(height=250, margin=dict(l=20,r=20,t=30,b=20), title="Monte Carlo Forecast", xaxis_title="Time [min]", yaxis_title="MAP [mmHg]")
        return fig

    @staticmethod
    def counterfactual(df, df_b, key):
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=df['MAP'].iloc[-60:], name="Rx", line=dict(color=CONFIG.COLORS['ok'])))
        fig.add_trace(go.Scatter(y=df_b['MAP'].iloc[-60:], name="No Rx", line=dict(dash='dot', color=CONFIG.COLORS['crit'])))
        fig.update_layout(height=250, margin=dict(l=20,r=20,t=30,b=20), title="Counterfactual", xaxis_title="Time [min]", yaxis_title="MAP [mmHg]")
        return fig

    @staticmethod
    def mspc(t2, spe, mewma_t2, key):
        fig = make_subplots(rows=1, cols=3, subplot_titles=("T² (Sys)", "SPE (Resid)", "MEWMA"))
        fig.add_trace(go.Scatter(y=t2), row=1, col=1); fig.add_hline(y=chi2.ppf(0.99, 3), line_color='red', row=1, col=1)
        fig.add_trace(go.Scatter(y=spe), row=1, col=2)
        fig.add_trace(go.Scatter(y=mewma_t2, line_color='orange'), row=1, col=3)
        fig.update_layout(height=250, margin=dict(l=10,r=10,t=30,b=20), title="Multivariate SPC & MEWMA")
        return fig

    @staticmethod
    def adv_control(df, key):
        d = df['MAP'].to_numpy()
        ewma = QualityAssurance.calc_ewma(d); cp, cm = QualityAssurance.calc_cusum(d)
        violations = QualityAssurance.check_westgard(d) + QualityAssurance.check_nelson(d)
        fig = make_subplots(rows=1, cols=2, subplot_titles=("EWMA", "CUSUM"))
        fig.add_trace(go.Scatter(y=d, line=dict(color='gray'), name="Raw"), row=1, col=1)
        fig.add_trace(go.Scatter(y=ewma, line=dict(color='blue'), name="EWMA"), row=1, col=1)
        # CUSUM simplified visualization
        fig.add_trace(go.Scatter(y=cp, name="C+", fill='tozeroy'), row=1, col=2)
        fig.add_trace(go.Scatter(y=cm, name="C-", fill='tozeroy'), row=1, col=2)
        title_str = "Stable" if not violations else f"Violations: {len(violations)}"
        fig.update_layout(height=250, margin=dict(l=10,r=10,t=30,b=20), title=f"Adv Control ({title_str})")
        return fig

    @staticmethod
    def adv_forecast(df, key):
        hist = df['MAP'].iloc[-60:].to_numpy()
        hw = ForecastingEngine.fit_predict(hist)
        arima = ForecastingEngine.fit_arima(hist)
        prophet = ForecastingEngine.prophet_lite(hist)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.arange(60), y=hist, name="Hx", line=dict(color='black')))
        fig.add_trace(go.Scatter(x=np.arange(60,90), y=hw, name="ETS", line=dict(dash='dot', color='green')))
        fig.add_trace(go.Scatter(x=np.arange(60,90), y=arima, name="ARIMA", line=dict(dash='dash', color='orange')))
        fig.add_trace(go.Scatter(x=np.arange(60,90), y=prophet, name="Prophet", line=dict(dash='dot', color='purple')))
        fig.update_layout(height=250, margin=dict(l=20,r=20,t=30,b=20), title="Ensemble Forecasting", xaxis_title="Steps", yaxis_title="MAP [mmHg]")
        return fig

    @staticmethod
    def wasserstein(df, k_filt, key):
        e = df['MAP'].iloc[:60]; l = df['MAP'].iloc[-60:]
        d = wasserstein_distance(e, l)
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=e, opacity=0.5, name="Baseline"))
        fig.add_trace(go.Histogram(x=l, opacity=0.5, name="Current"))
        fig.add_trace(go.Scatter(y=k_filt, mode='lines', name='Kalman State', yaxis='y2'))
        fig.update_layout(height=200, margin=dict(l=10,r=10,t=30,b=20), title=f"Dist Shift (W={d:.1f}) & Kalman", barmode='overlay', xaxis_title="MAP [mmHg]", yaxis_title="Count", yaxis2=dict(overlaying='y', side='right', showgrid=False))
        return fig
    
    @staticmethod
    def ml_dashboard(res, key):
        # FIX: Added specs for 'domain' type trace (Indicator)
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Feature Imp (RF)", "Shock Prob (LogReg)"),
                            specs=[[{"type": "xy"}, {"type": "domain"}]])
        fig.add_trace(go.Bar(x=list(res['RF_Imp'].values()), y=list(res['RF_Imp'].keys()), orientation='h'), row=1, col=1)
        fig.add_trace(go.Indicator(mode="gauge+number", value=res['Shock_Prob']*100, title="Prob %"), row=1, col=2)
        fig.update_layout(height=200, margin=dict(l=10,r=10,t=30,b=20), title=f"ML Engine (XGB Pred MAP: {res['XGB_Pred']:.1f})")
        return fig

# ==========================================
# 8. APP ORCHESTRATION
# ==========================================
class App:
    def __init__(self):
        self.sim = None
        self.df = None
        self.drugs = {}
        
    def run(self):
        st.markdown(STYLING, unsafe_allow_html=True)
        if 'events' not in st.session_state: st.session_state['events'] = []
        if 'fluids' not in st.session_state: st.session_state['fluids'] = 0
        
        with st.sidebar:
            st.title("TITAN | L8")
            res_mins = st.select_slider("Resolution", [60, 180, 360, 720], value=360)
            case_id = st.selectbox("Profile", ["65M Post-CABG", "24F Septic Shock", "82M HFpEF Sepsis", "50M Trauma"])
            
            c1, c2 = st.columns(2)
            with c1: h = st.number_input("Ht", 150, 200, 175); norepi = st.number_input("Norepi", 0.0, 2.0, 0.0)
            with c2: w = st.number_input("Wt", 50, 150, 80); vaso = st.number_input("Vaso", 0.0, 0.1, 0.0)
            bsa = np.sqrt((h*w)/3600)
            
            dobu = st.number_input("Dobutamine", 0.0, 10.0, 0.0); bb = st.number_input("Esmolol", 0.0, 1.0, 0.0)
            fio2 = st.slider("FiO2", 0.21, 1.0, 0.4); peep = st.slider("PEEP", 0, 20, 5)
            is_paced = st.checkbox("Pacemaker"); vent_mode = st.selectbox("Vent", ["Spontaneous", "Control (AC)"])
            
            if st.button("Bolus 500mL"): st.session_state['fluids'] += 500
            live = st.checkbox("LIVE")

        self.sim = PatientSimulator(res_mins)
        self.drugs = {'norepi':norepi, 'vaso':vaso, 'dobu':dobu, 'bb':bb, 'fio2':fio2}
        
        df = self.sim.run(case_id, self.drugs, st.session_state['fluids'], bsa, peep, is_paced, vent_mode)
        
        sim_b = PatientSimulator(60)
        base = {'norepi':0, 'vaso':0, 'dobu':0, 'bb':0, 'fio2':0.21}
        df_b = sim_b.run(case_id, base, 0, bsa, peep, False, 'Spontaneous')
        
        # Analytics Pipeline
        df = Analytics.detect_anomalies(df)
        probs = Analytics.bayes_shock(df.iloc[-1])
        sugg, conf = Analytics.rl_advisor(df.iloc[-1], self.drugs)
        p10, p50, p90 = Analytics.monte_carlo_forecast(df)
        t2, spe, mewma_t2 = Analytics.spc_multivariate(df)
        src, _, reason, spec_ent = Analytics.signal_forensics(df['HR'].iloc[-120:], is_paced)
        centroids = Analytics.inverse_centroids(df)
        granger = Analytics.check_granger(df)
        k_filt = Analytics.kalman_filter(df['MAP'].values)
        ml_res = MLEngine.run_models(df)
        
        if live:
            holder = st.empty()
            for i in range(max(10, res_mins-60), res_mins):
                # Slicing data for live simulation
                with holder.container(): self.layout(df.iloc[:i], df_b, probs, sugg, conf, p10, p50, p90, t2[:i], spe[:i], mewma_t2[:i], src, reason, centroids, i, granger, k_filt[:i], spec_ent, ml_res)
                time.sleep(0.1)
        else:
            self.layout(df, df_b, probs, sugg, conf, p10, p50, p90, t2, spe, mewma_t2, src, reason, centroids, len(df), granger, k_filt, spec_ent, ml_res)

    def layout(self, df, df_b, probs, sugg, conf, p10, p50, p90, t2, spe, mewma_t2, src, reason, centroids, ix, granger, k_filt, spec_ent, ml_res):
        curr = df.iloc[-1]; prev = df.iloc[-60] if len(df)>60 else df.iloc[0]
        
        drift_status = "DRIFT" if curr.get('drift_flag',0)==1 else "STABLE"
        
        st.markdown(f"""
        <div class="status-banner" style="border-left-color: {CONFIG.COLORS['ai']};">
            <div><div style="font-size:0.8rem; font-weight:800; color:{CONFIG.COLORS['ai']}">BAYESIAN STATE</div>
            <div style="font-size:1.5rem; font-weight:800;">{max(probs, key=probs.get).upper()}</div></div>
            <div style="text-align:right">
                <div style="font-size:0.8rem; font-weight:700;">ANOMALY / DRIFT</div>
                <div class="{'crit-pulse' if curr['anomaly']==-1 else ''}" style="font-size:2rem; font-weight:800; color:{CONFIG.COLORS['crit'] if curr['anomaly']==-1 else CONFIG.COLORS['ok']}">{ 'DETECTED' if curr['anomaly']==-1 else 'NORMAL' } / {drift_status}</div>
            </div>
        </div>""", unsafe_allow_html=True)

        t_main, t_resp, t_ai, t_spc, t_adv = st.tabs(["🫀 Clinical Command", "🫁 Respiratory", "🤖 AI & Forensics", "📊 SPC & Quality", "🧪 Advanced Research"])
        
        with t_main:
            cols = st.columns(6)
            mets = [("MAP", curr['MAP']), ("CI", curr['CI']), ("SVRI", curr['SVRI']), ("Lactate", curr['Lactate']), ("O2ER", curr['O2ER']*100), ("CPO", curr['CPO'])]
            for i, (l, v) in enumerate(mets):
                cols[i].metric(l, f"{v:.1f}", f"{v - prev[l]:.1f}")
                cols[i].plotly_chart(Viz.spark(df[l].iloc[-60:], CONFIG.COLORS['hemo'], f"s{i}_{ix}"), use_container_width=True)
            
            c1, c2, c3 = st.columns(3)
            c1.plotly_chart(Viz.hemodynamic_profile(df, ix), use_container_width=True)
            c1.markdown("<div class='clinical-hint'><b>Significance:</b> Classifies shock state into clinical quadrants. <br><b>Action:</b> Low CI/High SVR (Cold/Wet) needs Inotropes. High CI/Low SVR (Warm/Dry) needs Vasopressors.</div>", unsafe_allow_html=True)
            
            c2.plotly_chart(Viz.phase_space(df, ix), use_container_width=True)
            c2.markdown("<div class='clinical-hint'><b>Significance:</b> Visualizes the coupling between Cardiac Power (Pump) and Lactate (Metabolism). <br><b>Action:</b> If trajectory moves to bottom-right (Low Power/High Lactate), immediate mechanical support (IABP/Impella) may be required.</div>", unsafe_allow_html=True)
            
            c3.plotly_chart(Viz.attractor_3d(df, ix), use_container_width=True)
            c3.markdown("<div class='clinical-hint'><b>Significance:</b> 3D Topological stability analysis. <br><b>Action:</b> Large, erratic orbits indicate a chaotic, unstable system prone to sudden crash. Tight orbits indicate stability.</div>", unsafe_allow_html=True)
            
            z1, z2 = st.columns(2)
            z1.plotly_chart(Viz.forecast(df, p10, p50, p90, ix), use_container_width=True)
            z1.markdown("<div class='clinical-hint'><b>Significance:</b> Stochastic Monte Carlo projection of MAP based on recent volatility. <br><b>Action:</b> A widening cone indicates loss of autonomic control. Downward trend requires pre-emptive vasopressor titration.</div>", unsafe_allow_html=True)
            z2.plotly_chart(Viz.counterfactual(df, df_b, ix), use_container_width=True)
            z2.markdown("<div class='clinical-hint'><b>Significance:</b> 'What If' analysis showing patient trajectory without current interventions. <br><b>Action:</b> The gap between lines represents the 'Value Add' of your current therapy. If lines converge, therapy is futile.</div>", unsafe_allow_html=True)

        with t_resp:
            c1, c2 = st.columns(2)
            c1.plotly_chart(Viz.vq_scatter(df, ix), use_container_width=True)
            c1.markdown("<div class='clinical-hint'><b>Significance:</b> Distinguishes Shunt from Dead Space. <br><b>Action:</b> Vertical scatter (Hypoxia) -> Increase PEEP/FiO2. Horizontal scatter (Hypercapnia) -> Increase Ventilation (RR/Vt).</div>", unsafe_allow_html=True)
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True)
            fig.add_trace(go.Scatter(y=df['SpO2'], name="SpO2"), row=1, col=1)
            fig.add_trace(go.Scatter(y=df['PaCO2'], name="PaCO2"), row=2, col=1)
            fig.add_trace(go.Scatter(y=df['Vd/Vt'], name="Vd/Vt"), row=3, col=1)
            fig.update_layout(height=400, margin=dict(l=0,r=0,t=0,b=0))
            c2.plotly_chart(fig, use_container_width=True, key=f"t_resp_{ix}")
            c2.markdown("<div class='clinical-hint'><b>Significance:</b> Real-time ventilation telemetry stack. <br><b>Action:</b> Monitor Dead Space fraction (Vd/Vt). Rising Vd/Vt indicates worsening ARDS or PE.</div>", unsafe_allow_html=True)

        with t_ai:
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(Viz.bayes(probs, ix), use_container_width=True)
                st.markdown("<div class='clinical-hint'><b>Significance:</b> Posterior probability of shock state given current hemodynamics. <br><b>Action:</b> Use to confirm diagnosis when clinical picture is ambiguous.</div>", unsafe_allow_html=True)
                st.info(f"RL Advisor: {sugg} ({conf}%) | Granger Causality (HR->MAP): {granger}")
                st.markdown("<div class='clinical-hint'><b>Significance:</b> RL Policy & Causality. <br><b>Action:</b> If Granger is Positive, HR control is effective for MAP.</div>", unsafe_allow_html=True)
                st.plotly_chart(Viz.chaos(df, src, ix), use_container_width=True)
                st.markdown(f"<div class='clinical-hint'><b>Significance:</b> Signal Forensics. <br><b>Action:</b> {src}. {reason}. Spec Entropy: {spec_ent:.2f}</div>", unsafe_allow_html=True)
            with c2:
                st.plotly_chart(Viz.spectral(df, ix), use_container_width=True)
                st.markdown("<div class='clinical-hint'><b>Significance:</b> Frequency Domain HRV (Fourier). <br><b>Action:</b> LF (Low Freq) = Sympathetic. HF (High Freq) = Parasympathetic. Loss of power = Autonomic Failure.</div>", unsafe_allow_html=True)
                st.plotly_chart(Viz.wasserstein(df, k_filt, ix), use_container_width=True)
                st.markdown("<div class='clinical-hint'><b>Significance:</b> Wasserstein Metric & Kalman Filter. <br><b>Action:</b> Kalman line (red) shows true estimated state filtering out noise.</div>", unsafe_allow_html=True)
                st.plotly_chart(Viz.adv_forecast(df, ix), use_container_width=True)
                st.markdown("<div class='clinical-hint'><b>Significance:</b> Ensemble Forecasting (ETS + ARIMA + Prophet). <br><b>Action:</b> Consensus model predicts collapse earlier than single models.</div>", unsafe_allow_html=True)

        with t_spc:
            st.plotly_chart(Viz.spc_charts(df, ix), use_container_width=True)
            st.markdown("<div class='clinical-hint'><b>Significance:</b> X-Bar/R-Charts & Gauge R&R. <br><b>Action:</b> High Gauge R&R means sensor error, not patient change.</div>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            c1.plotly_chart(Viz.mspc(t2, spe, mewma_t2, ix), use_container_width=True)
            c1.markdown("<div class='clinical-hint'><b>Significance:</b> Multivariate SPC & MEWMA. <br><b>Action:</b> MEWMA (Orange) is sensitive to small covariance shifts (e.g., Early Sepsis).</div>", unsafe_allow_html=True)
            c2.plotly_chart(Viz.adv_control(df, ix), use_container_width=True)
            c2.markdown("<div class='clinical-hint'><b>Significance:</b> Nelson Rules & Westgard. <br><b>Action:</b> Violations indicate systematic error or physiological trend (6 points increasing).</div>", unsafe_allow_html=True)
            
            c3, c4 = st.columns(2)
            c3.plotly_chart(Viz.method_comp(df, ix), use_container_width=True)
            c3.markdown("<div class='clinical-hint'><b>Significance:</b> Bland-Altman Analysis. <br><b>Action:</b> Validates Invasive vs Non-Invasive BP. Wide limits of agreement = Sensor Failure.</div>", unsafe_allow_html=True)
            c4.plotly_chart(Viz.cpk_tol(df, ix), use_container_width=True)
            c4.markdown("<div class='clinical-hint'><b>Significance:</b> Process Capability Index (Cpk). <br><b>Action:</b> Cpk < 1.33 means the patient is hemodynamically unstable and likely to breach safety limits.</div>", unsafe_allow_html=True)
            
        with t_adv:
            c1, c2 = st.columns(2)
            c1.plotly_chart(Viz.recurrence_plot(df, ix), use_container_width=True)
            c1.markdown("<div class='clinical-hint'><b>Phase Space Recurrence:</b> Visualizes periodicity and determinism in MAP signal.</div>", unsafe_allow_html=True)
            c2.plotly_chart(Viz.ml_dashboard(ml_res, ix), use_container_width=True)
            c2.markdown("<div class='clinical-hint'><b>ML Engine:</b> Real-time RF Feature Importance and Logistic Shock Prediction.</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    app = App()
    app.run()
