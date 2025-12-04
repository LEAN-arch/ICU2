import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.signal import welch, find_peaks
from scipy.stats import norm, multivariate_normal, wasserstein_distance, linregress, chi2, ks_2samp
from scipy.fft import fft, fftfreq
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
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
    page_title="TITAN | L8 CLINICAL COMMAND",
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
        "ext": "#d946ef", "spc": "#059669", "gold": "#d4af37"
    }
    
    # Physics Constants
    ATM_PRESSURE = 760.0; H2O_PRESSURE = 47.0; R_QUOTIENT = 0.8; MAX_PAO2 = 600.0
    HB_CONVERSION = 1.34; LAC_PROD_THRESH = 330.0; LAC_CLEAR_RATE = 0.08; VCO2_CONST = 130
    
    # Advanced PK/PD: Ke0 determines the delay between plasma and effect site
    # Emax model: Hill equation parameters for realistic non-linear dose response
    DRUG_PK = {
        'norepi': {'ke0': 0.2,  'ec50': 0.3, 'n': 1.5, 'emax': {'svr': 2500.0, 'map': 80.0, 'hr': 10.0}}, 
        'vaso':   {'ke0': 0.05, 'ec50': 0.04, 'n': 2.0, 'emax': {'svr': 3500.0, 'map': 90.0, 'co': -0.8}}, 
        'dobu':   {'ke0': 0.15, 'ec50': 5.0,  'n': 1.2, 'emax': {'svr': -600.0, 'co': 3.5, 'hr': 35.0, 'map': 5.0}}, 
        'bb':     {'ke0': 0.1,  'ec50': 0.5,  'n': 1.0, 'emax': {'svr': 50.0, 'map': -30.0, 'co': -1.5, 'hr': -45.0}}
    }
    
    # Clinical Limits
    MAP_LSL = 65.0; MAP_USL = 110.0
    LACTATE_CRIT = 2.0; SCVO2_CRIT = 70.0

STYLING = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=Roboto+Mono:wght@500;700&display=swap');
    .stApp {{ background-color: {CONFIG.COLORS['bg']}; color: {CONFIG.COLORS['text']}; font-family: 'Inter', sans-serif; }}
    
    div[data-testid="stMetric"] {{ background-color: {CONFIG.COLORS['card']}; border-left: 4px solid {CONFIG.COLORS['info']}; border-radius: 6px; padding: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }}
    div[data-testid="stMetric"] label {{ color: {CONFIG.COLORS['muted']}; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.05em; }}
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{ font-family: 'Roboto Mono', monospace; font-size: 1.6rem; font-weight: 800; }}
    
    .status-banner {{ padding: 12px 20px; border-radius: 8px; background: {CONFIG.COLORS['card']}; border-left: 6px solid {CONFIG.COLORS['ai']}; margin-bottom: 20px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); display: flex; justify-content: space-between; align-items: center; }}
    .clinical-hint {{ font-size: 0.8rem; color: #334155; background: #f1f5f9; padding: 12px; border-radius: 6px; margin-top: 5px; border-left: 4px solid {CONFIG.COLORS['info']}; }}
    .action-header {{ font-weight: 800; text-transform: uppercase; color: {CONFIG.COLORS['crit']}; font-size: 0.75rem; margin-top: 4px; }}
    .sig-header {{ font-weight: 800; text-transform: uppercase; color: {CONFIG.COLORS['info']}; font-size: 0.75rem; }}
    
    .crit-pulse {{ animation: pulse-red 1.5s infinite; color: {CONFIG.COLORS['crit']}; font-weight: 900; }}
    @keyframes pulse-red {{ 0% {{ opacity: 1; transform: scale(1); }} 50% {{ opacity: 0.6; transform: scale(1.05); }} 100% {{ opacity: 1; transform: scale(1); }} }}
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
    def pink_noise(n):
        # 1/f noise for physiological realism
        uneven = n % 2
        X = Utils._rng.standard_normal(n // 2 + 1 + uneven) + 1j * Utils._rng.standard_normal(n // 2 + 1 + uneven)
        S = np.sqrt(np.arange(len(X)) + 1.)
        y = (np.fft.irfft(X / S)).real
        if uneven: y = y[:-1]
        return (y - np.mean(y)) / (np.std(y) + 1e-8)

# ==========================================
# 3. PHYSIOLOGY ENGINES (COUPLED ODE)
# ==========================================
class Physiology:
    class Autonomic:
        @staticmethod
        def baroreflex(current_map, current_hr, target_map=75, gain=0.8):
            # Dynamic Baroreflex with hysteresis
            error = target_map - current_map
            # Sympathetic activation takes time, Vagal is fast. Simplified here.
            delta_hr = error * gain
            return np.clip(current_hr + (delta_hr * 0.1), 40, 190)

        @staticmethod
        def frank_starling(preload_vol, contractility):
            # Sigmoidal relationship: SV = (a * Preload^n) / (b^n + Preload^n)
            # Contractility shifts the curve up/down
            sv = (130 * contractility * (preload_vol**2)) / (40**2 + preload_vol**2)
            return sv

    class PKPD:
        @staticmethod
        def hill_effect(ce, emax, ec50, n):
            # Sigmoidal Emax model
            return (emax * (ce**n)) / (ec50**n + ce**n)

        @staticmethod
        def step_drug(cp, target_conc, ke0):
            # First order transfer to effect site
            # dCe/dt = Ke0 * (Cp - Ce)
            return cp + ke0 * (target_conc - cp)

    class Metabolic:
        @staticmethod
        def lactate_kinetics(current_lac, do2, vo2, clearance_rate):
            # Anaerobic threshold logic
            production = 0.0
            if do2 < vo2: # Shock state
                production = (vo2 - do2) * 0.02 # Conversion factor
            else:
                production = 0.1 # Basal
            
            # Michaelis-Menten like clearance
            clearance = current_lac * clearance_rate
            return max(0.5, current_lac + production - clearance)

# ==========================================
# 4. ANALYTICS & FORENSICS
# ==========================================
class Analytics:
    @staticmethod
    def calc_hemodynamic_coherence(df):
        # Correlation between Macro (MAP) and Micro (Lactate)
        if len(df) < 20: return 0
        w = df.iloc[-30:]
        corr, _ = linregress(w['MAP'], w['Lactate'])[:2]
        # Negative correlation is GOOD (Higher MAP = Lower Lactate)
        # Positive correlation implies Micro-circulatory failure (Shunting)
        return corr

    @staticmethod
    def signal_forensics(ts, is_paced):
        arr = np.array(ts)
        N = len(arr)
        if N < 60: return "Initializing...", 0, "Wait", 0
        
        yf = fft(arr)
        psd = np.abs(yf)**2
        psd_norm = psd / np.sum(psd)
        spec_ent = -np.sum(psd_norm * np.log2(psd_norm + 1e-12)) / np.log2(N)

        if is_paced: 
            return "PACEMAKER CAPTURE", 99, "Artificial Regularity", spec_ent
        if spec_ent < 0.3:
            return "METRONOMIC RIGIDITY", 95, "Loss of Autonomic Variance", spec_ent
        if spec_ent > 0.8:
            return "AUTONOMIC INTACT", 85, "Physiologic Pink Noise", spec_ent
        
        return "STRESS PATTERN", 60, "Sympathetic Dominance", spec_ent

    @staticmethod
    def detect_anomalies(df):
        # Advanced Drift + Point Anomaly
        data = df[['MAP','CI','SVRI']].fillna(0)
        
        # 1. Isolation Forest for point anomalies
        iso = IsolationForest(contamination=0.05, random_state=42)
        df['anomaly'] = iso.fit_predict(data)
        
        # 2. Concept Drift via Rolling KS Test
        w_ref = df['MAP'].iloc[:30]
        w_cur = df['MAP'].iloc[-30:]
        drift_p = ks_2samp(w_ref, w_cur).pvalue
        df['drift_flag'] = 1 if drift_p < 0.01 else 0
        
        return df

    @staticmethod
    def monte_carlo_forecast(df, n_sims=100):
        # Geometric Brownian Motion simulation for forecast
        last_price = df['MAP'].iloc[-1]
        returns = np.diff(np.log(df['MAP'].iloc[-60:]))
        mu = np.mean(returns)
        sigma = np.std(returns)
        
        simulations = np.zeros((30, n_sims))
        for i in range(n_sims):
            path = [last_price]
            for _ in range(29):
                price = path[-1] * np.exp(mu + sigma * np.random.normal())
                path.append(price)
            simulations[:, i] = path
            
        return np.percentile(simulations, 10, axis=1), np.percentile(simulations, 50, axis=1), np.percentile(simulations, 90, axis=1)

    @staticmethod
    def spc_multivariate(df):
        # Hotelling's T2 & SPE
        X = df[['MAP', 'CI', 'SVRI']].values
        if len(X) < 10: return np.zeros(len(X)), np.zeros(len(X)), np.zeros(len(X))
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # T2 statistic
        lambda_inv = np.linalg.inv(np.diag(pca.explained_variance_))
        t2 = np.array([x @ lambda_inv @ x.T for x in X_pca])
        
        # SPE (Q-statistic)
        X_recon = pca.inverse_transform(X_pca)
        error = X_scaled - X_recon
        spe = np.sum(error**2, axis=1)
        
        # MEWMA
        lambda_ = 0.2
        z = np.zeros_like(X_scaled)
        z[0] = X_scaled[0]
        for i in range(1, len(X)):
            z[i] = lambda_ * X_scaled[i] + (1-lambda_) * z[i-1]
        mewma_dist = np.sum(z**2, axis=1) # Simplified
        
        return t2, spe, mewma_dist

    @staticmethod
    def check_granger(df):
        # Does HR predict MAP? (Responsiveness check)
        try:
            d = df[['MAP', 'HR']].diff().dropna()
            if len(d) > 25:
                gc = grangercausalitytests(d, maxlag=2, verbose=False)
                p = gc[2][0]['ssr_ftest'][1]
                return f"Positive (p={p:.3f})" if p < 0.05 else "Uncoupled"
        except: pass
        return "Calc..."

# ==========================================
# 5. QA & PROCESS CONTROL
# ==========================================
class QA:
    @staticmethod
    def gauge_rr(df):
        # Synthetic Gauge R&R based on signal noise ratio
        signal = np.std(df['MAP'].iloc[-20:])
        noise = 2.0 # Assumed sensor noise
        if signal + noise == 0: return 0
        return (noise / (signal + noise)) * 100

    @staticmethod
    def method_comp(df):
        # Bland-Altman: Invasive MAP vs NIBP (Simulated)
        inv = df['MAP'].values[-60:]
        # NIBP tends to overestimate lows and underestimate highs
        nibp = 0.8 * inv + 15 + np.random.normal(0, 5, len(inv)) 
        return inv, nibp

# ==========================================
# 6. PATIENT SIMULATOR (EULER INTEGRATION)
# ==========================================
class PatientSimulator:
    def __init__(self, mins=360):
        self.mins = mins
        self.dt = 1.0 # 1 minute steps
        
    def run(self, case_id, drugs, fluids, bsa, peep, is_paced, vent_mode):
        # 1. Initialize State Vector
        utils = Utils()
        utils.set_seed(42)
        
        # Baseline Parameters based on Case
        cases = {
            "65M Post-CABG": {'svr_b': 1800, 'co_b': 4.5, 'map_b': 75, 'vo2': 140, 'hb': 10},
            "24F Septic Shock": {'svr_b': 600, 'co_b': 6.5, 'map_b': 50, 'vo2': 180, 'hb': 11},
            "82M HFpEF Sepsis": {'svr_b': 1400, 'co_b': 3.0, 'map_b': 60, 'vo2': 130, 'hb': 9},
            "50M Trauma": {'svr_b': 2200, 'co_b': 3.5, 'map_b': 65, 'vo2': 150, 'hb': 7}
        }
        c = cases[case_id]
        
        # Arrays for history
        n = self.mins
        map_arr = np.zeros(n); ci_arr = np.zeros(n); hr_arr = np.zeros(n)
        svri_arr = np.zeros(n); lac_arr = np.zeros(n); spo2_arr = np.zeros(n)
        sv_arr = np.zeros(n)
        
        # Initial States
        curr_map = c['map_b']
        curr_co = c['co_b']
        curr_svr = c['svr_b']
        curr_hr = 90.0
        curr_lac = 1.5
        curr_vol = 50.0 + (fluids/200) # Effective circulating volume index
        
        # Drug Effect Site Concentrations (Ce)
        ce = {k: 0.0 for k in drugs}
        
        noise_map = utils.pink_noise(n) * 4.0
        noise_hr = utils.pink_noise(n) * 3.0
        
        for t in range(n):
            # A. Update Pharmacokinetics (Cp -> Ce)
            for d, dose in drugs.items():
                pk = CONFIG.DRUG_PK.get(d)
                if pk:
                    # Euler step for Ce
                    ce[d] = Physiology.PKPD.step_drug(ce[d], dose, pk['ke0'])
            
            # B. Calculate Drug Effects (Emax)
            eff_svr = 0; eff_map = 0; eff_co = 0; eff_hr = 0
            for d in drugs:
                pk = CONFIG.DRUG_PK.get(d)
                if pk:
                    conc = ce[d]
                    if 'svr' in pk['emax']: eff_svr += Physiology.PKPD.hill_effect(conc, pk['emax']['svr'], pk['ec50'], pk['n'])
                    if 'map' in pk['emax']: eff_map += Physiology.PKPD.hill_effect(conc, pk['emax']['map'], pk['ec50'], pk['n'])
                    if 'co' in pk['emax']: eff_co += Physiology.PKPD.hill_effect(conc, pk['emax']['co'], pk['ec50'], pk['n'])
                    if 'hr' in pk['emax']: eff_hr += Physiology.PKPD.hill_effect(conc, pk['emax']['hr'], pk['ec50'], pk['n'])

            # C. Sepsis/Disease Progression (Degradation over time)
            if "Sepsis" in case_id:
                progression = t / 60.0 # Hours
                curr_svr = c['svr_b'] * (0.95 ** progression) # Vasoplegia
                curr_vol = curr_vol * (0.98 ** progression) # Capillary Leak
            
            # D. Physiology Coupling
            # 1. Preload -> SV (Frank-Starling)
            contractility = 1.0 + (eff_co * 0.1)
            sv = Physiology.Autonomic.frank_starling(curr_vol, contractility)
            
            # 2. SV + HR -> CO
            # Baroreflex affects HR
            if not is_paced:
                curr_hr = Physiology.Autonomic.baroreflex(curr_map, 90 + eff_hr) + noise_hr[t]
            else:
                curr_hr = 80 + utils._rng.normal(0, 0.5)
            
            curr_co = (sv * curr_hr) / 1000.0 # L/min
            
            # 3. CO + SVR -> MAP
            total_svr = curr_svr + eff_svr
            # MAP = CO * SVR / 80 + CVP (assume 5)
            target_map = (curr_co * total_svr / 80.0) + 5.0
            
            # Lag in MAP change (Arterial Compliance)
            curr_map = curr_map + 0.2 * (target_map - curr_map) + noise_map[t]
            
            # E. Respiratory & Metabolic
            do2 = curr_co * c['hb'] * 1.34 * 0.95 * 10
            curr_lac = Physiology.Metabolic.lactate_kinetics(curr_lac, do2, c['vo2'], 0.05)
            
            # Store
            map_arr[t] = curr_map
            ci_arr[t] = curr_co / bsa
            svri_arr[t] = total_svr * bsa
            hr_arr[t] = curr_hr
            lac_arr[t] = curr_lac
            spo2_arr[t] = 98 - (2 if "Trauma" in case_id else 0)
            sv_arr[t] = sv

        # Create DataFrame
        df = pd.DataFrame({
            "Time": np.arange(n), "HR": hr_arr, "MAP": map_arr, "CI": ci_arr, "SVRI": svri_arr,
            "Lactate": lac_arr, "SpO2": spo2_arr, "SV": sv_arr,
            "CPO": map_arr * ci_arr * bsa / 451,
            "PPV": np.abs(np.sin(np.arange(n)/5))*15 # Simulated PPV
        })
        return df

# ==========================================
# 7. VISUALIZATION LAYER
# ==========================================
class Viz:
    @staticmethod
    def render_guidance(title, sig, action):
        st.markdown(f"""
        <div class="clinical-hint">
            <div style="font-weight:700; border-bottom:1px solid #cbd5e1; margin-bottom:5px;">{title}</div>
            <div class="sig-header">CLINICAL SIGNIFICANCE</div>
            <div style="font-size:0.8rem; margin-bottom:4px;">{sig}</div>
            <div class="action-header">ACTION REQUIRED</div>
            <div style="font-size:0.8rem;">{action}</div>
        </div>
        """, unsafe_allow_html=True)

    @staticmethod
    def spark(data, color, title):
        fig = px.line(x=np.arange(len(data)), y=data)
        fig.update_traces(line_color=color, line_width=2)
        fig.update_layout(showlegend=False, xaxis_visible=False, yaxis_visible=False, 
                          margin=dict(l=0,r=0,t=0,b=0), height=40)
        return fig

    @staticmethod
    def target_bullseye(df):
        r = df.iloc[-1]
        fig = go.Figure()
        # Green Zone
        fig.add_shape(type="circle", x0=2.2, y0=1600, x1=4.5, y1=2400, fillcolor="rgba(0,128,0,0.2)", line_color="green")
        # Current Point
        fig.add_trace(go.Scatter(x=[r['CI']], y=[r['SVRI']], mode='markers+text', 
                                 marker=dict(size=15, color='black'), text=["YOU"], textposition="top center"))
        fig.update_layout(title="Hemo-Target (Forrester)", xaxis_title="CI", yaxis_title="SVRI", 
                          xaxis_range=[1, 6], yaxis_range=[400, 3500], height=250, margin=dict(l=20,r=20,t=30,b=20))
        return fig

    @staticmethod
    def starling_curve(df):
        curr_sv = df['SV'].iloc[-1]
        vol = np.linspace(0, 100, 100)
        sv_curve = (130 * (vol**2)) / (40**2 + vol**2)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=vol, y=sv_curve, name="Contractility"))
        fig.add_trace(go.Scatter(x=[50], y=[curr_sv], mode='markers', marker=dict(size=12, color='red'), name="Status"))
        fig.update_layout(title="Frank-Starling", xaxis_title="Preload", yaxis_title="Stroke Vol", height=250, margin=dict(l=20,r=20,t=30,b=20))
        return fig

    @staticmethod
    def spc_dashboard(df):
        data = df['MAP'].values
        # Subplots with DOMAIN spec for Gauge
        fig = make_subplots(rows=1, cols=3, subplot_titles=("X-Bar (Acute)", "EWMA (Chronic)", "Sensor MSA"),
                            specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "domain"}]])
        
        # X-Bar
        subgroups = data[:len(data)//5*5].reshape(-1, 5)
        means = np.mean(subgroups, axis=1)
        fig.add_trace(go.Scatter(y=means, mode='lines+markers', name="X-Bar"), row=1, col=1)
        fig.add_hline(y=np.mean(data)+3*np.std(means), line_color='red', line_dash='dot', row=1, col=1)
        
        # EWMA
        lambda_ = 0.2
        z = np.zeros_like(data)
        z[0] = data[0]
        for i in range(1, len(data)): z[i] = lambda_*data[i] + (1-lambda_)*z[i-1]
        fig.add_trace(go.Scatter(y=z, line_color='orange', name="EWMA"), row=1, col=2)
        
        # Gauge
        rr = QA.gauge_rr(df)
        fig.add_trace(go.Indicator(mode="gauge+number", value=rr, title="Gauge R&R %", 
                                   gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "black"}, 
                                          'steps': [{'range': [0, 10], 'color': "lightgreen"}, {'range': [10, 30], 'color': "yellow"}]}), row=1, col=3)
        
        fig.update_layout(height=250, margin=dict(l=10,r=10,t=30,b=20))
        return fig
    
    @staticmethod
    def method_comp_plot(df):
        inv, nibp = QA.method_comp(df)
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Correlation", "Bland-Altman"))
        
        # Regression
        fig.add_trace(go.Scatter(x=inv, y=nibp, mode='markers', opacity=0.5, name="Points"), row=1, col=1)
        fig.add_trace(go.Scatter(x=[40, 120], y=[40, 120], mode='lines', line=dict(dash='dash', color='black'), name="Identity"), row=1, col=1)
        
        # Bland-Altman
        avg = (inv + nibp) / 2
        diff = inv - nibp
        mean_diff = np.mean(diff)
        sd_diff = np.std(diff)
        fig.add_trace(go.Scatter(x=avg, y=diff, mode='markers', opacity=0.5, name="Diff"), row=1, col=2)
        fig.add_hline(y=mean_diff, line_color='blue', row=1, col=2)
        fig.add_hline(y=mean_diff + 1.96*sd_diff, line_color='red', line_dash='dot', row=1, col=2)
        fig.add_hline(y=mean_diff - 1.96*sd_diff, line_color='red', line_dash='dot', row=1, col=2)
        
        fig.update_layout(height=250, margin=dict(l=10,r=10,t=30,b=20))
        return fig

# ==========================================
# 8. MAIN APP LOGIC
# ==========================================
class App:
    def run(self):
        st.markdown(STYLING, unsafe_allow_html=True)
        
        # --- SIDEBAR CONTROLS ---
        with st.sidebar:
            st.title("TITAN | L8")
            st.caption("Advanced Clinical Decision Support")
            
            case_id = st.selectbox("Patient Profile", 
                                   ["65M Post-CABG", "24F Septic Shock", "82M HFpEF Sepsis", "50M Trauma"])
            
            st.markdown("---")
            st.markdown("**Infusions (mcg/kg/min)**")
            c1, c2 = st.columns(2)
            norepi = c1.number_input("Norepi", 0.0, 2.0, 0.05, 0.01)
            vaso = c2.number_input("Vaso", 0.0, 0.1, 0.0, 0.01)
            dobu = c1.number_input("Dobutamine", 0.0, 20.0, 0.0, 0.5)
            bb = c2.number_input("Esmolol", 0.0, 0.5, 0.0, 0.05)
            
            st.markdown("---")
            st.markdown("**Resuscitation**")
            if 'fluids' not in st.session_state: st.session_state.fluids = 0
            if st.button("🌊 Bolus 500mL Crystal"): st.session_state.fluids += 500
            st.progress(min(st.session_state.fluids / 3000, 1.0))
            st.caption(f"Total Volume: {st.session_state.fluids} mL")

            is_paced = st.checkbox("Pacemaker Active")

        # --- SIMULATION ENGINE ---
        sim = PatientSimulator(mins=300)
        drugs = {'norepi': norepi, 'vaso': vaso, 'dobu': dobu, 'bb': bb}
        # BSA Calculation (Standard)
        bsa = 1.9 
        
        df = sim.run(case_id, drugs, st.session_state.fluids, bsa, 5, is_paced, "AC")
        
        # --- ANALYTICS PIPELINE ---
        df = Analytics.detect_anomalies(df)
        t2, spe, mewma = Analytics.spc_multivariate(df)
        p10, p50, p90 = Analytics.monte_carlo_forecast(df)
        forensic_msg, forensic_conf, forensic_reason, spec_ent = Analytics.signal_forensics(df['HR'].iloc[-120:], is_paced)
        coherence = Analytics.calc_hemodynamic_coherence(df)
        granger = Analytics.check_granger(df)
        
        curr = df.iloc[-1]
        prev = df.iloc[-60]

        # --- DASHBOARD HEADER ---
        # Determining Shock State
        shock_state = "STABLE"
        if curr['CI'] < 2.2 and curr['SVRI'] > 2000: shock_state = "CARDIOGENIC SHOCK"
        elif curr['CI'] > 2.2 and curr['SVRI'] < 1200: shock_state = "DISTRIBUTIVE SHOCK"
        elif curr['CI'] < 2.2 and curr['SVRI'] > 2500: shock_state = "HYPOVOLEMIC SHOCK"
        
        # Drift Status
        drift = "DETECTED" if curr['drift_flag'] else "NONE"
        
        st.markdown(f"""
        <div class="status-banner">
            <div>
                <div style="font-size:0.75rem; font-weight:800; color:{CONFIG.COLORS['muted']}">CLINICAL STATE</div>
                <div style="font-size:1.8rem; font-weight:900; color:{CONFIG.COLORS['crit'] if 'SHOCK' in shock_state else CONFIG.COLORS['ok']}">{shock_state}</div>
                <div style="font-size:0.8rem;">Lactate: {curr['Lactate']:.1f} mmol/L | SOFA Trend: { "↗" if curr['Lactate'] > 2 else "→"}</div>
            </div>
            <div style="text-align:right">
                <div style="font-size:0.75rem; font-weight:800; color:{CONFIG.COLORS['muted']}">SYSTEM INTEGRITY</div>
                <div style="font-size:1.2rem; font-weight:800;">DRIFT: <span style="color:{CONFIG.COLORS['warn'] if curr['drift_flag'] else CONFIG.COLORS['ok']}">{drift}</span></div>
                <div style="font-size:0.8rem;">Signal Entropy: {spec_ent:.2f} (Quality: {forensic_conf}%)</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # --- TABS ---
        t1, t2_tab, t3, t4 = st.tabs(["🩺 Hemodynamics & Perfusion", "🧪 Forensics & AI", "📉 Quality & SPC", "🔬 Research"])
        
        with t1:
            # 1. Vital Signs Row
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("MAP", f"{curr['MAP']:.0f}", f"{curr['MAP']-prev['MAP']:.0f} mmHg")
            c1.plotly_chart(Viz.spark(df['MAP'].iloc[-60:], CONFIG.COLORS['hemo'], "map"), use_container_width=True)
            
            c2.metric("Cardiac Index", f"{curr['CI']:.1f}", f"{curr['CI']-prev['CI']:.1f} L/m/m²")
            c2.plotly_chart(Viz.spark(df['CI'].iloc[-60:], CONFIG.COLORS['info'], "ci"), use_container_width=True)
            
            c3.metric("SVRI", f"{curr['SVRI']:.0f}", f"{curr['SVRI']-prev['SVRI']:.0f}")
            c3.plotly_chart(Viz.spark(df['SVRI'].iloc[-60:], CONFIG.COLORS['warn'], "svri"), use_container_width=True)
            
            c4.metric("Lactate", f"{curr['Lactate']:.1f}", f"{curr['Lactate']-prev['Lactate']:.1f}")
            c4.plotly_chart(Viz.spark(df['Lactate'].iloc[-60:], CONFIG.COLORS['crit'], "lac"), use_container_width=True)
            
            c5.metric("Shock Index", f"{curr['HR']/curr['MAP']:.2f}", "Normal < 0.7")
            c6.metric("CPO (Power)", f"{curr['CPO']:.2f}", "Watts (Crit < 0.6)")
            
            st.caption("Sparklines represent 60-minute trend. **Downward Slope on MAP** requires immediate pressor readiness. **Upward Slope on Lactate** indicates washout or worsening hypoperfusion.")

            # 2. Main Visualization Row
            c1, c2, c3 = st.columns(3)
            with c1:
                st.plotly_chart(Viz.target_bullseye(df), use_container_width=True)
                Viz.render_guidance(
                    "HEMO-TARGET (Forrester)",
                    "Diagnoses shock type via Quadrants. Top-Left: Cardiogenic (Cold/Wet). Bottom-Right: Distributive (Warm/Dry).",
                    f"Current State: <b>{shock_state}</b>.<br>• If Top-Left: Start Inotropes (Dobutamine).<br>• If Bottom-Right: Start Vasopressors (Norepi).<br>• If High/High: Check fluid status/sedation."
                )
            
            with c2:
                # Forecast
                fig_fc = go.Figure()
                fig_fc.add_trace(go.Scatter(y=df['MAP'].iloc[-60:], name="History", line_color="black"))
                # Cone
                x_fut = np.arange(60, 90)
                fig_fc.add_trace(go.Scatter(x=np.concatenate([x_fut, x_fut[::-1]]), 
                                            y=np.concatenate([p90, p10[::-1]]), 
                                            fill='toself', fillcolor='rgba(0,0,255,0.2)', line_width=0, name="90% CI"))
                fig_fc.update_layout(title="Monte Carlo MAP Forecast (30min)", height=250, margin=dict(l=20,r=20,t=30,b=20))
                st.plotly_chart(fig_fc, use_container_width=True)
                Viz.render_guidance(
                    "PREDICTIVE ANALYTICS",
                    "Projects MAP based on current volatility (Geometric Brownian Motion). Answers: 'Where will patient be in 30m?'",
                    f"• <b>Cone < 65 mmHg:</b> Start/Increase pressors NOW. Do not wait for hypotension.<br>• <b>Wide Cone:</b> High volatility. Stop titrating to allow equilibration."
                )

            with c3:
                st.plotly_chart(Viz.starling_curve(df), use_container_width=True)
                ppv = curr['PPV']
                Viz.render_guidance(
                    "FRANK-STARLING / PRELOAD",
                    "Determines Fluid Responsiveness via Stroke Volume curve. Steep = Responder. Flat = Non-Responder.",
                    f"PPV is <b>{ppv:.1f}%</b>.<br>• <b>>12% (Steep):</b> GIVE FLUIDS (500mL Bolus).<br>• <b><12% (Flat):</b> STOP FLUIDS. Start Inotropes/Pressors. Risk of Pulmonary Edema."
                )

        with t2_tab:
            c1, c2 = st.columns(2)
            with c1:
                st.info(f"Forensic Analysis: {forensic_msg}")
                # Anomaly Timeline
                fig_anom = px.scatter(df.iloc[-100:], x="Time", y="MAP", color=df['anomaly'].iloc[-100:].astype(str), 
                                      color_discrete_map={'-1': 'red', '1': 'blue'}, title="Isolation Forest Anomalies")
                fig_anom.update_layout(height=250, margin=dict(l=20,r=20,t=30,b=20))
                st.plotly_chart(fig_anom, use_container_width=True)
                Viz.render_guidance(
                    "ISOLATION FOREST",
                    "Unsupervised ML detects data points that deviate from patient's physiological history.",
                    "• <b>Red Clusters:</b> Acute Event (Arrhythmia, Bleeding, Kinked Line). INSPECT PATIENT.<br>• <b>Sparse Red Dots:</b> Likely sensor artifact/movement."
                )
                
            with c2:
                # Spectral Entropy Trend
                fig_spec = go.Figure(go.Scatter(y=df['HR'].rolling(30).std().iloc[-100:], name="HRV"))
                fig_spec.update_layout(title="Autonomic Complexity (HRV)", height=250, margin=dict(l=20,r=20,t=30,b=20))
                st.plotly_chart(fig_spec, use_container_width=True)
                Viz.render_guidance(
                    "AUTONOMIC COMPLEXITY",
                    f"Measures Nervous System Reserve via HRV. Current Entropy: {spec_ent:.2f}.",
                    "• <b>High Variability:</b> Intact Autonomic System.<br>• <b>Flatline (Metronomic Rigidity):</b> Loss of regulation. Precedes Cardiac Arrest. Reduce sedation, check electrolytes."
                )
                
                st.metric("Hemodynamic Coherence (r)", f"{coherence:.2f}", "Ideal < -0.5")

        with t3:
            st.plotly_chart(Viz.spc_dashboard(df), use_container_width=True)
            c1, c2, c3 = st.columns(3)
            with c1:
                Viz.render_guidance("SPC (X-Bar vs EWMA)", "Differentiates Signal vs Noise.", "• <b>X-Bar Violation:</b> Acute event (Embolus, Bolus).<br>• <b>EWMA Drift:</b> 'Silent Killer'. Detects occult bleeding/sepsis early.")
            with c2:
                Viz.render_guidance("GAUGE R&R (MSA)", "Ratio of Patient Variance vs Sensor Noise.", "• <b>Green (<10%):</b> Trust Data.<br>• <b>Red (>30%):</b> NOISE. Do not treat number. Flush line/Zero transducer.")
            with c3:
                Viz.render_guidance("SENSOR MSA", "Measurement System Analysis.", "Ensures data integrity before treatment.")

            st.plotly_chart(Viz.method_comp_plot(df), use_container_width=True)
            Viz.render_guidance(
                "BLAND-ALTMAN (Method Comp)", 
                "Compares Invasive Art-Line vs NIBP Cuff.",
                "• <b>Clusters at 0:</b> Accurate.<br>• <b>Large Bias:</b> Damping issue. Perform Square-Wave Flush Test. Trust NIBP until resolved."
            )
            
            # Multivariate SPC
            fig_mspc = make_subplots(rows=1, cols=2, subplot_titles=("T² (State Dev)", "SPE (Residual)"))
            fig_mspc.add_trace(go.Scatter(y=t2[-60:], fill='tozeroy', name="T2"), row=1, col=1)
            fig_mspc.add_hline(y=10, line_color='red', row=1, col=1)
            fig_mspc.add_trace(go.Scatter(y=spe[-60:], fill='tozeroy', name="SPE"), row=1, col=2)
            fig_mspc.update_layout(height=200, margin=dict(l=10,r=10,t=30,b=20))
            st.plotly_chart(fig_mspc, use_container_width=True)
            Viz.render_guidance(
                "MULTIVARIATE SPC (Physiological Coupling)",
                "T2 = Total distance from normal state. SPE = Break in correlation between vitals.",
                "• <b>High T2:</b> Severe Illness/Multi-organ failure.<br>• <b>High SPE:</b> 'Uncoupling'. Regulatory mechanisms broken. Aggressive resuscitation required."
            )

        with t4:
            st.write("Research Data - Phase Space")
            fig_3d = go.Figure(go.Scatter3d(x=df['MAP'], y=df['Lactate'], z=df['HR'], 
                                            mode='lines', line=dict(color=df.index, colorscale='Viridis', width=3)))
            fig_3d.update_layout(scene=dict(xaxis_title='MAP', yaxis_title='Lactate', zaxis_title='HR'), height=400)
            st.plotly_chart(fig_3d, use_container_width=True)
            Viz.render_guidance(
                "PHASE SPACE ATTRACTOR",
                "Visualizes the trajectory of the physiological system.",
                "• <b>Tight Orbit:</b> Stable/Compensated.<br>• <b>Chaotic/Wide Loops:</b> Unstable. Simplify drug regimen to re-stabilize."
            )

if __name__ == "__main__":
    app = App()
    app.run()
