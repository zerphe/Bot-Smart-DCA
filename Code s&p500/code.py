import json
import os
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Wedge
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# --- CONFIGURATION ---
FICHIER_JSON = "portfolio_history.json"

# 1. LE CERVEAU (On analyse l'indice officiel pour les signaux techniques)
TICKER_ANALYSE = "^GSPC"

# 2. LE PORTEFEUILLE (L'ETF que tu achètes vraiment) - utilisé seulement pour l'achat/price fetch
TICKER_ACHAT = "PSP5.PA"

BUDGET_BASE = 300  # Ton DCA de base

# --- GESTION DES DONNÉES (JSON) ---
def charger_portfolio():
    if os.path.exists(FICHIER_JSON):
        with open(FICHIER_JSON, "r") as f:
            return json.load(f)
    else:
        return {
            "resume": {"cash_total": 0.0, "parts_totales": 0.0, "pru": 0.0},
            "historique": []
        }

def sauvegarder_portfolio(data):
    with open(FICHIER_JSON, "w") as f:
        json.dump(data, f, indent=4)
    print("💾 Données sauvegardées.")

# --- RECUPERATION PRIX ETF (AUTOMATIQUE) ---
def get_etf_price():
    """Récupère le prix de l'ETF PSP5.PA pour le calcul d'achat"""
    try:
        print(f"⏳ Récupération prix ETF ({TICKER_ACHAT})...")
        etf = yf.Ticker(TICKER_ACHAT)
        hist = etf.history(period="5d")
        if not hist.empty:
            price = float(hist['Close'].iloc[-1])
            print(f"✅ Prix actuel {TICKER_ACHAT} : {price:.2f} €")
            return price
        else:
            print("⚠️ Historique ETF vide.")
            return None
    except Exception as e:
        print(f"⚠️ Erreur get_etf_price: {e}")
        return None

# --- ANALYSE TECHNIQUE (Sur l'Indice ^GSPC) ---
def get_market_data():
    print(f"⏳ Récupération des données d'analyse ({TICKER_ANALYSE})...")
    
    # 1. Téléchargement de l'historique
    df = yf.download(TICKER_ANALYSE, period="5y", interval="1d", progress=False)

    if df.empty:
        raise RuntimeError("Données de marché vides pour ticker_analysis.")

    # Nettoyage des colonnes MultiIndex si nécessaire
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 2. CORRECTIF PRIX : On récupère le prix "Live" exact via fast_info
    try:
        ticker_obj = yf.Ticker(TICKER_ANALYSE)
        # fast_info['last_price'] est beaucoup plus fiable pour le prix actuel
        prix_reel = ticker_obj.fast_info['last_price']
        
        # On remplace la valeur 'Close' de la toute dernière ligne par le prix réel
        if prix_reel is not None:
            df.iloc[-1, df.columns.get_loc('Close')] = prix_reel
    except Exception as e:
        print(f"⚠️ Impossible de forcer le prix live, utilisation du prix historique ({e})")

    # Assurer index en tz-naive
    df.index = pd.to_datetime(df.index).tz_localize(None)

    # --- CALCULS TECHNIQUES ---

    # 1. RSI (14)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # 2. MM200
    df['MM200'] = df['Close'].rolling(window=200).mean()

    # 3. BOLLINGER BANDS (20 jours)
    df['MM20'] = df['Close'].rolling(window=20).mean()
    df['STD20'] = df['Close'].rolling(window=20).std()
    df['Bollinger_Up'] = df['MM20'] + (df['STD20'] * 2)
    df['Bollinger_Low'] = df['MM20'] - (df['STD20'] * 2)

    # 4. DRAWDOWN FROM ATH
    df['ATH'] = df['Close'].cummax()
    df['Drawdown'] = ((df['Close'] - df['ATH']) / df['ATH']) * 100

    # Clean NaNs at top (important pour ne pas fausser les moyennes)
    df = df.dropna(subset=['Close']).copy()
    
    return df

def get_vix_data():
    """Récupère le VIX (Indice de la Peur)"""
    try:
        print("⏳ Récupération du VIX...")
        vix = yf.download("^VIX", period="5d", interval="1d", progress=False)

        if vix.empty:
            print("⚠️ VIX vide.")
            return None

        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.get_level_values(0)

        vix.index = pd.to_datetime(vix.index).tz_localize(None)
        vix_value = float(vix['Close'].iloc[-1])
        print(f"✅ VIX récupéré: {vix_value:.2f}")
        return vix_value
    except Exception as e:
        print(f"⚠️ VIX non disponible: {e}")
        return None

def reconstruire_performance(df_market, portfolio):
    """
    Reconstruit la performance du portefeuille en se basant SUR L'INDICE df_market.
    df_market : dataframe du ticker_analysis (Close utilisé pour calcul des valeurs)
    portfolio : dictionnaire chargé depuis JSON
    """
    # Préparer dataframe de performance indexé sur le marché
    perf_df = pd.DataFrame(index=df_market.index)
    perf_df['Cash_Investi_Jour'] = 0.0
    perf_df['Parts_Achetees_Jour'] = 0.0
    historique = portfolio.get("historique", [])

    if not historique:
        perf_df['Cumul_Cash'] = 0.0
        perf_df['Cumul_Parts'] = 0.0
        perf_df['Valeur_Portfolio'] = 0.0
        return perf_df

    # Pour chaque achat, on place montant/parts sur la date de marché la plus proche <= date_achat
    for h in historique:
        # certains historiques peuvent stocker date sans timezone
        date_achat = pd.to_datetime(h['date']).tz_localize(None)
        # trouver l'indice le plus proche à gauche (pad) pour la date de marché
        try:
            pos = df_market.index.get_indexer([date_achat], method='pad')[0]
            if pos == -1:
                # si date_achat est antérieure au premier jour de df_market, on l'ignore
                continue
            date_market = df_market.index[pos]
            perf_df.at[date_market, 'Cash_Investi_Jour'] += float(h.get('montant', 0.0))
            perf_df.at[date_market, 'Parts_Achetees_Jour'] += float(h.get('parts', 0.0))
        except Exception as e:
            # ignore si problème d'index
            print(f"⚠️ Warning mapping purchase date {h.get('date')} : {e}")
            continue

    perf_df['Cash_Investi_Jour'] = perf_df['Cash_Investi_Jour'].fillna(0.0)
    perf_df['Parts_Achetees_Jour'] = perf_df['Parts_Achetees_Jour'].fillna(0.0)
    perf_df['Cumul_Cash'] = perf_df['Cash_Investi_Jour'].cumsum()
    perf_df['Cumul_Parts'] = perf_df['Parts_Achetees_Jour'].cumsum()

    # Utiliser la série Close du df_market pour calculer valeur portefeuille
    close_series = df_market['Close'].reindex(perf_df.index).fillna(method='ffill').fillna(0.0)
    perf_df['Valeur_Portfolio'] = perf_df['Cumul_Parts'] * close_series

    return perf_df.fillna(0.0)

# --- CERVEAU QUANTITATIF (Formule Z-Score & Elasticité) ---
def calculer_conseil(df, pru_actuel, vix_value=None):
    last_row = df.iloc[-1]

    price = float(last_row['Close'])
    drawdown = float(last_row['Drawdown'])
    b_low = float(last_row['Bollinger_Low'])
    b_up = float(last_row['Bollinger_Up'])
    rsi = float(last_row['RSI'])

    # Pour le Z-Score (Bollinger)
    ma20 = float(last_row['MM20'])
    std20 = float(last_row['STD20'])

    # --- 1. CALCUL DU Z-SCORE (BOLLINGER) ---
    z_score = (price - ma20) / std20 if std20 > 0 else 0
    score_bollinger = 0
    if z_score < -1.0:
        score_bollinger = 0.15 * (abs(z_score) - 1) ** 1.5

    # --- 2. CALCUL ÉLASTICITÉ (DRAWDOWN) ---
    score_drawdown = (abs(drawdown) / 30.0) ** 2.0

    # --- 3. PRIME DE PEUR (VIX) ---
    score_vix = 0
    if vix_value and vix_value > 20:
        score_vix = (vix_value - 20) / 40.0

    # --- 4. PRU (Bonus si on achète en dessous de notre PRU) ---
    score_pru = 0
    if pru_actuel > 0 and price < pru_actuel:
        # Si le prix actuel est inférieur au PRU, on a un bonus
        ecart_pru_pct = ((pru_actuel - price) / pru_actuel) * 100
        # Bonus progressif : plus on achète en dessous du PRU, plus le bonus est fort
        score_pru = min(ecart_pru_pct / 10.0, 0.5)  # Max 50% de bonus (soit +150€)

    # --- TOTAL ---
    alpha_total = 1.0 + score_bollinger + score_drawdown + score_vix + score_pru

    if alpha_total > 3.5:
        alpha_total = 3.5
    if alpha_total < 1.0:
        alpha_total = 1.0

    montant_final = BUDGET_BASE * alpha_total

    # --- SIGNAUX ---
    signals = {
        'base': {'montant': BUDGET_BASE, 'raison': 'Socle DCA', 'pct': 0},
        'drawdown': {
            'montant': BUDGET_BASE * score_drawdown,
            'raison': f"Élasticité (DD {drawdown:.1f}%)",
            'pct': int(score_drawdown * 100)
        },
        'bollinger': {
            'montant': BUDGET_BASE * score_bollinger,
            'raison': f"Statistique (Z: {z_score:.2f})",
            'pct': int(score_bollinger * 100)
        },
        'vix': {
            'montant': BUDGET_BASE * score_vix,
            'raison': f"Prime de Peur (VIX {vix_value:.1f})",
            'pct': int(score_vix * 100)
        },
        'pru': {
            'montant': BUDGET_BASE * score_pru,
            'raison': f"Prix sous PRU ({((pru_actuel - price) / pru_actuel * 100):.1f}%)" if score_pru > 0 else "Prix >= PRU",
            'pct': int(score_pru * 100)
        }
    }
    return montant_final, signals, price, rsi, drawdown, b_low, b_up, vix_value

# --- FONCTIONS GRAPHIQUES ---
def draw_drawdown_gauge(ax, drawdown_value):
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-0.2, 1.2); ax.axis('off')
    ax.add_patch(Wedge((0, 0), 1, 0, 180, width=0.3, facecolor='#e0e0e0', edgecolor='none'))
    ax.add_patch(Wedge((0, 0), 1, 135, 180, width=0.3, facecolor='#ff6b6b', alpha=0.8))
    ax.add_patch(Wedge((0, 0), 1, 90, 135, width=0.3, facecolor='#ffd43b', alpha=0.8))
    ax.add_patch(Wedge((0, 0), 1, 0, 90, width=0.3, facecolor='#51cf66', alpha=0.8))

    dd_capped = max(-40, min(0, drawdown_value))
    angle = 180 + (dd_capped * 4.5)
    x = 0.85 * np.cos(np.radians(angle)); y = 0.85 * np.sin(np.radians(angle))
    ax.plot([0, x], [0, y], color='#2c3e50', linewidth=3, zorder=10)
    ax.plot(0, 0, 'o', color='#2c3e50', markersize=8, zorder=11)
    ax.text(0, -0.1, f'{drawdown_value:.1f}%', ha='center', va='top', fontsize=26, fontweight='bold', color='#2c3e50')
    ax.text(0, 0.45, 'DRAWDOWN', ha='center', fontsize=11, color='gray', fontweight='bold')

def draw_kpi_card(ax, title, value, subtitle, change=None, color='#1f77b4'):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    ax.add_patch(FancyBboxPatch((0.05, 0.1), 0.9, 0.8, boxstyle="round,pad=0.05", facecolor='white', edgecolor='#e0e0e0', linewidth=2))
    ax.text(0.5, 0.75, title, ha='center', va='center', fontsize=10, color='#6c757d', fontweight='bold')
    ax.text(0.5, 0.45, value, ha='center', va='center', fontsize=22, color=color, fontweight='bold')
    ax.text(0.5, 0.25, subtitle, ha='center', va='center', fontsize=9, color='#868e96')
    if change is not None:
        c_col = '#51cf66' if change >= 0 else '#ff6b6b'
        sym = '▲' if change >= 0 else '▼'
        ax.text(0.5, 0.12, f'{sym} {abs(change):.1f}%', ha='center', va='center', fontsize=10, color=c_col, fontweight='bold')

# --- VISUALISATION DASHBOARD ---
def tracer_dashboard(df, portfolio, vix_value=None):
    """
    Génère un dashboard complet analysant la performance du portefeuille 
    par rapport à l'indice réel (TICKER_ANALYSE).
    """
    # 1. Préparation des données
    perf_df = reconstruire_performance(df, portfolio)
    historique = portfolio.get("historique", [])
    resume = portfolio.get("resume", {})
    
    # Gestion robuste des dernières valeurs (évite les crashs si portfolio vide)
    if not perf_df.empty and perf_df['Valeur_Portfolio'].sum() > 0:
        last_val = float(perf_df['Valeur_Portfolio'].replace(0, np.nan).dropna().iloc[-1])
    else:
        last_val = 0.0
        
    last_cash = float(perf_df['Cumul_Cash'].iloc[-1]) if not perf_df.empty else 0.0
    gain_loss = last_val - last_cash
    gain_loss_pct = (gain_loss / last_cash * 100) if last_cash > 0 else 0
    drawdown = float(df['Drawdown'].iloc[-1]) if 'Drawdown' in df.columns else 0.0

    # 2. Création de la figure et du Layout
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 2, 2], width_ratios=[2, 1, 1], 
                         hspace=0.3, wspace=0.35)

    # --- ZONE 1: KPIs (Haut) ---
    ax_kpi1 = fig.add_subplot(gs[0, 0])
    ax_kpi2 = fig.add_subplot(gs[0, 1])
    ax_drawdown_gauge = fig.add_subplot(gs[0, 2])

    # KPI 1: État du Portefeuille
    draw_kpi_card(ax_kpi1, "VALEUR PORTEFEUILLE", f"{last_val:.0f} €",
                  f"Investi: {last_cash:.0f} € | P&L: {gain_loss:+.0f}€",
                  change=gain_loss_pct, color='#1f77b4')

    # KPI 2: Analyse du PRU (L'écart par rapport à l'indice)
    pru = resume.get("pru", 0.0)
    val_close = df['Close'].iloc[-1]
    prix_actuel = float(val_close.item()) if hasattr(val_close, "item") else float(val_close)
    
    # Calcul de l'écart (Déviation)
    ecart_pru = ((prix_actuel - pru) / pru * 100) if pru > 0 else 0
    draw_kpi_card(ax_kpi2, "PRU vs INDICE RÉEL", f"{pru:.2f}",
                  f"Prix actuel: {prix_actuel:.2f}", 
                  change=ecart_pru, color='#fd7e14')

    # Jauge de Drawdown (Risque actuel du marché)
    draw_drawdown_gauge(ax_drawdown_gauge, drawdown)

    # --- ZONE 2: ANALYSE TECHNIQUE & PRÉVISIONS (Gauche) ---
    ax_main = fig.add_subplot(gs[1:, 0])
    end_date = df.index[-1]
    start_date = end_date - timedelta(days=365)
    df_zoom = df[df.index >= start_date].copy()

    if not df_zoom.empty:
        # Courbes de prix et moyennes
        ax_main.plot(df_zoom.index, df_zoom['Close'], label=f'Cours {TICKER_ANALYSE}', 
                    color='#1f77b4', linewidth=2.5, zorder=4)
        
        if 'MM200' in df_zoom.columns:
            ax_main.plot(df_zoom.index, df_zoom['MM200'], label='MM 200 (Tendance)', 
                        color='#ff9800', linestyle='--', alpha=0.8, linewidth=2)

        # Bandes de Bollinger (Zone de volatilité)
        if 'Bollinger_Up' in df_zoom.columns and 'Bollinger_Low' in df_zoom.columns:
            ax_main.fill_between(df_zoom.index, df_zoom['Bollinger_Low'], df_zoom['Bollinger_Up'], 
                                color='gray', alpha=0.1, label='Zone Bollinger')

        # Affichage des achats historiques
        if historique:
            hist_zoom = [h for h in historique if pd.to_datetime(h["date"]).tz_localize(None) >= start_date]
            if hist_zoom:
                dates_achat = [pd.to_datetime(h["date"]).tz_localize(None) for h in hist_zoom]
                prix_achat = [h.get("prix_achat", np.nan) for h in hist_zoom]
                ax_main.scatter(dates_achat, prix_achat, color='#51cf66', marker='^', 
                               s=130, label='Points d\'achat DCA', zorder=5, edgecolors='white')

        # Ligne du PRU (Visualisation de la déviation)
        if pru > 0:
            ax_main.axhline(y=pru, color='#e74c3c', linestyle=':', linewidth=2, 
                           label=f'Mon PRU ({pru:.2f})', alpha=0.9, zorder=6)

        # 🔮 CÔNE DE PRÉVISION (Modèle Black-Scholes)
        sigma = (vix_value / 100.0) if (vix_value and vix_value > 0) else 0.20
        mu, days_proj = 0.10, 180  # Hypothèse 10% rendement annuel
        dates_future = [end_date + timedelta(days=x) for x in range(days_proj)]
        
        upper_cone, lower_cone, mean_path = [], [], []
        for i in range(days_proj):
            t = i / 365.0
            drift = (mu - 0.5 * sigma**2) * t
            vol = sigma * np.sqrt(t)
            mean_path.append(prix_actuel * np.exp(mu * t))
            upper_cone.append(prix_actuel * np.exp(drift + vol))
            lower_cone.append(prix_actuel * np.exp(drift - vol))

        ax_main.plot(dates_future, mean_path, color='gold', linestyle='--', alpha=0.7, label='Projection (+10%)')
        ax_main.fill_between(dates_future, lower_cone, upper_cone, color='gold', alpha=0.1, label='Cône de probabilité (VIX)')

        # Formatage graphique
        ax_main.set_ylabel(f"Valeur {TICKER_ANALYSE}", fontsize=12, fontweight='bold')
        ax_main.legend(loc='upper left', framealpha=0.9, fontsize=9, ncol=2)
        ax_main.grid(True, alpha=0.2, linestyle='--')
        ax_main.set_title(f"📊 ANALYSE TECHNIQUE & PRÉVISIONS (VIX: {sigma*100:.1f}%)", 
                         fontsize=13, fontweight='bold', pad=15)
        ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.setp(ax_main.xaxis.get_majorticklabels(), rotation=45)

    # --- ZONE 3: PERFORMANCE TEMPORELLE (Droite) ---
    
    # A. Performance Année en cours
    ax_monthly = fig.add_subplot(gs[1, 1:])
    current_year = datetime.now().year
    current_year_start = pd.Timestamp(f'{current_year}-01-01')
    
    if not perf_df.empty:
        perf_year = perf_df[perf_df.index >= current_year_start].copy()
        if not perf_year.empty:
            ax_monthly.plot(perf_year.index, perf_year['Cumul_Cash'], label='💰 Investi', color='#ff6b6b', linestyle='--')
            ax_monthly.plot(perf_year.index, perf_year['Valeur_Portfolio'], label='📈 Valeur', color='#51cf66', linewidth=2)
            ax_monthly.fill_between(perf_year.index, perf_year['Cumul_Cash'], perf_year['Valeur_Portfolio'], 
                                   where=(perf_year['Valeur_Portfolio'] >= perf_year['Cumul_Cash']), color='green', alpha=0.15)
            ax_monthly.set_title(f"💰 PERFORMANCE {current_year}", fontsize=12, fontweight='bold')
            ax_monthly.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
            ax_monthly.grid(True, alpha=0.2)

    # B. Historique Complet
    ax_yearly = fig.add_subplot(gs[2, 1:])
    if not perf_df.empty and len(historique) > 0:
        first_date = pd.to_datetime(historique[0]['date']).tz_localize(None)
        perf_full = perf_df[perf_df.index >= (first_date - timedelta(days=5))].copy()
        
        ax_yearly.plot(perf_full.index, perf_full['Cumul_Cash'], color='#ff6b6b', alpha=0.6)
        ax_yearly.plot(perf_full.index, perf_full['Valeur_Portfolio'], color='#51cf66', linewidth=2)
        ax_yearly.fill_between(perf_full.index, perf_full['Cumul_Cash'], perf_full['Valeur_Portfolio'], 
                               where=(perf_full['Valeur_Portfolio'] >= perf_full['Cumul_Cash']), color='green', alpha=0.1)
        ax_yearly.set_title("📈 HISTORIQUE GLOBAL", fontsize=12, fontweight='bold')
        ax_yearly.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax_yearly.grid(True, alpha=0.2)

    # 4. Finalisation
    plt.suptitle(f"🤖 SMART DCA DASHBOARD - {datetime.now().strftime('%d/%m/%Y %H:%M')}", 
                 fontsize=18, fontweight='bold', y=0.98, color='#2c3e50')
    plt.tight_layout()
    plt.show()
    
# --- MAIN ---
def main():
    print("--- DÉMARRAGE DU BOT ---")

    # 1. Chargement et Récupération des Données
    portfolio = charger_portfolio()
    df = get_market_data()
    vix_value = get_vix_data()

    # Récupération automatique du prix ETF
    etf_price = get_etf_price()
    last_price_indice = df['Close'].iloc[-1]
    prev_price_indice = df['Close'].iloc[-2]
    variation_indice = ((last_price_indice - prev_price_indice) / prev_price_indice) * 100
    color_var = "🟢" if variation_indice >= 0 else "🔴"
    print("\n" + "-"*30)
    print(f"📈 LIVE MARKET : {TICKER_ANALYSE}")
    print(f"👉 Prix actuel : {last_price_indice:,.2f} $")
    print(f"👉 Variation J : {color_var} {variation_indice:+.2f}%")
    print("-" * 30)
    
    pru = portfolio["resume"].get("pru", 0.0)
    
    # 2. Gestion du Premier Investissement ou Import
    if not portfolio["historique"]:
        print("\n" + "="*70)
        print("📋 CONFIGURATION INITIALE DU PORTFOLIO")
        print("="*70)
        
        reponse = input("\n❓ Avez-vous déjà investi sur cet ETF ? (oui/non) : ").strip().lower()
        
        if reponse in ['oui', 'o', 'yes', 'y']:
            try:
                montant_total_investi = float(input("💰 Montant total investi (€) : "))
                nb_parts_reel = float(input("📊 Nombre de parts possédées : "))
                
                # --- CALCUL DU PRU INDICE ESTIMÉ ---
                # On calcule ton PRU ETF Réel (ex: 500€ / 3 parts = 166.66€)
                pru_etf_reel = montant_total_investi / nb_parts_reel
                
                # On calcule le ratio actuel entre l'Indice et l'ETF
                # Cela permet de traduire ton PRU en € vers un PRU en points d'indice
                if etf_price and etf_price > 0:
                    ratio_indice_etf = last_price_indice / etf_price
                    pru_indice_estime = pru_etf_reel * ratio_indice_etf
                else:
                    pru_indice_estime = last_price_indice
                
                # Calcul des parts virtuelles basées sur ce PRU Indice
                nb_parts_virtuelles = montant_total_investi / pru_indice_estime
                
                # Mise à jour du portfolio
                portfolio["resume"]["cash_total"] = montant_total_investi
                portfolio["resume"]["parts_totales"] = nb_parts_virtuelles
                portfolio["resume"]["pru"] = pru_indice_estime
                
                portfolio["historique"].append({
                    "date": datetime.now().strftime('%Y-%m-%d'),
                    "montant": montant_total_investi,
                    "prix_achat": pru_indice_estime,
                    "parts": nb_parts_virtuelles,
                    "pru_apres_achat": pru_indice_estime,
                    "note": "Import initial"
                })
                
                sauvegarder_portfolio(portfolio)
                
                print("\n" + "="*70)
                print("✅ PORTFOLIO SYNCHRONISÉ")
                print(f"💰 Cash investi : {montant_total_investi:.2f} €")
                print(f"📈 Parts ETF réelles : {nb_parts_reel:.4f}")
                print(f"💵 PRU ETF calculé : {pru_etf_reel:.2f} €/part")
                print(f"📊 PRU Indice (référence système) : {pru_indice_estime:.2f}")
                print(f"📐 Parts virtuelles (indice) : {nb_parts_virtuelles:.6f}")
                print("="*70)
                print("\n🔄 Configuration terminée. Relancez le programme pour obtenir vos conseils.")
                return

            except Exception as e:
                print(f"❌ Erreur lors de l'import : {e}")
                return
            
        else:
            # L'utilisateur commence vraiment de zéro
            print("\n" + "="*70)
            print("🎉 PREMIER INVESTISSEMENT DÉTECTÉ !")
            print(f"💰 Montant recommandé: {BUDGET_BASE:.0f} €")
            print("="*70)
            
            if etf_price:
                nb_reel = int(BUDGET_BASE // etf_price)
                print(f"👉 Info Achat : Achète {nb_reel} parts de {TICKER_ACHAT}")

            choix = input("\n✍️ Montant investi : ")
            if choix.lower() != 'non':
                try:
                    m = float(choix)
                    p_indice = last_price_indice
                    nb_virtuel = m / p_indice if p_indice > 0 else 0
                    
                    portfolio["resume"]["cash_total"] = m
                    portfolio["resume"]["parts_totales"] = nb_virtuel
                    portfolio["resume"]["pru"] = p_indice
                    
                    portfolio["historique"].append({
                        "date": datetime.now().strftime('%Y-%m-%d'), 
                        "montant": m, 
                        "prix_achat": p_indice,
                        "parts": nb_virtuel, 
                        "pru_apres_achat": p_indice
                    })
                    sauvegarder_portfolio(portfolio)
                    print(f"✅ Premier achat sauvegardé (Indice réf: {p_indice:.2f})")
                except Exception as e:
                    print(f"❌ Erreur. {e}")
            return

    # 3. Analyse Complète (Cerveau)
    montant_final, signals, prix, rsi, dd, blo, bup, vix = calculer_conseil(df, pru, vix_value)

    # --- 4. AFFICHAGE DES RÉSULTATS OPTIMISÉ ---
    import math

    print("\n" + "="*60)
    print(f"📊 ANALYSE DU MARCHÉ ({datetime.now().strftime('%d/%m/%Y')})")
    print("="*60)
    print(f"{'INDICATEUR':<15} | {'VALEUR':<15} | {'DÉCISION':<25} | {'BONUS'}")
    print("-" * 75)
    print(f"{'SOCLE':<15} | {'-':<15} | {'Mise mensuelle fixe':<25} | {BUDGET_BASE:>4.0f} €")
    print(f"{'PRU':<15} | {pru:.2f} €         | {signals['pru']['raison']:<25} | {signals['pru']['montant']:>4.0f} €")
    vix_d = f"{vix:.1f}" if vix else "N/A"
    print(f"{'VIX (Peur)':<15} | {vix_d:<15} | {signals['vix']['raison']:<25} | {signals['vix']['montant']:>4.0f} €")
    print(f"{'DRAWDOWN':<15} | {dd:.1f}%           | {signals['drawdown']['raison']:<25} | {signals['drawdown']['montant']:>4.0f} €")
    print(f"{'BOLLINGER':<15} | Bas: {blo:.0f}      | {signals['bollinger']['raison']:<25} | {signals['bollinger']['montant']:>4.0f} €")
    print("-" * 75)
    print(f"💰 TOTAL THÉORIQUE : {montant_final:.2f} €")

    # --- LOGIQUE D'ARRONDI OPTIMISÉ ---
    if etf_price:
        nb_parts_float = montant_final / etf_price
        nb_actions = math.ceil(nb_parts_float) 
        cout_reel = nb_actions * etf_price
        reste_apres_achat = cout_reel - montant_final 

        print(f"\n🛒 OPTIMISATION DES PARTS ({TICKER_ACHAT}) :")
        print(f"👉 PRIX UNITAIRE : {etf_price:.2f} €")
        print(f"👉 ORDRE CONSEILLÉ : {nb_actions} ACTIONS")
        print(f"👉 COÛT TOTAL RÉEL : {cout_reel:.2f} €")
        if reste_apres_achat > 0:
            print(f"ℹ️ Note : Arrondi supérieur (+{reste_apres_achat:.2f}€) pour minimiser le cash dormant.")
    else:
        print("\n⚠️ Prix ETF indisponible pour calculer le nombre de parts.")
    print("="*60)

    # --- 5. SAISIE ET SAUVEGARDE DYNAMIQUE ---
    choix = input("\n✍️ Montant réellement investi (Appuie sur Entrée pour valider ou tape 'non') : ")
    
    if choix.lower() != 'non':
        try:
            p_indice_actuel = float(df['Close'].iloc[-1])
            
            if choix == "" and etf_price:
                m = cout_reel
            else:
                m = float(choix)

            nb_parts_indice_achat = m / p_indice_actuel if p_indice_actuel > 0 else 0
            
            ancien_cash = portfolio["resume"].get("cash_total", 0.0)
            ancien_pru = portfolio["resume"].get("pru", 0.0)
            ancien_parts = portfolio["resume"].get("parts_totales", 0.0)
            
            nouveau_cash = ancien_cash + m
            nouvelles_parts = ancien_parts + nb_parts_indice_achat
            
            if nouvelles_parts > 0:
                nouveau_pru = ((ancien_pru * ancien_parts) + (p_indice_actuel * nb_parts_indice_achat)) / nouvelles_parts
            else:
                nouveau_pru = p_indice_actuel

            print(f"\n📊 DÉTAILS DE L'OPÉRATION :")
            print(f"🔹 Prix Indice à l'achat : {p_indice_actuel:.2f} $")
            print(f"🔹 Nouveau PRU Indice : {nouveau_pru:.2f} $ ({((nouveau_pru - ancien_pru)/ancien_pru*100 if ancien_pru > 0 else 0):+.2f}%)")
            
            portfolio["resume"]["cash_total"] = nouveau_cash
            portfolio["resume"]["parts_totales"] = nouvelles_parts
            portfolio["resume"]["pru"] = nouveau_pru
            
            portfolio["historique"].append({
                "date": datetime.now().strftime('%Y-%m-%d'), 
                "montant": m, 
                "prix_achat": p_indice_actuel,
                "parts": nb_parts_indice_achat,
                "pru_apres_achat": nouveau_pru
            })
            
            sauvegarder_portfolio(portfolio)
            print("✅ Portefeuille mis à jour avec succès.")
            
        except Exception as e:
            print(f"❌ Erreur lors de la saisie : {e}")

    tracer_dashboard(df, portfolio, vix_value)

if __name__ == "__main__":
    main()