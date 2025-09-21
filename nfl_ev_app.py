# nfl_ev_app.py
import streamlit as st
import pandas as pd
import numpy as np
import requests

st.set_page_config(page_title="NFL +EV Bets", layout="wide")
st.title("NFL +EV Bets Dashboard (Live Odds)")

# -------------------------------
# Config
# -------------------------------
STAKE_UNIT = 100
MIN_ALPHA = 0.02
HOME_FIELD_ADVANTAGE = 25
CALIBRATION_FACTOR = 0.95  # adjust probabilities based on historical accuracy

# -------------------------------
# Fetch FiveThirtyEight Elo
# -------------------------------
@st.cache_data
def fetch_projections():
    url = 'https://projects.fivethirtyeight.com/nfl-api/nfl_elo.csv'
    df = pd.read_csv(url)
    latest = df.groupby('team1').tail(1)
    projections = latest[['team1','team2','elo1_pre','elo2_pre']].copy()
    projections.rename(columns={'team1':'home','team2':'away',
                                'elo1_pre':'elo_home','elo2_pre':'elo_away'}, inplace=True)
    projections['elo_home'] += HOME_FIELD_ADVANTAGE
    projections['model_prob_home'] = 1 / (1 + 10 ** ((projections['elo_away'] - projections['elo_home']) / 400))
    projections['model_prob_away'] = 1 - projections['model_prob_home']
    return projections[['home','away','model_prob_home','model_prob_away']]

projections = fetch_projections()
st.subheader("Elo-Based Probabilities")
st.dataframe(projections)

# -------------------------------
# Fetch Live Fanatics Odds
# -------------------------------
@st.cache_data(ttl=300)
def fetch_fanatics_odds():
    url = "https://www.fanatics.com/sportsbook/api/events/nfl"  # example JSON endpoint
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers)
    data = r.json()
    events = []
    for game in data['events']:
        home = game['homeTeam']['name']
        away = game['awayTeam']['name']
        for market in game['markets']:
            if market['name'] == "Moneyline":
                for outcome in market['outcomes']:
                    team = outcome['name']
                    price = outcome['price']
                    events.append({"home": home, "away": away, "team": team, "price": price})
    return pd.DataFrame(events)

try:
    odds_df = fetch_fanatics_odds()
    st.subheader("Live Odds")
    st.dataframe(odds_df)
except Exception as e:
    st.error(f"Failed to fetch live odds: {e}")
    st.stop()

# -------------------------------
# Helper functions
# -------------------------------
def implied_prob(odds):
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)

def kelly_fraction(p, b):
    return max((p * (b + 1) - 1) / b, 0)

# -------------------------------
# Merge and calculate EV
# -------------------------------
ev_df = odds_df.merge(projections, how='left', on=['home','away'])
ev_df['model_prob'] = np.where(ev_df['team'] == ev_df['home'],
                               ev_df['model_prob_home'], ev_df['model_prob_away'])

# Apply calibration
ev_df['model_prob'] = ev_df['model_prob'] * CALIBRATION_FACTOR
ev_df['model_prob'] = ev_df['model_prob'].clip(0,1)

ev_df['implied_prob'] = ev_df['price'].apply(implied_prob)
ev_df['alpha'] = ev_df['model_prob'] - ev_df['implied_prob']
ev_df['expected_value'] = ev_df['alpha'] * (ev_df['price']/100.0)

# Only show positive edge above threshold
top_ev = ev_df[ev_df['alpha'] >= MIN_ALPHA].copy()
top_ev['dec_odds'] = np.where(top_ev['price'] > 0,
                              top_ev['price']/100 + 1,
                              100/abs(top_ev['price']) + 1)
top_ev['kelly_stake'] = top_ev.apply(lambda row: round(STAKE_UNIT * kelly_fraction(row['model_prob'], row['dec_odds']-1),2), axis=1)
top_ev = top_ev.sort_values('alpha', ascending=False)

st.subheader("Top +EV Bets")
st.dataframe(top_ev[['home','away','team','price','model_prob','implied_prob','alpha','kelly_stake']])

# -------------------------------
# CSV download
# -------------------------------
st.download_button(
    label="Download +EV Bets",
    data=top_ev.to_csv(index=False),
    file_name='nfl_ev_bets.csv',
    mime='text/csv'
)
