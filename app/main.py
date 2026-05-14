"""
IPL Match Prediction Dashboard - IMPROVED VERSION
Uses comprehensive ball-by-ball data from IPL_FINAL.csv (10 teams only)
Enhanced UI with colors, emojis, better visualizations
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analyzer import IPLAnalyzer

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & THEME
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="IPL Prediction Dashboard",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# TEAM CONFIGURATION
TEAM_COLORS = {
    'Chennai Super Kings': '#FFFF00',
    'Delhi Capitals': '#0066FF',
    'Gujarat Titans': '#0099FF',
    'Kolkata Knight Riders': '#6600FF',
    'Lucknow Super Giants': '#00AA55',
    'Mumbai Indians': '#0033AA',
    'Punjab Kings': '#EE0000',
    'Rajasthan Royals': '#FF66FF',
    'Royal Challengers Bengaluru': '#220000',
    'Sunrisers Hyderabad': '#FF6600'
}

TEAM_EMOJI = {
    'Chennai Super Kings': '🐯',
    'Delhi Capitals': '🏛️',
    'Gujarat Titans': '💪',
    'Kolkata Knight Riders': '👑',
    'Lucknow Super Giants': '⭐',
    'Mumbai Indians': '🦁',
    'Punjab Kings': '👸',
    'Rajasthan Royals': '👑',
    'Royal Challengers Bengaluru': '❤️',
    'Sunrisers Hyderabad': '🌅'
}

# CUSTOM CSS
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    
    .team-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid #FF6B6B;
    }
    
    .winner-box {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        color: white;
    }
    
    .loser-box {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        color: white;
    }
    
    .team-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF6B6B;
        text-align: center;
        padding: 20px;
        margin-bottom: 20px;
    }
    
    h1 {
        color: #FF6B6B;
        text-align: center;
    }
    
    h2 {
        color: #667eea;
        border-bottom: 3px solid #667eea;
        padding-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_analyzer():
    """Load and cache the data analyzer"""
    return IPLAnalyzer('data/IPL_FINAL.csv')

try:
    analyzer = load_analyzer()
    data_loaded = analyzer.df is not None
except Exception as e:
    st.error(f"Error loading data: {e}")
    data_loaded = False

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    st.markdown("<div class='team-header'>🏏 IPL MATCH PREDICTION DASHBOARD 🏏</div>", unsafe_allow_html=True)
    st.markdown("**Predict IPL match outcomes using AI & historical data (2008-2026)**", unsafe_allow_html=False)
    
    if not data_loaded:
        st.error("❌ Failed to load data. Please ensure IPL_FINAL.csv is in the data/ directory.")
        return
    
    # Sidebar navigation
    st.sidebar.header("📊 NAVIGATION")
    page = st.sidebar.radio(
        "Select Page",
        ["🎯 Match Prediction", "📈 Team Analysis", "🏟️ Venue Stats", "🔍 2026 Season", "📋 Historical Data"]
    )
    
    # Route to appropriate page
    if page == "🎯 Match Prediction":
        prediction_page()
    elif page == "📈 Team Analysis":
        team_analysis_page()
    elif page == "🏟️ Venue Stats":
        venue_stats_page()
    elif page == "🔍 2026 Season":
        season_2026_page()
    elif page == "📋 Historical Data":
        historical_data_page()

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1: MATCH PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════

def prediction_page():
    st.header("🎯 Match Prediction")
    st.markdown("Select teams and get AI-powered win probability predictions")
    
    # Get team list
    team_stats = analyzer.get_team_statistics()
    all_teams = sorted(team_stats.keys())
    
    col1, col2 = st.columns(2)
    
    with col1:
        home_team = st.selectbox(
            "🏠 Home Team",
            all_teams,
            format_func=lambda x: f"{TEAM_EMOJI.get(x, '🏏')} {x}",
            key="home"
        )
    
    with col2:
        away_team = st.selectbox(
            "🚗 Away Team",
            [t for t in all_teams if t != home_team],
            format_func=lambda x: f"{TEAM_EMOJI.get(x, '🏏')} {x}",
            key="away"
        )
    
    # Toss details
    st.subheader("🎲 Toss Information")
    col1, col2 = st.columns(2)
    
    with col1:
        toss_winner = st.radio("Who won toss?", [home_team, away_team], horizontal=True)
    
    with col2:
        toss_decision = st.radio("Toss decision?", ["bat", "field"], horizontal=True)
    
    # Predict button
    if st.button("🔮 PREDICT MATCH", use_container_width=True, key="predict_btn"):
        with st.spinner("⚡ Analyzing match data..."):
            prediction = analyzer.predict_match(home_team, away_team, venue="Unknown")
            
            st.success("✅ Prediction Complete!")
            
            # Display probabilities with colors
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class='metric-card'>
                    <h3>{TEAM_EMOJI.get(home_team, '🏏')} {home_team}</h3>
                    <h1 style='color: #FF6B6B; margin: 0;'>{prediction['team1_win_prob']:.1%}</h1>
                    <p>Win Probability</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class='metric-card'>
                    <h3>{TEAM_EMOJI.get(away_team, '🏏')} {away_team}</h3>
                    <h1 style='color: #667eea; margin: 0;'>{prediction['team2_win_prob']:.1%}</h1>
                    <p>Win Probability</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Chart
            st.subheader("📊 Win Probability Comparison")
            chart_data = pd.DataFrame({
                'Team': [home_team, away_team],
                'Probability': [prediction['team1_win_prob'], prediction['team2_win_prob']]
            })
            st.bar_chart(chart_data.set_index('Team'), use_container_width=True)
            
            # Score analysis
            st.subheader("📈 Expected Score Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**{home_team}** Average Score: **{prediction['team1_avg_runs']:.0f}** runs")
            with col2:
                st.info(f"**{away_team}** Average Score: **{prediction['team2_avg_runs']:.0f}** runs")
            
            # Chasing thresholds
            st.subheader("🎯 Chasing Difficulty Levels")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.success("🟢 **EASY CHASE**\n\n< 160 runs\n\n✓ 80% win rate")
            with col2:
                st.warning("🟡 **MODERATE CHASE**\n\n160-180 runs\n\n≈ 50% win rate")
            with col3:
                st.error("🔴 **DIFFICULT CHASE**\n\n> 180 runs\n\n✗ 35% win rate")
            
            # Key factors
            st.subheader("🔍 Key Factors")
            
            team1_stats = team_stats[home_team]
            team2_stats = team_stats[away_team]
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                ### {home_team}
                - **Win Rate:** {team1_stats['win_rate']:.1%}
                - **Avg Runs:** {team1_stats['avg_runs']:.0f}
                - **Matches:** {team1_stats['matches']}
                """)
            
            with col2:
                st.markdown(f"""
                ### {away_team}
                - **Win Rate:** {team2_stats['win_rate']:.1%}
                - **Avg Runs:** {team2_stats['avg_runs']:.0f}
                - **Matches:** {team2_stats['matches']}
                """)
            
            st.markdown(f"""
            ### Match Context
            - **Toss:** {toss_winner} won toss, chose to **{toss_decision}**
            - **Home Advantage:** +3% for {home_team}
            """)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2: TEAM ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def team_analysis_page():
    st.header("📈 Team Analysis")
    st.markdown("Detailed statistics for each IPL team (all time)")
    
    team_stats = analyzer.get_team_statistics()
    all_teams = sorted(team_stats.keys())
    
    selected_team = st.selectbox(
        "Select Team",
        all_teams,
        format_func=lambda x: f"{TEAM_EMOJI.get(x, '🏏')} {x}"
    )
    
    if selected_team in team_stats:
        stats = team_stats[selected_team]
        
        # Team header with color
        color = TEAM_COLORS.get(selected_team, '#FF6B6B')
        st.markdown(f"<div style='background: {color}; padding: 20px; border-radius: 10px; text-align: center;'><h2 style='color: white; margin: 0;'>{TEAM_EMOJI.get(selected_team, '🏏')} {selected_team}</h2></div>", unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Total Matches", stats['matches'])
        with col2:
            st.metric("✅ Wins", stats['wins'])
        with col3:
            st.metric("❌ Losses", stats['losses'])
        with col4:
            st.metric("🎯 Win Rate", f"{stats['win_rate']:.1%}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📈 Avg Runs Scored", f"{stats['avg_runs']:.0f}")
        with col2:
            st.metric("⚡ Expected Score", f"{stats['avg_runs']:.0f} runs")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3: VENUE STATISTICS
# ═══════════════════════════════════════════════════════════════════════════════

def venue_stats_page():
    st.header("🏟️ Venue Statistics")
    st.markdown("Analyze performance at different IPL venues")
    
    with st.spinner("Loading venue data..."):
        venues = analyzer.get_venue_statistics()
        
        if venues is not None and len(venues) > 0:
            st.subheader("Top 10 Venues by Match Count")
            
            chart_data = venues.head(10)
            st.bar_chart(chart_data['total_matches'])
            
            st.subheader("Detailed Venue Information")
            st.dataframe(venues.head(15), use_container_width=True)
        else:
            st.info("No venue data available")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4: 2026 SEASON ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def season_2026_page():
    st.header("🔍 IPL 2026 Season Analysis")
    st.markdown("Current season standings and match results")
    
    with st.spinner("Loading 2026 season data..."):
        standings, matches = analyzer.get_2026_season_data()
        
        if standings is not None and len(standings) > 0:
            st.subheader("📊 Current League Standings")
            
            # Create standings table with ranking
            standings_display = standings.copy()
            standings_display.insert(0, 'Rank', range(1, len(standings_display) + 1))
            
            st.dataframe(standings_display, use_container_width=True, hide_index=True)
            
            # Pie chart
            st.subheader("📈 Win Distribution (Top 5)")
            fig_data = standings_display.head(5).set_index('winner')
            st.pie_chart(fig_data['wins'])
            
            # Recent matches
            st.subheader("⚡ Recent Match Results")
            
            if len(matches) > 0:
                recent_matches = matches.tail(10).sort_values('date', ascending=False)
                
                display_cols = ['date', 'batting_team', 'bowling_team', 'match_won_by', 'win_outcome', 'venue']
                available_cols = [c for c in display_cols if c in recent_matches.columns]
                
                st.dataframe(recent_matches[available_cols], use_container_width=True)
        else:
            st.info("No 2026 season data available yet")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5: HISTORICAL DATA
# ═══════════════════════════════════════════════════════════════════════════════

def historical_data_page():
    st.header("📋 Historical Data Explorer")
    st.markdown("Browse and analyze historical IPL data (2008-2026)")
    
    # Dataset overview
    st.subheader("📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📝 Total Records", f"{len(analyzer.df):,}")
    with col2:
        st.metric("🏏 Unique Matches", analyzer.df['match_id'].nunique())
    with col3:
        st.metric("🎭 Active Teams", analyzer.df['batting_team'].nunique())
    with col4:
        st.metric("📅 Years Covered", "2008-2026")
    
    # Top players
    st.subheader("⭐ Top 10 Batsmen (All Time)")
    batsmen, bowlers = analyzer.get_player_statistics()
    
    if len(batsmen) > 0:
        display_batsmen = batsmen.head(10).copy()
        display_batsmen.columns = ['Runs', 'Matches', 'Balls', 'Average', 'Strike Rate']
        display_batsmen = display_batsmen.round(2)
        st.dataframe(display_batsmen, use_container_width=True)
    
    st.subheader("🎯 Top 10 Bowlers (All Time)")
    if len(bowlers) > 0:
        st.dataframe(bowlers.head(10), use_container_width=True)
    
    st.subheader("📈 Data Statistics")
    st.info(f"""
    - **Ball-by-ball records:** {len(analyzer.df):,}
    - **Total matches:** {analyzer.df['match_id'].nunique()}
    - **Active teams:** {analyzer.df['batting_team'].nunique()}
    - **Unique batters:** {analyzer.df['batter'].nunique()}
    - **Unique bowlers:** {analyzer.df['bowler'].nunique()}
    - **Seasons covered:** 2008-2026 (19 years)
    """)

# ═══════════════════════════════════════════════════════════════════════════════
# RUN APP
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
