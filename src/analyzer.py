"""
Complete IPL Data Analysis and Processing
Works with comprehensive IPL.csv containing 283,678 ball-by-ball records
"""

import pandas as pd
import numpy as np
import warnings
import logging
from pathlib import Path

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IPLAnalyzer:
    """Comprehensive IPL data analyzer"""
    
    def __init__(self, data_path='data/IPL_FINAL.csv'):
        self.data_path = data_path
        self.df = None
        self.load_data()
    
    def load_data(self):
        """Load comprehensive IPL data"""
        logger.info(f"Loading IPL data from {self.data_path}...")
        try:
            self.df = pd.read_csv(self.data_path, index_col=0, low_memory=False)
            self.df['year'] = pd.to_numeric(self.df['year'], errors='coerce')
            logger.info(f"✓ Loaded {len(self.df):,} ball-by-ball records")
            logger.info(f"✓ Seasons: {self.df['year'].min():.0f} - {self.df['year'].max():.0f}")
            return True
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def get_match_summary(self, match_id):
        """Get detailed match summary"""
        match_data = self.df[self.df['match_id'] == match_id]
        
        if len(match_data) == 0:
            return None
        
        sample = match_data.iloc[0]
        
        # Inning 1 data
        inning1 = match_data[match_data['innings'] == 1]
        team1 = inning1['batting_team'].iloc[0] if len(inning1) > 0 else 'Unknown'
        
        if len(inning1) > 0:
            inning1_runs = inning1['runs_total'].sum()
            inning1_wickets = len(inning1[inning1['wicket_kind'].notna()])
            inning1_overs = inning1['over'].max() + 1
        else:
            inning1_runs = inning1_wickets = inning1_overs = 0
        
        # Inning 2 data
        inning2 = match_data[match_data['innings'] == 2]
        team2 = inning2['batting_team'].iloc[0] if len(inning2) > 0 else 'Unknown'
        
        if len(inning2) > 0:
            inning2_runs = inning2['runs_total'].sum()
            inning2_wickets = len(inning2[inning2['wicket_kind'].notna()])
            inning2_overs = inning2['over'].max() + 1
        else:
            inning2_runs = inning2_wickets = inning2_overs = 0
        
        return {
            'match_id': match_id,
            'date': sample['date'],
            'venue': sample['venue'],
            'team1': team1,
            'team1_runs': inning1_runs,
            'team1_wickets': inning1_wickets // 6 if inning1_wickets > 0 else 0,
            'team2': team2,
            'team2_runs': inning2_runs,
            'team2_wickets': inning2_wickets // 6 if inning2_wickets > 0 else 0,
            'winner': sample['match_won_by'],
            'result': sample['win_outcome'],
            'toss_winner': sample['toss_winner'],
            'toss_decision': sample['toss_decision']
        }
    
    def get_powerplay_analysis(self, match_id, inning=1):
        """Analyze powerplay performance"""
        match_data = self.df[(self.df['match_id'] == match_id) & (self.df['innings'] == inning)]
        
        if len(match_data) == 0:
            return None
        
        # First 6 overs
        powerplay = match_data[match_data['over'] < 6]
        
        pp_runs = powerplay['runs_total'].sum()
        pp_wickets = len(powerplay[powerplay['wicket_kind'].notna()]) // 6
        pp_overs = powerplay['over'].max() + 1
        
        return {
            'runs': pp_runs,
            'wickets': pp_wickets,
            'overs': pp_overs,
            'run_rate': pp_runs / pp_overs if pp_overs > 0 else 0
        }
    
    def get_team_statistics(self):
        """Calculate team statistics from all data"""
        logger.info("Calculating team statistics...")
        
        team_stats = {}
        
        # Get all unique teams
        all_teams = set(self.df['batting_team'].dropna().unique())
        
        for team in all_teams:
            # All matches where team batted
            team_matches = self.df[self.df['batting_team'] == team].drop_duplicates('match_id')
            
            if len(team_matches) == 0:
                continue
            
            wins = len(team_matches[team_matches['match_won_by'] == team])
            total = len(team_matches)
            
            # Average runs and wickets
            avg_runs = self.df[self.df['batting_team'] == team].groupby('match_id')['runs_total'].sum().mean()
            
            team_stats[team] = {
                'matches': total,
                'wins': wins,
                'losses': total - wins,
                'win_rate': wins / total if total > 0 else 0,
                'avg_runs': avg_runs
            }
        
        logger.info(f"✓ Calculated stats for {len(team_stats)} teams")
        return team_stats
    
    def get_player_statistics(self):
        """Get top players statistics"""
        logger.info("Calculating player statistics...")
        
        # Top batsmen
        batsmen = self.df.groupby('batter').agg({
            'runs_batter': 'sum',
            'match_id': 'nunique',
            'over': 'count'
        }).rename(columns={
            'runs_batter': 'runs',
            'match_id': 'matches',
            'over': 'balls_faced'
        })
        
        batsmen['average'] = batsmen['runs'] / batsmen['matches']
        batsmen['strike_rate'] = (batsmen['runs'] / batsmen['balls_faced'] * 100) if len(batsmen) > 0 else 0
        batsmen = batsmen.sort_values('runs', ascending=False).head(20)
        
        # Top bowlers
        bowlers = self.df[self.df['wicket_kind'].notna()].groupby('bowler').agg({
            'wicket_kind': 'count',
            'match_id': 'nunique'
        }).rename(columns={
            'wicket_kind': 'wickets',
            'match_id': 'matches'
        })
        
        bowlers = bowlers.sort_values('wickets', ascending=False).head(20)
        
        return batsmen, bowlers
    
    def get_venue_statistics(self):
        """Analyze venue data"""
        logger.info("Analyzing venue statistics...")
        
        venues = self.df.drop_duplicates('match_id').groupby('venue').agg({
            'match_id': 'count',
            'match_won_by': lambda x: x.notna().sum()
        }).rename(columns={
            'match_id': 'total_matches',
            'match_won_by': 'decisive_matches'
        })
        
        return venues.sort_values('total_matches', ascending=False).head(15)
    
    def get_venue_characteristics(self, venue):
        """Analyze venue characteristics"""
        
        venue_matches = self.df[self.df['venue'] == venue]
        
        if len(venue_matches) == 0:
            return {
                'venue_name': venue,
                'total_matches': 0,
                'avg_score': 170,
                'highest_score': 0,
                'lowest_score': 0,
                'batting_friendly': 'Moderate',
                'winning_team_avg_score': 170
            }
        
        total_matches = venue_matches['match_id'].nunique()
        avg_score = venue_matches.groupby('match_id')['runs_total'].max().mean()
        highest_score = venue_matches.groupby('match_id')['runs_total'].max().max()
        lowest_score = venue_matches.groupby('match_id')['runs_total'].max().min()
        
        # Determine if venue is batting or bowling friendly
        if avg_score > 180:
            batting_friendly = "Very Batting Friendly"
        elif avg_score > 165:
            batting_friendly = "Batting Friendly"
        elif avg_score > 150:
            batting_friendly = "Moderate"
        else:
            batting_friendly = "Bowling Friendly"
        
        # Average score of winning teams
        winning_matches = venue_matches.drop_duplicates('match_id')
        winning_scores = []
        for _, match in winning_matches.iterrows():
            winner = match['match_won_by']
            winner_score = venue_matches[(venue_matches['match_id'] == match['match_id']) & 
                                        (venue_matches['batting_team'] == winner)]['runs_total'].max()
            if winner_score > 0:
                winning_scores.append(winner_score)
        
        winning_team_avg_score = np.mean(winning_scores) if winning_scores else avg_score
        
        return {
            'venue_name': venue,
            'total_matches': total_matches,
            'avg_score': avg_score,
            'highest_score': highest_score,
            'lowest_score': lowest_score,
            'batting_friendly': batting_friendly,
            'winning_team_avg_score': winning_team_avg_score
        }
    
    def get_2026_season_data(self):
        """Get current season (2026) detailed data"""
        logger.info("Analyzing IPL 2026 season...")
        
        df_2026 = self.df[self.df['year'] == 2026]
        
        # Unique matches
        matches_2026 = df_2026.drop_duplicates('match_id')[['match_id', 'date', 'batting_team', 'bowling_team', 'match_won_by', 'win_outcome', 'venue']]
        
        # Team standings
        standings = []
        for match_id in matches_2026['match_id'].unique():
            match_info = matches_2026[matches_2026['match_id'] == match_id].iloc[0]
            if pd.notna(match_info['match_won_by']):
                standings.append({
                    'winner': match_info['match_won_by'],
                    'result': match_info['win_outcome']
                })
        
        standings_df = pd.DataFrame(standings)
        if len(standings_df) > 0:
            season_standings = standings_df.groupby('winner').size().reset_index(name='wins').sort_values('wins', ascending=False)
            return season_standings, matches_2026
        
        return None, matches_2026
    
    def predict_match(self, team1, team2, team1_recent_form=0.5, venue='Unknown'):
        """Simple prediction based on historical data"""
        
        # Get team win rates
        team_stats = self.get_team_statistics()
        
        team1_stats = team_stats.get(team1, {'win_rate': 0.5, 'avg_runs': 170})
        team2_stats = team_stats.get(team2, {'win_rate': 0.5, 'avg_runs': 170})
        
        # Calculate probability
        team1_win_prob = (
            team1_stats['win_rate'] * 0.4 +  # Win rate
            (team1_stats['avg_runs'] / 200) * 0.3 +  # Batting strength
            team1_recent_form * 0.2 +  # Recent form
            0.03  # Home advantage
        )
        
        team1_win_prob = np.clip(team1_win_prob, 0.3, 0.7)
        team2_win_prob = 1 - team1_win_prob
        
        return {
            'team1': team1,
            'team2': team2,
            'team1_win_prob': team1_win_prob,
            'team2_win_prob': team2_win_prob,
            'team1_avg_runs': team1_stats['avg_runs'],
            'team2_avg_runs': team2_stats['avg_runs'],
            'venue': venue
        }


def main():
    """Test analyzer"""
    analyzer = IPLAnalyzer('data/IPL_FINAL.csv')
    
    if analyzer.df is None:
        print("Failed to load data")
        return
    
    print("\n" + "="*70)
    print("IPL COMPREHENSIVE DATA ANALYSIS")
    print("="*70)
    
    # Team stats
    team_stats = analyzer.get_team_statistics()
    print(f"\nTop 5 Teams by Win Rate:")
    sorted_teams = sorted(team_stats.items(), key=lambda x: x[1]['win_rate'], reverse=True)
    for i, (team, stats) in enumerate(sorted_teams[:5], 1):
        print(f"  {i}. {team}: {stats['win_rate']:.1%} ({stats['wins']}/{stats['matches']})")
    
    # Player stats
    batsmen, bowlers = analyzer.get_player_statistics()
    print(f"\nTop 5 Batsmen:")
    for i, (name, row) in enumerate(batsmen.head().iterrows(), 1):
        print(f"  {i}. {name}: {row['runs']:.0f} runs @ {row['strike_rate']:.1f} SR")
    
    # 2026 Season
    standings, matches = analyzer.get_2026_season_data()
    if standings is not None:
        print(f"\nIPL 2026 Standings:")
        for i, row in standings.head(5).iterrows():
            print(f"  {i+1}. {row['winner']}: {row['wins']} wins")
    
    print("\n✓ Analysis complete!")


if __name__ == '__main__':
    main()
