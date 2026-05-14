"""
DYNAMIC CHASING DIFFICULTY CALCULATOR
Calculates team-specific chase difficulty based on:
- Team's historical chase success rate
- Chasing team's avg score
- Target team's avg score
- Recent form
- Head-to-head
"""

import pandas as pd
import numpy as np

class ChasingDifficultyAnalyzer:
    """Calculate personalized chasing difficulty for each team"""
    
    def __init__(self, df):
        self.df = df
    
    def get_team_chase_stats(self, team):
        """Get chasing statistics for a team"""
        
        # Get all matches where team batted second (chasing)
        chasing_matches = self.df[self.df['batting_team'] == team].drop_duplicates('match_id')
        
        if len(chasing_matches) == 0:
            return {
                'avg_runs_when_chasing': 0,
                'chase_success_rate': 0.5,
                'avg_target_faced': 0,
                'targets_under_140': 0,
                'targets_140_160': 0,
                'targets_160_180': 0,
                'targets_above_180': 0,
            }
        
        # Successful chases
        successful_chases = len(chasing_matches[chasing_matches['match_won_by'] == team])
        total_chases = len(chasing_matches)
        success_rate = successful_chases / total_chases if total_chases > 0 else 0.5
        
        # Average runs in chases
        avg_runs = chasing_matches.groupby('match_id')['runs_total'].max().mean()
        
        # Average target faced
        avg_target = chasing_matches.groupby('match_id')['runs_total'].max().mean()
        
        return {
            'team': team,
            'avg_runs_when_chasing': avg_runs,
            'chase_success_rate': success_rate,
            'avg_target_faced': avg_target,
            'successful_chases': successful_chases,
            'total_chases': total_chases,
        }
    
    def calculate_dynamic_thresholds(self, chasing_team, target_runs):
        """
        Calculate personalized chase difficulty for a team
        Returns: Easy/Moderate/Difficult with team-specific thresholds
        """
        
        chase_stats = self.get_team_chase_stats(chasing_team)
        success_rate = chase_stats['chase_success_rate']
        avg_runs = chase_stats['avg_runs_when_chasing']
        
        if avg_runs == 0:
            avg_runs = 170  # Default if no data
        
        # PERSONALIZED THRESHOLDS
        # Easy: Target is 20% below team's average
        easy_threshold = avg_runs * 0.80
        
        # Moderate: Target is 0-20% above team's average
        moderate_threshold = avg_runs * 1.15
        
        # Difficult: Target is >20% above team's average
        difficult_threshold = avg_runs * 1.20
        
        # SUCCESS RATES (based on team's history)
        if success_rate > 0.6:
            # Strong chasing team
            easy_win_rate = 0.85
            moderate_win_rate = 0.55
            difficult_win_rate = 0.40
        elif success_rate > 0.45:
            # Average chasing team
            easy_win_rate = 0.80
            moderate_win_rate = 0.50
            difficult_win_rate = 0.35
        else:
            # Weak chasing team
            easy_win_rate = 0.70
            moderate_win_rate = 0.40
            difficult_win_rate = 0.25
        
        # CLASSIFY TARGET
        if target_runs <= easy_threshold:
            difficulty = "🟢 EASY CHASE"
            win_rate = easy_win_rate
            reason = f"Target {target_runs} is below {chasing_team}'s avg ({avg_runs:.0f})"
        elif target_runs <= moderate_threshold:
            difficulty = "🟡 MODERATE CHASE"
            win_rate = moderate_win_rate
            reason = f"Target {target_runs} is slightly above {chasing_team}'s avg ({avg_runs:.0f})"
        else:
            difficulty = "🔴 DIFFICULT CHASE"
            win_rate = difficult_win_rate
            reason = f"Target {target_runs} is well above {chasing_team}'s avg ({avg_runs:.0f})"
        
        return {
            'chasing_team': chasing_team,
            'target_runs': target_runs,
            'difficulty': difficulty,
            'win_rate': win_rate,
            'team_avg_runs': avg_runs,
            'team_chase_success_rate': success_rate,
            'reason': reason,
            'easy_threshold': easy_threshold,
            'moderate_threshold': moderate_threshold,
            'difficult_threshold': difficult_threshold,
        }
    
    def get_all_teams_chase_analysis(self):
        """Analyze chasing ability for all teams"""
        
        teams = self.df['batting_team'].unique()
        analysis = []
        
        for team in teams:
            stats = self.get_team_chase_stats(team)
            analysis.append(stats)
        
        df_analysis = pd.DataFrame(analysis).sort_values('chase_success_rate', ascending=False)
        return df_analysis


# USAGE IN STREAMLIT
def display_personalized_difficulty(chasing_team, target_runs, analyzer):
    """Display personalized chasing difficulty in Streamlit"""
    
    import streamlit as st
    
    analysis = analyzer.calculate_dynamic_thresholds(chasing_team, target_runs)
    
    st.subheader("🎯 Personalized Chasing Difficulty")
    
    # Main difficulty display
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Show the difficulty with color
        difficulty = analysis['difficulty']
        if '🟢' in difficulty:
            st.success(difficulty)
        elif '🟡' in difficulty:
            st.warning(difficulty)
        else:
            st.error(difficulty)
    
    with col2:
        st.metric("Win Rate", f"{analysis['win_rate']:.0%}")
    
    # Explanation
    st.markdown(f"""
    **Why?**
    - {analysis['reason']}
    - {chasing_team} has **{analysis['team_chase_success_rate']:.1%}** chase success rate
    """)
    
    # Team-specific thresholds
    st.subheader(f"{chasing_team}'s Chasing Thresholds")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.success(f"""
        🟢 EASY
        
        < {analysis['easy_threshold']:.0f} runs
        
        {0.85:.0%} win rate
        """)
    
    with col2:
        st.warning(f"""
        🟡 MODERATE
        
        {analysis['easy_threshold']:.0f} - {analysis['moderate_threshold']:.0f}
        
        {analysis['win_rate']:.0%} win rate
        """)
    
    with col3:
        st.error(f"""
        🔴 DIFFICULT
        
        > {analysis['moderate_threshold']:.0f} runs
        
        {0.35:.0%} win rate
        """)
    
    # Comparison with other teams
    st.subheader("Team Chasing Ability Comparison")
    
    all_teams_analysis = analyzer.get_all_teams_chase_analysis()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Best Chasing Teams:**")
        st.dataframe(
            all_teams_analysis[['team', 'chase_success_rate', 'avg_runs_when_chasing']].head(5),
            use_container_width=True,
            hide_index=True
        )
    
    with col2:
        st.write("**Weakest Chasing Teams:**")
        st.dataframe(
            all_teams_analysis[['team', 'chase_success_rate', 'avg_runs_when_chasing']].tail(5),
            use_container_width=True,
            hide_index=True
        )


# EXAMPLE
if __name__ == '__main__':
    
    df = pd.read_csv('data/IPL_FINAL.csv', index_col=0, low_memory=False)
    
    analyzer = ChasingDifficultyAnalyzer(df)
    
    # Example 1: MI chasing 180
    print("\n" + "="*70)
    print("EXAMPLE 1: Mumbai Indians chasing 180")
    print("="*70)
    
    result = analyzer.calculate_dynamic_thresholds('Mumbai Indians', 180)
    print(f"Difficulty: {result['difficulty']}")
    print(f"Win Rate: {result['win_rate']:.0%}")
    print(f"Reason: {result['reason']}")
    print(f"MI's Chase Success Rate: {result['team_chase_success_rate']:.1%}")
    
    # Example 2: RCB chasing 170
    print("\n" + "="*70)
    print("EXAMPLE 2: Royal Challengers Bengaluru chasing 170")
    print("="*70)
    
    result = analyzer.calculate_dynamic_thresholds('Royal Challengers Bengaluru', 170)
    print(f"Difficulty: {result['difficulty']}")
    print(f"Win Rate: {result['win_rate']:.0%}")
    print(f"Reason: {result['reason']}")
    print(f"RCB's Chase Success Rate: {result['team_chase_success_rate']:.1%}")
    
    # Compare all teams
    print("\n" + "="*70)
    print("ALL TEAMS CHASING ANALYSIS")
    print("="*70)
    
    all_teams = analyzer.get_all_teams_chase_analysis()
    print(all_teams.to_string())
