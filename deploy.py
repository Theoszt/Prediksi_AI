import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import base64
import joblib
import numpy as np
from RandomForest import random_forest_baru,decisiontreebaru
from RandomForest2 import RandomForest,DecisionTree,Node


team_images = {
    'Ath Bilbao': r'Gambar/Ath Bilbao.png',
    'Betis': r'Gambar/Betis.png',
    'Celta': r'Gambar/Celta.png',
    'Las Palmas': r'Gambar/Las Palmas.png',
    'Osasuna': r'Gambar/Osasuna.png',
    'Valencia': r'Gambar/Valencia.png',
    'Sociedad': r'Gambar/Sociedad.png',
    'Mallorca': r'Gambar/Mallorca.png',
    'Valladolid': r'Gambar/Valladolid.png',
    'Villarreal': r'Gambar/Villarreal.png',
    'Sevilla': r'Gambar/Sevilla.png',
    'Barcelona': r'Gambar/Barcelona.png',
    'Espanol': r'Gambar/Espanol.png',
    'Getafe': r'Gambar/Getafe.png',
    'Real Madrid': r'Gambar/Real Madrid.png',
    'Leganes': r'Gambar/Leganes.png',
    'Alaves': r'Gambar/Alaves.png',
    'Ath Madrid': r'Gambar/Ath Madrid.png',
    'Vallecano': r'Gambar/Vallecano.png',
    'Girona': r'Gambar/Girona.png',
    'Cadiz': r'Gambar/Cadiz.png',
    'Granada': r'Gambar/Granada.png'
}
def get_image_as_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

st.set_page_config(page_title='FotMob Clone', layout='wide')
st.markdown("""
    <style>
    .header-container {
        background-color: #1e1e1e;
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .match-card {
        background-color: #292929;
        padding: 15px;
        border-radius: 10px;
        margin: 10px;
        text-align: center;
    }
    .standings-table-container {
        background-color: #1e1e1e;
        padding: 20px;
        border-radius: 10px;
        width: 100%;
    }
    .standings-table {
        width: 100%; 
        margin: auto;
        border-collapse: collapse;
        border: none;
    }
    .standings-table th, .standings-table td {
        padding: 10px;
        text-align: center;
        border: none;
        min-width: 150px;
        max-width: 200px; 
    }
    .standings-table th {
        color: white;
        background-color: #333333;
    }
    .standings-table tr {
        border: none; 
    }
    .dataframe {
        width: 100%; 
        overflow-x: auto; 
        border: none;
    }
    .dataframe td, .dataframe th {
        text-align: center;
        font-weight: bold;
        border: none;
    }
    .form-cell {
        display: inline-block;
        padding: 5px 10px;
        margin: 2px;
        border-radius: 5px;
        border: none;
    }
    /* General style for all elements with class 'st-emotion-cache' */
    .st-emotion-cache {
        padding-left: 22px;
    }
    .W { background-color: #4CAF50; color: white; }
    .D { background-color: #9E9E9E; color: white; }
    .L { background-color: #F44336; color: white; }
    </style>

""", unsafe_allow_html=True)

# Header Section
la_liga_logo_base64 = get_image_as_base64(r'Gambar/87.png')
st.markdown(f"""
<div class="header-container" style="text-align: center; width: 100%; position: relative; box-sizing: border-box; display: flex; align-items: center; justify-content: space-between; height: 90px; padding: 0 10px;">  <!-- Mengatur tinggi menjadi 60px -->
    <div style="display: flex; align-items: center; height: 100%;">  <!-- Mengatur tinggi 100% untuk child -->
        <img src="data:image/png;base64,{la_liga_logo_base64}" alt="La Liga Logo" style="width: 60px; height: auto; margin-right: 10px;">  <!-- Mengatur ukuran logo -->
        <div style="display: flex; flex-direction: column; align-items: left; height: 100%;">
            <h1 style="color: white; margin: 0; font-size: 1.5rem; padding:20px 0px 0px 0px;">La Liga</h1>
            <p style="color: white; margin: 0; text-align: left ; font-size: 1rem;">Spain</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)



# Load and Process Data
@st.cache_data
def load_data():
    data = pd.read_csv(r'SP1 (14).csv')
    data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')
    data_cleaned = data.drop(['B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 'BFH', 'BFD', 'BFA', 'PSH', 'PSD', 'PSA',
        'WHH', 'WHD', 'WHA', '1XBH', '1XBD', '1XBA', 'MaxH', 'MaxD', 'MaxA', 'AvgH', 'AvgD', 'AvgA',
        'BFEH', 'BFED', 'BFEA', 'B365>2.5', 'B365<2.5', 'P>2.5', 'P<2.5', 'Max>2.5', 'Max<2.5',
        'Avg>2.5', 'Avg<2.5', 'BFE>2.5', 'BFE<2.5', 'AHh', 'B365AHH', 'B365AHA', 'PAHH', 'PAHA',
        'MaxAHH', 'MaxAHA', 'AvgAHH', 'AvgAHA', 'BFEAHH', 'BFEAHA', 'B365CH', 'B365CD', 'B365CA',
        'BWCH', 'BWCD', 'BWCA', 'BFCH', 'BFCD', 'BFCA', 'PSCH', 'PSCD', 'PSCA', 'WHCH', 'WHCD',
        'WHCA', '1XBCH', '1XBCD', '1XBCA', 'MaxCH', 'MaxCD', 'MaxCA', 'AvgCH', 'AvgCD', 'AvgCA',
        'BFECH', 'BFECD', 'BFECA', 'B365C>2.5', 'B365C<2.5', 'PC>2.5', 'PC<2.5', 'MaxC>2.5', 'MaxC<2.5',
        'AvgC>2.5', 'AvgC<2.5', 'BFEC>2.5', 'BFEC<2.5', 'AHCh', 'B365CAHH', 'B365CAHA', 'PCAHH',
        'PCAHA', 'MaxCAHH', 'MaxCAHA', 'AvgCAHH', 'AvgCAHA', 'BFECAHH', 'BFECAHA'], axis=1)
    return data_cleaned

data_cleaned = load_data()

# Calculate Standings
# Calculate Standings
teams = pd.concat([data_cleaned['HomeTeam'], data_cleaned['AwayTeam']]).unique()
standings = pd.DataFrame(teams, columns=['Team'])

standings['PL'] = 0
standings['W'] = 0
standings['D'] = 0
standings['L'] = 0
standings['+'] = 0
standings['-'] = 0
standings['GD'] = 0
standings['PTS'] = 0
standings['Form'] = ''
logos = {team: get_image_as_base64(path) for team, path in team_images.items()}
standings['Logo'] = standings['Team'].map(logos)

for _, match in data_cleaned.iterrows():
    home_team = match['HomeTeam']
    away_team = match['AwayTeam']
    home_goals = match['FTHG']
    away_goals = match['FTAG']
    result = match['FTR']

    standings.loc[standings['Team'] == home_team, 'PL'] += 1
    standings.loc[standings['Team'] == away_team, 'PL'] += 1

    standings.loc[standings['Team'] == home_team, '+'] += home_goals
    standings.loc[standings['Team'] == home_team, '-'] += away_goals
    standings.loc[standings['Team'] == away_team, '+'] += away_goals
    standings.loc[standings['Team'] == away_team, '-'] += home_goals

    if result == 'H':
        standings.loc[standings['Team'] == home_team, 'PTS'] += 3
        standings.loc[standings['Team'] == home_team, 'W'] += 1
        standings.loc[standings['Team'] == away_team, 'L'] += 1
        standings.loc[standings['Team'] == home_team, 'Form'] += 'W'
        standings.loc[standings['Team'] == away_team, 'Form'] += 'L'
    elif result == 'A':
        standings.loc[standings['Team'] == away_team, 'PTS'] += 3
        standings.loc[standings['Team'] == away_team, 'W'] += 1
        standings.loc[standings['Team'] == home_team, 'L'] += 1
        standings.loc[standings['Team'] == away_team, 'Form'] += 'W'
        standings.loc[standings['Team'] == home_team, 'Form'] += 'L'
    elif result == 'D':
        standings.loc[standings['Team'] == home_team, 'PTS'] += 1
        standings.loc[standings['Team'] == away_team, 'PTS'] += 1
        standings.loc[standings['Team'] == home_team, 'D'] += 1
        standings.loc[standings['Team'] == away_team, 'D'] += 1
        standings.loc[standings['Team'] == home_team, 'Form'] += 'D'
        standings.loc[standings['Team'] == away_team, 'Form'] += 'D'

standings['GD'] = standings['+'] - standings['-']
standings['Form'] = standings['Form'].apply(lambda x: ''.join([f'<span class="form-cell {char}">{char}</span>' for char in x[-5:]]))

# Debugging: Print kolom yang ada di DataFrame
print("Kolom yang ada di DataFrame:", standings.columns.tolist())

# Pastikan kolom yang ingin diurutkan ada
standings = standings.sort_values(by=['PTS', 'GD', '+'], ascending=False).reset_index(drop=True)


standings_html = standings.to_html(escape=False, index=False, 
                                    columns=['Logo', 'Team', 'PL','W','D','L','+','-','GD', 'PTS','Form'],
                                    formatters={'Logo': lambda x: f'<img src="data:image/png;base64,{x}" width="30" height="30">'})
standings_html = standings_html.replace('<th>Logo</th>', '<th></th>')

# Menampilkan tabel

def load_yang_akan_datang():
    
    
    jadwal_path = r'la_liga_matches_updated.csv'
    jadwal_pertandingan_df = pd.read_csv(jadwal_path)

    # Convert the 'date' column to datetime and extract the date part
    jadwal_pertandingan_df['date'] = pd.to_datetime(jadwal_pertandingan_df['date']).dt.date
    
    today = datetime.now().date()
    jadwal_tomorrow_df = jadwal_pertandingan_df[jadwal_pertandingan_df['date'] > today]
    return jadwal_tomorrow_df


def load_sudah_terjadi():
    data = pd.read_csv(r'SP1 (14).csv')
    # data = pd.read_csv(r'Laliga_2011-2025.csv')
    # Memilih kolom yang diperlukan
    selected_columns = data[['Date', 'Time', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']]
    
    return selected_columns


def fitur():
    data = pd.read_csv(r'SP1 (14).csv')

    data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')
    data = data.sort_values(by='Date')

    # Function to calculate home performance for each team
    def calculate_home_performance(team):
        home_games = data[data['HomeTeam'] == team]
        home_wins = (home_games['FTR'] == 'H').sum()
        home_draws = (home_games['FTR'] == 'D').sum()
        home_losses = (home_games['FTR'] == 'A').sum()
        home_games_played = home_games.shape[0]
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        numeric_sums = home_games[numeric_columns].sum()

        result = {
            'Team': team,
            'home_wins': home_wins,
            'home_draws': home_draws,
            'home_losses': home_losses,
            'home_games_played': home_games_played
        }
        for col in numeric_columns:
            result[f'home_{col}'] = numeric_sums[col]

        return result

    # Function to calculate away performance for each team
    def calculate_away_performance(team):
        away_games = data[data['AwayTeam'] == team]
        away_wins = (away_games['FTR'] == 'A').sum()
        away_draws = (away_games['FTR'] == 'D').sum()
        away_losses = (away_games['FTR'] == 'H').sum()
        away_games_played = away_games.shape[0]
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        numeric_sums = away_games[numeric_columns].sum()

        result = {
            'Team': team,
            'away_wins': away_wins,
            'away_draws': away_draws,
            'away_losses': away_losses,
            'away_games_played': away_games_played
        }
        for col in numeric_columns:
            result[f'away_{col}'] = numeric_sums[col]

        return result

    # Calculate home performance for all teams
    home_teams = data['HomeTeam'].unique()
    home_performance = [calculate_home_performance(team) for team in home_teams]
    home_full_performance = pd.DataFrame(home_performance)

    # Calculate away performance for all teams
    away_teams = data['AwayTeam'].unique()
    away_performance = [calculate_away_performance(team) for team in away_teams]
    away_full_performance = pd.DataFrame(away_performance)

    # Merge home and away performance
    full_performance = pd.merge(home_full_performance, away_full_performance, on='Team', how='outer')

    # Calculate additional metrics
    full_performance['home_win_ratio'] = full_performance['home_wins'] / full_performance['home_games_played']
    full_performance['away_win_ratio'] = full_performance['away_wins'] / full_performance['away_games_played']
    full_performance['home_goals_avg'] = full_performance['home_FTHG'] / full_performance['home_games_played']
    full_performance['away_goals_avg'] = full_performance['away_FTAG'] / full_performance['away_games_played']
    full_performance['home_concede_avg'] = full_performance['home_FTAG'] / full_performance['home_games_played']
    full_performance['away_concede_avg'] = full_performance['away_FTHG'] / full_performance['away_games_played']

    # Function to get last 5 games results
    def get_last_5_games(team, results_column='FTR', home_column='HomeTeam', away_column='AwayTeam', n=5):
        team_home = data[data[home_column] == team][['Date', results_column]]
        team_home['Result'] = team_home[results_column].replace({'H': 'W', 'A': 'L', 'D': 'D'})
        team_away = data[data[away_column] == team][['Date', results_column]]
        team_away['Result'] = team_away[results_column].replace({'H': 'L', 'A': 'W', 'D': 'D'})
        all_results = pd.concat([team_home, team_away]).sort_values(by='Date')
        return ''.join(all_results['Result'].tail(n)) if len(all_results) >= n else ''.join(all_results['Result'])

    # Add 'last_5_games' column to the performance data
    full_performance['last_5_games'] = full_performance['Team'].apply(
        lambda team: get_last_5_games(team)
    )

    def convert_last_5_to_points(results):
        points_map = {'W': 2, 'D': 1, 'L': 0}
        return sum(points_map[result] for result in results)

    # Replace last_5_games with total points
    full_performance['last_5_games'] = full_performance['last_5_games'].apply(
        lambda results: convert_last_5_to_points(results)
    )

    # Define features
    home_feature = [
        'home_wins', 'home_draws', 'home_losses', 'home_FTHG', 'home_FTAG', 'home_HTHG',
        'home_HTAG', 'home_HS', 'home_win_ratio', 'home_goals_avg', 'home_concede_avg', 'last_5_games'
    ]
    away_feature = [
        'away_wins', 'away_draws', 'away_losses', 'away_FTHG', 'away_FTAG', 'away_HTHG',
        'away_HTAG', 'away_HS', 'away_win_ratio', 'away_goals_avg', 'away_concede_avg', 'last_5_games'
    ]

    # Load jadwal pertandingan
    jadwal_path = r'la_liga_matches_updated.csv'
    jadwal_pertandingan_df = pd.read_csv(jadwal_path)

    # Convert the 'date' column to datetime and extract the date part
    jadwal_pertandingan_df['date'] = pd.to_datetime(jadwal_pertandingan_df['date']).dt.date
    
    today = datetime.now().date()
    jadwal_besok = jadwal_pertandingan_df[jadwal_pertandingan_df['date'] > today]

    # Prepare home and away data for merging
    home_data = full_performance[home_feature + ['Team']].rename(columns=lambda x: x if x == 'Team' else f"{x}_home")
    away_data = full_performance[away_feature + ['Team']].rename(columns=lambda x: x if x == 'Team' else f"{x}_away")

    # Merge home and away data into jadwal_besok
    jadwal_besok = jadwal_besok.merge(home_data, left_on='Home Team', right_on='Team', how='left') /
                                 .merge(away_data, left_on='Away Team', right_on='Team', how='left', suffixes=('_home', '_away'))

    return jadwal_besok

def model():
    standard=joblib.load(r'standard_scal_rendra.joblib')
    isi_fitur=fitur()
    model=joblib.load(r'model_rendra.joblib')
    feature_columns = [
        'home_wins_home', 'home_draws_home', 'home_losses_home', 
        'home_FTHG_home', 'home_FTAG_home', 'home_HTHG_home', 
        'home_HTAG_home', 'home_HS_home', 'home_win_ratio_home', 
        'home_goals_avg_home', 'home_concede_avg_home', 'last_5_games_home',
        'away_wins_away', 'away_draws_away', 'away_losses_away', 
        'away_FTHG_away', 'away_FTAG_away', 'away_HTHG_away', 
        'away_HTAG_away', 'away_HS_away', 'away_win_ratio_away', 
        'away_goals_avg_away', 'away_concede_avg_away', 'last_5_games_away'
    ]
    
    # Transform the features using the scaler
    X_upcoming = standard.transform(isi_fitur[feature_columns])
    prediksi = model.predict_probabaru(X_upcoming)
    prediksi = np.array(prediksi)

    # Threshold adjustments
    minimal_homewin = 0.15
    minimal_draw = 0.25
    minimal_awaywin = 0.15

    prediksi[:, 0] = np.maximum(prediksi[:, 0], minimal_homewin)  # Home Win
    prediksi[:, 1] = np.maximum(prediksi[:, 1], minimal_draw)      # Draw
    prediksi[:, 2] = np.maximum(prediksi[:, 2], minimal_awaywin)   # Away Win

    # Normalize probabilities
    probability_sum = prediksi.sum(axis=1).reshape(-1, 1)
    prediksi = prediksi / probability_sum

    # Create results DataFrame
    results = pd.DataFrame(prediksi, columns=['Away Win', 'Draw', 'Home Win'])
    results['round'] = isi_fitur['round']  # Assuming 'round' is in isi_fitur
    results['Home Team'] = isi_fitur['Home Team']  # Assuming 'Home Team' is in isi_fitur
    results['Away Team'] = isi_fitur['Away Team']  # Assuming 'Away Team' is in isi_fitur
    results['date'] = isi_fitur['date']  # Assuming 'date' is in isi_fitur

    # Format the probabilities as percentages
    results['Home Win'] = (results['Home Win'] * 100).round(2).astype(str) + '%'
    results['Draw'] = (results['Draw'] * 100).round(2).astype(str) + '%'
    results['Away Win'] = (results['Away Win'] * 100).round(2).astype(str) + '%'
    
    # Select relevant columns for the final output
    results = results[['round', 'date', 'Home Team', 'Away Team', 'Home Win', 'Draw', 'Away Win']]
    
    return results
def model2():
    model2=joblib.load(r'RandomForest_Model_2.joblib')
    away=joblib.load(r'label_encoder_away.joblib')
    home=joblib.load(r'label_encoder_home.joblib')
    winner=joblib.load(r'label_encoder_winners.joblib')
    dataset=pd.read_csv(r'Stat_laliga.csv')
    scaler=joblib.load(r'standard_scaler.joblib')
    if 'Home Team' in dataset.columns and 'Away Team' in dataset.columns:
        dataset['HomeTeam'] = home.transform(dataset['Home Team'])
        dataset['AwayTeam'] = away.transform(dataset['Away Team'])
    else:
        print("Kolom 'HomeTeam' atau 'AwayTeam' tidak ditemukan di data.")

    # Daftar kolom fitur yang diharapkan oleh model
    expected_feature_order = [
        'HomeTeam', 'AwayTeam', 'mean_points_per_match_tot_home',
        'mean_points_per_match_tot_away', 'matches_won_tot_home',
        'matches_won_tot_away', 'matches_lost_tot_home',
        'matches_lost_tot_away', 'mean_shots_home', 'mean_shots_away',
        'mean_shots_on_target_home', 'mean_shots_on_target_away',
        'mean_fouls_home', 'mean_fouls_away', 'h2h_home_wins', 'h2h_away_wins',
        'h2h_draws', 'home_win_ratio', 'away_win_ratio',
        'avg_goals_conceded_home', 'avg_goals_conceded_away'
    ]

    # Tambahkan kolom yang hilang
    for col in expected_feature_order:
        if col not in dataset.columns:
            dataset[col] = 0  

    df_dummy = dataset[expected_feature_order]

    # Standard Scale the features
    df_dummy[expected_feature_order] = scaler.transform(df_dummy[expected_feature_order])


    X = df_dummy[expected_feature_order].to_numpy()  
    probabilities = model2.predict_proba(X)

    # Membuat prediksi
    predictions = model2.predict(X)

    # Tampilkan prediksi
    print("\nPrediksi untuk data dummy:")
    print(predictions)
    result_labels = winner.inverse_transform(predictions)  
    prediksi=np.array(probabilities)
    minimal_homewin = 0.15
    minimal_draw = 0.25
    minimal_awaywin = 0.15

    prediksi[:, 0] = np.maximum(prediksi[:, 0], minimal_homewin)  # Home Win
    prediksi[:, 1] = np.maximum(prediksi[:, 1], minimal_draw)      # Draw
    prediksi[:, 2] = np.maximum(prediksi[:, 2], minimal_awaywin)   # Away Win

    # Normalize probabilities
    probability_sum = prediksi.sum(axis=1).reshape(-1, 1)
    prediksi = prediksi / probability_sum
    probability_sum = prediksi.sum(axis=1).reshape(-1, 1)
    prediksi = prediksi / probability_sum

    # Create results DataFrame
    results = pd.DataFrame(prediksi, columns=['Away Win', 'Draw', 'Home Win'])
    results['round'] = dataset['round']  
    results['date'] = pd.to_datetime(dataset['date'], errors='coerce')
    results['Home Team'] = dataset['Home Team'] 
    results['Away Team'] = dataset['Away Team']  

    # Format the probabilities as percentages
    results['Home Win'] = (results['Home Win'] * 100).round(2).astype(str) + '%'
    results['Draw'] = (results['Draw'] * 100).round(2).astype(str) + '%'
    results['Away Win'] = (results['Away Win'] * 100).round(2).astype(str) + '%'
    results['Winners']=result_labels
    # Select relevant columns for the final output
    results = results[['round', 'date', 'Home Team', 'Away Team', 'Home Win', 'Draw', 'Away Win']]
    return results
yesterday= load_sudah_terjadi()
test=fitur()
tomorrow= load_yang_akan_datang()
print(tomorrow)

# Display content based on selected tab
def klasmen():
    st.subheader('Standings')
    st.markdown(f"""
        <div class="standings-table-container">
            <table class="standings-table">
                {standings_html}
            </table>
        </div>
    """, unsafe_allow_html=True)
# Add additional sections for "tomorrow" and "Tomorrow" tabs as needed
def page2():
    model_choice = st.selectbox("Select Prediction", ["prediksi A", "prediksi B"])

    if model_choice == "prediksi A":
        predictions = model()
    else:
        predictions = model2()

    # Kolom pencarian untuk nama tim

    # Filter berdasarkan input pencarian
    if search_team:
        filtered_predictions = predictions[
            (predictions['Home Team'].str.contains(search_team, case=False)) |
            (predictions['Away Team'].str.contains(search_team, case=False))
        ]
    else:
        filtered_predictions = predictions

    # Cek apakah ada pertandingan
    if filtered_predictions.empty:
        st.write("No matches scheduled for predictions.")
    else:
        # Dapatkan matchday unik dari kolom 'round'
        unique_matchdays = filtered_predictions['round'].unique()
        
        # Buat dropdown untuk memilih matchday
        selected_matchday = st.selectbox("Select Matchday", unique_matchdays)

        # Filter pertandingan berdasarkan matchday yang dipilih
        filtered_matches = filtered_predictions[filtered_predictions['round'] == selected_matchday]

        # Cek apakah ada pertandingan untuk matchday yang dipilih
        if filtered_matches.empty:
            st.write("No matches scheduled for the selected matchday.")
        else:
            # Buat kotak untuk setiap pertandingan yang dijadwalkan
            for _, match in filtered_matches.iterrows():
                home_team = match['Home Team']
                away_team = match['Away Team']
                match_time = match['date']

                match_baru = match_time.strftime('%d/%m/%Y')
                # Ganti dengan fungsi untuk mendapatkan gambar tim
                home_team_image_base64 = get_image_as_base64(team_images[home_team])
                away_team_image_base64 = get_image_as_base64(team_images[away_team])
                
                home_win_prob = float(match['Home Win'].replace('%', ''))  # Hapus simbol persen dan konversi
                draw_prob = float(match['Draw'].replace('%', ''))  # Hapus simbol persen dan konversi
                away_win_prob = float(match['Away Win'].replace('%', ''))  # Hapus simbol persen dan konversi
                total_prob = home_win_prob + draw_prob + away_win_prob
                
                # Buat kotak pertandingan untuk setiap pertandingan
                st.markdown(f"""
                <div style="background-color:#292929; padding: 20px; border-radius: 10px; margin-bottom: 10px; width: 100%; display: flex; align-items: center; justify-content: space-between;">
                    <div style="display: flex; flex-direction: column; align-items: center;">
                        <img src="data:image/png;base64,{home_team_image_base64}" width="80">
                        <h4 style="color: white; text-align: center; width: 90px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 13px; display: flex; flex-direction: column; justify-content: center;  align-items: center;">{home_team}</h4>
                    </div>
                    <div style="flex-grow: 1; text-align: center;">
                        <p style="color: white;">Match Date: {match_baru}</p>
                        <p style="color: white; font-weight: bold;">PELUANG MENANG</p>
                        <table style="width: 100%; border-collapse: collapse;">
                            <tr>
                                <td style="background-color: orange; height: 5px; border-radius: 0px; width: {home_win_prob / total_prob * 100:.0f}%; padding: 0;">H {home_win_prob}%</td>
                                <td style="background-color: gray; height: 5px; border-radius: 0px; width: {draw_prob / total_prob * 100:.0f}%; padding: 0;">D {draw_prob}%</td>
                                <td style="background-color: blue; height: 5px; border-radius: 0px; width: {away_win_prob / total_prob * 100:.0f}%; padding: 0;">A {away_win_prob}%</td>
                            </tr>
                        </table>
                    </div>
                    <div style="display: flex; flex-direction: column; align-items: center;">
                        <img src="data:image/png;base64,{away_team_image_base64}" width="80">
                        <h4 style="color: white; text-align: center; width: 90px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 13px; display: flex; flex-direction: column; justify-content: center;  align-items: center;">{away_team}</h4>
                    </div>
                </div>
                """, unsafe_allow_html=True)

def page3():
    st.subheader('Yesterday\'s Matches')

    # Set to yesterday's date
    yesterday['Date'] = pd.to_datetime(yesterday['Date'], errors='coerce')
    yesterday['Month'] = yesterday['Date'].dt.month_name()  # Get month names

    # Get unique teams for the dropdown
    teams = pd.concat([yesterday['HomeTeam'], yesterday['AwayTeam']]).unique()

    # User selects a month
    col1, col2 = st.columns(2)

    # User selects a month di kolom pertama
    with col1:
        selected_month = st.selectbox("Select Month", sorted(yesterday['Month'].unique()))

    # User selects a team di kolom kedua

    # Filter matches based on selected month and team
    filtered_matches = yesterday[
        (yesterday['Month'] == selected_month) 
    ]

    # Tambahkan filter berdasarkan pencarian tim
    if search_team:
        filtered_matches = filtered_matches[
            (filtered_matches['HomeTeam'].str.contains(search_team, case=False)) |
            (filtered_matches['AwayTeam'].str.contains(search_team, case=False))
        ]

    # If no matches are found
    if filtered_matches.empty:
        st.write(f"No matches found for in {selected_month}.")
    else:
        # Display match information
        for _, match in filtered_matches.iterrows():
            home_team = match['HomeTeam']
            away_team = match['AwayTeam']
            skorhome = match['FTHG']
            skoraway = match['FTAG']
            time = match['Time']
            tanggal = match['Date']
            tanggal_baru = tanggal.strftime('%d/%m/%Y')
            home_team_image_base64 = get_image_as_base64(team_images[home_team])
            away_team_image_base64 = get_image_as_base64(team_images[away_team])

            # Displaying match information
            st.markdown(f"""
            <div style="background-color:#292929; padding: 20px; border-radius: 10px; margin-bottom: 10px; width: 100%; display: flex; align-items: center; justify-content: space-between; gap: 3rem;">
                <div style="display: flex; flex-direction: column; align-items: center;">
                    <img src="data:image/png;base64,{home_team_image_base64}" alt="Team Logo" width="80">
                    <h4 style="color: white; text-align: center; width: 90px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 13px; display: flex; flex-direction: column; justify-content: center; align-items: center;">{home_team}</h4>
                </div>
                <h4 style="color: white; text-align: center; width: 50px; font-size: 2rem;">{skorhome}</h4>
                <div style="flex-grow: 1; text-align: center;">
                    <p style="color: white; font-size: 1rem;">Match Time: {time}</p>
                    <p style="color: white; font-size: 1rem;">{tanggal_baru}</p>
                </div>
                <h4 style="color: white; text-align: center; width: 50px; font-size: 2rem;">{skoraway}</h4>
                <div style="display: flex; flex-direction: column; align-items: center;">
                    <img src="data:image/png;base64,{away_team_image_base64}" alt="Team Logo" width="80">
                    <h4 style="color: white; text-align: center; width: 90px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 13px; display: flex; flex-direction: column; justify-content: center; align-items: center;">{away_team}</h4>
                </div>
            </div>
            """, unsafe_allow_html=True)
search_team = st.sidebar.text_input("Search Team", "")
pg = st.navigation([st.Page(klasmen, title="Klasemen"), st.Page(page3, title="Yesterday"), st.Page(page2, title="Tomorrow")])
pg.run()

