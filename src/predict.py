import joblib
import pandas as pd
import sys

# Load the model and encoder
model = joblib.load('artifacts/xgb_model.joblib')
encoder = joblib.load('artifacts/label_encoder.joblib')

# Read recent match history
recent_matches = pd.read_csv('data/RecentMatches.csv')

# Function to build features
# (Assuming build_features is defined elsewhere)
features = build_features(recent_matches)

# Select the most recent match per Date+HomeTeam+AwayTeam
recent_matches['Date'] = pd.to_datetime(recent_matches['Date'])
recent_matches = recent_matches.sort_values('Date')
latest_matches = recent_matches.groupby(['Date', 'HomeTeam', 'AwayTeam']).last().reset_index()

# Encode teams
latest_matches['HomeTeam'] = encoder.transform(latest_matches['HomeTeam'])
latest_matches['AwayTeam'] = encoder.transform(latest_matches['AwayTeam'])

# Drop non-feature columns
latest_matches = latest_matches.drop(columns=['Date', 'some_other_columns_to_drop'])  # Update with actual columns to drop

# Predict
predictions = model.predict(latest_matches)
probabilities = model.predict_proba(latest_matches)

# Output predictions to stdout
for pred, prob in zip(predictions, probabilities):
    print(f'Predicted class: {pred}, Probabilities: {prob}')

# Output predictions to CSV
output_df = pd.DataFrame({'Predicted Class': predictions, 'Probabilities': probabilities.tolist()})
output_df.to_csv('data/predictions.csv', index=False)