import os
from pathlib import Path

import joblib
import pandas as pd

def get_points(hg, ag):
    if hg > ag:
        return (3, 0)
    if hg < ag:
        return (0, 3)
    return (1, 1)

def compute_team_stats(history: pd.DataFrame, rolling_window: int) -> pd.DataFrame:
    history = history.copy()
    history["Date"] = pd.to_datetime(history["Date"], dayfirst=True, errors="coerce")
    history = history.dropna(subset=["Date"]).sort_values("Date")

    history["HomePoints"], history["AwayPoints"] = zip(
        *history.apply(lambda r: get_points(r.FTHG, r.FTAG), axis=1)
    )

    home_stats = history[["Date", "HomeTeam", "FTHG", "FTAG", "HomePoints"]].copy()
    home_stats.rename(
        columns={
            "HomeTeam": "Team",
            "FTHG": "GoalsFor",
            "FTAG": "GoalsAgainst",
            "HomePoints": "Points",
        },
        inplace=True,
    )

    away_stats = history[["Date", "AwayTeam", "FTAG", "FTHG", "AwayPoints"]].copy()
    away_stats.rename(
        columns={
            "AwayTeam": "Team",
            "FTAG": "GoalsFor",
            "FTHG": "GoalsAgainst",
            "AwayPoints": "Points",
        },
        inplace=True,
    )

    team_stats = pd.concat([home_stats, away_stats], ignore_index=True)
    team_stats.sort_values(["Team", "Date"], inplace=True)

    team_stats["GoalDiff"] = team_stats["GoalsFor"] - team_stats["GoalsAgainst"]
    team_stats["Win"] = (team_stats["Points"] == 3).astype(int)

    for col in ["GoalsFor", "GoalsAgainst", "GoalDiff", "Points", "Win"]:
        team_stats[f"{col}_rolling"] = team_stats.groupby("Team")[col].transform(
            lambda x: x.shift().rolling(rolling_window, min_periods=1).mean()
        )

    for col in ["GoalsFor", "GoalsAgainst", "Points"]:
        team_stats[f"{col}_cumu"] = team_stats.groupby("Team")[col].transform(
            lambda x: x.shift().cumsum() / (x.shift().expanding().count())
        )

    return team_stats

def merge_latest_stats(fixtures: pd.DataFrame, team_stats: pd.DataFrame, side: str) -> pd.DataFrame:
    side_col = f"{side}Team"

    request = fixtures[["MatchKey", "Date", side_col]].rename(columns={side_col: "Team"})
    request = request.sort_values(["Team", "Date"]) 

    team_stats_sorted = team_stats.sort_values(["Team", "Date"]) 

    merged = pd.merge_asof(
        request,
        team_stats_sorted,
        on="Date",
        by="Team",
        direction="backward",
    )

    prefix = "Home" if side == "Home" else "Away"

    merged = merged.rename(
        columns={
            "GoalsFor_rolling": f"{prefix}GF_avg",
            "GoalsAgainst_rolling": f"{prefix}GA_avg",
            "GoalDiff_rolling": f"{prefix}GD_avg",
            "Points_rolling": f"{prefix}Pts_avg",
            "Win_rolling": f"{prefix}WinRate",
            "GoalsFor_cumu": f"{prefix}GF_cumu",
            "GoalsAgainst_cumu": f"{prefix}GA_cumu",
            "Points_cumu": f"{prefix}Pts_cumu",
        }
    )

    return merged[[
        "MatchKey",
        f"{prefix}GF_avg",
        f"{prefix}GA_avg",
        f"{prefix}GD_avg",
        f"{prefix}Pts_avg",
        f"{prefix}WinRate",
        f"{prefix}GF_cumu",
        f"{prefix}GA_cumu",
        f"{prefix}Pts_cumu",
    ]]

def main():
    data_dir = os.environ.get("DATA_DIR", "data")
    artifacts_dir = Path(os.environ.get("ARTIFACTS_DIR", "artifacts"))
    rolling_window = int(os.environ.get("ROLLING_WINDOW", "5"))

    history_path = Path(data_dir) / "History.csv"
    fixtures_path = Path(data_dir) / "Fixtures.csv"

    if not history_path.exists():
        raise FileNotFoundError(
            f"Missing history file: {history_path}. It must include Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR."
        )
    if not fixtures_path.exists():
        raise FileNotFoundError(
            f"Missing fixtures file: {fixtures_path}. It must include Date, HomeTeam, AwayTeam."
        )

    history = pd.read_csv(history_path)
    fixtures = pd.read_csv(fixtures_path)

    fixtures = fixtures.copy()
    fixtures["Date"] = pd.to_datetime(fixtures["Date"], dayfirst=True, errors="coerce")
    fixtures = fixtures.dropna(subset=["Date"])
    fixtures["MatchKey"] = (
        fixtures["Date"].dt.strftime("%Y-%m-%d") + "|" + fixtures["HomeTeam"] + "|" + fixtures["AwayTeam"]
    )

    team_stats = compute_team_stats(history, rolling_window=rolling_window)

    home_features = merge_latest_stats(fixtures, team_stats, "Home")
    away_features = merge_latest_stats(fixtures, team_stats, "Away")

    feature_frame = (
        fixtures[["MatchKey", "Date", "HomeTeam", "AwayTeam"]]
        .merge(home_features, on="MatchKey", how="left")
        .merge(away_features, on="MatchKey", how="left")
    )

    model = joblib.load(artifacts_dir / "xgb_model.joblib")
    encoder = joblib.load(artifacts_dir / "label_encoder.joblib")

    unseen = set(feature_frame["HomeTeam"]).union(feature_frame["AwayTeam"]) - set(encoder.classes_)
    if unseen:
        raise ValueError(
            "Unseen teams in fixtures: " + ", ".join(sorted(unseen)) + ". Retrain or update the encoder."
        )

    feature_frame["HomeTeamEnc"] = encoder.transform(feature_frame["HomeTeam"])
    feature_frame["AwayTeamEnc"] = encoder.transform(feature_frame["AwayTeam"])

    exclude = ["MatchKey", "Date", "HomeTeam", "AwayTeam"]
    X = feature_frame.drop(columns=exclude)

    preds = model.predict(X)
    probs = model.predict_proba(X)

    label_map = {0: "H", 1: "D", 2: "A"}
    pred_labels = [label_map[p] for p in preds]

    output = feature_frame[["Date", "HomeTeam", "AwayTeam"]].copy()
    output["PredictedResult"] = pred_labels
    output["Prob_H"] = probs[:, 0]
    output["Prob_D"] = probs[:, 1]
    output["Prob_A"] = probs[:, 2]

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    output_path = artifacts_dir / "predictions.csv"
    output.to_csv(output_path, index=False)

    print(output)
    print(f"Saved predictions to {output_path}")

if __name__ == "__main__":
    main()