import pandas as pd


def build_features(df: pd.DataFrame, rolling_window: int = 5) -> pd.DataFrame:
    """
    Create rolling and cumulative pre-match features for each match.

    Notes:
    - Uses shifted rolling/expanding windows to prevent leakage.
    - Adds MatchKey to enable correct validation filtering by match identity.
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")

    def get_points(hg, ag):
        if hg > ag:
            return (3, 0)
        if hg < ag:
            return (0, 3)
        return (1, 1)

    df["HomePoints"], df["AwayPoints"] = zip(
        *df.apply(lambda r: get_points(r.FTHG, r.FTAG), axis=1)
    )

    df["MatchKey"] = (
        df["Date"].dt.strftime("%Y-%m-%d") + "|" + df["HomeTeam"] + "|" + df["AwayTeam"]
    )

    home_stats = df[["Date", "HomeTeam", "FTHG", "FTAG", "HomePoints"]].copy()
    home_stats.rename(
        columns={
            "HomeTeam": "Team",
            "FTHG": "GoalsFor",
            "FTAG": "GoalsAgainst",
            "HomePoints": "Points",
        },
        inplace=True,
    )

    away_stats = df[["Date", "AwayTeam", "FTAG", "FTHG", "AwayPoints"]].copy()
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

    team_stats["DateKey"] = team_stats["Date"].dt.strftime("%Y-%m-%d")

    features = (
        df[["Date", "HomeTeam", "AwayTeam", "FTR", "MatchKey"]]
        .assign(DateKey=df["Date"].dt.strftime("%Y-%m-%d"))
        .merge(
            team_stats[
                [
                    "Team",
                    "DateKey",
                    "GoalsFor_rolling",
                    "GoalsAgainst_rolling",
                    "GoalDiff_rolling",
                    "Points_rolling",
                    "Win_rolling",
                    "GoalsFor_cumu",
                    "GoalsAgainst_cumu",
                    "Points_cumu",
                ]
            ],
            left_on=["HomeTeam", "DateKey"],
            right_on=["Team", "DateKey"],
            how="left",
        )
        .rename(
            columns={
                "GoalsFor_rolling": "HomeGF_avg",
                "GoalsAgainst_rolling": "HomeGA_avg",
                "GoalDiff_rolling": "HomeGD_avg",
                "Points_rolling": "HomePts_avg",
                "Win_rolling": "HomeWinRate",
                "GoalsFor_cumu": "HomeGF_cumu",
                "GoalsAgainst_cumu": "HomeGA_cumu",
                "Points_cumu": "HomePts_cumu",
            }
        )
        .drop(columns=["Team"])
        .merge(
            team_stats[
                [
                    "Team",
                    "DateKey",
                    "GoalsFor_rolling",
                    "GoalsAgainst_rolling",
                    "GoalDiff_rolling",
                    "Points_rolling",
                    "Win_rolling",
                    "GoalsFor_cumu",
                    "GoalsAgainst_cumu",
                    "Points_cumu",
                ]
            ],
            left_on=["AwayTeam", "DateKey"],
            right_on=["Team", "DateKey"],
            how="left",
            suffixes=("", "_away"),
        )
        .rename(
            columns={
                "GoalsFor_rolling": "AwayGF_avg",
                "GoalsAgainst_rolling": "AwayGA_avg",
                "GoalDiff_rolling": "AwayGD_avg",
                "Points_rolling": "AwayPts_avg",
                "Win_rolling": "AwayWinRate",
                "GoalsFor_cumu": "AwayGF_cumu",
                "GoalsAgainst_cumu": "AwayGA_cumu",
                "Points_cumu": "AwayPts_cumu",
            }
        )
        .drop(columns=["Team", "DateKey", "DateKey_away"])
    )

    return features
