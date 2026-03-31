import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from features import build_features

def load_csvs(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_path = Path(data_dir)

    train_files = [data_path / f"Train{i}.csv" for i in range(1, 6)]
    missing = [str(p) for p in train_files if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing training files: " + ", ".join(missing)
        )

    df_train = pd.concat([pd.read_csv(p) for p in train_files], ignore_index=True)

    test_path = data_path / "Test.csv"
    if not test_path.exists():
        raise FileNotFoundError(f"Missing validation file: {test_path}")

    df_val = pd.read_csv(test_path)
    return df_train, df_val

def add_match_key(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"])
    df["MatchKey"] = (
        df["Date"].dt.strftime("%Y-%m-%d") + "|" + df["HomeTeam"] + "|" + df["AwayTeam"]
    )
    return df

def prepare_features(df_train: pd.DataFrame, df_val: pd.DataFrame, rolling_window: int):
    df_train = add_match_key(df_train)
    df_val = add_match_key(df_val)

    train_features = build_features(df_train, rolling_window=rolling_window)

    combined = pd.concat([df_train, df_val], ignore_index=True)
    combined_features = build_features(combined, rolling_window=rolling_window)

    val_keys = set(df_val["MatchKey"].unique())
    val_features = combined_features[combined_features["MatchKey"].isin(val_keys)].copy()

    return train_features, val_features

def train_xgb(train_features: pd.DataFrame, val_features: pd.DataFrame):
    le = LabelEncoder()
    all_teams = pd.concat(
        [
            train_features["HomeTeam"],
            train_features["AwayTeam"],
            val_features["HomeTeam"],
            val_features["AwayTeam"],
        ]
    )
    le.fit(all_teams)

    for df in [train_features, val_features]:
        df["HomeTeamEnc"] = le.transform(df["HomeTeam"])
        df["AwayTeamEnc"] = le.transform(df["AwayTeam"])

    target_map = {"H": 0, "D": 1, "A": 2}
    train_features["target"] = train_features["FTR"].map(target_map)
    val_features["target"] = val_features["FTR"].map(target_map)

    exclude = ["Date", "HomeTeam", "AwayTeam", "FTR", "target", "MatchKey"]
    X_train = train_features.drop(columns=exclude)
    y_train = train_features["target"]
    X_val = val_features.drop(columns=exclude)
    y_val = val_features["target"]

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        learning_rate=0.05,
        max_depth=6,
        n_estimators=500,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        reg_alpha=0.3,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=30,
        verbose=False,
    )

    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)

    print("Val Accuracy:", accuracy_score(y_val, y_val_pred))
    print("Val Log Loss:", log_loss(y_val, y_val_proba))
    print("Confusion Matrix:\n", confusion_matrix(y_val, y_val_pred))

    return model, le

def main():
    data_dir = os.environ.get("DATA_DIR", "data")
    artifacts_dir = Path(os.environ.get("ARTIFACTS_DIR", "artifacts"))
    rolling_window = int(os.environ.get("ROLLING_WINDOW", "5"))

    df_train, df_val = load_csvs(data_dir)
    train_features, val_features = prepare_features(
        df_train, df_val, rolling_window=rolling_window
    )

    model, label_encoder = train_xgb(train_features, val_features)

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifacts_dir / "xgb_model.joblib")
    joblib.dump(label_encoder, artifacts_dir / "label_encoder.joblib")

    print(f"Saved model to {artifacts_dir / 'xgb_model.joblib'}")
    print(f"Saved label encoder to {artifacts_dir / 'label_encoder.joblib'}")

if __name__ == "__main__":
    main()