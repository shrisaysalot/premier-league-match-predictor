# Premier League Match Predictor

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/shrisaysalot/premier-league-match-predictor.git
   cd premier-league-match-predictor
   ```
2. Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Place your CSV files in the `data/` directory:
   - `Train1.csv`
   - `Train2.csv`
   - `Train3.csv`
   - `Train4.csv`
   - `Train5.csv`
   - `Test.csv`

5. Run the training script:
   ```bash
   python src/train.py
   ```

## Run Instructions

### Train

```bash
python src/train.py
```

### Predict (inference)

Place the following files in `data/`:

- `History.csv` (past matches with `Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR`)
- `Fixtures.csv` (upcoming matches with `Date, HomeTeam, AwayTeam`)

Then run:

```bash
python src/predict.py
```

Predictions will be saved to:

```
artifacts/predictions.csv
```
