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

- To train the model, ensure your data files are in the `data/` directory and execute:
  ```bash
  python src/train.py
  ```