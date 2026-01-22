# FPL 2025-2026 ML Project - Architecture

Overview
This project builds per-position RandomForest models to predict FPL player performance over upcoming horizons (1, 5, 10, 15 GWs). It ingests finished historical data, generates leakage-free features and labels, trains models, evaluates them, and outputs ranked prediction tables. Predictions are not scaled by play probability; instead, availability uncertainty flags are produced for manual decision-making.

High-Level Data Flow
1) Raw CSVs (per GW) -> 2) Unified tables -> 3) Features + Labels -> 4) Train/Eval -> 5) Predictions + Reports

Directory Layout
```
.
├── data/
│   ├── 2025-2026/
│   │   └── By Gameweek/
│   └── processed/
├── src/
│   ├── io/
│   ├── features/
│   ├── labels/
│   ├── models/
│   ├── evaluation/
│   ├── pipelines/
│   └── reporting/
├── configs/
│   ├── features.yaml
│   └── models.yaml
├── outputs/
│   └── predictions/
├── reports/
│   └── figures/
├── ARCHITECTURE.md
└── IMPLEMENTATION_PLAN.md
```

Core Components

1) IO Layer (`src/io/`)
Purpose: load raw CSVs, validate schemas, and construct unified tables.
Key modules:
- `loaders.py`: read all GWs for each table, return concatenated DataFrames.
- `schema.py`: assert required columns and dtypes.
- `index.py`: compute `max_finished_gw` from matches.csv finished flags.

2) Feature Engineering (`src/features/`)
Purpose: create player, team, and fixture features with no leakage.
Key modules:
- `player_features.py`: rolling player stats (3/5/10) using only past matches where minutes > 0.
- `team_features.py`: rolling team stats (xG/xGA/goals/clean sheets) from matches.
- `fixture_features.py`: upcoming fixture difficulty (home/away, opponent elo, strength).
- `availability_flags.py`: uncertainty indicators (status, missed_last_5, avg_minutes_last_5, sub_early_rate).

3) Label Builder (`src/labels/`)
Purpose: build horizon targets for next 1/5/10/15 GWs.
Key modules:
- `targets.py`: create horizon sums and per-week averages for:
  - points (event_points)
  - expected goals (expected_goals)
  - expected assists (expected_assists)
  - defensive contribution (defensive_contribution)
Rules:
- Only label rows where t+N <= max_finished_gw.
- No leakage from future beyond finished GWs.

4) Models (`src/models/`)
Purpose: train and run per-position, per-horizon RandomForest models.
Key modules:
- `train.py`: loop over positions, horizons, targets; train RandomForestRegressor.
- `predict.py`: load trained models and produce predictions for latest GW snapshot.
- `registry.py`: consistent naming conventions for saved models.

5) Evaluation (`src/evaluation/`)
Purpose: quantify model performance and create charts.
Key modules:
- `metrics.py`: R2, MAE, RMSE, Spearman rank.
- `plots.py`: residuals, predicted vs actual, feature importances.

6) Pipelines (`src/pipelines/`)
Purpose: orchestrate end-to-end runs.
Key modules:
- `build_dataset.py`: create feature matrix + labels.
- `train_models.py`: train and evaluate.
- `predict_next.py`: generate predictions for upcoming fixtures.

7) Reporting (`src/reporting/`)
Purpose: format and export outputs.
Key modules:
- `export_predictions.py`: per-position CSVs with required columns.
- `export_summaries.py`: top-N tables and aggregated insights.

Data Model (Key Tables)

teams.csv
- code, id, name, strength, strength_overall_home/away, strength_attack_home/away, strength_defence_home/away, elo

players.csv
- player_id, web_name, team_code, position

matches.csv / fixtures.csv
- gameweek, kickoff_time, home_team, away_team, home_team_elo, away_team_elo, finished, match_id, team stats (xG, shots, possession, etc)

player_gameweek_stats.csv (canonical)
- id, gw, minutes, event_points, total_points, expected_goals, expected_assists, defensive_contribution, influence, creativity, threat, ict_index, starts, bonus, bps, etc.

playermatchstats.csv
- player_id, match_id, minutes_played, xg, xa, start_min, finish_min, and detailed action stats

Key Joins
- player_gameweek_stats.id -> players.player_id
- players.team_code -> teams.code
- matches.home_team/away_team -> teams.id
- playermatchstats.player_id -> players.player_id

Feature Construction Details
- Rolling windows: compute over last 3/5/10 played matches (minutes > 0).
- Per-90 stats: normalize by minutes to reduce sub effects.
- Trend signals: last_3 - last_10 for points/xG/xA.
- Fixture difficulty: average opponent elo and strength for next N fixtures.
- Home/away: binary flag and counts over next N fixtures.

Uncertainty Flags (No Scaling)
Output flags to surface risk without altering predictions:
- `status` not equal to "a".
- `chance_of_playing_next_round` < 100 (if present).
- `missed_last_5` > 0.
- `avg_minutes_last_5` < 60.
- `sub_early_rate` > 0.4 (finish_min < 70).
- `sub_on_rate` > 0.4 (start_min > 1 for many appearances).

Modeling Strategy
- Per position, per target, per horizon RandomForestRegressor.
- Feature set shared across targets to keep outputs consistent.
- Hyperparameters in `configs/models.yaml`, including n_estimators, max_depth, min_samples_leaf, max_features, random_state.
- Save models to `outputs/models/` with a deterministic naming scheme.

Evaluation Strategy
- Rolling time splits (train on early GWs, validate on later GWs).
- Report metrics per position/target/horizon.
- Visual diagnostics to detect bias or instability.

Inference Strategy
- Use latest finished GW as the feature cutoff.
- Generate predictions for all players, including those with DNP history.
- Provide predictions and uncertainty flags per player, per horizon.

Output Schema (Predictions)
Per position CSV:
- player_id, web_name, team, position, price
- expected_points_next_gw
- expected_points_5_total, expected_points_5_per_week
- expected_points_10_total, expected_points_10_per_week
- expected_points_15_total, expected_points_15_per_week
- expected_xg_1/5/10/15 totals + per-week
- expected_xa_1/5/10/15 totals + per-week
- expected_defcon_1/5/10/15 totals + per-week
- uncertainty flags (status, missed_last_5, avg_minutes_last_5, sub_early_rate)

Reproducibility
- Fixed random seeds in configs.
- Cached processed data with a run timestamp.
- Model artifacts versioned by season and GW cutoff.

Extensibility
- Add alternative models (XGBoost, LightGBM) under `src/models/`.
- Add new seasons by pointing to a different data root.
- Add transfer strategy optimization as a downstream module.
