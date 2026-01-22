# FPL 2025-2026 ML Project - Implementation Plan

Goal
Build a data-to-predictions pipeline that uses historic FPL data to predict, per player and per position, expected points and key stats for the next 1, 5, 10, and 15 gameweeks. Output ranked lists by position with predicted points and expected goals, assists, defensive contribution, and price. Do not scale predictions by play probability; instead, flag uncertainty indicators so final judgment remains manual.

Scope Decisions (Resolved)
- Season scope: 2025-2026 only.
- Training window: all finished gameweeks. Current data shows finished matches through GW22; later GWs are unfinished and should be used only for future fixtures.
- Per-position models: separate models for Goalkeeper, Defender, Midfielder, Forward.
- Targets: expected points, expected goals, expected assists, expected defensive contribution for horizons 1/5/10/15 (totals and per-week averages).
- No leakage: features at GW t may use only information available up to the end of GW t.
- Do not multiply predictions by play probability. Provide uncertainty flags based on availability and recent minutes.

Data Inventory (2025-2026, By Gameweek)
Files per GW:
- teams.csv: team metadata (id, code, strength, elo).
- players.csv: player metadata (player_id, team_code, position).
- matches.csv: match results and team stats; finished flag indicates completed matches.
- fixtures.csv: same schema as matches.csv; use for upcoming fixtures when finished=False.
- player_gameweek_stats.csv: per-player per-GW stats (event_points, minutes, xG/xA, defensive_contribution, etc).
- playerstats.csv: duplicates player_gameweek_stats; use player_gameweek_stats as canonical.
- playermatchstats.csv: per-player per-match stats; useful for start/finish times and detailed actions.

Key Join Keys
- players.csv.player_id <-> player_gameweek_stats.id <-> playermatchstats.player_id
- players.csv.team_code <-> teams.csv.code
- matches.csv.home_team / away_team <-> teams.csv.id

Implementation Steps

Step 1: Project Scaffold
- Create `src/` with submodules for io, features, labels, models, evaluation, pipelines, reporting.
- Add `configs/` for model and feature settings (YAML).
- Add `outputs/` and `reports/` for predictions and charts.
- Dependencies: pandas, numpy, scikit-learn, scipy, matplotlib (or seaborn), pyyaml.
Acceptance criteria:
- New folder structure exists and imports resolve.
- Config files load without errors.

Step 2: Data Loading + Schema Validation
- Implement CSV loaders for each table across all GWs.
- Validate headers and required columns; fail fast if mismatched.
- Identify `max_finished_gw` by checking matches.csv where finished=True for all matches.
- Cache unified tables to `data/processed/` (optional but recommended).
Acceptance criteria:
- Loader returns consistent DataFrames for teams, players, matches, player_gameweek_stats, playermatchstats.
- max_finished_gw computed correctly.

Step 3: Canonical Player-GW Table
- Base table: player_gameweek_stats for finished GWs only.
- Join players (position, team_code) and teams (team strength, elo).
- Add team and opponent features for GW t from matches.csv:
  - team_id, opponent_id, home/away, team_elo, opponent_elo, team strength.
- Keep only rows where minutes > 0 for modeling (exclude DNPs from training rows).
Acceptance criteria:
- One row per player per GW t (for finished GWs).
- All rows include position and team identifiers.

Step 4: Feature Engineering
Player form features (computed using only past data up to GW t):
- Rolling means over last 3/5/10 played matches (minutes > 0):
  - minutes, event_points, xG, xA, shots, assists, goals, defensive_contribution, saves.
- Per-90 rates for key stats.
- Trend features: last_3 minus last_10 for points/xG/xA.
Team context features:
- Rolling team xG/xGA, goals, clean sheets, shots (last 3/5/10).
- Elo delta: team_elo - opponent_elo.
Fixture features:
- Home/away flag.
- Next-N fixture difficulty: average opponent strength and elo for horizons 1/5/10/15.
Uncertainty indicators (do not scale predictions):
- status, chance_of_playing_next_round.
- missed_last_5 (count of DNPs in last 5 GWs).
- starts_last_5, avg_minutes_last_5.
- sub_early_rate (finish_min < 70) and sub_on_rate (start_min > 0 and start_min > 1) derived from playermatchstats aggregated to GW.
Acceptance criteria:
- Feature table contains rolling features with no forward leakage.
- Uncertainty flags are present for every player-GW row.

Step 5: Label Construction (Targets)
For each player-GW row at time t:
- Horizon 1: sum of event_points/xG/xA/defensive_contribution at GW t+1.
- Horizon 5/10/15: sum over GWs t+1..t+N.
- Also compute per-week averages (sum / N) for each horizon.
Only include rows where t+N <= max_finished_gw to avoid leakage.
Acceptance criteria:
- Label columns exist for each target and horizon.
- No labels computed using future beyond finished data.

Step 6: Train/Validation Splits
- Time-based backtesting:
  - Rolling splits: train on GWs 1..k, validate on k+1..k+H for H in {1,5,10,15}.
  - Final holdout: last finished 3-5 GWs (if enough data).
Acceptance criteria:
- Split indices are time-ordered and non-overlapping.

Step 7: Modeling
- Train separate RandomForestRegressor models per position, horizon, and target:
  - Targets: points, expected_goals, expected_assists, defensive_contribution.
  - Horizons: 1, 5, 10, 15.
- Use consistent feature set across models; allow target-specific tweaks if needed.
- Hyperparameters from config; set random_state for reproducibility.
Acceptance criteria:
- Models train without errors for all positions and horizons.
- Feature importances produced for interpretability.

Step 8: Evaluation + Diagnostics
- Metrics: R2, MAE, RMSE per model.
- Ranking evaluation: Spearman correlation of predicted vs actual points.
- Plots: predicted vs actual, residuals, feature importance, error by position.
Acceptance criteria:
- Metrics and charts saved to `reports/`.

Step 9: Inference Pipeline
- Build latest feature snapshot at max_finished_gw.
- Generate next-1/5/10/15 predictions per player and target.
- Add uncertainty flags (status, missed_last_5, avg_minutes_last_5, sub_early_rate).
Acceptance criteria:
- Predictions run end-to-end using only finished data + upcoming fixtures.

Step 10: Output Tables
Generate CSVs per position:
- Columns:
  - player_id, web_name, team, position, price
  - expected_points_next_gw
  - expected_points_5/10/15 (total + per-week)
  - expected_xg_1/5/10/15 (total + per-week)
  - expected_xa_1/5/10/15 (total + per-week)
  - expected_defcon_1/5/10/15 (total + per-week)
  - uncertainty flags
Acceptance criteria:
- One output per position in `outputs/predictions/`.
- Columns match agreed schema.

Step 11: QA and Tests
- Unit tests for:
  - join keys
  - rolling features (no future leakage)
  - label windows
  - fixture difficulty aggregation
- Data sanity checks:
  - counts per GW
  - no NaNs in required columns (after imputation)
Acceptance criteria:
- Tests pass locally.

Step 12: Documentation
- Usage instructions in README:
  - data prerequisites
  - training command
  - inference command
  - where outputs are stored
Acceptance criteria:
- README includes a minimal quickstart.

Risks and Mitigations
- Incomplete future GWs: use only finished GWs for labels; treat future fixtures as inference only.
- DNP bias: exclude DNP rows from training but flag uncertainty for predictions.
- Double gameweeks: allow multiple matches per GW via match aggregation in playermatchstats.
- Data drift: rerun pipeline when new GWs are finished.

Expected Deliverables
- `ARCHITECTURE.md`
- `IMPLEMENTATION_PLAN.md` (this file)
- Project scaffold in `src/`, `configs/`, `outputs/`, `reports/`
- Trained models and prediction CSVs
