# Architecture — NBA Over/Under Prediction Engine

**Version:** 2.0 (ML Ensemble)  
**Last Updated:** February 2026

---

## System Overview

The NBA Predictor is a fully automated, serverless sports analytics platform that generates daily over/under predictions for NBA player props. It combines a **Gradient Boosted ML model** with a **weighted rolling average baseline** in a 60/40 ensemble blend, pulls live betting lines from The Odds API, and delivers picks via a real-time dashboard and Discord notifications.

The entire system runs autonomously on AWS with zero manual intervention — data ingestion, stat processing, model training, prediction generation, result verification, and accuracy tracking all happen on automated schedules.

---

## Architecture Diagram

```
                     ┌─────────────────────────────────────────┐
                     │           EXTERNAL DATA SOURCES          │
                     │                                          │
                     │  ┌──────────────┐   ┌────────────────┐  │
                     │  │ balldontlie  │   │  The Odds API  │  │
                     │  │ (Box Scores, │   │  (Live Player  │  │
                     │  │  Advanced    │   │   Prop Lines)  │  │
                     │  │  Stats)      │   │                │  │
                     │  └──────┬───────┘   └───────┬────────┘  │
                     └─────────┼───────────────────┼───────────┘
                               │                   │
          ┌────────────────────┼───────────────────┼────────────────────┐
          │  AWS CLOUD         │                   │                    │
          │                    ▼                   │                    │
          │  ┌──────────────────────┐              │                    │
          │  │  ① FETCH DATA λ     │──────────┐   │                    │
          │  │  Daily 10 AM UTC     │          │   │                    │
          │  │  Box scores + adv.   │          ▼   │                    │
          │  │  stats + injuries    │   ┌──────────┴──┐                │
          │  └──────────┬───────────┘   │  S3 Bucket   │                │
          │             │               │  (Raw JSON,  │                │
          │             │               │   ML Models)  │                │
          │             ▼               └──────────────┘                │
          │  ┌──────────────────────┐          ▲                       │
          │  │  ② PROCESS STATS λ  │          │                       │
          │  │  Daily 11 AM UTC     │          │                       │
          │  │  Rolling averages    │          │                       │
          │  │  Trends + consistency│          │                       │
          │  └──────────┬───────────┘          │                       │
          │             │                      │                       │
          │             ▼                      │                       │
          │  ┌────────────────────┐            │                       │
          │  │     DynamoDB       │            │                       │
          │  │                    │            │                       │
          │  │  Stats Table       │◄───────────┤                       │
          │  │  Predictions Table │            │                       │
          │  │  Results Table     │            │                       │
          │  └─────┬──────┬───────┘            │                       │
          │        │      │                    │                       │
          │        │      ▼                    │                       │
          │        │  ┌──────────────────────┐ │                       │
          │        │  │  ③ PREDICT λ         │─┘                       │
          │        │  │  Daily 12 PM UTC     │◄── Loads ML models      │
          │        │  │  Ensemble:           │    from S3               │
          │        │  │  60% ML + 40% WA     │                         │
          │        │  │  + Live odds lines   │                         │
          │        │  └──────────┬───────────┘                         │
          │        │             │                                      │
          │        │             ▼                                      │
          │        │  ┌──────────────────────┐                         │
          │        │  │  ④ DISCORD NOTIFY λ  │──► Discord Webhook      │
          │        │  │  Daily 12:15 PM UTC  │    (Top 15 picks)       │
          │        │  └──────────────────────┘                         │
          │        │                                                    │
          │        │  ┌──────────────────────┐                         │
          │        │  │  ⑤ VERIFY RESULTS λ  │                         │
          │        │  │  Daily 2 PM UTC      │                         │
          │        │  │  Checks yesterday's  │                         │
          │        │  │  predictions vs real  │                         │
          │        │  └──────────────────────┘                         │
          │        │                                                    │
          │        │  ┌──────────────────────┐                         │
          │        │  │  ⑥ TRAIN MODEL λ     │                         │
          │        │  │  Weekly (Sunday)      │──► Saves .pkl models    │
          │        │  │  Gradient Boosted     │    to S3                │
          │        │  │  Regressor × 4 stats  │                         │
          │        │  └──────────────────────┘                         │
          │        │                                                    │
          │        ▼                                                    │
          │  ┌──────────────────────┐                                  │
          │  │  API Gateway (REST)  │   Rate limited: 5 req/sec        │
          │  │  GET only            │                                  │
          │  └──────────┬───────────┘                                  │
          │             │                                              │
          │             ▼                                              │
          │  ┌──────────────────────┐                                  │
          │  │  ⑦ API λ            │   Input validation               │
          │  │  Read-only queries   │   Sanitized error responses      │
          │  └──────────┬───────────┘                                  │
          │             │                                              │
          └─────────────┼──────────────────────────────────────────────┘
                        │
                        ▼
               ┌──────────────────┐
               │  S3 Static Site  │   Frontend Dashboard
               │  (Public HTML)   │   Dark-themed, real-time
               └──────────────────┘
```

---

## Data Pipeline

### Schedule (all times UTC)

| Time | Function | Duration | What It Does |
|------|----------|----------|--------------|
| 10:00 AM | **FetchData** | ~60s | Pulls box scores, advanced stats (pace, usage%, defensive rating), injuries from balldontlie API. Stores raw JSON in S3, structured records in DynamoDB. |
| 11:00 AM | **ProcessStats** | ~30s | Scans players with recent games. Computes rolling averages (5/10/20 game windows), trend analysis, consistency scores (coefficient of variation). |
| 12:00 PM | **Predict** | ~35s | Loads ML models from S3. Builds 26-feature vectors per player. Blends ML + weighted average predictions. Pulls live betting lines from The Odds API. Generates confidence scores and OVER/UNDER recommendations. |
| 12:15 PM | **DiscordNotify** | ~5s | Sends top 15 picks to Discord channel with confidence levels and accuracy stats. |
| 2:00 PM | **VerifyResults** | ~15s | Compares yesterday's predictions against actual game results. Stores correct/incorrect outcomes for accuracy tracking. |
| Sunday 8 AM | **TrainModel** | ~100s | Retrains 4 Gradient Boosted Regressors on accumulated historical data. Saves serialized models (.pkl) to S3. |

---

## ML Ensemble Model

### Architecture

```
Player Game Logs + Context  →  26-Feature Vector  →  GradientBoostingRegressor
                                                            │
                                                     ML Prediction (60%)
                                                            │
Rolling Averages (5/10/20) →  Weighted Average  ──► WA Prediction (40%)
                                                            │
                                                     ┌──────┴──────┐
                                                     │   ENSEMBLE   │
                                                     │  Final Pred  │
                                                     └─────────────┘
```

### 26-Feature Vector

| # | Feature | Source |
|---|---------|--------|
| 1-4 | Points: rolling avg (3/5/10 games) + trend | Player game logs |
| 5-8 | Rebounds: rolling avg (3/5/10 games) + trend | Player game logs |
| 9-12 | Assists: rolling avg (3/5/10 games) + trend | Player game logs |
| 13-16 | 3-Pointers: rolling avg (3/5/10 games) + trend | Player game logs |
| 17 | Average minutes (last 10) | Player game logs |
| 18 | Usage rate % (last 10) | Advanced stats |
| 19 | True shooting % (last 10) | Advanced stats |
| 20 | Games available (capped at 50) | Player game logs |
| 21 | Home/away indicator (1.0 / 0.0) | Game schedule |
| 22 | Rest days since last game (capped at 7) | Game dates |
| 23 | Opponent defensive rating | Team aggregated stats |
| 24 | Opponent pace | Team aggregated stats |
| 25 | Opponent points allowed average | Team aggregated stats |
| 26 | Team injury count (players ruled out) | Injury report |

### Model Configuration

```python
GradientBoostingRegressor(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    min_samples_split=10,
    min_samples_leaf=5,
    subsample=0.8,
    random_state=42,
)
```

### Separate Models Per Stat

Four independent models are trained:

| Model | Target | Rationale |
|-------|--------|-----------|
| `pts_model.pkl` | Points scored | Driven by usage rate, pace, minutes |
| `reb_model.pkl` | Rebounds | Driven by position, opponent rebounding |
| `ast_model.pkl` | Assists | Driven by usage rate, team pace |
| `fg3m_model.pkl` | 3-pointers made | Driven by shooting %, attempts |

PRA (Points + Rebounds + Assists) uses the weighted average only — no dedicated model.

### Ensemble Blending

```
Final Prediction = (0.6 × ML Prediction) + (0.4 × Weighted Average)
```

**Rationale:** The ML model captures contextual intelligence (opponent strength, rest, injuries, trends) while the weighted average provides stability against overfitting. If the ML model is unavailable for a player (insufficient game history), it falls back to weighted average only.

---

## DynamoDB Schema

### Single-Table Design

All data uses a composite primary key (`pk` + `sk`) with one Global Secondary Index (GSI1).

| Entity | PK | SK | GSI1PK | GSI1SK |
|--------|----|----|--------|--------|
| Game | `GAME#{id}` | `DATE#{date}` | `DATE#{date}` | `GAME#{id}` |
| Player Game | `PLAYER#{id}` | `GAME#{gameId}#{date}` | `DATE#{date}` | `PLAYER#{id}` |
| Processed Stats | `PLAYER#{id}` | `PROCESSED#{date}` | `PROCESSED#{date}` | `PLAYER#{id}` |
| Team Profile | `TEAM#{id}` | `PROFILE` | — | — |
| Injury | `INJURY#{playerId}` | `DATE#{date}` | `INJURIES#{date}` | `PLAYER#{id}` |
| Prediction | `DATE#{date}` | `PLAYER_PROP#{id}` | — | — |
| Result | `DATE#{date}` | `RESULT#{playerId}_{stat}` | — | — |

### Access Patterns

| Query | Key Condition |
|-------|--------------|
| Get player's recent games | `pk = PLAYER#{id}`, `sk begins_with GAME#` |
| Get all games on a date | `GSI1: gsi1pk = DATE#{date}`, `gsi1sk begins_with GAME#` |
| Get all players who played on a date | `GSI1: gsi1pk = DATE#{date}`, `gsi1sk begins_with PLAYER#` |
| Get today's predictions | `pk = DATE#{today}` |
| Get yesterday's results | `pk = DATE#{yesterday}` |
| Get current injuries | `GSI1: gsi1pk = INJURIES#{today}` |
| Get latest processed stats | `pk = PLAYER#{id}`, `sk begins_with PROCESSED#`, Limit=1, DESC |

---

## API Design

### Endpoints

| Method | Path | Lambda | Description |
|--------|------|--------|-------------|
| `GET` | `/predictions` | API | Today's predictions (player props + team totals) |
| `GET` | `/predictions/{date}` | API | Predictions for a specific date |
| `GET` | `/players/{playerId}/stats` | API | Player game logs + processed analytics |
| `GET` | `/accuracy` | API | Rolling 30-day accuracy breakdown |
| `GET` | `/teams` | API | All NBA teams |

### Input Validation

All path and query parameters are validated with regex before use:

```python
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")      # YYYY-MM-DD only
PLAYER_ID_PATTERN = re.compile(r"^\d{1,10}$")            # Numeric, max 10 digits
```

Invalid input returns `400 Bad Request`. Server errors return a generic `500 Internal Server Error` — no internal details are leaked to the client.

---

## Security Design

### Threat Model

| Threat | Mitigation |
|--------|------------|
| **API abuse / DDoS** | API Gateway stage-level rate limiting (5 req/sec, burst 10) |
| **Data tampering** | API is read-only (GET only). All writes happen via scheduled Lambdas with no public trigger. |
| **Secret exposure** | All API keys stored in Lambda environment variables (encrypted at rest by AWS). No secrets in source code. `.gitignore` excludes `samconfig.toml`, `.env`. |
| **Injection attacks** | Input validation on all API parameters. DynamoDB key-condition queries are not susceptible to injection. |
| **Error information leakage** | API returns generic error messages. Full stack traces logged server-side only. |
| **CORS abuse** | CORS restricted to GET and OPTIONS methods only. |
| **Credential theft in CI/CD** | GitHub Actions uses OIDC federation (`id-token: write`) — no static AWS access keys stored anywhere. |
| **S3 data exposure** | Raw data bucket has all 4 public access blocks enabled. Only Lambda IAM roles can read/write. |
| **Unencrypted data** | S3 server-side encryption (AES-256). API Gateway enforces TLS for all API traffic. |
| **Database loss** | DynamoDB Point-in-Time Recovery enabled on all 3 tables (35 days of continuous backups). |

### IAM Principle of Least Privilege

Each Lambda function has a scoped IAM policy generated by SAM:

| Function | Permissions |
|----------|------------|
| FetchData | DynamoDB CRUD (Stats), S3 CRUD (Raw Bucket) |
| ProcessStats | DynamoDB CRUD (Stats), S3 Read (Raw Bucket) |
| Predict | DynamoDB CRUD (Stats, Predictions), S3 Read (Models) |
| TrainModel | DynamoDB Read (Stats), S3 Write (Models) |
| VerifyResults | DynamoDB Read (Stats, Predictions), DynamoDB Write (Results) |
| DiscordNotify | DynamoDB Read (Predictions, Results) |
| **API** | **DynamoDB Read-Only** (Predictions, Stats, Results) |

The API function — the only publicly accessible function — has **read-only** database access.

### Secrets Management

| Secret | Storage | Rotation |
|--------|---------|----------|
| Odds API key | Lambda environment variable | Manual (regenerated after exposure) |
| balldontlie API key | Lambda environment variable | Manual |
| Discord webhook URL | Lambda environment variable | Manual |
| AWS credentials (CI/CD) | GitHub OIDC federation | Automatic (temporary STS tokens) |

No secrets are stored in source code, configuration files, or the frontend.

---

## Frontend

### Stack

Single-file HTML/CSS/JS dashboard with no build step and no framework dependencies:

- **Fonts:** Google Fonts (Outfit + JetBrains Mono)
- **Styling:** Custom CSS variables, dark theme
- **Data:** Fetches from API Gateway on page load, falls back to demo data if API unavailable

### Features

- Filter by stat type, confidence level, or prediction method (ML Ensemble vs Weighted Avg)
- Player detail modal showing ML prediction, WA prediction, and blended result
- Accuracy tab with ring chart, confidence breakdown, and per-stat breakdown
- "How It Works" tab explaining the ML ensemble methodology

---

## Infrastructure as Code

The entire infrastructure is defined in a single `template.yaml` (AWS SAM / CloudFormation):

- 7 Lambda functions
- 3 DynamoDB tables
- 2 S3 buckets
- 1 API Gateway (REST)
- 6 EventBridge schedules
- 1 SNS topic
- IAM roles auto-generated by SAM

### Deployment

```bash
sam build && sam deploy
```

CI/CD via GitHub Actions: push to `main` triggers automatic build and deploy using OIDC federation.

---

## Monitoring

- **CloudWatch Logs:** All Lambda functions log to CloudWatch with structured JSON output including prediction summaries, ML model metrics, and error traces.
- **Prediction Summary:** Each daily run logs counts of ML-enhanced vs weighted-average-only predictions, models loaded, games processed.
- **Training Metrics:** Weekly training logs cross-validation MAE, feature importances, and sample counts per model.
- **Accuracy Tracking:** VerifyResults stores correct/incorrect outcomes daily. Dashboard displays rolling 30-day accuracy by confidence level and stat type.

---

## Cost

All services operate within AWS Free Tier for personal use:

| Service | Monthly Usage | Free Tier Limit |
|---------|--------------|-----------------|
| Lambda | ~1,000 invocations, ~500s compute | 1M invocations, 400K GB-s |
| DynamoDB | ~50K read/write units | 25 RCU/WCU always free |
| S3 | <1 GB storage | 5 GB |
| API Gateway | <1,000 requests | 1M requests |
| EventBridge | 7 scheduled rules | Free |
| CloudWatch | ~500 MB logs | 5 GB |

**Estimated monthly cost: $0.00**

---

## File Structure

```
nba-betting-predictor/
├── template.yaml                    # SAM template — all AWS resources
├── SECURITY.md                      # Security audit documentation
├── ARCHITECTURE.md                  # This file
├── README.md                        # Getting started guide
├── lambdas/
│   ├── fetch_data/handler.py        # ① Data ingestion (balldontlie API)
│   ├── process_stats/handler.py     # ② Rolling averages, trends, consistency
│   ├── predict/handler.py           # ③ Ensemble prediction engine
│   ├── discord_notify/handler.py    # ④ Discord webhook notifications
│   ├── verify_results/handler.py    # ⑤ Accuracy verification
│   ├── train_model/handler.py       # ⑥ ML model training (weekly)
│   └── api/handler.py               # ⑦ REST API (read-only)
├── frontend/
│   └── index.html                   # Dashboard (single-file, no build step)
├── tests/
│   └── test_predictions.py          # Unit tests for prediction logic
├── scripts/
│   ├── deploy.ps1                   # Windows backend deploy
│   ├── deploy.sh                    # Linux/Mac backend deploy
│   ├── deploy-frontend.ps1          # Windows frontend deploy
│   ├── deploy-frontend.sh           # Linux/Mac frontend deploy
│   └── build_layer.py               # sklearn Lambda layer builder
├── .github/workflows/
│   └── deploy.yml                   # CI/CD pipeline (OIDC auth)
└── .gitignore                       # Excludes samconfig.toml, .env, etc.
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Separate models per stat** | Points, rebounds, assists have different drivers. A single model would underfit each target. |
| **60/40 ensemble blend** | ML captures context (opponent, rest days). WA provides stability. The blend outperforms either alone. |
| **Module-level model caching** | Lambda containers persist across invocations. Loading models once per container saves 2-3s per warm start. |
| **Single-table DynamoDB** | Minimizes costs. Composite keys and GSIs support all access patterns without table scans. |
| **No framework frontend** | Zero build step, zero dependencies. Deploys instantly to S3. One file to maintain. |
| **Read-only public API** | All data mutations happen via scheduled Lambdas. The public surface has no write capability. |
| **OIDC for CI/CD** | No long-lived AWS credentials in GitHub. STS issues temporary tokens per deployment. |
| **No data expiration** | Historical data improves ML accuracy over time. S3 and DynamoDB records persist indefinitely. |
