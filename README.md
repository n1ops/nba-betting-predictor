# 🏀 NBA Over/Under Prediction Engine

A **fully automated, serverless NBA analytics platform** that predicts player prop over/unders using an **ML ensemble model** — blending a Gradient Boosted Regressor with weighted rolling averages. Pulls live betting lines, delivers picks via dashboard and Discord, and tracks accuracy over time.

> The entire system runs autonomously on AWS — data ingestion, stat processing, ML training, prediction generation, result verification, and Discord notifications — all on automated schedules, for $0/month.

---

## How the Prediction Model Works

The engine uses two methods to predict how a player will perform, then combines them for the best result.

### 1. Weighted Rolling Average (the stable baseline)

This looks at what a player has actually done recently and averages it. If LeBron scored 25, 30, 28, 22, and 35 in his last 5 games, his average is 28 points. We compute this for his last 5, 10, and 20 games, then blend them — weighting recent games more heavily because last week matters more than last month:

```
Weighted Average = (Last 5 avg × 45%) + (Last 10 avg × 30%) + (Last 20 avg × 25%)
```

This is reliable and stable, but it's also simple. It doesn't know the player is facing a terrible defensive team, or that he's on a back-to-back, or that a teammate is injured.

### 2. Gradient Boosted ML Model (the contextual intelligence)

This is a machine learning algorithm that learns patterns from historical data. Instead of just averaging past scores, it looks at **26 different factors** at once — the player's recent trends, opponent defensive rating, game pace, home/away, rest days, usage rate, true shooting percentage, and more.

"Gradient Boosted" refers to how it learns. It builds many small decision trees, one at a time. The first tree makes predictions, then the second tree focuses on correcting the mistakes the first one made, then the third tree fixes the remaining errors, and so on for 100 rounds. Each tree "boosts" the overall accuracy by learning from prior mistakes.

Four separate models are trained — one each for points, rebounds, assists, and 3-pointers — because different stats are driven by different factors. The models retrain weekly on accumulated historical data.

### 3. The 60/40 Ensemble Blend

Instead of trusting only one method, the engine takes **60% of the ML prediction** and **40% of the weighted average** and combines them:

```
Final Prediction = (ML Prediction × 60%) + (Weighted Average × 40%)
```

**Example — LeBron's points tonight:**
- ML model says **30.0** (sees he's facing a bad defense, at home, with extra rest)
- Weighted average says **27.5** (his recent scoring average)
- Final: (30.0 × 0.6) + (27.5 × 0.4) = **29.0**

Why not 100% ML? Because ML models can sometimes overreact to patterns. The weighted average acts as a safety net — it anchors predictions to what the player actually does on a regular basis. If the ML model ever says a player will score 50, the weighted average pulls it back to something reasonable.

### 4. Confidence & Recommendations

Each prediction is compared to the live betting line from The Odds API. If the prediction deviates significantly from the posted line, it's flagged as an edge:

| Condition | Recommendation |
|-----------|---------------|
| Prediction > Line by 8%+ | **OVER** ▲ |
| Prediction < Line by 8%+ | **UNDER** ▼ |
| Within 8% | **HOLD** — |

Confidence is scored 0-100 based on the player's consistency (coefficient of variation) and edge size. Picks are labeled HIGH, MEDIUM, or LOW confidence.

---

## Architecture

```
  ┌──────────────────────────────────────────────────────────────────┐
  │                      EXTERNAL DATA SOURCES                       │
  │   balldontlie API (box scores, advanced stats, injuries)         │
  │   The Odds API (live player prop betting lines)                  │
  └──────────────┬──────────────────────────────────┬────────────────┘
                 │                                  │
  ┌──────────────┼──────────────────────────────────┼────────────────┐
  │  AWS         │                                  │                │
  │              ▼                                  │                │
  │   ① Fetch Data λ ──► S3 (raw JSON) + DynamoDB  │                │
  │   (10 AM UTC)                                   │                │
  │              │                                  │                │
  │              ▼                                  │                │
  │   ② Process Stats λ ──► Rolling avgs, trends    │                │
  │   (11 AM UTC)                                   │                │
  │              │                                  │                │
  │              ▼                     ┌────────────┘                │
  │   ③ Predict λ ◄── ML models (S3) ◄┘                             │
  │   (12 PM UTC)    + live odds lines                               │
  │   Ensemble: 60% ML + 40% WA                                     │
  │              │                                                   │
  │              ├──► ④ Discord Notify λ ──► Discord Webhook         │
  │              │    (12:15 PM UTC)                                  │
  │              │                                                   │
  │              ▼                                                   │
  │   ⑤ Verify Results λ ──► Accuracy tracking                      │
  │   (2 PM UTC)                                                     │
  │                                                                  │
  │   ⑥ Train Model λ ──► S3 (.pkl models)                          │
  │   (Sundays 8 AM UTC)                                             │
  │                                                                  │
  │   DynamoDB ──► ⑦ API λ (read-only) ──► API Gateway              │
  │                                            │                     │
  └────────────────────────────────────────────┼─────────────────────┘
                                               │
                                    ┌──────────┴──────────┐
                                    │  S3 Static Site     │
                                    │  Frontend Dashboard │
                                    └─────────────────────┘
```

### Daily Pipeline Schedule (UTC)

| Time | Function | What It Does |
|------|----------|--------------|
| 10:00 AM | **FetchData** | Pulls box scores, advanced stats (pace, usage%, def rating), and injury reports |
| 11:00 AM | **ProcessStats** | Calculates rolling averages (5/10/20 games), trends, and consistency scores |
| 12:00 PM | **Predict** | Loads ML models, builds features, blends predictions, pulls live betting lines |
| 12:15 PM | **DiscordNotify** | Sends top 15 picks to Discord with confidence levels |
| 2:00 PM | **VerifyResults** | Compares yesterday's predictions against actual game results |
| Sunday 8 AM | **TrainModel** | Retrains 4 Gradient Boosted models on all accumulated data |

---

## Tech Stack

| Layer | Service |
|-------|---------|
| **Compute** | AWS Lambda (Python 3.11) × 7 functions |
| **ML** | scikit-learn (GradientBoostingRegressor) |
| **API** | API Gateway (REST, rate-limited) |
| **Database** | DynamoDB (single-table design, 3 tables) |
| **Storage** | S3 (raw data, ML models, frontend) |
| **Scheduling** | EventBridge (6 cron rules) |
| **Notifications** | Discord webhook |
| **IaC** | AWS SAM (CloudFormation) |
| **CI/CD** | GitHub Actions (OIDC auth, no static keys) |
| **Data** | balldontlie API, The Odds API |

---

## Features

- **ML ensemble predictions** — Gradient Boosted model (26 features) blended with weighted rolling averages
- **Live betting lines** — Real-time player prop lines from The Odds API
- **6 stat types** — Points, rebounds, assists, 3-pointers, PRA (pts+reb+ast), team totals
- **Confidence scoring** — HIGH / MEDIUM / LOW based on edge size and player consistency
- **Discord notifications** — Daily top picks delivered automatically
- **Accuracy tracking** — Rolling 30-day accuracy by stat type and confidence level
- **Dark-themed dashboard** — Filter by stat, confidence, or prediction method (ML vs WA)
- **Player drill-down** — Click any prediction to see ML vs WA breakdown, rolling averages, and trend data
- **Fully automated** — Zero manual intervention from data ingestion to prediction delivery
- **Weekly model retraining** — Models improve as more historical data accumulates

---

## Getting Started

### Prerequisites

- [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) configured with credentials
- [AWS SAM CLI](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html)
- Python 3.11+
- An AWS account (free tier eligible)

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/nba-betting-predictor.git
cd nba-betting-predictor
```

### 2. Deploy the backend

**Windows (PowerShell):**
```powershell
.\scripts\deploy.ps1
```

**Mac/Linux:**
```bash
chmod +x scripts/*.sh
./scripts/deploy.sh
```

On first run, SAM will ask guided setup questions. Use these defaults:

| Prompt | Value |
|--------|-------|
| Stack Name | `nba-predictor` |
| AWS Region | `us-east-1` |
| Confirm changes | `Y` |
| Allow SAM CLI IAM role creation | `Y` |
| Save arguments to config file | `Y` |

### 3. Set API keys

```bash
# Required: Odds API key (get one free at https://the-odds-api.com/)
aws lambda update-function-configuration \
  --function-name nba-predictor-PredictFunction-XXXXX \
  --environment Variables="{ODDS_API_KEY=your_key_here}"

# Optional: balldontlie API key (for higher rate limits)
aws lambda update-function-configuration \
  --function-name nba-predictor-FetchDataFunction-XXXXX \
  --environment Variables="{BALLDONTLIE_API_KEY=your_key_here}"

# Optional: Discord webhook (for daily pick notifications)
aws lambda update-function-configuration \
  --function-name nba-predictor-DiscordNotify \
  --environment Variables="{DISCORD_WEBHOOK_URL=your_webhook_url}"
```

### 4. Update the frontend

Copy the API Gateway URL from the deployment output and paste it into `frontend/index.html`:

```javascript
const API_BASE = 'https://YOUR-API-ID.execute-api.us-east-1.amazonaws.com/prod';
```

### 5. Deploy the frontend

```powershell
.\scripts\deploy-frontend.ps1
```

### 6. Trigger the first data fetch

```bash
aws lambda invoke --function-name nba-predictor-FetchDataFunction-XXXXX --payload '{}' response.json
```

After that, everything runs automatically on schedule.

---

## Project Structure

```
nba-betting-predictor/
├── template.yaml                    # SAM template — all AWS infrastructure
├── ARCHITECTURE.md                  # Detailed technical architecture
├── SECURITY.md                      # Security audit documentation
├── README.md                        # This file
├── lambdas/
│   ├── fetch_data/handler.py        # ① Data ingestion (balldontlie API)
│   ├── process_stats/handler.py     # ② Rolling averages, trends, consistency
│   ├── predict/handler.py           # ③ Ensemble prediction engine
│   ├── discord_notify/handler.py    # ④ Discord webhook notifications
│   ├── verify_results/handler.py    # ⑤ Accuracy verification
│   ├── train_model/handler.py       # ⑥ ML model training (weekly)
│   └── api/handler.py               # ⑦ REST API (read-only, validated)
├── frontend/
│   └── index.html                   # Dashboard (single-file, no build step)
├── tests/
│   └── test_predictions.py          # Unit tests for prediction logic
├── scripts/
│   ├── deploy.ps1 / deploy.sh       # Backend deploy scripts
│   ├── deploy-frontend.ps1 / .sh    # Frontend deploy scripts
│   └── build_layer.py               # sklearn Lambda layer builder
└── .github/workflows/
    └── deploy.yml                   # CI/CD (OIDC auth, no static AWS keys)
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/predictions` | Today's predictions (player props + team totals) |
| `GET` | `/predictions/{date}` | Predictions for a specific date (YYYY-MM-DD) |
| `GET` | `/players/{playerId}/stats` | Player game logs + processed analytics |
| `GET` | `/accuracy` | Prediction accuracy over last 30 days |
| `GET` | `/teams` | All NBA teams |

All endpoints are read-only, rate-limited (5 req/sec), and input-validated.

---

## Security

See [SECURITY.md](SECURITY.md) for the full audit. Highlights:

- **Read-only API** — No public endpoint can modify data
- **Rate limiting** — API Gateway throttled at 5 req/sec, burst 10
- **Input validation** — All API parameters validated with regex
- **No hardcoded secrets** — All API keys in Lambda environment variables
- **OIDC CI/CD** — GitHub Actions uses federated auth, no static AWS keys
- **S3 encryption** — AES-256 server-side encryption on raw data
- **DynamoDB backups** — Point-in-Time Recovery on all tables
- **Sanitized errors** — API returns generic error messages, no internal details leaked
- **Least-privilege IAM** — Each Lambda has scoped permissions (API function is read-only)

---

## ML Model Details

### 26-Feature Vector

| Features | Description |
|----------|-------------|
| Rolling averages (×12) | 3/5/10-game averages for pts, reb, ast, fg3m |
| Trends (×4) | Recent vs older performance per stat |
| Minutes, usage%, true shooting% | Player workload and efficiency |
| Games available | Sample size indicator |
| Home/away, rest days | Game context |
| Opponent def rating, pace, pts allowed | Matchup difficulty |
| Team injury count | Roster context |

### Training

- **Algorithm:** GradientBoostingRegressor (100 estimators, max depth 4)
- **Validation:** 5-fold cross-validation
- **Schedule:** Weekly retraining every Sunday
- **Models:** 4 separate (points, rebounds, assists, 3-pointers)
- **Storage:** Serialized .pkl files in S3

---

## Running Tests

```bash
python -m pytest tests/ -v
```

Or directly:

```bash
python tests/test_predictions.py
```

---

## What I Learned

- **ML ensemble methods** — Gradient boosting, feature engineering, cross-validation, model serialization
- **Event-driven architecture** — EventBridge schedules, Lambda chaining, automated pipelines
- **DynamoDB single-table design** — Composite keys, GSIs, efficient query patterns
- **Serverless API design** — API Gateway, rate limiting, input validation, CORS
- **Security engineering** — OIDC CI/CD, least-privilege IAM, encryption at rest, sanitized errors
- **Infrastructure as Code** — SAM templates, CloudFormation, automated deployments

---

## License

MIT
