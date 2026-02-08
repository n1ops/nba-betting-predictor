"""
NBA Predictor ML Training Lambda
Trains Gradient Boosted Regressor with enhanced features:
- Rolling averages (3/5/10 game windows)
- Trends and consistency
- Opponent defensive rating
- Pace (possessions per game)
- Player usage rate
- Home/away indicator
- Rest days (back-to-back detection)
- Injury-adjusted context
"""

import json
import os
import logging
import pickle
import io
from datetime import datetime, timedelta
from decimal import Decimal
import boto3
from boto3.dynamodb.conditions import Key

logger = logging.getLogger()
logger.setLevel(logging.INFO)

dynamodb = boto3.resource("dynamodb")
s3 = boto3.client("s3")

STATS_TABLE = os.environ.get("STATS_TABLE", "")
RAW_DATA_BUCKET = os.environ.get("RAW_DATA_BUCKET", "")
MODEL_PREFIX = "models/"

STAT_TARGETS = ["pts", "reb", "ast", "fg3m"]

# Feature names for logging and interpretability
FEATURE_NAMES = [
    # Per-stat rolling averages (4 stats × 3 windows = 12)
    "pts_avg3", "pts_avg5", "pts_avg10", "pts_trend",
    "reb_avg3", "reb_avg5", "reb_avg10", "reb_trend",
    "ast_avg3", "ast_avg5", "ast_avg10", "ast_trend",
    "fg3m_avg3", "fg3m_avg5", "fg3m_avg10", "fg3m_trend",
    # Player context (4)
    "minutes_avg", "usage_pct_avg", "true_shooting_avg", "games_available",
    # Game context (5)
    "is_home", "rest_days", "opp_def_rating", "opp_pace", "opp_pts_allowed_avg",
    # Team injury context (1)
    "team_injuries_count",
]


def decimal_to_float(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, dict):
        return {k: decimal_to_float(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [decimal_to_float(i) for i in obj]
    return obj


def parse_minutes(min_str):
    if not min_str or min_str == "0":
        return 0.0
    try:
        if ":" in str(min_str):
            parts = str(min_str).split(":")
            return float(parts[0]) + float(parts[1]) / 60
        return float(min_str)
    except (ValueError, IndexError):
        return 0.0


def get_all_player_ids():
    table = dynamodb.Table(STATS_TABLE)
    player_ids = set()
    for i in range(30):
        date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        try:
            response = table.query(
                IndexName="GSI1",
                KeyConditionExpression=Key("gsi1pk").eq(f"DATE#{date}") & Key("gsi1sk").begins_with("PLAYER#"),
            )
            for item in response.get("Items", []):
                pid = item.get("player_id")
                if pid:
                    player_ids.add(pid)
        except Exception:
            continue
    return player_ids


def get_player_game_logs(player_id, limit=50):
    table = dynamodb.Table(STATS_TABLE)
    response = table.query(
        KeyConditionExpression=Key("pk").eq(f"PLAYER#{player_id}") & Key("sk").begins_with("GAME#"),
        ScanIndexForward=False,
        Limit=limit,
    )
    return [decimal_to_float(item) for item in response.get("Items", [])]


def compute_team_def_ratings():
    """
    Compute opponent defensive rating from stored game data.
    Returns dict: team_id -> {def_rating, pace, pts_allowed_avg}
    """
    table = dynamodb.Table(STATS_TABLE)
    team_stats = {}

    for i in range(30):
        date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        try:
            response = table.query(
                IndexName="GSI1",
                KeyConditionExpression=Key("gsi1pk").eq(f"DATE#{date}") & Key("gsi1sk").begins_with("GAME#"),
            )
            for game in response.get("Items", []):
                home_id = game.get("home_team_id")
                away_id = game.get("visitor_team_id")
                home_score = float(game.get("home_score", 0) or 0)
                away_score = float(game.get("visitor_score", 0) or 0)

                # Points allowed by each team
                if home_id:
                    home_id = str(home_id)
                    if home_id not in team_stats:
                        team_stats[home_id] = {"pts_allowed": [], "def_ratings": [], "paces": []}
                    team_stats[home_id]["pts_allowed"].append(away_score)

                if away_id:
                    away_id = str(away_id)
                    if away_id not in team_stats:
                        team_stats[away_id] = {"pts_allowed": [], "def_ratings": [], "paces": []}
                    team_stats[away_id]["pts_allowed"].append(home_score)

        except Exception:
            continue

    # Also pull advanced stats (def_rating, pace) from player games
    for i in range(14):
        date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        try:
            response = table.query(
                IndexName="GSI1",
                KeyConditionExpression=Key("gsi1pk").eq(f"DATE#{date}") & Key("gsi1sk").begins_with("PLAYER#"),
            )
            for item in response.get("Items", []):
                opp_id = str(item.get("opponent_id", ""))
                team_id = str(item.get("team_id", ""))
                def_r = float(item.get("def_rating", 0) or 0)
                pace = float(item.get("pace", 0) or 0)

                if team_id and def_r > 0:
                    if team_id not in team_stats:
                        team_stats[team_id] = {"pts_allowed": [], "def_ratings": [], "paces": []}
                    team_stats[team_id]["def_ratings"].append(def_r)

                if team_id and pace > 0:
                    team_stats[team_id]["paces"].append(pace)
        except Exception:
            continue

    # Aggregate
    result = {}
    for tid, stats in team_stats.items():
        pts_allowed = stats["pts_allowed"]
        def_ratings = stats["def_ratings"]
        paces = stats["paces"]

        result[tid] = {
            "def_rating": sum(def_ratings) / len(def_ratings) if def_ratings else 110.0,
            "pace": sum(paces) / len(paces) if paces else 100.0,
            "pts_allowed_avg": sum(pts_allowed) / len(pts_allowed) if pts_allowed else 110.0,
        }

    logger.info(f"Computed defensive stats for {len(result)} teams")
    return result


def get_team_injuries_count():
    """Count current injuries per team."""
    table = dynamodb.Table(STATS_TABLE)
    today = datetime.now().strftime("%Y-%m-%d")
    counts = {}

    try:
        response = table.query(
            IndexName="GSI1",
            KeyConditionExpression=Key("gsi1pk").eq(f"INJURIES#{today}"),
        )
        for item in response.get("Items", []):
            abbr = item.get("team_abbr", "")
            status = item.get("status_abbr", "")
            if abbr and status in ("O", "OFS"):  # Out or Out for Season
                counts[abbr] = counts.get(abbr, 0) + 1
    except Exception as e:
        logger.warning(f"Could not fetch injuries: {e}")

    return counts


def build_features(prior_games, current_game, team_def_stats, injury_counts):
    """
    Build feature vector (26 features) from prior games and context.
    """
    try:
        features = []

        # Per-stat rolling averages + trends (16 features)
        for stat in STAT_TARGETS:
            values = [float(g.get(stat, 0) or 0) for g in prior_games]
            avg_3 = sum(values[:3]) / min(3, len(values)) if values else 0
            avg_5 = sum(values[:5]) / min(5, len(values)) if values else 0
            avg_10 = sum(values[:10]) / min(10, len(values)) if values else 0

            if len(values) >= 6:
                recent = sum(values[:3]) / 3
                older = sum(values[3:6]) / 3
                trend = ((recent - older) / older * 100) if older > 0 else 0
            else:
                trend = 0

            features.extend([avg_3, avg_5, avg_10, trend])

        # Player context (4 features)
        minutes = [parse_minutes(g.get("min", "0")) for g in prior_games[:10]]
        avg_min = sum(minutes) / len(minutes) if minutes else 0
        features.append(avg_min)

        usage_pcts = [float(g.get("usage_pct", 0) or 0) for g in prior_games[:10]]
        avg_usage = sum(usage_pcts) / len(usage_pcts) if usage_pcts else 0
        features.append(avg_usage)

        ts_pcts = [float(g.get("true_shooting_pct", 0) or 0) for g in prior_games[:10]]
        avg_ts = sum(ts_pcts) / len(ts_pcts) if ts_pcts else 0
        features.append(avg_ts)

        features.append(min(len(prior_games), 50))

        # Game context (5 features)
        is_home = 1.0 if current_game.get("is_home") else 0.0
        features.append(is_home)

        # Rest days
        if len(prior_games) >= 1:
            current_date = current_game.get("date", "")
            prev_date = prior_games[0].get("date", "")
            try:
                d1 = datetime.strptime(current_date[:10], "%Y-%m-%d")
                d2 = datetime.strptime(prev_date[:10], "%Y-%m-%d")
                rest = (d1 - d2).days
                rest = min(rest, 7)  # Cap at 7
            except (ValueError, TypeError):
                rest = 1
        else:
            rest = 2
        features.append(float(rest))

        # Opponent defensive stats
        opp_id = str(current_game.get("opponent_id", ""))
        opp_stats = team_def_stats.get(opp_id, {})
        features.append(opp_stats.get("def_rating", 110.0))
        features.append(opp_stats.get("pace", 100.0))
        features.append(opp_stats.get("pts_allowed_avg", 110.0))

        # Injury context
        team_abbr = current_game.get("team_abbr", "")
        features.append(float(injury_counts.get(team_abbr, 0)))

        return features

    except Exception as e:
        logger.error(f"Error building features: {e}")
        return None


def build_training_data(player_ids, team_def_stats, injury_counts):
    datasets = {stat: {"X": [], "y": []} for stat in STAT_TARGETS}
    processed = 0

    for pid in player_ids:
        games = get_player_game_logs(pid, limit=50)
        if len(games) < 8:
            continue

        for i in range(5, len(games)):
            current_game = games[i]
            prior_games = games[i + 1:] if i + 1 < len(games) else []
            if len(prior_games) < 5:
                continue

            features = build_features(prior_games, current_game, team_def_stats, injury_counts)
            if features is None:
                continue

            for stat in STAT_TARGETS:
                actual = float(current_game.get(stat, 0) or 0)
                datasets[stat]["X"].append(features)
                datasets[stat]["y"].append(actual)

        processed += 1

    logger.info(f"Processed {processed} players")
    for stat in STAT_TARGETS:
        logger.info(f"  {stat}: {len(datasets[stat]['X'])} training samples")
    return datasets


def train_models(datasets):
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import cross_val_score
    import numpy as np

    models = {}
    metrics = {}

    for stat in STAT_TARGETS:
        X = datasets[stat]["X"]
        y = datasets[stat]["y"]

        if len(X) < 50:
            logger.warning(f"Not enough data for {stat}: {len(X)} samples")
            continue

        X_arr = np.array(X)
        y_arr = np.array(y)
        logger.info(f"Training {stat}: {len(X)} samples, {X_arr.shape[1]} features")

        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            min_samples_split=10,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=42,
        )

        try:
            cv_scores = cross_val_score(model, X_arr, y_arr, cv=5, scoring="neg_mean_absolute_error")
            mae = -cv_scores.mean()
            metrics[stat] = {
                "mae": round(mae, 3),
                "cv_std": round(cv_scores.std(), 3),
                "samples": len(X),
                "features": X_arr.shape[1],
            }
            logger.info(f"  {stat} CV MAE: {mae:.3f} +/- {cv_scores.std():.3f}")
        except Exception as e:
            logger.warning(f"CV failed for {stat}: {e}")
            metrics[stat] = {"mae": None, "samples": len(X)}

        model.fit(X_arr, y_arr)
        models[stat] = model

        # Feature importance
        importances = model.feature_importances_
        names = FEATURE_NAMES if len(FEATURE_NAMES) == len(importances) else [f"f{i}" for i in range(len(importances))]
        top_idx = importances.argsort()[-5:][::-1]
        logger.info(f"  Top features: {[(names[i], round(importances[i], 3)) for i in top_idx]}")

    return models, metrics


def save_models_to_s3(models, metrics):
    today = datetime.now().strftime("%Y-%m-%d")

    for stat, model in models.items():
        buffer = io.BytesIO()
        pickle.dump(model, buffer)
        buffer.seek(0)
        key = f"{MODEL_PREFIX}{stat}_model.pkl"
        s3.put_object(Bucket=RAW_DATA_BUCKET, Key=key, Body=buffer.getvalue())
        logger.info(f"Saved: s3://{RAW_DATA_BUCKET}/{key}")

    metadata = {
        "trained_date": today,
        "metrics": metrics,
        "model_type": "GradientBoostingRegressor",
        "features": FEATURE_NAMES,
        "feature_count": len(FEATURE_NAMES),
    }
    s3.put_object(
        Bucket=RAW_DATA_BUCKET,
        Key=f"{MODEL_PREFIX}metadata.json",
        Body=json.dumps(metadata, indent=2),
        ContentType="application/json",
    )
    s3.put_object(
        Bucket=RAW_DATA_BUCKET,
        Key=f"{MODEL_PREFIX}archive/{today}_metadata.json",
        Body=json.dumps(metadata, indent=2),
        ContentType="application/json",
    )


def lambda_handler(event, context):
    logger.info("Starting ML model training")

    player_ids = get_all_player_ids()
    logger.info(f"Found {len(player_ids)} active players")

    if not player_ids:
        return {"statusCode": 200, "body": json.dumps({"message": "No player data found"})}

    team_def_stats = compute_team_def_ratings()
    injury_counts = get_team_injuries_count()
    logger.info(f"Injury counts by team: {injury_counts}")

    datasets = build_training_data(player_ids, team_def_stats, injury_counts)

    models, metrics = train_models(datasets)

    if not models:
        return {"statusCode": 200, "body": json.dumps({"message": "Not enough data to train models"})}

    save_models_to_s3(models, metrics)

    summary = {
        "message": "Training complete",
        "models_trained": list(models.keys()),
        "metrics": metrics,
    }
    logger.info(f"Training summary: {json.dumps(summary, default=str)}")
    return {"statusCode": 200, "body": json.dumps(summary, default=str)}
