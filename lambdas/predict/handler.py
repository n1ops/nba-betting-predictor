"""
NBA Prediction Engine Lambda
Generates over/under predictions for player props and team totals.
ENSEMBLE MODEL: Blends ML (Gradient Boosted) + weighted average predictions.
Pulls live betting lines from The Odds API.
"""

import json
import os
import logging
import time
import pickle
import io
from datetime import datetime, timedelta
from decimal import Decimal
import urllib.request
import boto3
from boto3.dynamodb.conditions import Key

logger = logging.getLogger()
logger.setLevel(logging.INFO)

dynamodb = boto3.resource("dynamodb")
s3 = boto3.client("s3")

STATS_TABLE = os.environ.get("STATS_TABLE", "")
PREDICTIONS_TABLE = os.environ.get("PREDICTIONS_TABLE", "")
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
RAW_DATA_BUCKET = os.environ.get("RAW_DATA_BUCKET", "")

WEIGHTS = {"last_5": 0.45, "last_10": 0.30, "last_20": 0.25}
HIGH_CONFIDENCE = 15
MEDIUM_CONFIDENCE = 8
TREND_MULTIPLIER = 0.05
ML_BLEND_WEIGHT = 0.6  # 60% ML, 40% weighted average

PROP_MARKETS = [
    "player_points", "player_rebounds", "player_assists",
    "player_threes", "player_points_rebounds_assists",
]
MARKET_TO_STAT = {
    "player_points": "pts", "player_rebounds": "reb",
    "player_assists": "ast", "player_threes": "fg3m",
    "player_points_rebounds_assists": "pra",
}
STAT_TARGETS = ["pts", "reb", "ast", "fg3m"]

_ml_models = {}
_models_loaded = False


def float_to_decimal(obj):
    if isinstance(obj, float):
        return Decimal(str(round(obj, 3)))
    if isinstance(obj, dict):
        return {k: float_to_decimal(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [float_to_decimal(i) for i in obj]
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


def load_ml_models():
    global _ml_models, _models_loaded
    if _models_loaded:
        return _ml_models
    if not RAW_DATA_BUCKET:
        _models_loaded = True
        return _ml_models
    for stat in STAT_TARGETS:
        key = f"models/{stat}_model.pkl"
        try:
            response = s3.get_object(Bucket=RAW_DATA_BUCKET, Key=key)
            _ml_models[stat] = pickle.loads(response["Body"].read())
            logger.info(f"Loaded ML model for {stat}")
        except Exception as e:
            logger.warning(f"Could not load ML model for {stat}: {e}")
    _models_loaded = True
    logger.info(f"Loaded {len(_ml_models)} ML models")
    return _ml_models


def get_player_game_logs(player_id, limit=15):
    table = dynamodb.Table(STATS_TABLE)
    response = table.query(
        KeyConditionExpression=Key("pk").eq(f"PLAYER#{player_id}") & Key("sk").begins_with("GAME#"),
        ScanIndexForward=False,
        Limit=limit,
    )
    result = []
    for item in response.get("Items", []):
        converted = {}
        for k, v in item.items():
            converted[k] = float(v) if isinstance(v, Decimal) else v
        result.append(converted)
    return result


def compute_team_def_stats():
    table = dynamodb.Table(STATS_TABLE)
    team_stats = {}
    for i in range(14):
        date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        try:
            response = table.query(
                IndexName="GSI1",
                KeyConditionExpression=Key("gsi1pk").eq(f"DATE#{date}") & Key("gsi1sk").begins_with("GAME#"),
            )
            for game in response.get("Items", []):
                home_id = str(game.get("home_team_id", ""))
                away_id = str(game.get("visitor_team_id", ""))
                home_score = float(game.get("home_score", 0) or 0)
                away_score = float(game.get("visitor_score", 0) or 0)
                for tid, pts in [(home_id, away_score), (away_id, home_score)]:
                    if tid:
                        if tid not in team_stats:
                            team_stats[tid] = {"pts_allowed": []}
                        team_stats[tid]["pts_allowed"].append(pts)
        except Exception:
            continue
    result = {}
    for tid, stats in team_stats.items():
        pa = stats["pts_allowed"]
        result[tid] = {
            "def_rating": 110.0,
            "pace": 100.0,
            "pts_allowed_avg": sum(pa) / len(pa) if pa else 110.0,
        }
    return result


def build_ml_features(prior_games, is_home, opponent_id, team_def_stats):
    if len(prior_games) < 5:
        return None
    try:
        features = []
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
        minutes = [parse_minutes(g.get("min", "0")) for g in prior_games[:10]]
        features.append(sum(minutes) / len(minutes) if minutes else 0)
        usage = [float(g.get("usage_pct", 0) or 0) for g in prior_games[:10]]
        features.append(sum(usage) / len(usage) if usage else 0)
        ts = [float(g.get("true_shooting_pct", 0) or 0) for g in prior_games[:10]]
        features.append(sum(ts) / len(ts) if ts else 0)
        features.append(min(len(prior_games), 50))
        features.append(1.0 if is_home else 0.0)
        if len(prior_games) >= 1:
            try:
                d2 = datetime.strptime(str(prior_games[0].get("date", ""))[:10], "%Y-%m-%d")
                rest = min((datetime.now() - d2).days, 7)
            except (ValueError, TypeError):
                rest = 1
        else:
            rest = 2
        features.append(float(rest))
        opp = team_def_stats.get(str(opponent_id), {})
        features.append(opp.get("def_rating", 110.0))
        features.append(opp.get("pace", 100.0))
        features.append(opp.get("pts_allowed_avg", 110.0))
        features.append(0.0)
        return features
    except Exception as e:
        logger.error(f"Error building ML features: {e}")
        return None


def ml_predict(models, features, stat):
    if stat not in models or features is None:
        return None
    try:
        import numpy as np
        return round(float(models[stat].predict(np.array([features]))[0]), 1)
    except Exception as e:
        logger.warning(f"ML prediction failed for {stat}: {e}")
        return None


def get_todays_games():
    today = datetime.now().strftime("%Y-%m-%d")
    url = f"https://api.balldontlie.io/v1/games?dates[]={today}"
    try:
        req = urllib.request.Request(url)
        api_key = os.environ.get("BALLDONTLIE_API_KEY", "")
        if api_key:
            req.add_header("Authorization", api_key)
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode()).get("data", [])
    except Exception as e:
        logger.error(f"Failed to fetch today's games: {e}")
        return []


def get_team_roster_recent_players(team_id, days=7):
    table = dynamodb.Table(STATS_TABLE)
    players = {}
    for i in range(days):
        date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        try:
            response = table.query(
                IndexName="GSI1",
                KeyConditionExpression=Key("gsi1pk").eq(f"DATE#{date}") & Key("gsi1sk").begins_with("PLAYER#"),
            )
            for item in response.get("Items", []):
                if str(item.get("team_id")) == str(team_id) or item.get("team_abbr") == str(team_id):
                    pid = item["player_id"]
                    if pid not in players:
                        players[pid] = {
                            "player_id": pid,
                            "player_name": item.get("player_name", ""),
                            "team_abbr": item.get("team_abbr", ""),
                        }
        except Exception as e:
            logger.error(f"Error querying date {date}: {e}")
    return list(players.values())


def get_processed_stats(player_id):
    table = dynamodb.Table(STATS_TABLE)
    response = table.query(
        KeyConditionExpression=Key("pk").eq(f"PLAYER#{player_id}") & Key("sk").begins_with("PROCESSED#"),
        ScanIndexForward=False,
        Limit=1,
    )
    items = response.get("Items", [])
    return items[0] if items else None


def predict_player_stat(processed_stats, stat_field, line=None, ml_prediction=None):
    rolling = processed_stats.get("rolling_averages", {})
    trends = processed_stats.get("trends", {})
    consistency_scores = processed_stats.get("consistency", {})

    weighted_sum = 0
    total_weight = 0
    for window_key, weight in WEIGHTS.items():
        window_data = rolling.get(window_key, {})
        value = float(window_data.get(stat_field, 0) or 0)
        if value > 0:
            weighted_sum += value * weight
            total_weight += weight
    if total_weight == 0:
        return None

    base_prediction = weighted_sum / total_weight
    trend = float(trends.get(stat_field, 0) or 0)
    trend_adjustment = base_prediction * (trend / 100) * TREND_MULTIPLIER
    wa_prediction = base_prediction + trend_adjustment

    if ml_prediction is not None and ml_prediction > 0:
        adjusted_prediction = (ML_BLEND_WEIGHT * ml_prediction) + ((1 - ML_BLEND_WEIGHT) * wa_prediction)
        prediction_method = "ensemble"
    else:
        adjusted_prediction = wa_prediction
        prediction_method = "weighted_avg"

    consistency = float(consistency_scores.get(stat_field, 50) or 50)
    confidence_score = consistency
    recommendation = "HOLD"
    edge = 0

    if line is not None and line > 0:
        edge = ((adjusted_prediction - line) / line) * 100
        if abs(edge) >= HIGH_CONFIDENCE:
            confidence_score = min(95, confidence_score + 15)
        elif abs(edge) >= MEDIUM_CONFIDENCE:
            confidence_score = min(90, confidence_score + 5)
        else:
            confidence_score = max(20, confidence_score - 10)
        if edge > MEDIUM_CONFIDENCE:
            recommendation = "OVER"
        elif edge < -MEDIUM_CONFIDENCE:
            recommendation = "UNDER"

    if confidence_score >= 75:
        confidence_label = "HIGH"
    elif confidence_score >= 50:
        confidence_label = "MEDIUM"
    else:
        confidence_label = "LOW"

    return {
        "prediction": round(adjusted_prediction, 1),
        "line": line,
        "edge_pct": round(edge, 1),
        "recommendation": recommendation,
        "confidence_score": round(confidence_score, 1),
        "confidence_label": confidence_label,
        "trend": round(trend, 1),
        "consistency": round(consistency, 1),
        "prediction_method": prediction_method,
        "ml_prediction": ml_prediction,
        "wa_prediction": round(wa_prediction, 1),
        "breakdown": {
            "last_5_avg": float(rolling.get("last_5", {}).get(stat_field, 0) or 0),
            "last_10_avg": float(rolling.get("last_10", {}).get(stat_field, 0) or 0),
            "last_20_avg": float(rolling.get("last_20", {}).get(stat_field, 0) or 0),
            "base_prediction": round(base_prediction, 1),
            "trend_adjustment": round(trend_adjustment, 2),
        },
    }


def predict_team_total(team_id, opponent_id):
    table = dynamodb.Table(STATS_TABLE)
    team_totals = []
    for i in range(20):
        date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        response = table.query(
            IndexName="GSI1",
            KeyConditionExpression=Key("gsi1pk").eq(f"DATE#{date}") & Key("gsi1sk").begins_with("GAME#"),
        )
        for game in response.get("Items", []):
            if str(game.get("home_team_id")) == str(team_id):
                team_totals.append(float(game.get("home_score", 0) or 0))
            elif str(game.get("visitor_team_id")) == str(team_id):
                team_totals.append(float(game.get("visitor_score", 0) or 0))
    if not team_totals:
        return None
    avg_5 = sum(team_totals[:5]) / min(5, len(team_totals))
    avg_10 = sum(team_totals[:10]) / min(10, len(team_totals))
    avg_20 = sum(team_totals[:20]) / min(20, len(team_totals))
    prediction = avg_5 * 0.45 + avg_10 * 0.30 + avg_20 * 0.25
    return {
        "prediction": round(prediction, 1),
        "last_5_avg": round(avg_5, 1),
        "last_10_avg": round(avg_10, 1),
        "last_20_avg": round(avg_20, 1),
    }


def normalize_name(name):
    if not name:
        return ""
    name = name.strip()
    for suffix in [" Jr.", " Sr.", " III", " II", " IV", " Jr", " Sr"]:
        name = name.replace(suffix, "")
    return name.lower().strip()


def fetch_player_prop_lines():
    if not ODDS_API_KEY:
        return {}
    all_lines = {}
    events_url = (
        f"https://api.the-odds-api.com/v4/sports/basketball_nba/odds/"
        f"?apiKey={ODDS_API_KEY}&regions=us&markets=totals&oddsFormat=american"
    )
    try:
        with urllib.request.urlopen(events_url, timeout=20) as resp:
            events = json.loads(resp.read().decode())
            logger.info(f"Found {len(events)} events from Odds API")
    except Exception as e:
        logger.warning(f"Failed to fetch events: {e}")
        return {}
    for event in events:
        event_id = event.get("id")
        if not event_id:
            continue
        markets_str = ",".join(PROP_MARKETS)
        prop_url = (
            f"https://api.the-odds-api.com/v4/sports/basketball_nba/events/{event_id}/odds"
            f"?apiKey={ODDS_API_KEY}&regions=us&markets={markets_str}&oddsFormat=american"
        )
        try:
            with urllib.request.urlopen(prop_url, timeout=20) as resp:
                event_data = json.loads(resp.read().decode())
        except Exception as e:
            logger.warning(f"Failed to fetch props for event {event_id}: {e}")
            continue
        bookmakers = event_data.get("bookmakers", [])
        if not bookmakers:
            continue
        for bookmaker in bookmakers:
            for mkt in bookmaker.get("markets", []):
                market_key = mkt.get("key", "")
                stat_key = MARKET_TO_STAT.get(market_key)
                if not stat_key:
                    continue
                for outcome in mkt.get("outcomes", []):
                    if outcome.get("name") == "Over":
                        player_name = outcome.get("description", "")
                        point = outcome.get("point")
                        if player_name and point is not None:
                            name_key = normalize_name(player_name)
                            if name_key not in all_lines:
                                all_lines[name_key] = {}
                            if stat_key not in all_lines[name_key]:
                                all_lines[name_key][stat_key] = float(point)
    logger.info(f"Fetched lines for {len(all_lines)} players")
    return all_lines


def match_player_line(player_name, lines_dict):
    name_key = normalize_name(player_name)
    if name_key in lines_dict:
        return lines_dict[name_key]
    parts = name_key.split()
    if len(parts) >= 2:
        last_name = parts[-1]
        first_initial = parts[0][0] if parts[0] else ""
        matches = [key for key in lines_dict if last_name in key.split()]
        if len(matches) == 1:
            return lines_dict[matches[0]]
        for match in matches:
            if match.split()[0].startswith(first_initial):
                return lines_dict[match]
    return {}


def store_predictions(predictions):
    table = dynamodb.Table(PREDICTIONS_TABLE)
    today = datetime.now().strftime("%Y-%m-%d")
    ttl = int(time.time()) + (30 * 86400)
    for pred in predictions:
        item = {"pk": f"DATE#{today}", "sk": f"{pred['type']}#{pred['id']}", "date": today, "ttl": ttl, **pred}
        table.put_item(Item=float_to_decimal(item))
    logger.info(f"Stored {len(predictions)} predictions for {today}")


def lambda_handler(event, context):
    logger.info("Starting prediction engine (ensemble mode)")
    today = datetime.now().strftime("%Y-%m-%d")
    games = get_todays_games()
    all_predictions = []

    if not games:
        store_predictions([{"type": "STATUS", "id": "no_games", "message": "No games scheduled"}])
        return {"statusCode": 200, "body": json.dumps({"message": "No games today"})}

    logger.info(f"Found {len(games)} games for today")

    models = load_ml_models()
    ml_available = len(models) > 0
    logger.info(f"ML models available: {ml_available} ({len(models)} models)")

    team_def_stats = compute_team_def_stats() if ml_available else {}
    lines = fetch_player_prop_lines()
    logger.info(f"Total players with lines: {len(lines)}")

    ml_used_count = 0
    wa_only_count = 0

    for game in games:
        home_team = game.get("home_team", {})
        visitor_team = game.get("visitor_team", {})
        game_id = game.get("id")
        logger.info(f"Processing {visitor_team.get('abbreviation')} @ {home_team.get('abbreviation')}")

        for team_info, is_home in [(home_team, True), (visitor_team, False)]:
            team_id = team_info.get("id")
            opponent_id = visitor_team.get("id") if is_home else home_team.get("id")
            players = get_team_roster_recent_players(team_id)

            for player in players:
                pid = player["player_id"]
                processed = get_processed_stats(pid)
                if not processed:
                    continue

                player_name = player.get("player_name", "")
                player_lines = match_player_line(player_name, lines)

                ml_features = None
                if ml_available:
                    game_logs = get_player_game_logs(pid, limit=15)
                    if len(game_logs) >= 5:
                        ml_features = build_ml_features(game_logs, is_home, opponent_id, team_def_stats)

                for stat in ["pts", "reb", "ast", "fg3m", "pra"]:
                    line = player_lines.get(stat)
                    ml_pred = None
                    if ml_features and stat in models and stat != "pra":
                        ml_pred = ml_predict(models, ml_features, stat)
                        if ml_pred is not None:
                            ml_used_count += 1
                        else:
                            wa_only_count += 1
                    else:
                        wa_only_count += 1

                    prediction = predict_player_stat(processed, stat, line=line, ml_prediction=ml_pred)
                    if prediction:
                        all_predictions.append({
                            "type": "PLAYER_PROP",
                            "id": f"{pid}_{stat}",
                            "player_id": pid,
                            "player_name": player_name,
                            "team_abbr": player.get("team_abbr", ""),
                            "stat": stat,
                            "stat_label": {
                                "pts": "Points", "reb": "Rebounds", "ast": "Assists",
                                "fg3m": "3-Pointers Made", "pra": "Pts+Reb+Ast",
                            }.get(stat, stat),
                            "game_id": game_id,
                            "opponent": visitor_team.get("abbreviation") if is_home else home_team.get("abbreviation"),
                            "is_home": is_home,
                            "matchup": f"{visitor_team.get('abbreviation')} @ {home_team.get('abbreviation')}",
                            **prediction,
                        })

        for team_info in [home_team, visitor_team]:
            team_pred = predict_team_total(
                team_info.get("id"),
                visitor_team.get("id") if team_info == home_team else home_team.get("id"),
            )
            if team_pred:
                all_predictions.append({
                    "type": "TEAM_TOTAL",
                    "id": f"team_{team_info.get('id')}_{game_id}",
                    "team_id": team_info.get("id"),
                    "team_name": team_info.get("full_name", ""),
                    "team_abbr": team_info.get("abbreviation", ""),
                    "game_id": game_id,
                    "matchup": f"{visitor_team.get('abbreviation')} @ {home_team.get('abbreviation')}",
                    **team_pred,
                })

    player_props = [p for p in all_predictions if p["type"] == "PLAYER_PROP"]
    player_props.sort(key=lambda x: x.get("confidence_score", 0), reverse=True)
    store_predictions(all_predictions)

    with_lines = len([p for p in player_props if p.get("line") is not None])
    summary = {
        "date": today,
        "games": len(games),
        "total_predictions": len(all_predictions),
        "predictions_with_lines": with_lines,
        "high_confidence_picks": len([p for p in player_props if p.get("confidence_label") == "HIGH"]),
        "over_picks": len([p for p in player_props if p.get("recommendation") == "OVER"]),
        "under_picks": len([p for p in player_props if p.get("recommendation") == "UNDER"]),
        "ml_predictions": ml_used_count,
        "wa_only_predictions": wa_only_count,
        "ml_models_loaded": len(models),
    }
    logger.info(f"Prediction summary: {json.dumps(summary)}")
    return {"statusCode": 200, "body": json.dumps(summary)}