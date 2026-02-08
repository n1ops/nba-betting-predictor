import json
import os
import logging
from datetime import datetime, timedelta
from decimal import Decimal
import boto3
from boto3.dynamodb.conditions import Key

logger = logging.getLogger()
logger.setLevel(logging.INFO)

dynamodb = boto3.resource("dynamodb")
STATS_TABLE = os.environ.get("STATS_TABLE", "")

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

def get_player_game_logs(player_id, limit=20):
    table = dynamodb.Table(STATS_TABLE)
    response = table.query(
        KeyConditionExpression=Key("pk").eq(f"PLAYER#{player_id}") & Key("sk").begins_with("GAME#"),
        ScanIndexForward=False,
        Limit=limit,
    )
    return response.get("Items", [])

def calculate_rolling_averages(game_logs, windows=(5, 10, 20)):
    stat_fields = ["pts", "reb", "ast", "stl", "blk", "turnover", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta"]
    averages = {}
    for window in windows:
        subset = game_logs[:window]
        if not subset:
            continue
        n = len(subset)
        window_avg = {}
        for field in stat_fields:
            total = sum(float(g.get(field, 0) or 0) for g in subset)
            window_avg[field] = round(total / n, 2)
        fga_total = sum(float(g.get("fga", 0) or 0) for g in subset)
        fg3a_total = sum(float(g.get("fg3a", 0) or 0) for g in subset)
        fta_total = sum(float(g.get("fta", 0) or 0) for g in subset)
        window_avg["fg_pct"] = round(sum(float(g.get("fgm", 0) or 0) for g in subset) / fga_total, 3) if fga_total > 0 else 0
        window_avg["fg3_pct"] = round(sum(float(g.get("fg3m", 0) or 0) for g in subset) / fg3a_total, 3) if fg3a_total > 0 else 0
        window_avg["ft_pct"] = round(sum(float(g.get("ftm", 0) or 0) for g in subset) / fta_total, 3) if fta_total > 0 else 0
        window_avg["min"] = round(sum(parse_minutes(g.get("min", "0")) for g in subset) / n, 1)
        window_avg["pra"] = round(window_avg["pts"] + window_avg["reb"] + window_avg["ast"], 2)
        averages[f"last_{window}"] = window_avg
    return averages

def calculate_trend(game_logs, stat_field, window=5):
    if len(game_logs) < window * 2:
        return 0.0
    recent = game_logs[:window]
    older = game_logs[window: window * 2]
    recent_avg = sum(float(g.get(stat_field, 0) or 0) for g in recent) / len(recent)
    older_avg = sum(float(g.get(stat_field, 0) or 0) for g in older) / len(older)
    if older_avg == 0:
        return 0.0
    return round((recent_avg - older_avg) / older_avg * 100, 2)

def calculate_consistency(game_logs, stat_field, window=10):
    subset = game_logs[:window]
    if len(subset) < 3:
        return 50.0
    values = [float(g.get(stat_field, 0) or 0) for g in subset]
    mean = sum(values) / len(values)
    if mean == 0:
        return 0.0
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    std_dev = variance ** 0.5
    cv = std_dev / mean
    return max(0, min(100, round((1 - cv) * 100, 1)))

def store_processed_stats(player_id, player_name, team_abbr, processed_data):
    table = dynamodb.Table(STATS_TABLE)
    today = datetime.now().strftime("%Y-%m-%d")
    item = {
        "pk": f"PLAYER#{player_id}",
        "sk": f"PROCESSED#{today}",
        "gsi1pk": f"PROCESSED#{today}",
        "gsi1sk": f"PLAYER#{player_id}",
        "player_id": player_id,
        "player_name": player_name,
        "team_abbr": team_abbr,
        "processed_date": today,
        "rolling_averages": processed_data["rolling_averages"],
        "trends": processed_data["trends"],
        "consistency": processed_data["consistency"],
        "games_analyzed": processed_data["games_analyzed"],
        "entity_type": "processed_stats",
    }
    table.put_item(Item=float_to_decimal(item))

def lambda_handler(event, context):
    logger.info("Starting scheduled stats processing")
    table = dynamodb.Table(STATS_TABLE)
    player_ids_to_process = set()
    for i in range(7):
        date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        try:
            response = table.query(
                IndexName="GSI1",
                KeyConditionExpression=Key("gsi1pk").eq(f"DATE#{date}") & Key("gsi1sk").begins_with("PLAYER#"),
            )
            for item in response.get("Items", []):
                player_ids_to_process.add(
                    (item.get("player_id"), item.get("player_name", ""), item.get("team_abbr", ""))
                )
        except Exception as e:
            logger.error(f"Error querying date {date}: {e}")
            continue
    logger.info(f"Found {len(player_ids_to_process)} players to process")
    processed_count = 0
    for player_id, player_name, team_abbr in player_ids_to_process:
        try:
            game_logs = get_player_game_logs(player_id, limit=20)
            if len(game_logs) < 3:
                continue
            rolling_averages = calculate_rolling_averages(game_logs)
            trends = {}
            consistency = {}
            for stat in ["pts", "reb", "ast", "fg3m"]:
                trends[stat] = calculate_trend(game_logs, stat)
                consistency[stat] = calculate_consistency(game_logs, stat)
            processed_data = {
                "rolling_averages": rolling_averages,
                "trends": trends,
                "consistency": consistency,
                "games_analyzed": len(game_logs),
            }
            store_processed_stats(player_id, player_name, team_abbr, processed_data)
            processed_count += 1
        except Exception as e:
            logger.error(f"Failed to process player {player_id}: {e}")
            continue
    return {
        "statusCode": 200,
        "body": json.dumps({"message": "Processing complete", "players_processed": processed_count}),
    }
