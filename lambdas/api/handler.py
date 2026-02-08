"""
NBA Predictor API Lambda
Serves predictions, stats, and accuracy data to the frontend dashboard.
Handles all API Gateway routes.
"""

import json
import re
import os
import logging
from datetime import datetime, timedelta
from decimal import Decimal
import boto3
from boto3.dynamodb.conditions import Key

logger = logging.getLogger()
logger.setLevel(logging.INFO)

dynamodb = boto3.resource("dynamodb")

PREDICTIONS_TABLE = os.environ.get("PREDICTIONS_TABLE", "")
STATS_TABLE = os.environ.get("STATS_TABLE", "")
RESULTS_TABLE = os.environ.get("RESULTS_TABLE", "")


class DecimalEncoder(json.JSONEncoder):
    """Handle Decimal serialization for DynamoDB items."""
    def default(self, o):
        if isinstance(o, Decimal):
            return float(o)
        return super().default(o)


def json_response(status_code, body):
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "http://nba-predictor-frontend-132141656493.s3-website-us-east-1.amazonaws.com",
            "Access-Control-Allow-Methods": "GET,OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        },
        "body": json.dumps(body, cls=DecimalEncoder),
    }

def validate_date(date_str):
    if not date_str:
        return None
    if not re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
        return None
    return date_str

def validate_player_id(pid):
    if not pid:
        return None
    if not re.match(r'^\d{1,10}$', str(pid)):
        return None
    return str(pid)


def get_predictions(date=None):
    """Get all predictions for a date."""
    table = dynamodb.Table(PREDICTIONS_TABLE)
    if not date:
        date = datetime.now().strftime("%Y-%m-%d")

    response = table.query(
        KeyConditionExpression=Key("pk").eq(f"DATE#{date}"),
    )
    items = response.get("Items", [])

    # Separate into categories
    player_props = [i for i in items if i.get("type") == "PLAYER_PROP"]
    team_totals = [i for i in items if i.get("type") == "TEAM_TOTAL"]

    # Sort player props by confidence
    player_props.sort(key=lambda x: float(x.get("confidence_score", 0)), reverse=True)

    return {
        "date": date,
        "player_props": player_props,
        "team_totals": team_totals,
        "total_predictions": len(player_props) + len(team_totals),
        "high_confidence": [p for p in player_props if p.get("confidence_label") == "HIGH"],
    }


def get_player_stats(player_id):
    """Get detailed stats for a specific player."""
    table = dynamodb.Table(STATS_TABLE)

    # Get game logs
    game_logs_resp = table.query(
        KeyConditionExpression=Key("pk").eq(f"PLAYER#{player_id}") & Key("sk").begins_with("GAME#"),
        ScanIndexForward=False,
        Limit=20,
    )
    game_logs = game_logs_resp.get("Items", [])

    # Get most recent processed stats
    processed_resp = table.query(
        KeyConditionExpression=Key("pk").eq(f"PLAYER#{player_id}") & Key("sk").begins_with("PROCESSED#"),
        ScanIndexForward=False,
        Limit=1,
    )
    processed = processed_resp.get("Items", [])
    processed_stats = processed[0] if processed else None

    player_name = game_logs[0].get("player_name", "") if game_logs else ""
    team_abbr = game_logs[0].get("team_abbr", "") if game_logs else ""

    return {
        "player_id": player_id,
        "player_name": player_name,
        "team_abbr": team_abbr,
        "game_logs": game_logs[:20],
        "processed_stats": processed_stats,
    }


def get_accuracy():
    """Calculate prediction accuracy from historical results."""
    table = dynamodb.Table(RESULTS_TABLE)

    # Scan recent results (last 30 days)
    results = []
    for i in range(30):
        date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        try:
            response = table.query(
                KeyConditionExpression=Key("pk").eq(f"DATE#{date}"),
            )
            results.extend(response.get("Items", []))
        except Exception:
            continue

    if not results:
        return {
            "total_predictions": 0,
            "correct": 0,
            "accuracy_pct": 0,
            "by_confidence": {},
            "by_stat": {},
            "message": "No tracked results yet. Accuracy will populate as predictions are verified.",
        }

    total = len(results)
    correct = len([r for r in results if r.get("correct")])

    # Accuracy by confidence level
    by_confidence = {}
    for level in ["HIGH", "MEDIUM", "LOW"]:
        level_results = [r for r in results if r.get("confidence_label") == level]
        level_correct = len([r for r in level_results if r.get("correct")])
        if level_results:
            by_confidence[level] = {
                "total": len(level_results),
                "correct": level_correct,
                "accuracy_pct": round(level_correct / len(level_results) * 100, 1),
            }

    # Accuracy by stat type
    by_stat = {}
    for stat in ["pts", "reb", "ast", "fg3m", "pra"]:
        stat_results = [r for r in results if r.get("stat") == stat]
        stat_correct = len([r for r in stat_results if r.get("correct")])
        if stat_results:
            by_stat[stat] = {
                "total": len(stat_results),
                "correct": stat_correct,
                "accuracy_pct": round(stat_correct / len(stat_results) * 100, 1),
            }

    return {
        "total_predictions": total,
        "correct": correct,
        "accuracy_pct": round(correct / total * 100, 1) if total > 0 else 0,
        "by_confidence": by_confidence,
        "by_stat": by_stat,
        "days_tracked": 30,
    }


def get_teams():
    """Get all NBA teams."""
    table = dynamodb.Table(STATS_TABLE)
    response = table.query(
        KeyConditionExpression=Key("pk").begins_with("TEAM#") if False else Key("pk").eq("TEAM#1"),
    )
    # Use scan with filter for teams (small dataset, acceptable)
    response = table.scan(
        FilterExpression="entity_type = :et",
        ExpressionAttributeValues={":et": "team"},
    )
    return response.get("Items", [])


def lambda_handler(event, context):
    """Route API Gateway requests to appropriate handlers."""
    logger.info(f"API event: {json.dumps(event, default=str)}")

    path = event.get("path", "")
    method = event.get("httpMethod", "GET")
    path_params = event.get("pathParameters") or {}
    query_params = event.get("queryStringParameters") or {}

    try:
        if path == "/predictions" and method == "GET":
            date = query_params.get("date")
            data = get_predictions(date)
            return json_response(200, data)

        elif path.startswith("/predictions/") and method == "GET":
            date = path_params.get("date")
            data = get_predictions(date)
            return json_response(200, data)

        elif path.startswith("/players/") and path.endswith("/stats") and method == "GET":
            player_id = path_params.get("playerId")
            data = get_player_stats(player_id)
            return json_response(200, data)

        elif path == "/accuracy" and method == "GET":
            data = get_accuracy()
            return json_response(200, data)

        elif path == "/teams" and method == "GET":
            data = get_teams()
            return json_response(200, data)

        else:
            return json_response(404, {"error": "Not found", "path": path})

    except Exception as e:
        logger.error(f"API error: {e}", exc_info=True)
        return json_response(500, {"error": "Internal server error"})
