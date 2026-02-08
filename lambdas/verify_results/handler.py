"""
NBA Results Verifier Lambda
Runs daily to check yesterday's predictions against actual game results.
Stores accuracy data in the Results table for the dashboard.
"""

import json
import os
import logging
import time
from datetime import datetime, timedelta
from decimal import Decimal
import boto3
from boto3.dynamodb.conditions import Key

logger = logging.getLogger()
logger.setLevel(logging.INFO)

dynamodb = boto3.resource("dynamodb")

STATS_TABLE = os.environ.get("STATS_TABLE", "")
PREDICTIONS_TABLE = os.environ.get("PREDICTIONS_TABLE", "")
RESULTS_TABLE = os.environ.get("RESULTS_TABLE", "")


def float_to_decimal(obj):
    if isinstance(obj, float):
        return Decimal(str(round(obj, 3)))
    if isinstance(obj, dict):
        return {k: float_to_decimal(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [float_to_decimal(i) for i in obj]
    return obj


def get_predictions_for_date(date):
    """Get all predictions that were made for a specific date."""
    table = dynamodb.Table(PREDICTIONS_TABLE)
    response = table.query(
        KeyConditionExpression=Key("pk").eq(f"DATE#{date}"),
    )
    items = response.get("Items", [])
    return [i for i in items if i.get("type") == "PLAYER_PROP" and i.get("line") is not None]


def get_actual_player_stats(player_id, date):
    """Get the actual stats a player recorded on a specific date."""
    table = dynamodb.Table(STATS_TABLE)
    response = table.query(
        KeyConditionExpression=Key("pk").eq(f"PLAYER#{player_id}") & Key("sk").begins_with(f"GAME#"),
    )
    # Find the game that matches this date
    for item in response.get("Items", []):
        if item.get("date", "") == date:
            return item
    return None


def verify_prediction(prediction, actual_stats):
    """
    Compare a prediction against actual results.
    Returns dict with correctness info.
    """
    stat = prediction.get("stat")
    line = prediction.get("line")
    recommendation = prediction.get("recommendation")
    predicted_value = prediction.get("prediction")

    if not actual_stats or not line or not recommendation or recommendation == "HOLD":
        return None

    # Get actual value
    if stat == "pra":
        actual = (
            float(actual_stats.get("pts", 0) or 0) +
            float(actual_stats.get("reb", 0) or 0) +
            float(actual_stats.get("ast", 0) or 0)
        )
    else:
        actual = float(actual_stats.get(stat, 0) or 0)

    line = float(line)

    # Did the actual go over or under the line?
    actual_result = "OVER" if actual > line else "UNDER" if actual < line else "PUSH"

    # Was our prediction correct?
    if actual_result == "PUSH":
        correct = None  # Push = no result
    else:
        correct = (recommendation == actual_result)

    return {
        "correct": correct,
        "actual_value": actual,
        "actual_result": actual_result,
        "line": line,
        "predicted_value": predicted_value,
        "recommendation": recommendation,
        "difference": round(actual - line, 1),
    }


def lambda_handler(event, context):
    """
    Verify predictions from yesterday (or a specified date).
    Compares predicted over/under vs actual results.
    """
    # Check yesterday by default, or accept a date parameter
    check_date = event.get("date")
    if not check_date:
        check_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    logger.info(f"Verifying predictions for {check_date}")

    # Get predictions that had lines and recommendations
    predictions = get_predictions_for_date(check_date)
    logger.info(f"Found {len(predictions)} predictions with lines for {check_date}")

    if not predictions:
        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": f"No predictions with lines found for {check_date}",
                "date": check_date,
            }),
        }

    results_table = dynamodb.Table(RESULTS_TABLE)
    total = 0
    correct = 0
    incorrect = 0
    pushes = 0
    no_data = 0

    for pred in predictions:
        player_id = pred.get("player_id")
        player_name = pred.get("player_name", "")
        stat = pred.get("stat")

        if not player_id:
            continue

        # Get actual stats
        actual = get_actual_player_stats(player_id, check_date)

        if not actual:
            no_data += 1
            continue

        # Verify
        result = verify_prediction(pred, actual)

        if not result:
            continue

        if result["correct"] is None:
            pushes += 1
            continue

        total += 1
        if result["correct"]:
            correct += 1
        else:
            incorrect += 1

        # Store result
        result_item = {
            "pk": f"DATE#{check_date}",
            "sk": f"RESULT#{player_id}_{stat}",
            "date": check_date,
            "player_id": player_id,
            "player_name": player_name,
            "team_abbr": pred.get("team_abbr", ""),
            "stat": stat,
            "stat_label": pred.get("stat_label", ""),
            "matchup": pred.get("matchup", ""),
            "correct": result["correct"],
            "recommendation": result["recommendation"],
            "predicted_value": result["predicted_value"],
            "actual_value": result["actual_value"],
            "line": result["line"],
            "actual_result": result["actual_result"],
            "difference": result["difference"],
            "confidence_score": pred.get("confidence_score"),
            "confidence_label": pred.get("confidence_label"),
            "edge_pct": pred.get("edge_pct"),
            "entity_type": "result",
        }
        results_table.put_item(Item=float_to_decimal(result_item))

    accuracy_pct = round(correct / total * 100, 1) if total > 0 else 0

    summary = {
        "date": check_date,
        "total_verified": total,
        "correct": correct,
        "incorrect": incorrect,
        "pushes": pushes,
        "no_data": no_data,
        "accuracy_pct": accuracy_pct,
    }

    logger.info(f"Verification summary: {json.dumps(summary)}")

    return {"statusCode": 200, "body": json.dumps(summary)}