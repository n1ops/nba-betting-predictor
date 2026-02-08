"""
NBA Predictor Discord Notifications
Sends daily top picks to a Discord webhook.
Runs after predictions are generated.
"""

import json
import os
import logging
import urllib.request
import urllib.error
from datetime import datetime
from decimal import Decimal
import boto3
from boto3.dynamodb.conditions import Key

logger = logging.getLogger()
logger.setLevel(logging.INFO)

dynamodb = boto3.resource("dynamodb")

PREDICTIONS_TABLE = os.environ.get("PREDICTIONS_TABLE", "")
RESULTS_TABLE = os.environ.get("RESULTS_TABLE", "")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")

MAX_PICKS = 15  # Top picks to show


class DecimalEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Decimal):
            return float(o)
        return super().default(o)


def get_todays_predictions():
    today = datetime.now().strftime("%Y-%m-%d")
    table = dynamodb.Table(PREDICTIONS_TABLE)
    response = table.query(
        KeyConditionExpression=Key("pk").eq(f"DATE#{today}"),
    )
    items = response.get("Items", [])
    # Filter to player props with lines and non-HOLD recommendations
    picks = [
        i for i in items
        if i.get("type") == "PLAYER_PROP"
        and i.get("line") is not None
        and i.get("recommendation") in ("OVER", "UNDER")
    ]
    # Sort by confidence then edge
    picks.sort(key=lambda x: (
        float(x.get("confidence_score", 0)),
        abs(float(x.get("edge_pct", 0)))
    ), reverse=True)
    return picks


def get_recent_accuracy(days=7):
    """Get accuracy from recent verified results."""
    table = dynamodb.Table(RESULTS_TABLE)
    total = 0
    correct = 0

    from datetime import timedelta
    for i in range(1, days + 1):
        date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        try:
            response = table.query(
                KeyConditionExpression=Key("pk").eq(f"DATE#{date}"),
            )
            for item in response.get("Items", []):
                if item.get("correct") is not None:
                    total += 1
                    if item.get("correct"):
                        correct += 1
        except Exception:
            continue

    if total == 0:
        return None, 0, 0
    return round(correct / total * 100, 1), correct, total


def format_discord_message(picks, accuracy_pct, correct, total):
    """Build a Discord embed message with today's top picks."""
    today = datetime.now().strftime("%A, %B %d, %Y")

    # Split into overs and unders
    overs = [p for p in picks if p.get("recommendation") == "OVER"][:8]
    unders = [p for p in picks if p.get("recommendation") == "UNDER"][:7]
    top_picks = (overs + unders)[:MAX_PICKS]

    if not top_picks:
        return {
            "embeds": [{
                "title": "🏀 NBA Predictor — No Strong Picks Today",
                "description": f"No high-confidence picks for {today}. Check back tomorrow!",
                "color": 0x3498db,
            }]
        }

    # Build picks text
    lines = []
    for p in top_picks:
        rec = p.get("recommendation", "")
        emoji = "🟢" if rec == "OVER" else "🔴"
        edge = float(p.get("edge_pct", 0))
        edge_str = f"+{edge:.1f}%" if edge > 0 else f"{edge:.1f}%"
        conf = p.get("confidence_label", "")
        conf_emoji = "🔥" if conf == "HIGH" else "⚡" if conf == "MEDIUM" else ""

        line_val = p.get("line", "")
        pred_val = p.get("prediction", "")
        stat_label = p.get("stat_label", p.get("stat", ""))
        matchup = p.get("matchup", "")

        lines.append(
            f"{emoji} **{p.get('player_name', '')}** ({p.get('team_abbr', '')}) — "
            f"{stat_label} **{rec}** {line_val}\n"
            f"　Pred: {pred_val} | Edge: {edge_str} | {conf_emoji} {conf}"
        )

    picks_text = "\n\n".join(lines)

    # Stats summary
    all_overs = len([p for p in picks if p.get("recommendation") == "OVER"])
    all_unders = len([p for p in picks if p.get("recommendation") == "UNDER"])
    high_conf = len([p for p in picks if p.get("confidence_label") == "HIGH"])

    # Accuracy footer
    if accuracy_pct is not None:
        accuracy_text = f"📊 Last 7 days: {accuracy_pct}% ({correct}/{total})"
    else:
        accuracy_text = "📊 Accuracy tracking starts after first verified day"

    embed = {
        "title": f"🏀 NBA Predictor — Today's Top Picks",
        "description": (
            f"**{today}**\n"
            f"🟢 {all_overs} Overs | 🔴 {all_unders} Unders | 🔥 {high_conf} High Confidence\n\n"
            f"{picks_text}"
        ),
        "color": 0x00ff88,  # Green
        "footer": {
            "text": f"{accuracy_text} | NBA Over/Under Prediction Engine",
        },
    }

    # Keep within Discord 4096 char limit
    if len(embed["description"]) > 4000:
        embed["description"] = embed["description"][:3997] + "..."

    return {"embeds": [embed]}


def send_discord_message(payload):
    """Send message to Discord webhook."""
    if not DISCORD_WEBHOOK_URL:
        logger.error("No DISCORD_WEBHOOK_URL set")
        return False

    data = json.dumps(payload, cls=DecimalEncoder).encode("utf-8")
    req = urllib.request.Request(
        DISCORD_WEBHOOK_URL,
        data=data,
        method="POST",
    )
    req.add_header("Content-Type", "application/json")
    req.add_header("User-Agent", "Mozilla/5.0 (compatible; NBA-Predictor/1.0)")
    req.add_header("Accept", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            logger.info(f"Discord response: {resp.status}")
            return resp.status == 204
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else "no body"
        logger.error(f"Discord HTTP Error {e.code}: {body}")
        return False
    except Exception as e:
        logger.error(f"Failed to send Discord message: {e}")
        return False


def lambda_handler(event, context):
    logger.info("Sending Discord notifications")

    picks = get_todays_predictions()
    logger.info(f"Found {len(picks)} actionable picks")

    accuracy_pct, correct, total = get_recent_accuracy()

    message = format_discord_message(picks, accuracy_pct, correct, total)
    success = send_discord_message(message)

    return {
        "statusCode": 200,
        "body": json.dumps({
            "message": "Discord notification sent" if success else "Failed to send",
            "picks_sent": min(len(picks), MAX_PICKS),
            "success": success,
        }),
    }