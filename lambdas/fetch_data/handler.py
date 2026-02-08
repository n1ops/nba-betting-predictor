"""
NBA Data Fetcher Lambda
Pulls player stats, advanced stats, and team data from balldontlie API.
Supports backfill via event parameter: {"backfill_days": 30}
"""

import json
import os
import logging
import time
from datetime import datetime, timedelta
from decimal import Decimal
import urllib.request
import urllib.error
import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3 = boto3.client("s3")
dynamodb = boto3.resource("dynamodb")

RAW_DATA_BUCKET = os.environ.get("RAW_DATA_BUCKET", "")
STATS_TABLE = os.environ.get("STATS_TABLE", "")
BASE_URL = "https://api.balldontlie.io/v1"
API_KEY = os.environ.get("BALLDONTLIE_API_KEY", "")


def make_request(url, retries=2):
    headers = {}
    if API_KEY:
        headers["Authorization"] = API_KEY
    req = urllib.request.Request(url, headers=headers)
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries:
                logger.warning(f"Rate limited, waiting 2s...")
                time.sleep(2)
                continue
            logger.error(f"HTTP {e.code} for {url}")
            raise
        except Exception as e:
            if attempt < retries:
                time.sleep(1)
                continue
            raise


def fetch_games(date_str):
    url = f"{BASE_URL}/games?dates[]={date_str}"
    data = make_request(url)
    return data.get("data", [])


def fetch_game_stats(game_id):
    url = f"{BASE_URL}/stats?game_ids[]={game_id}&per_page=100"
    data = make_request(url)
    return data.get("data", [])


def fetch_advanced_stats(game_id):
    """Fetch advanced stats (pace, usage, offensive/defensive rating) for a game."""
    url = f"{BASE_URL}/stats/advanced?game_ids[]={game_id}&per_page=100"
    try:
        data = make_request(url)
        return data.get("data", [])
    except Exception as e:
        logger.warning(f"Advanced stats not available for game {game_id}: {e}")
        return []


def fetch_all_teams():
    url = f"{BASE_URL}/teams"
    data = make_request(url)
    return data.get("data", [])


def fetch_player_injuries():
    """Fetch current injury report."""
    url = f"{BASE_URL}/player_injuries"
    try:
        data = make_request(url)
        return data.get("data", [])
    except Exception as e:
        logger.warning(f"Injuries endpoint not available: {e}")
        return []


def float_to_decimal(obj):
    if isinstance(obj, float):
        return Decimal(str(round(obj, 3)))
    if isinstance(obj, dict):
        return {k: float_to_decimal(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [float_to_decimal(i) for i in obj]
    return obj


def store_raw_to_s3(data, prefix, filename):
    key = f"raw/{prefix}/{filename}"
    s3.put_object(
        Bucket=RAW_DATA_BUCKET,
        Key=key,
        Body=json.dumps(data, default=str),
        ContentType="application/json",
    )
    return key


def store_game_stats_to_dynamo(game, player_stats, advanced_stats=None):
    table = dynamodb.Table(STATS_TABLE)
    today = datetime.now().strftime("%Y-%m-%d")
    game_date = game.get("date", today)[:10]

    home_team = game.get("home_team", {})
    visitor_team = game.get("visitor_team", {})

    # Store game record
    game_item = {
        "pk": f"GAME#{game['id']}",
        "sk": f"DATE#{game_date}",
        "gsi1pk": f"DATE#{game_date}",
        "gsi1sk": f"GAME#{game['id']}",
        "game_id": game["id"],
        "date": game_date,
        "season": game.get("season", 2025),
        "home_team_id": home_team.get("id"),
        "home_team": home_team.get("full_name", ""),
        "home_team_abbr": home_team.get("abbreviation", ""),
        "visitor_team_id": visitor_team.get("id"),
        "visitor_team": visitor_team.get("full_name", ""),
        "visitor_team_abbr": visitor_team.get("abbreviation", ""),
        "home_score": game.get("home_team_score", 0),
        "visitor_score": game.get("visitor_team_score", 0),
        "status": game.get("status", ""),
        "entity_type": "game",
    }
    table.put_item(Item=float_to_decimal(game_item))

    # Index advanced stats by player ID for quick lookup
    adv_by_player = {}
    if advanced_stats:
        for adv in advanced_stats:
            player = adv.get("player", {})
            if player.get("id"):
                adv_by_player[player["id"]] = adv

    # Store each player's stats
    for stat in player_stats:
        player = stat.get("player", {})
        team = stat.get("team", {})
        if not player.get("id"):
            continue

        # Determine if home or away
        is_home = (team.get("id") == home_team.get("id"))
        opponent_id = visitor_team.get("id") if is_home else home_team.get("id")
        opponent_abbr = visitor_team.get("abbreviation", "") if is_home else home_team.get("abbreviation", "")

        # Get advanced stats for this player
        adv = adv_by_player.get(player["id"], {})

        player_item = {
            "pk": f"PLAYER#{player['id']}",
            "sk": f"GAME#{game['id']}#{game_date}",
            "gsi1pk": f"DATE#{game_date}",
            "gsi1sk": f"PLAYER#{player['id']}",
            "player_id": player["id"],
            "player_name": f"{player.get('first_name', '')} {player.get('last_name', '')}",
            "team_id": team.get("id"),
            "team_abbr": team.get("abbreviation", ""),
            "game_id": game["id"],
            "date": game_date,
            "is_home": is_home,
            "opponent_id": opponent_id,
            "opponent_abbr": opponent_abbr,
            # Basic stats
            "min": stat.get("min", "0"),
            "pts": stat.get("pts", 0),
            "reb": stat.get("reb", 0),
            "ast": stat.get("ast", 0),
            "stl": stat.get("stl", 0),
            "blk": stat.get("blk", 0),
            "turnover": stat.get("turnover", 0),
            "fgm": stat.get("fgm", 0),
            "fga": stat.get("fga", 0),
            "fg3m": stat.get("fg3m", 0),
            "fg3a": stat.get("fg3a", 0),
            "ftm": stat.get("ftm", 0),
            "fta": stat.get("fta", 0),
            "pf": stat.get("pf", 0),
            "fg_pct": stat.get("fg_pct", 0),
            "fg3_pct": stat.get("fg3_pct", 0),
            "ft_pct": stat.get("ft_pct", 0),
            # Advanced stats (if available)
            "pace": adv.get("pace", 0),
            "usage_pct": adv.get("usage_percentage", 0),
            "off_rating": adv.get("offensive_rating", 0),
            "def_rating": adv.get("defensive_rating", 0),
            "net_rating": adv.get("net_rating", 0),
            "true_shooting_pct": adv.get("true_shooting_percentage", 0),
            "assist_pct": adv.get("assist_percentage", 0),
            "reb_pct": adv.get("rebound_percentage", 0),
            "entity_type": "player_game",
        }
        table.put_item(Item=float_to_decimal(player_item))

    logger.info(f"Stored game {game['id']} with {len(player_stats)} players, {len(adv_by_player)} advanced")


def store_team_stats(teams):
    """Store team profiles and compute team-level defensive ratings from recent games."""
    table = dynamodb.Table(STATS_TABLE)
    for team in teams:
        team_item = {
            "pk": f"TEAM#{team['id']}",
            "sk": "PROFILE",
            "team_id": team["id"],
            "full_name": team.get("full_name", ""),
            "abbreviation": team.get("abbreviation", ""),
            "city": team.get("city", ""),
            "conference": team.get("conference", ""),
            "division": team.get("division", ""),
            "entity_type": "team",
        }
        table.put_item(Item=float_to_decimal(team_item))


def store_injuries(injuries):
    """Store current injury data in DynamoDB."""
    table = dynamodb.Table(STATS_TABLE)
    today = datetime.now().strftime("%Y-%m-%d")

    for injury in injuries:
        player = injury.get("player", {})
        if not player.get("id"):
            continue

        teams = player.get("teams", [])
        team_abbr = ""
        if teams:
            team_abbr = teams[0].get("abbreviation", "")

        injury_item = {
            "pk": f"INJURY#{player['id']}",
            "sk": f"DATE#{today}",
            "gsi1pk": f"INJURIES#{today}",
            "gsi1sk": f"PLAYER#{player['id']}",
            "player_id": player["id"],
            "player_name": f"{player.get('first_name', '')} {player.get('last_name', '')}",
            "team_abbr": team_abbr,
            "status": injury.get("status", ""),
            "status_abbr": injury.get("status_abbreviation", ""),
            "injury_type": injury.get("injury_type", ""),
            "return_date": injury.get("return_date", ""),
            "comment": injury.get("comment", ""),
            "entity_type": "injury",
        }
        table.put_item(Item=float_to_decimal(injury_item))

    logger.info(f"Stored {len(injuries)} injury records")


def lambda_handler(event, context):
    logger.info(f"Starting NBA data fetch. Event: {json.dumps(event, default=str)}")

    # Support backfill via event parameter
    backfill_days = event.get("backfill_days", 7)
    fetch_advanced = event.get("fetch_advanced", True)

    dates_to_fetch = []
    for i in range(backfill_days):
        d = datetime.now() - timedelta(days=i)
        dates_to_fetch.append(d.strftime("%Y-%m-%d"))

    total_games = 0
    total_players = 0

    for date_str in dates_to_fetch:
        logger.info(f"Fetching games for {date_str}")
        try:
            games = fetch_games(date_str)
        except Exception as e:
            logger.error(f"Failed to fetch games for {date_str}: {e}")
            continue

        for game in games:
            game_id = game.get("id")
            if not game_id:
                continue
            if game.get("status") != "Final":
                continue

            try:
                player_stats = fetch_game_stats(game_id)

                # Fetch advanced stats if enabled
                advanced_stats = []
                if fetch_advanced:
                    advanced_stats = fetch_advanced_stats(game_id)

                store_raw_to_s3(
                    {"game": game, "player_stats": player_stats, "advanced_stats": advanced_stats},
                    prefix=f"games/{date_str}",
                    filename=f"game_{game_id}.json",
                )
                store_game_stats_to_dynamo(game, player_stats, advanced_stats)

                total_games += 1
                total_players += len(player_stats)

            except Exception as e:
                logger.error(f"Failed to process game {game_id}: {e}")
                continue

    # Fetch teams
    try:
        teams = fetch_all_teams()
        store_raw_to_s3(teams, prefix="teams", filename="all_teams.json")
        store_team_stats(teams)
    except Exception as e:
        logger.error(f"Failed to fetch teams: {e}")

    # Fetch injuries
    try:
        injuries = fetch_player_injuries()
        if injuries:
            store_raw_to_s3(injuries, prefix="injuries", filename=f"injuries_{datetime.now().strftime('%Y-%m-%d')}.json")
            store_injuries(injuries)
    except Exception as e:
        logger.error(f"Failed to fetch injuries: {e}")

    summary = {
        "statusCode": 200,
        "body": json.dumps({
            "message": "Data fetch complete",
            "dates_processed": dates_to_fetch,
            "total_games": total_games,
            "total_player_records": total_players,
            "backfill_days": backfill_days,
        }),
    }
    logger.info(f"Fetch complete: {total_games} games, {total_players} player records")
    return summary