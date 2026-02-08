"""
Tests for the prediction engine logic.
Run: python -m pytest tests/ -v
"""

import sys
import os

# Add lambdas to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lambdas', 'predict'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lambdas', 'process_stats'))


def test_predict_player_stat_basic():
    """Test basic prediction with all three windows."""
    from handler import predict_player_stat

    processed_stats = {
        "rolling_averages": {
            "last_5": {"pts": 28.0, "reb": 8.0, "ast": 5.0},
            "last_10": {"pts": 25.0, "reb": 7.5, "ast": 4.5},
            "last_20": {"pts": 24.0, "reb": 7.0, "ast": 4.0},
        },
        "trends": {"pts": 5.0, "reb": 2.0, "ast": 3.0},
        "consistency": {"pts": 72.0, "reb": 65.0, "ast": 58.0},
    }

    result = predict_player_stat(processed_stats, "pts", line=25.5)

    assert result is not None
    assert "prediction" in result
    assert "recommendation" in result
    assert "confidence_score" in result
    assert "confidence_label" in result
    assert result["prediction"] > 0

    # With these averages (28*0.45 + 25*0.30 + 24*0.25 = 25.1), prediction should be around 25-26
    assert 24 < result["prediction"] < 30

    # With line at 25.5 and prediction around 25.7, edge should be positive but small
    print(f"Prediction: {result['prediction']}, Edge: {result['edge_pct']}%")


def test_predict_player_stat_no_line():
    """Test prediction without a betting line."""
    from handler import predict_player_stat

    processed_stats = {
        "rolling_averages": {
            "last_5": {"pts": 30.0},
            "last_10": {"pts": 28.0},
            "last_20": {"pts": 26.0},
        },
        "trends": {"pts": 0},
        "consistency": {"pts": 60},
    }

    result = predict_player_stat(processed_stats, "pts")
    assert result is not None
    assert result["recommendation"] == "HOLD"
    assert result["line"] is None


def test_predict_over_recommendation():
    """Test that a high prediction vs low line gives OVER."""
    from handler import predict_player_stat

    processed_stats = {
        "rolling_averages": {
            "last_5": {"pts": 35.0},
            "last_10": {"pts": 33.0},
            "last_20": {"pts": 32.0},
        },
        "trends": {"pts": 10.0},
        "consistency": {"pts": 80.0},
    }

    result = predict_player_stat(processed_stats, "pts", line=25.0)
    assert result["recommendation"] == "OVER"
    assert result["edge_pct"] > 0
    print(f"OVER test - Prediction: {result['prediction']}, Edge: {result['edge_pct']}%")


def test_predict_under_recommendation():
    """Test that a low prediction vs high line gives UNDER."""
    from handler import predict_player_stat

    processed_stats = {
        "rolling_averages": {
            "last_5": {"pts": 18.0},
            "last_10": {"pts": 19.0},
            "last_20": {"pts": 20.0},
        },
        "trends": {"pts": -5.0},
        "consistency": {"pts": 70.0},
    }

    result = predict_player_stat(processed_stats, "pts", line=25.0)
    assert result["recommendation"] == "UNDER"
    assert result["edge_pct"] < 0
    print(f"UNDER test - Prediction: {result['prediction']}, Edge: {result['edge_pct']}%")


def test_predict_missing_data():
    """Test prediction with empty data returns None."""
    from handler import predict_player_stat

    result = predict_player_stat({"rolling_averages": {}, "trends": {}, "consistency": {}}, "pts")
    assert result is None


def test_rolling_averages():
    """Test rolling average calculation."""
    from handler import calculate_rolling_averages

    game_logs = [
        {"pts": 30, "reb": 10, "ast": 5, "stl": 2, "blk": 1, "turnover": 3,
         "fgm": 12, "fga": 22, "fg3m": 3, "fg3a": 8, "ftm": 3, "fta": 4, "min": "35:20"},
        {"pts": 25, "reb": 8, "ast": 7, "stl": 1, "blk": 0, "turnover": 2,
         "fgm": 10, "fga": 20, "fg3m": 2, "fg3a": 6, "ftm": 3, "fta": 3, "min": "32:10"},
        {"pts": 28, "reb": 12, "ast": 4, "stl": 3, "blk": 2, "turnover": 1,
         "fgm": 11, "fga": 21, "fg3m": 4, "fg3a": 9, "ftm": 2, "fta": 2, "min": "38:00"},
        {"pts": 22, "reb": 6, "ast": 8, "stl": 1, "blk": 1, "turnover": 4,
         "fgm": 9, "fga": 19, "fg3m": 1, "fg3a": 5, "ftm": 3, "fta": 4, "min": "30:45"},
        {"pts": 35, "reb": 9, "ast": 6, "stl": 2, "blk": 0, "turnover": 2,
         "fgm": 14, "fga": 24, "fg3m": 5, "fg3a": 10, "ftm": 2, "fta": 2, "min": "36:00"},
    ]

    averages = calculate_rolling_averages(game_logs, windows=(5,))
    assert "last_5" in averages

    avg = averages["last_5"]
    assert avg["pts"] == (30 + 25 + 28 + 22 + 35) / 5  # 28.0
    assert avg["reb"] == (10 + 8 + 12 + 6 + 9) / 5     # 9.0
    assert avg["pra"] == avg["pts"] + avg["reb"] + avg["ast"]

    print(f"Rolling avg PTS: {avg['pts']}, REB: {avg['reb']}, AST: {avg['ast']}, PRA: {avg['pra']}")


def test_trend_calculation():
    """Test trend calculation (recent vs older performance)."""
    from handler import calculate_trend

    # Player trending up in points
    game_logs = [
        {"pts": 30}, {"pts": 32}, {"pts": 28}, {"pts": 31}, {"pts": 29},  # recent: avg 30
        {"pts": 22}, {"pts": 24}, {"pts": 20}, {"pts": 23}, {"pts": 21},  # older: avg 22
    ]

    trend = calculate_trend(game_logs, "pts", window=5)
    assert trend > 0  # Should be trending up
    print(f"Trend (up): {trend}%")

    # Player trending down
    game_logs_down = list(reversed(game_logs))
    trend_down = calculate_trend(game_logs_down, "pts", window=5)
    assert trend_down < 0  # Should be trending down
    print(f"Trend (down): {trend_down}%")


def test_consistency_calculation():
    """Test consistency scoring."""
    from handler import calculate_consistency

    # Very consistent player
    consistent_logs = [{"pts": 25}, {"pts": 26}, {"pts": 24}, {"pts": 25},
                       {"pts": 26}, {"pts": 25}, {"pts": 24}, {"pts": 25},
                       {"pts": 26}, {"pts": 24}]
    high_consistency = calculate_consistency(consistent_logs, "pts")

    # Very inconsistent player
    inconsistent_logs = [{"pts": 40}, {"pts": 10}, {"pts": 35}, {"pts": 8},
                         {"pts": 42}, {"pts": 12}, {"pts": 38}, {"pts": 5},
                         {"pts": 45}, {"pts": 7}]
    low_consistency = calculate_consistency(inconsistent_logs, "pts")

    assert high_consistency > low_consistency
    print(f"High consistency: {high_consistency}, Low consistency: {low_consistency}")


def test_parse_minutes():
    """Test minute string parsing."""
    from handler import parse_minutes

    assert parse_minutes("32:15") == 32.25
    assert parse_minutes("0") == 0.0
    assert parse_minutes("") == 0.0
    assert parse_minutes(None) == 0.0
    assert parse_minutes("40:00") == 40.0


if __name__ == "__main__":
    print("Running prediction engine tests...\n")
    test_predict_player_stat_basic()
    print("✅ Basic prediction test passed")

    test_predict_player_stat_no_line()
    print("✅ No-line prediction test passed")

    test_predict_over_recommendation()
    print("✅ OVER recommendation test passed")

    test_predict_under_recommendation()
    print("✅ UNDER recommendation test passed")

    test_predict_missing_data()
    print("✅ Missing data test passed")

    test_rolling_averages()
    print("✅ Rolling averages test passed")

    test_trend_calculation()
    print("✅ Trend calculation test passed")

    test_consistency_calculation()
    print("✅ Consistency calculation test passed")

    test_parse_minutes()
    print("✅ Parse minutes test passed")

    print("\n🎉 All tests passed!")
