from src.core.types import format_timestamp


def test_format_timestamp_carries_rounded_second_into_minutes():
    assert format_timestamp(59.9999) == "00:01:00.000"
