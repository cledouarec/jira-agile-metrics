import datetime
import numpy as np
import pandas as pd
import pytest

from .utils import (
    get_extension,
    to_json_string,
    to_days_since_epoch,
    extend_dict,
    filter_by_columns,
    breakdown_by_month,
    breakdown_by_month_sum_days,
    to_bin,
)


@pytest.fixture
def issues():
    return pd.DataFrame(
        [
            {
                "key": "ABC-1",
                "priority": "high",
                "start": pd.Timestamp(2018, 1, 1),
                "end": pd.Timestamp(2018, 3, 20),
            },
            {
                "key": "ABC-2",
                "priority": "med",
                "start": pd.Timestamp(2018, 1, 2),
                "end": pd.Timestamp(2018, 1, 20),
            },
            {
                "key": "ABC-3",
                "priority": "high",
                "start": pd.Timestamp(2018, 2, 3),
                "end": pd.Timestamp(2018, 3, 20),
            },
            {
                "key": "ABC-4",
                "priority": "med",
                "start": pd.Timestamp(2018, 1, 4),
                "end": pd.Timestamp(2018, 3, 20),
            },
            {
                "key": "ABC-5",
                "priority": "high",
                "start": pd.Timestamp(2018, 2, 5),
                "end": pd.Timestamp(2018, 2, 20),
            },
            {
                "key": "ABC-6",
                "priority": "med",
                "start": pd.Timestamp(2018, 3, 6),
                "end": pd.Timestamp(2018, 3, 20),
            },
        ],
        columns=["key", "priority", "start", "end"],
    )


def test_extend_dict():
    assert extend_dict({"one": 1}, {"two": 2}) == {"one": 1, "two": 2}


def test_get_extension():
    assert get_extension("foo.csv") == ".csv"
    assert get_extension("/path/to/foo.csv") == ".csv"
    assert get_extension("\\path\\to\\foo.csv") == ".csv"
    assert get_extension("foo") == ""
    assert get_extension("foo.CSV") == ".csv"


def test_to_json_string():
    assert to_json_string(1) == "1"
    assert to_json_string("foo") == "foo"
    assert to_json_string(None) == ""
    assert to_json_string(np.NaN) == ""
    assert to_json_string(pd.NaT) == ""
    assert to_json_string(pd.Timestamp(2018, 2, 1)) == "2018-02-01"


def test_to_days_since_epoch():
    assert to_days_since_epoch(datetime.date(1970, 1, 1)) == 0
    assert to_days_since_epoch(datetime.date(1970, 1, 15)) == 14


def test_filter_by_columns():
    df = pd.DataFrame(
        [
            {"high": 1, "med": 2, "low": 0},
            {"high": 3, "med": 1, "low": 2},
            {"high": 2, "med": 2, "low": 3},
        ],
        columns=["high", "med", "low"],
    )

    # Check without values, original data frame will be returned.
    result = filter_by_columns(df, None)
    assert result.equals(df)

    # Check with values, columns will be filtered and reordered
    result = filter_by_columns(df, ["med", "high"])
    assert list(result.columns) == ["med", "high"]
    assert result.to_dict("records") == [
        {"high": 1, "med": 2},
        {"high": 3, "med": 1},
        {"high": 2, "med": 2},
    ]


def test_breakdown_by_month(issues):
    breakdown = breakdown_by_month(issues, "start", "end", "key", "priority")
    assert list(breakdown.columns) == ["high", "med"]  # alphabetical

    assert list(breakdown.index) == [
        pd.Timestamp(2018, 1, 1),
        pd.Timestamp(2018, 2, 1),
        pd.Timestamp(2018, 3, 1),
    ]
    assert breakdown.to_dict("records") == [
        {"high": 1, "med": 2},
        {"high": 3, "med": 1},
        {"high": 2, "med": 2},
    ]


def test_breakdown_by_month_open_ended(issues):
    # Replace ABC-6 end date to None
    issues["end"][5] = None

    breakdown = breakdown_by_month(issues, "start", "end", "key", "priority")
    assert list(breakdown.columns) == ["high", "med"]  # alphabetical

    # Note: We will get columns until the current month; assume this test is
    # run from June onwards ;)
    assert list(breakdown.index)[:5] == [
        pd.Timestamp(2018, 1, 1),
        pd.Timestamp(2018, 2, 1),
        pd.Timestamp(2018, 3, 1),
        pd.Timestamp(2018, 4, 1),
        pd.Timestamp(2018, 5, 1),
    ]
    assert breakdown.to_dict("records")[:5] == [
        {"high": 1, "med": 2},
        {"high": 3, "med": 1},
        {"high": 2, "med": 2},
        {"high": 0, "med": 1},
        {"high": 0, "med": 1},
    ]


def test_breakdown_by_month_none_values(issues):
    # Replace all priorities to None
    issues["priority"] = None

    breakdown = breakdown_by_month(issues, "start", "end", "key", "priority")
    assert list(breakdown.columns) == [None]

    assert list(breakdown.index) == [
        pd.Timestamp(2018, 1, 1),
        pd.Timestamp(2018, 2, 1),
        pd.Timestamp(2018, 3, 1),
    ]
    assert breakdown.to_dict("records") == [{None: 3}, {None: 4}, {None: 4}]


def test_breakdown_by_month_sum_days(issues):
    breakdown = breakdown_by_month_sum_days(issues, "start", "end", "priority")
    assert list(breakdown.columns) == ["high", "med"]  # alphabetical

    assert list(breakdown.index) == [
        pd.Timestamp(2018, 1, 1),
        pd.Timestamp(2018, 2, 1),
        pd.Timestamp(2018, 3, 1),
    ]

    assert breakdown.to_dict("records") == [
        {"high": 31.0, "med": 47.0},
        {"high": 70.0, "med": 28.0},
        {"high": 40.0, "med": 35.0},
    ]


def test_breakdown_by_month_sum_day_open_ended(issues):
    # Replace ABC-6 end date to None
    issues["end"][5] = None

    breakdown = breakdown_by_month_sum_days(issues, "start", "end", "priority")
    assert list(breakdown.columns) == ["high", "med"]  # alphabetical

    # Note: We will get columns until the current month; assume this test is
    # run from June onwards ;)
    assert list(breakdown.index)[:5] == [
        pd.Timestamp(2018, 1, 1),
        pd.Timestamp(2018, 2, 1),
        pd.Timestamp(2018, 3, 1),
        pd.Timestamp(2018, 4, 1),
        pd.Timestamp(2018, 5, 1),
    ]
    assert breakdown.to_dict("records")[:5] == [
        {"high": 31.0, "med": 47.0},
        {"high": 70.0, "med": 28.0},
        {"high": 40.0, "med": 46.0},
        {"high": 0, "med": 30.0},
        {"high": 0, "med": 31.0},
    ]


def test_breakdown_by_month_sum_days_none_values(issues):
    # Replace all high priorities to None
    issues["priority"].mask(issues["priority"] == "high", None, inplace=True)

    breakdown = breakdown_by_month_sum_days(issues, "start", "end", "priority")
    assert list(breakdown.columns) == [None, "med"]

    assert list(breakdown.index) == [
        pd.Timestamp(2018, 1, 1),
        pd.Timestamp(2018, 2, 1),
        pd.Timestamp(2018, 3, 1),
    ]

    assert breakdown.to_dict("records") == [
        {None: 31.0, "med": 47.0},
        {None: 70.0, "med": 28.0},
        {None: 40.0, "med": 35.0},
    ]


def test_to_bin():
    assert to_bin(0, [10, 20, 30]) == (0, 10)
    assert to_bin(9, [10, 20, 30]) == (0, 10)
    assert to_bin(10, [10, 20, 30]) == (0, 10)

    assert to_bin(11, [10, 20, 30]) == (10, 20)
    assert to_bin(20, [10, 20, 30]) == (10, 20)

    assert to_bin(30, [10, 20, 30]) == (20, 30)

    assert to_bin(31, [10, 20, 30]) == (30, None)
