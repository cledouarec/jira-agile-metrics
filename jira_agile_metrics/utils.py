import datetime
import os.path

import numpy as np
import pandas as pd
import pandas.tseries.offsets as offsets
import seaborn as sns


class StatusTypes:
    backlog = "backlog"
    accepted = "accepted"
    complete = "complete"


def extend_dict(d, e):
    r = d.copy()
    r.update(e)
    return r


def to_json_string(value):
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if value in (None, np.NaN, pd.NaT):
        return ""

    try:
        return str(value)
    except TypeError:
        return value


def get_extension(filename):
    return os.path.splitext(filename)[1].lower()


def to_days_since_epoch(d):
    return (d - datetime.date(1970, 1, 1)).days


class Chart:
    _current_palette = None

    @classmethod
    def set_style(cls, context=None, style=None, palette=None, despine=True):
        """Defines chart style to use. By default, it is optimized for display
        and printer, the `despine` value is used to remove the contour.
        """
        if context is None:
            context = "paper"
        if style is None:
            style = "darkgrid"
        if palette is not None and len(palette) == 1:
            palette = palette[0]
        cls._current_palette = palette
        sns.set(context=context, style=style, palette=cls._current_palette)
        if despine:
            sns.despine()

    @classmethod
    def use_palette(cls, palette=None, n_colors=None):
        """Defines the color palette to use and the number of colors in the
        palette and return it to use with `with`.
        """
        if palette is None:
            palette = cls._current_palette
        elif len(palette) == 1:
            palette = palette[0]
        return sns.color_palette(palette=palette, n_colors=n_colors)


def filter_by_columns(df, output_columns):
    """To restrict (and order) the value columns, pass a list of valid values
    as `output_columns`.
    """
    if output_columns:
        return df[[s for s in output_columns if s in df.columns]]
    return df


def filter_by_threshold(df, threshold):
    """To restrict (and order) the value columns, pass a threshold in percent
    to filter. All columns under the threshold will be put in `Others` column.
    """
    if threshold:
        total = df.sum(axis=1)
        threshold_mask = (df * 100.0 / total[1] < threshold).all()
        df["Others"] = df.loc[:, threshold_mask].sum(axis=1)
        threshold_mask["Others"] = False
        return df.loc[:, ~threshold_mask]
    return df


def filter_by_window(df, window):
    """To restrict to last N rows."""
    if window:
        return df[-window:]
    return df


def sort_colums_by_last_row(df, ascending=False):
    """Reorder columns based on values of last row."""
    if len(df.index) > 0 and len(df.columns) > 0:
        return df.sort_values(
            by=df.last_valid_index(), axis=1, ascending=ascending
        )
    return df


def breakdown_by_month(
    df, start_column, end_column, key_column, value_column, aggfunc="count"
):
    """If `df` is a DataFrame of items that are valid/active between the
    timestamps stored in `start_column` and `end_column`, and where each item
    is uniquely identified by `key_column` and has a categorical value in
    `value_column`, return a new DataFrame counting the number of items in
    each month broken down by each unique value in `value_column`.
    """

    def build_df(t):
        start_date = getattr(t, start_column)
        end_date = getattr(t, end_column)
        key = getattr(t, key_column)
        value = getattr(t, value_column)

        if end_date is pd.NaT:
            end_date = pd.Timestamp.today()

        first_month = (
            start_date.normalize().to_period("M").to_timestamp("D", "S")
        )
        last_month = end_date.normalize().to_period("M").to_timestamp("D", "S")

        index = pd.date_range(first_month, last_month, freq="MS")

        return pd.DataFrame(index=index, data=[[key]], columns=[value])

    return (
        pd.concat([build_df(t) for t in df.itertuples()], sort=True)
        .resample("MS")
        .agg(aggfunc)
    )


def breakdown_by_month_sum_days(
    df, start_column, end_column, value_column, aggfunc="sum"
):
    """If `df` is a DataFrame of items that are valid/active between the
    timestamps stored in `start_column` and `end_column`, and where each has a
    categorical value in `value_column`, return a new DataFrame summing the
    overlapping days of items in each month broken down by each unique value in
    `value_column`.
    """

    def build_df(t):
        start_date = getattr(t, start_column)
        end_date = getattr(t, end_column)
        value = getattr(t, value_column)

        if end_date is pd.NaT:
            end_date = pd.Timestamp.today()

        days_range = pd.date_range(start_date, end_date, freq="D")
        first_month = (
            start_date.normalize().to_period("M").to_timestamp("D", "S")
        )
        last_month = end_date.normalize().to_period("M").to_timestamp("D", "S")

        index = pd.date_range(first_month, last_month, freq="MS")

        return pd.DataFrame(
            index=index,
            data=[
                [
                    len(
                        pd.date_range(
                            month_start,
                            month_start + offsets.MonthEnd(1),
                            freq="D",
                        ).intersection(days_range)
                    )
                ]
                for month_start in index
            ],
            columns=[value],
        )

    return (
        pd.concat([build_df(t) for t in df.itertuples()], sort=True)
        .resample("MS")
        .agg(aggfunc)
    )


def to_bin(value, edges):
    """Pass a list of numbers in `edges` and return which of them `value` falls
    between. If < the first item, return (0, <first>). If > last item, return
    (<last>, None).
    """

    previous = 0
    for v in edges:
        if previous <= value <= v:
            return previous, v
        previous = v
    return previous, None
