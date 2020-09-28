import logging
import dateutil.parser

import pandas as pd
import matplotlib.pyplot as plt

from ..calculator import Calculator
from ..utils import (
    breakdown_by_month,
    Chart,
    filter_by_columns,
    filter_by_threshold,
    filter_by_window,
    sort_colums_by_last_row,
)

logger = logging.getLogger(__name__)


class DefectsCalculator(Calculator):
    """Calculate defect concentration

    Queries JIRA with JQL set in `defects_query` and creates three stacked
    bar charts, presuming their file name values are set. Each shows the
    concentration of defects by month. The number of months to show can be
    limited with `defects_window`.

    - `defects_by_priority_chart`: Grouped by priority
      (`defects_priority_field`), optionally limited to a list of known values
      in order (`defects_priority_values`) and with title
      `defects_by_priority_chart_title`.
    - `defects_by_type_chart`: Grouped by type
      (`defects_type_field`), optionally limited to a list of known values
      in order (`defects_type_values`) and with title
      `defects_by_type_chart_title`.
    - `defects_by_environment_chart`: Grouped by environment
      (`defects_environment_field`), optionally limited to a list of known
      values in order (`defects_environment_values`) and with title
      `defects_by_environment_chart_title`.
    """

    def run(self):

        query = self.settings["defects_query"]

        # This calculation is expensive. Only run it if we have a query.
        if not query:
            logger.debug(
                "Not calculating defects chart data as no query specified"
            )
            return None

        # Get the fields
        priority_field = self.settings["defects_priority_field"]
        priority_field_id = (
            self.query_manager.field_name_to_id(priority_field)
            if priority_field
            else None
        )

        type_field = self.settings["defects_type_field"]
        type_field_id = (
            self.query_manager.field_name_to_id(type_field)
            if type_field
            else None
        )

        environment_field = self.settings["defects_environment_field"]
        environment_field_id = (
            self.query_manager.field_name_to_id(environment_field)
            if environment_field
            else None
        )

        # Build data frame
        columns = [
            "key",
            "priority",
            "type",
            "environment",
            "created",
            "resolved",
        ]
        series = {
            "key": {"data": [], "dtype": "str"},
            "priority": {"data": [], "dtype": "str"},
            "type": {"data": [], "dtype": "str"},
            "environment": {"data": [], "dtype": "str"},
            "created": {"data": [], "dtype": "datetime64[ns]"},
            "resolved": {"data": [], "dtype": "datetime64[ns]"},
        }

        for issue in self.query_manager.find_issues(query, expand=None):
            series["key"]["data"].append(issue.key)
            series["priority"]["data"].append(
                self.query_manager.resolve_field_value(
                    issue, priority_field_id
                )
                if priority_field
                else None
            )
            series["type"]["data"].append(
                self.query_manager.resolve_field_value(issue, type_field_id)
                if type_field
                else None
            )
            series["environment"]["data"].append(
                self.query_manager.resolve_field_value(
                    issue, environment_field_id
                )
                if environment_field
                else None
            )
            series["created"]["data"].append(
                dateutil.parser.parse(issue.fields.created)
            )
            series["resolved"]["data"].append(
                dateutil.parser.parse(issue.fields.resolutiondate)
                if issue.fields.resolutiondate
                else None
            )

        data = {}
        for k, v in series.items():
            data[k] = pd.Series(v["data"], dtype=v["dtype"])

        return pd.DataFrame(data, columns=columns)

    def write(self):
        chart_data = self.get_result()
        if chart_data is None:
            return

        if len(chart_data.index) == 0:
            logger.warning("Cannot draw defect charts with zero items")
            return

        if self.settings["defects_by_priority_chart"]:
            self.write_defects_by_priority_chart(
                chart_data, self.settings["defects_by_priority_chart"]
            )

        if self.settings["defects_by_type_chart"]:
            self.write_defects_by_type_chart(
                chart_data, self.settings["defects_by_type_chart"]
            )

        if self.settings["defects_by_environment_chart"]:
            self.write_defects_by_environment_chart(
                chart_data, self.settings["defects_by_environment_chart"]
            )

    def write_defects_by_priority_chart(self, chart_data, output_file):
        self._write_defects_by_field_chart(
            chart_data=chart_data,
            output_file=output_file,
            field="priority",
            title=self.settings["defects_by_priority_chart_title"],
            palette=self.settings["defects_by_priority_chart_palette"],
            field_values=self.settings["defects_priority_values"],
            window=self.settings["defects_window"],
            threshold=self.settings["defects_priority_threshold"],
            sort=False,
        )

    def write_defects_by_type_chart(self, chart_data, output_file):
        self._write_defects_by_field_chart(
            chart_data=chart_data,
            output_file=output_file,
            field="type",
            title=self.settings["defects_by_type_chart_title"],
            palette=self.settings["defects_by_type_chart_palette"],
            field_values=self.settings["defects_type_values"],
            window=self.settings["defects_window"],
            threshold=self.settings["defects_type_threshold"],
        )

    def write_defects_by_environment_chart(self, chart_data, output_file):
        self._write_defects_by_field_chart(
            chart_data=chart_data,
            output_file=output_file,
            field="environment",
            title=self.settings["defects_by_environment_chart_title"],
            palette=self.settings["defects_by_environment_chart_palette"],
            field_values=self.settings["defects_environment_values"],
            window=self.settings["defects_window"],
            threshold=self.settings["defects_environment_threshold"],
        )

    @staticmethod
    def _write_defects_by_field_chart(
        chart_data,
        output_file,
        field,
        title,
        palette,
        field_values,
        window,
        threshold,
        sort=True,
    ):
        breakdown = filter_by_window(
            filter_by_columns(
                breakdown_by_month(
                    chart_data, "created", "resolved", "key", field
                ),
                field_values,
            ),
            window,
        )
        if sort:
            breakdown = sort_colums_by_last_row(breakdown)
        breakdown = filter_by_threshold(breakdown, threshold)

        n_columns = len(breakdown.columns)
        if len(breakdown.index) == 0 or n_columns == 0:
            logger.warning(
                "Cannot draw defects by %s chart with zero items", field
            )
            return

        # Trick to avoid middle value in case of using diverging palette to
        # increase readability
        n_columns = n_columns + 1 if n_columns % 2 else n_columns
        with Chart.use_palette(palette, n_columns):
            fig, ax = plt.subplots()
            breakdown.plot.bar(ax=ax, stacked=True)

        if title:
            ax.set_title(title)
        ax.set_xlabel("Month", labelpad=20)
        ax.set_ylabel("Number of items", labelpad=10)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            reversed(handles),
            reversed(labels),
            loc="center left",
            bbox_to_anchor=(1, 0.5),
        )
        labels = [d.strftime("%b %y") for d in breakdown.index]
        ax.set_xticklabels(labels, rotation=90, size="small")

        # Write file
        logger.info("Writing defects by %s chart to %s", field, output_file)
        fig.savefig(output_file, bbox_inches="tight", dpi=300)
        plt.close(fig)
