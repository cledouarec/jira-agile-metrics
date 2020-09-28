import logging
import dateutil.parser
import pandas as pd
from matplotlib import pyplot as plt

from ..calculator import Calculator
from ..utils import Chart, filter_by_window

logger = logging.getLogger(__name__)


class WasteCalculator(Calculator):
    """Calculate stories withdrawn, grouped by the time of withdrawal and
    stage prior to withdrawal.

    Chart is drawn in `waste_chart`, with title `waste_chart_title`, using
    tickets fetched with `waste_chart_query`, and grouped by month, limited to
    `waste_chart_window` months (if given).
    """

    def run(self):

        query = self.settings["waste_query"]
        if not query:
            logger.debug(
                "Not calculating waste chart data as no query specified"
            )
            return None

        backlog_column = self.settings["backlog_column"]
        done_column = self.settings["done_column"]

        cycle_lookup = {}
        for idx, cycle_step in enumerate(self.settings["cycle"]):
            for status in cycle_step["statuses"]:
                cycle_lookup[status.lower()] = dict(
                    index=idx,
                    name=cycle_step["name"],
                    type=cycle_step["type"],
                )

        columns = ["key", "last_status", "resolution", "withdrawn_date"]
        series = {
            "key": {"data": [], "dtype": "str"},
            "last_status": {"data": [], "dtype": "str"},
            "resolution": {"data": [], "dtype": "str"},
            "withdrawn_date": {"data": [], "dtype": "datetime64[ns]"},
        }

        for issue in self.query_manager.find_issues(query):
            # Assume all waste items are resolved somehow
            if not issue.fields.resolution:
                continue

            last_status = None
            status_changes = list(
                self.query_manager.iter_changes(issue, ["status"])
            )
            if len(status_changes) > 0:
                last_status = status_changes[-1].from_string

            if last_status is not None and last_status.lower() in cycle_lookup:
                last_status = cycle_lookup.get(last_status.lower())["name"]
            else:
                logger.warning(
                    "Issue %s transitioned from unknown JIRA status %s",
                    issue.key,
                    last_status,
                )

            # Skip if last_status was the backlog or done column
            # (not really withdrawn)
            if last_status in (backlog_column, done_column):
                continue

            series["key"]["data"].append(issue.key)
            series["last_status"]["data"].append(last_status)
            series["resolution"]["data"].append(issue.fields.resolution.name)
            series["withdrawn_date"]["data"].append(
                dateutil.parser.parse(issue.fields.resolutiondate)
            )

        data = {}
        for k, v in series.items():
            data[k] = pd.Series(v["data"], dtype=v["dtype"])

        return pd.DataFrame(data, columns=columns)

    def write(self):
        chart_data = self.get_result()
        if chart_data is None:
            return

        output_file = self.settings["waste_chart"]
        if not output_file:
            logger.debug("No output file specified for waste chart")
            return

        if len(chart_data.index) == 0:
            logger.warning("Cannot draw waste chart with zero items")
            return

        cycle_names = [s["name"] for s in self.settings["cycle"]]
        backlog_column = self.settings["backlog_column"]
        done_column = self.settings["done_column"]

        cycle_names.remove(backlog_column)
        cycle_names.remove(done_column)

        breakdown = filter_by_window(
            chart_data.pivot_table(
                index="withdrawn_date",
                columns="last_status",
                values="key",
                aggfunc="count",
            )
            .groupby(
                pd.Grouper(
                    freq=self.settings["waste_frequency"],
                    closed="left",
                    label="left",
                )
            )
            .sum()
            .reindex(cycle_names, axis=1),
            self.settings["waste_window"],
        )

        n_columns = len(breakdown.columns)
        if len(breakdown.index) == 0 or n_columns == 0:
            logger.warning("Cannot draw waste chart with zero items")
            return

        with Chart.use_palette(
            self.settings["waste_chart_palette"], n_columns
        ):
            fig, ax = plt.subplots()
            breakdown.plot.bar(ax=ax, stacked=True)

        if self.settings["waste_chart_title"]:
            ax.set_title(self.settings["waste_chart_title"])
        ax.set_xlabel("Month", labelpad=20)
        ax.set_ylabel("Number of items", labelpad=10)
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        labels = [d.strftime("%b %y") for d in breakdown.index]
        ax.set_xticklabels(labels, rotation=90, size="small")

        # Write file
        logger.info("Writing waste chart to %s", output_file)
        fig.savefig(output_file, bbox_inches="tight", dpi=300)
        plt.close(fig)
