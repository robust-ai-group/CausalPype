from __future__ import annotations
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import base64
import io

import polars as pl
import numpy as np
import matplotlib.pyplot as plt



@dataclass
class ReportConfig:
    title: str = "Causal Analysis Report"
    author: str = ""
    include_plots: bool = True
    include_tables: bool = True
    include_metadata: bool = True
    plot_format: str = "png"
    plot_width: int = 10
    plot_height: int = 6
    max_table_rows: int = 20
    decimal_places: int = 4


class MarkdownReport:
    
    def __init__(self, config: Optional[ReportConfig] = None):
        self._config = config or ReportConfig()
        self._sections: List[str] = []
        self._plots_dir: Optional[Path] = None
    
    def add_title(self, title: Optional[str] = None) -> MarkdownReport:
        title = title or self._config.title
        self._sections.append(f"# {title}\n")
        
        if self._config.author:
            self._sections.append(f"**Author:** {self._config.author}\n")
        
        self._sections.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self._sections.append("---\n")
        
        return self
    
    def add_section(self, title: str, level: int = 2) -> MarkdownReport:
        prefix = "#" * level
        self._sections.append(f"\n{prefix} {title}\n")
        return self
    
    def add_text(self, text: str) -> MarkdownReport:
        self._sections.append(f"{text}\n")
        return self
    
    def add_table(
        self,
        data: Union[Dict[str, List], pl.DataFrame, List[Dict]],
        caption: Optional[str] = None
    ) -> MarkdownReport:
        
        if not self._config.include_tables:
            return self
        
        if isinstance(data, pl.DataFrame):
            df = data
        elif isinstance(data, dict):
            df = pl.DataFrame(data)
        elif isinstance(data, list):
            df = pl.DataFrame(data)
        else:
            return self
        
        if len(df) > self._config.max_table_rows:
            df = df.head(self._config.max_table_rows)
            truncated = True
        else:
            truncated = False
        
        headers = df.columns
        header_row = "| " + " | ".join(headers) + " |"
        separator = "| " + " | ".join(["---"] * len(headers)) + " |"
        
        rows = []
        for i in range(len(df)):
            row_values = []
            for col in headers:
                val = df[col][i]
                if isinstance(val, float):
                    row_values.append(f"{val:.{self._config.decimal_places}f}")
                else:
                    row_values.append(str(val))
            rows.append("| " + " | ".join(row_values) + " |")
        
        table_md = "\n".join([header_row, separator] + rows)
        
        if caption:
            self._sections.append(f"\n*{caption}*\n")
        
        self._sections.append(f"\n{table_md}\n")
        
        if truncated:
            self._sections.append(f"\n*Showing first {self._config.max_table_rows} rows*\n")
        
        return self
    
    def add_key_value_table(self, data: Dict[str, Any], caption: Optional[str] = None) -> MarkdownReport:
        rows = []
        for key, value in data.items():
            if isinstance(value, float):
                value = f"{value:.{self._config.decimal_places}f}"
            rows.append({"Metric": key, "Value": str(value)})
        
        return self.add_table(rows, caption)
    
    def add_plot(
        self,
        fig,
        filename: str,
        caption: Optional[str] = None,
        embed: bool = True
    ) -> MarkdownReport:
        
        if not self._config.include_plots:
            return self
        

        
        if embed:
            buf = io.BytesIO()
            fig.savefig(
                buf,
                format=self._config.plot_format,
                dpi=100,
                bbox_inches="tight"
            )
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode("utf-8")
            
            self._sections.append(f"\n![{caption or filename}](data:image/{self._config.plot_format};base64,{img_base64})\n")
        else:
            if self._plots_dir:
                plot_path = self._plots_dir / f"{filename}.{self._config.plot_format}"
                fig.savefig(plot_path, dpi=100, bbox_inches="tight")
                self._sections.append(f"\n![{caption or filename}]({plot_path})\n")
        
        if caption:
            self._sections.append(f"\n*{caption}*\n")
        
        plt.close(fig)
        
        return self
    
    def add_code(self, code: str, language: str = "") -> MarkdownReport:
        self._sections.append(f"\n```{language}\n{code}\n```\n")
        return self
    
    def add_quote(self, text: str) -> MarkdownReport:
        lines = text.split("\n")
        quoted = "\n".join([f"> {line}" for line in lines])
        self._sections.append(f"\n{quoted}\n")
        return self
    
    def add_divider(self) -> MarkdownReport:
        self._sections.append("\n---\n")
        return self
    
    def build(self) -> str:
        return "\n".join(self._sections)
    
    def save(self, path: Union[str, Path], plots_dir: Optional[Union[str, Path]] = None) -> None:
        path = Path(path)
        
        if plots_dir:
            self._plots_dir = Path(plots_dir)
            self._plots_dir.mkdir(parents=True, exist_ok=True)
        
        path.write_text(self.build())


class ReportGenerator:
    
    def __init__(self, config: Optional[ReportConfig] = None):
        self._config = config or ReportConfig()
    
    def generate(
        self,
        results: List[Any],
        output_path: Union[str, Path],
        title: Optional[str] = None
    ) -> None:
        
        report = MarkdownReport(self._config)
        report.add_title(title)
        
        for result in results:
            self._add_result_section(report, result)
        
        report.save(output_path)
    
    def _add_result_section(self, report: MarkdownReport, result: Any) -> None:
        if result.analysis_type == "root_cause":
            self._add_root_cause_section(report, result)
        elif result.analysis_type == "treatment_effect":
            self._add_treatment_effect_section(report, result)
        elif result.analysis_type == "what_if":
            self._add_what_if_section(report, result)
    
    def _add_root_cause_section(self, report: MarkdownReport, result) -> None:
        report.add_section("Root Cause Analysis")
        
        direction = "increased" if result.total_change > 0 else "decreased"
        pct = abs(result.total_change / result.baseline_mean * 100) if result.baseline_mean != 0 else 0
        
        report.add_text(
            f"**Target variable:** `{result.target}`\n\n"
            f"The target {direction} from **{result.baseline_mean:.{self._config.decimal_places}f}** "
            f"to **{result.comparison_mean:.{self._config.decimal_places}f}** "
            f"(change of **{result.total_change:+.{self._config.decimal_places}f}**, {pct:.1f}%)."
        )
        
        report.add_section("Contributions by Variable", level=3)
        
        sorted_contribs = sorted(
            result.contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        table_data = []
        for node, contrib in sorted_contribs:
            pct_of_change = abs(contrib / result.total_change * 100) if result.total_change != 0 else 0
            table_data.append({
                "Variable": node,
                "Contribution": contrib,
                "% of Total Change": f"{pct_of_change:.1f}%"
            })
        
        report.add_table(table_data, "Contribution of each variable to the observed change")
        
        if self._config.include_plots:
            fig = self._create_contribution_plot(result)
            if fig:
                report.add_plot(fig, "root_cause_contributions", "Contributions to change")
        
        if self._config.include_metadata and result.metadata:
            report.add_section("Metadata", level=3)
            report.add_key_value_table(result.metadata)
    
    def _add_treatment_effect_section(self, report: MarkdownReport, result) -> None:
        report.add_section("Treatment Effect Analysis")
        
        direction = "increases" if result.ate > 0 else "decreases"
        
        report.add_text(
            f"**Treatment:** `{result.treatment}`\n\n"
            f"**Outcome:** `{result.outcome}`\n\n"
            f"The treatment {direction} the outcome by **{abs(result.ate):.{self._config.decimal_places}f}** on average."
        )
        
        report.add_section("Results Summary", level=3)
        
        summary_data = {
            "Average Treatment Effect (ATE)": result.ate,
            "Treated Group Mean": result.treated_mean,
            "Control Group Mean": result.control_mean,
            "N (Treated)": result.n_treated,
            "N (Control)": result.n_control
        }
        report.add_key_value_table(summary_data)
        
        if self._config.include_plots:
            fig = self._create_treatment_effect_plot(result)
            if fig:
                report.add_plot(fig, "treatment_effect", "Treatment vs Control comparison")
        
        if self._config.include_metadata and result.metadata:
            report.add_section("Metadata", level=3)
            report.add_key_value_table(result.metadata)
    
    def _add_what_if_section(self, report: MarkdownReport, result) -> None:
        report.add_section("What-If Analysis")
        
        interventions_str = ", ".join([f"`{k}={v}`" for k, v in result.interventions.items()])
        report.add_text(f"**Interventions:** {interventions_str}")
        
        if result.samples is not None:
            report.add_text(f"**Samples generated:** {len(result.samples)}")
        
        report.add_section("Effects on Variables", level=3)
        
        table_data = []
        for var in result.intervention_means:
            if var not in result.interventions:
                baseline = result.baseline_means.get(var, 0)
                intervention = result.intervention_means[var]
                diff = intervention - baseline
                table_data.append({
                    "Variable": var,
                    "Baseline": baseline,
                    "After Intervention": intervention,
                    "Change": diff
                })
        
        if table_data:
            report.add_table(table_data, "Effect of interventions on non-intervened variables")
        
        if self._config.include_plots and table_data:
            fig = self._create_what_if_plot(result)
            if fig:
                report.add_plot(fig, "what_if_effects", "Baseline vs Intervention comparison")
        
        if self._config.include_metadata and result.metadata:
            report.add_section("Metadata", level=3)
            report.add_key_value_table(result.metadata)
    
    def _create_contribution_plot(self, result):
       
        sorted_contribs = sorted(
            result.contributions.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        nodes = [x[0] for x in sorted_contribs]
        values = [x[1] for x in sorted_contribs]
        colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in values]
        
        fig, ax = plt.subplots(figsize=(self._config.plot_width, self._config.plot_height))
        ax.barh(nodes, values, color=colors)
        ax.axvline(x=0, color="black", linewidth=0.5)
        ax.set_xlabel("Contribution")
        ax.set_title(f"Root Causes of {result.target} Change")
        ax.invert_yaxis()
        plt.tight_layout()
        
        return fig
    
    def _create_treatment_effect_plot(self, result):
        
        fig, ax = plt.subplots(figsize=(self._config.plot_width, self._config.plot_height))
        
        x = [0, 1]
        y = [result.control_mean, result.treated_mean]
        colors = ["#3498db", "#e74c3c"]
        labels = ["Control", "Treated"]
        
        bars = ax.bar(x, y, color=colors, width=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel(result.outcome)
        ax.set_title(f"Effect of {result.treatment} on {result.outcome}")
        
        for bar, val in zip(bars, y):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01 * max(y),
                f"{val:.{self._config.decimal_places}f}",
                ha="center",
                va="bottom"
            )
        
        plt.tight_layout()
        return fig
    
    def _create_what_if_plot(self, result):
        variables = []
        baseline_vals = []
        intervention_vals = []
        
        for var in result.intervention_means:
            if var not in result.interventions:
                variables.append(var)
                baseline_vals.append(result.baseline_means.get(var, 0))
                intervention_vals.append(result.intervention_means[var])
        
        if not variables:
            return None
        
        fig, ax = plt.subplots(figsize=(self._config.plot_width, self._config.plot_height))
        
        x = np.arange(len(variables))
        width = 0.35
        
        ax.bar(x - width/2, baseline_vals, width, label="Baseline", color="#3498db")
        ax.bar(x + width/2, intervention_vals, width, label="Intervention", color="#e74c3c")
        
        ax.set_xticks(x)
        ax.set_xticklabels(variables, rotation=45, ha="right")
        ax.set_ylabel("Value")
        ax.set_title("What-If: Baseline vs Intervention")
        ax.legend()
        
        plt.tight_layout()
        return fig