"""
Shared UI helpers — blue/slate theme.
Colors:
  accent   #58a6ff  (blue)
  success  #3fb950  (muted green — only for pass)
  warning  #d29922  (amber)
  danger   #f85149  (red)
  bg0      #0d1117  (darkest)
  bg1      #161b22
  bg2      #21262d
  border   #30363d
  text0    #e6edf3  (bright)
  text1    #c9d1d9  (normal)
  text2    #8b949e  (muted)
"""

from __future__ import annotations
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from core.models import (
    RAGASMetrics,
    BeforeAfterComparison,
    CostAccuracyReport,
    PrescriptionSheet,
    ProductionReadinessReport,
    HealthScore,
)


BG0, BG1, BG2 = "#0d1117", "#161b22", "#21262d"
BORDER = "#30363d"
TEXT0 = "#e6edf3"
TEXT1 = "#c9d1d9"
TEXT2 = "#8b949e"
ACCENT = "#58a6ff"
SUCCESS = "#3fb950"
WARNING = "#d29922"
DANGER = "#f85149"


def score_color(score: float) -> str:
    if score >= 80:
        return SUCCESS
    if score >= 60:
        return WARNING
    return DANGER


def metric_color(val: float) -> str:
    if val >= 0.70:
        return SUCCESS
    if val >= 0.45:
        return WARNING
    return DANGER


def health_score_html(health: HealthScore) -> str:
    color = score_color(health.score)
    return f"""
<div style="background:{BG1};border:1px solid {color};border-radius:10px;
            padding:24px;text-align:center">
  <div style="font-size:52px;font-weight:800;color:{color};line-height:1">
    {health.score:.0f}
  </div>
  <div style="font-size:16px;color:{TEXT2};margin-top:4px">/ 100</div>
  <div style="margin-top:10px">
    <span style="background:{BG2};color:{color};border:1px solid {color};
                 border-radius:20px;padding:4px 14px;font-size:13px;font-weight:600">
      Grade {health.grade.value} — {health.category.value}
    </span>
  </div>
</div>"""


def build_radar_chart(metrics: RAGASMetrics, title: str = "Metric Profile") -> go.Figure:
    cats = ["Faithfulness", "Answer Relevancy", "Context Recall", "Context Precision", "Answer Correctness"]
    vals = [
        metrics.faithfulness,
        metrics.answer_relevancy,
        metrics.context_recall,
        metrics.context_precision,
        metrics.answer_correctness,
    ]
    v2 = vals + [vals[0]]
    c2 = cats + [cats[0]]

    fig = go.Figure(go.Scatterpolar(
        r=v2, theta=c2, fill="toself",
        fillcolor="rgba(88,166,255,0.12)",
        line=dict(color=ACCENT, width=2),
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, range=[0, 1],
                gridcolor=BORDER, linecolor=BORDER,
                tickfont=dict(size=9, color=TEXT2),
            ),
            angularaxis=dict(linecolor=BORDER, gridcolor=BORDER),
            bgcolor=BG1,
        ),
        paper_bgcolor=BG0,
        font=dict(color=TEXT1, size=11),
        title=dict(text=title, font=dict(size=13, color=TEXT0)),
        showlegend=False,
        margin=dict(t=50, b=20, l=20, r=20),
    )
    return fig


def build_before_after_chart(c: BeforeAfterComparison) -> go.Figure:
    labels = ["Faithfulness", "Answer Relevancy", "Context Recall", "Context Precision", "Answer Correctness"]
    before = [
        c.before_metrics.faithfulness, c.before_metrics.answer_relevancy,
        c.before_metrics.context_recall, c.before_metrics.context_precision,
        c.before_metrics.answer_correctness,
    ]
    after = [
        c.after_metrics.faithfulness, c.after_metrics.answer_relevancy,
        c.after_metrics.context_recall, c.after_metrics.context_precision,
        c.after_metrics.answer_correctness,
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Before optimization", x=labels, y=before,
                         marker_color="#f85149", opacity=0.85))
    fig.add_trace(go.Bar(name="After optimization", x=labels, y=after,
                         marker_color="#58a6ff", opacity=0.85))
    fig.update_layout(
        barmode="group",
        paper_bgcolor=BG0, plot_bgcolor=BG1,
        font=dict(color=TEXT1),
        title=dict(text="Before vs After Optimization", font=dict(size=13, color=TEXT0)),
        yaxis=dict(range=[0, 1], gridcolor=BORDER, title="Score"),
        xaxis=dict(gridcolor=BORDER),
        legend=dict(bgcolor=BG2, bordercolor=BORDER, borderwidth=1),
        margin=dict(t=50, b=60, l=50, r=20),
    )
    return fig


def build_cost_accuracy_chart(r: CostAccuracyReport) -> go.Figure:
    if not r.points:
        fig = go.Figure()
        fig.update_layout(title="No data yet", paper_bgcolor=BG0)
        return fig

    df = pd.DataFrame([p.model_dump() for p in r.points])
    colors = [ACCENT if p else "#484f58" for p in df["is_pareto_optimal"]]
    sizes = [14 if p else 8 for p in df["is_pareto_optimal"]]

    fig = go.Figure(go.Scatter(
        x=df["token_cost"], y=df["health_score"],
        mode="markers+text",
        marker=dict(color=colors, size=sizes, line=dict(color=BG2, width=1)),
        text=df["config_id"],
        textposition="top center",
        textfont=dict(size=9, color=TEXT2),
        customdata=df[["embedding_model", "chunk_size", "top_k"]].values,
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Cost: $%{x:.4f}<br>Score: %{y:.1f}/100<br>"
            "Model: %{customdata[0]}<br>"
            "Chunk: %{customdata[1]} | Top-K: %{customdata[2]}"
            "<extra></extra>"
        ),
    ))
    fig.update_layout(
        paper_bgcolor=BG0, plot_bgcolor=BG1,
        font=dict(color=TEXT1),
        title=dict(text="Cost vs Accuracy  (blue = Pareto optimal)", font=dict(size=13, color=TEXT0)),
        xaxis=dict(title="Estimated cost per run (USD)", gridcolor=BORDER),
        yaxis=dict(title="Health Score (0–100)", gridcolor=BORDER),
        margin=dict(t=50, b=50, l=60, r=20),
    )
    return fig


def prescription_html(s: PrescriptionSheet) -> str:
    p_colors = {"P1": DANGER, "P2": WARNING, "P3": ACCENT}
    c_color = score_color(s.current_health_score)
    pr_color = score_color(s.projected_health_score)

    rows = ""
    for item in s.prescriptions:
        pc = p_colors.get(item.priority.value, TEXT2)
        sign = "+" if item.expected_gain >= 0 else ""
        rows += f"""
        <tr style="border-bottom:1px solid {BG2}">
          <td style="padding:10px 8px;white-space:nowrap">
            <span style="background:{pc}22;color:{pc};border:1px solid {pc}44;
                         padding:2px 8px;border-radius:4px;font-size:11px;font-weight:700">
              {item.priority.value}
            </span>
          </td>
          <td style="padding:10px 8px;color:{TEXT1}">{item.fix}</td>
          <td style="padding:10px 8px;color:{ACCENT};text-align:right;
                     font-family:monospace;white-space:nowrap">
            {sign}{item.expected_gain:.0f} {item.expected_metric}
          </td>
        </tr>"""

    return f"""
<div style="background:{BG1};border:1px solid {BORDER};border-radius:10px;
            padding:24px;font-family:'Inter',sans-serif">
  <div style="display:flex;align-items:center;gap:14px;margin-bottom:20px">
    <div style="width:40px;height:40px;background:#1f6feb;border-radius:8px;
                display:flex;align-items:center;justify-content:center;
                font-size:18px;font-weight:900;color:white;font-family:monospace">Rx</div>
    <div>
      <div style="color:{TEXT0};font-size:15px;font-weight:700">AI Prescription Sheet</div>
      <div style="color:{TEXT2};font-size:12px">
        {s.pipeline_name} · Evaluated {s.evaluated_year} · Navigate Labs Nexus
      </div>
    </div>
  </div>
  <table style="width:100%;border-collapse:collapse">
    <tr style="border-bottom:1px solid {BORDER}">
      <td colspan="2" style="padding:10px 8px;color:{TEXT2}">Current Health Score</td>
      <td style="padding:10px 8px;text-align:right;color:{c_color};font-weight:700">
        {s.current_health_score:.0f} / 100 — {s.current_category.value}
      </td>
    </tr>
    {rows}
    <tr style="border-top:1px solid {BORDER}">
      <td colspan="2" style="padding:12px 8px;color:{TEXT2}">
        Projected score after applying all fixes
      </td>
      <td style="padding:12px 8px;text-align:right;color:{pr_color};font-weight:700">
        {s.projected_health_score:.0f} / 100 — {s.projected_category.value}
      </td>
    </tr>
  </table>
</div>"""


def production_readiness_html(r: ProductionReadinessReport) -> str:
    v_colors = {"Ready": SUCCESS, "Needs Work": WARNING, "Not Ready": DANGER}
    v_color = v_colors.get(r.verdict.value, TEXT2)

    rows = ""
    for item in r.checklist:
        icon = "✓" if item.passed else "✗"
        i_color = SUCCESS if item.passed else DANGER
        rows += f"""
        <tr style="border-bottom:1px solid {BG2}">
          <td style="padding:8px;text-align:center">
            <span style="color:{i_color};font-weight:700">{icon}</span>
          </td>
          <td style="padding:8px;color:{TEXT1}">{item.dimension}</td>
          <td style="padding:8px;color:{TEXT2};font-family:monospace">{item.value}</td>
          <td style="padding:8px;color:{TEXT2};font-size:12px">{item.note}</td>
        </tr>"""

    notes = "".join(
        f"<li style='color:{TEXT2};margin:4px 0'>{n}</li>"
        for n in r.deployment_notes
    )

    return f"""
<div style="background:{BG1};border:1px solid {BORDER};border-radius:10px;padding:24px">
  <div style="display:flex;align-items:center;gap:12px;margin-bottom:18px">
    <div style="font-size:26px;font-weight:800;color:{v_color}">{r.verdict.value}</div>
    <div style="color:{TEXT2};font-size:13px">Production Readiness Verdict</div>
  </div>
  <table style="width:100%;border-collapse:collapse">
    <thead>
      <tr style="border-bottom:1px solid {BORDER}">
        <th style="padding:8px;color:{TEXT2};width:32px"></th>
        <th style="padding:8px;color:{TEXT2};text-align:left">Check</th>
        <th style="padding:8px;color:{TEXT2};text-align:left">Value</th>
        <th style="padding:8px;color:{TEXT2};text-align:left">Note</th>
      </tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>
  <div style="margin-top:16px;padding-top:14px;border-top:1px solid {BORDER}">
    <div style="color:{TEXT2};font-size:12px;margin-bottom:6px;font-weight:600;
                text-transform:uppercase;letter-spacing:0.5px">Deployment Notes</div>
    <ul style="margin:0;padding-left:18px">{notes}</ul>
  </div>
</div>"""
