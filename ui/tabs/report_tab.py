"""Tab 5 — Report"""
from __future__ import annotations
import gradio as gr
from api.endpoints.evaluate import _evaluation_store
from api.endpoints.diagnose import _diagnosis_store
from api.endpoints.optimize import _optimization_store
from ui.components import prescription_html, production_readiness_html

BG1, BG2, BORDER = "#161b22", "#21262d", "#30363d"
TEXT0, TEXT1, TEXT2 = "#e6edf3", "#c9d1d9", "#8b949e"
ACCENT, SUCCESS, DANGER = "#58a6ff", "#3fb950", "#f85149"


def _log(lines):
    items = "".join(
        f'<div style="padding:3px 0;border-bottom:1px solid {BG2};'
        f'color:{TEXT1};font-size:13px">{l}</div>'
        for l in lines
    )
    return (
        f'<div style="background:{BG1};border:1px solid {BORDER};border-radius:8px;'
        f'padding:14px;font-family:monospace">{items}</div>'
    )


def generate_report(pipeline_id, pipeline_name):
    name = pipeline_name.strip() or "RAG Pipeline"

    if not pipeline_id:
        yield _log(["⚠  No pipeline. Complete all steps first."]), "", ""
        return

    evaluation = _evaluation_store.get(pipeline_id)
    diagnosis = _diagnosis_store.get(pipeline_id)
    optimization = _optimization_store.get(pipeline_id)

    missing = []
    if not evaluation:
        missing.append("Evaluation")
    if not diagnosis:
        missing.append("Diagnosis")
    if not optimization:
        missing.append("Optimization")

    if missing:
        yield _log([
            f"⚠  These steps are missing: {', '.join(missing)}",
            "   Complete them before generating the report.",
        ]), "", ""
        return

    log = []
    log.append(f"→  Generating AI Prescription Sheet for '{name}' ...")
    log.append("   The LLM will analyze all evaluation and diagnosis data ...")
    yield _log(log), "", ""

    try:
        from agents.report_generator import generate_prescription_sheet
        rx = generate_prescription_sheet(
            pipeline_id=pipeline_id, pipeline_name=name,
            evaluation=evaluation, diagnosis=diagnosis, optimization=optimization,
        )
        log.append(f"✓  Prescription Sheet ready — {len(rx.prescriptions)} fixes identified")
        for item in rx.prescriptions:
            sign = "+" if item.expected_gain >= 0 else ""
            log.append(
                f"   [{item.priority.value}] {item.fix[:70]}{'...' if len(item.fix) > 70 else ''}"
                f"  →  {sign}{item.expected_gain:.0f} {item.expected_metric}"
            )
        log.append(f"   Current score : {rx.current_health_score:.0f}/100")
        log.append(f"   Projected     : {rx.projected_health_score:.0f}/100")
        yield _log(log), "", ""
    except Exception as e:
        log.append(f"✗  Prescription Sheet failed: {e}")
        yield _log(log), "", ""
        return

    log.append("")
    log.append("→  Running Production Readiness Assessment (8 dimensions) ...")
    yield _log(log), "", ""

    try:
        from agents.report_generator import build_production_readiness
        prod = build_production_readiness(pipeline_id, evaluation, diagnosis)
        passed = sum(1 for c in prod.checklist if c.passed)
        total = len(prod.checklist)
        log.append(f"✓  Assessment complete — {passed}/{total} checks passed")
        log.append(f"   Verdict: {prod.verdict.value}")
        for c in prod.checklist:
            icon = "✓" if c.passed else "✗"
            log.append(f"   {icon}  {c.dimension}  (value: {c.value})")
        log.append("")
        log.append("✓  Report ready. Scroll down to see the full output.")
        yield _log(log), prescription_html(rx), production_readiness_html(prod)
    except Exception as e:
        log.append(f"✗  Readiness assessment failed: {e}")
        yield _log(log), "", ""


def build_report_tab(pipeline_id_state):
    with gr.Tab("Report"):
        gr.HTML(f"""
        <div style="padding:16px 0 10px">
          <div style="color:{TEXT0};font-size:17px;font-weight:700">
            AI Prescription Sheet &amp; Production Readiness
          </div>
          <div style="color:{TEXT2};font-size:13px;margin-top:4px">
            The LLM reads all evaluation and diagnosis data and writes a prioritized
            fix list (P1/P2/P3) with expected metric improvements.
            Also runs an 8-point production readiness checklist.
          </div>
        </div>""")

        with gr.Row():
            name_input = gr.Textbox(
                label="Pipeline name  (optional)",
                placeholder="e.g. Customer Support Bot",
                value="RAG Pipeline",
                scale=3,
            )
            report_btn = gr.Button("Generate report", variant="primary", scale=1)

        with gr.Row():
            with gr.Column(scale=1):
                log_panel = gr.HTML()

        gr.HTML(
            f'<div style="color:{TEXT0};font-size:14px;font-weight:700;'
            f'padding:14px 0 6px">Prescription Sheet (Rx)</div>'
        )
        rx_html = gr.HTML()

        gr.HTML(
            f'<div style="color:{TEXT0};font-size:14px;font-weight:700;'
            f'padding:14px 0 6px">Production Readiness Assessment</div>'
        )
        prod_html = gr.HTML()

        report_btn.click(
            fn=generate_report,
            inputs=[pipeline_id_state, name_input],
            outputs=[log_panel, rx_html, prod_html],
        )
