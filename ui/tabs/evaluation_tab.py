"""Tab 2 — Evaluate"""
from __future__ import annotations

import gradio as gr

from api.endpoints.evaluate import _evaluation_store
from ui.components import health_score_html, build_radar_chart, metric_color
from agents.evaluation_agent import generate_eval_dataset
from services.evaluation_service import collect_rag_outputs_parallel, run_ragas_evaluation
from services.rag_service import get_pipeline_config, get_raw_pages
from core.exceptions import EvaluationError

BG1, BG2, BORDER = "#161b22", "#21262d", "#30363d"
TEXT0, TEXT1, TEXT2 = "#e6edf3", "#c9d1d9", "#8b949e"
ACCENT, SUCCESS, DANGER, WARNING = "#58a6ff", "#3fb950", "#f85149", "#d29922"


def _log(lines: list[str]) -> str:
    items = "".join(
        f'<div style="padding:3px 0;border-bottom:1px solid {BG2};'
        f'color:{TEXT1};font-size:13px">{line}</div>'
        for line in lines
    )
    return (
        f'<div style="background:{BG1};border:1px solid {BORDER};border-radius:8px;'
        f'padding:14px;font-family:monospace">{items}</div>'
    )


def _bar(val: float) -> str:
    color = metric_color(val)
    pct = int(val * 100)
    return (
        f'<div style="display:flex;align-items:center;gap:10px">'
        f'<div style="flex:1;background:{BG2};border-radius:4px;height:8px">'
        f'<div style="width:{pct}%;background:{color};height:8px;border-radius:4px"></div>'
        f'</div>'
        f'<span style="color:{color};font-family:monospace;width:44px;text-align:right">'
        f'{val:.3f}</span></div>'
    )


def run_evaluation(pipeline_id, file_path, num_questions):
    num_q = int(num_questions or 0)

    if not pipeline_id:
        yield _log(["⚠  No pipeline found. Upload a document first."]), "", None, "", ""
        return

    if not file_path:
        yield _log(["⚠  File path missing. Re-upload the document."]), "", None, "", ""
        return

    log: list[str] = []
    log.append("→  Loading document text from disk ...")
    yield _log(log), "", None, "", ""

    try:
        pages = get_raw_pages(file_path)
        doc_text = " ".join([page.page_content for page in pages])
        log.append(f"✓  Loaded {len(pages)} page(s)  ({len(doc_text):,} characters)")
        yield _log(log), "", None, "", ""
    except Exception as exc:
        log.append(f"✗  Could not read document: {exc}")
        yield _log(log), "", None, "", ""
        return

    log.append(f"→  Asking the LLM to create {num_q} evaluation questions ...")
    yield _log(log), "", None, "", ""

    try:
        dataset = generate_eval_dataset(pipeline_id, doc_text, num_q)
        log.append(f"✓  Generated {len(dataset.questions)} questions")
        for i, q in enumerate(dataset.questions):
            log.append(
                f"   Q{i+1}: {q.question[:80]}{'...' if len(q.question) > 80 else ''}"
            )
        yield _log(log), "", None, "", ""
    except Exception as exc:
        log.append(f"✗  Question generation failed: {exc}")
        yield _log(log), "", None, "", ""
        return

    log.append(f"→  Running {num_q} questions through the RAG pipeline in parallel ...")
    yield _log(log), "", None, "", ""

    try:
        per_query = collect_rag_outputs_parallel(
            pipeline_id, dataset.questions, max_workers=min(num_q, 8)
        )
        log.append(f"✓  Got answers and retrieved contexts for all {len(per_query)} questions")
        yield _log(log), "", None, "", ""
    except Exception as exc:
        log.append(f"✗  RAG pipeline query failed: {exc}")
        yield _log(log), "", None, "", ""
        return

    log.append("→  Scoring with RAGAS (5 metrics, all running at the same time) ...")
    log.append(
        "   Faithfulness · Answer Relevancy · Context Recall · Context Precision · Answer Correctness"
    )
    yield _log(log), "", None, "", ""

    try:
        config = get_pipeline_config(pipeline_id)
        result = run_ragas_evaluation(config, dataset.questions, per_query)
    except Exception as exc:
        log.append(f"✗  RAGAS scoring failed: {exc}")
        yield _log(log), "", None, "", ""
        return

    _evaluation_store[pipeline_id] = result
    m = result.aggregate_metrics
    h = result.health_score

    log.append("")
    log.append("✓  Evaluation complete")
    log.append(
        f"   Health Score   : {h.score:.1f}/100  (Grade {h.grade.value} — {h.category.value})"
    )
    log.append(f"   Faithfulness   : {m.faithfulness:.3f}")
    log.append(f"   Answer Relev.  : {m.answer_relevancy:.3f}")
    log.append(f"   Context Recall : {m.context_recall:.3f}")
    log.append(f"   Ctx Precision  : {m.context_precision:.3f}")
    log.append(f"   Ans. Correct.  : {m.answer_correctness:.3f}")
    log.append("")
    log.append("   Go to the Diagnose tab to find out why scores are low.")

    metric_rows = "".join(
        f'<tr style="border-bottom:1px solid {BG2}">'
        f'<td style="padding:8px;color:{TEXT2};width:160px">{label}</td>'
        f'<td style="padding:8px;min-width:200px">{_bar(val)}</td>'
        f'</tr>'
        for label, val in [
            ("Faithfulness", m.faithfulness),
            ("Answer Relevancy", m.answer_relevancy),
            ("Context Recall", m.context_recall),
            ("Context Precision", m.context_precision),
            ("Answer Correctness", m.answer_correctness),
        ]
    )

    metrics_html = (
        f'<div style="background:{BG1};border:1px solid {BORDER};border-radius:8px;padding:18px">'
        f'<div style="color:{TEXT0};font-weight:700;margin-bottom:12px">RAGAS Metrics</div>'
        f'<table style="width:100%;border-collapse:collapse">{metric_rows}</table>'
        f'<div style="color:{TEXT2};font-size:12px;margin-top:10px">'
        f'Questions evaluated: {len(result.per_query_results)}</div></div>'
    )

    q_rows = "".join(
        f'<tr style="border-bottom:1px solid {BG2}">'
        f'<td style="padding:8px;color:{TEXT2}">{i+1}</td>'
        f'<td style="padding:8px;color:{TEXT1};max-width:320px">'
        f'{pq.question[:80]}{"..." if len(pq.question) > 80 else ""}</td>'
        f'<td style="padding:8px;text-align:center;color:{metric_color(pq.faithfulness)}">'
        f'{pq.faithfulness:.2f}</td>'
        f'<td style="padding:8px;text-align:center;color:{metric_color(pq.context_recall)}">'
        f'{pq.context_recall:.2f}</td>'
        f'<td style="padding:8px;text-align:center;color:{metric_color(pq.answer_relevancy)}">'
        f'{pq.answer_relevancy:.2f}</td>'
        f'</tr>'
        for i, pq in enumerate(result.per_query_results)
    )

    per_q_html = (
        f'<div style="background:{BG1};border:1px solid {BORDER};border-radius:8px;'
        f'padding:18px;overflow-x:auto">'
        f'<div style="color:{TEXT0};font-weight:700;margin-bottom:12px">Per-Question Results</div>'
        f'<table style="width:100%;border-collapse:collapse">'
        f'<thead><tr style="border-bottom:1px solid {BORDER}">'
        f'<th style="padding:8px;color:{TEXT2};text-align:left">#</th>'
        f'<th style="padding:8px;color:{TEXT2};text-align:left">Question</th>'
        f'<th style="padding:8px;color:{TEXT2}">Faithful.</th>'
        f'<th style="padding:8px;color:{TEXT2}">Recall</th>'
        f'<th style="padding:8px;color:{TEXT2}">Relevancy</th>'
        f'</tr></thead>'
        f'<tbody>{q_rows}</tbody></table></div>'
    )

    yield _log(log), health_score_html(result.health_score), build_radar_chart(result.aggregate_metrics), metrics_html, per_q_html


def build_evaluation_tab(pipeline_id_state, file_path_state):
    with gr.Tab("Evaluate"):
        gr.HTML(f"""
        <div style="padding:16px 0 10px">
          <div style="color:{TEXT0};font-size:17px;font-weight:700">Evaluate the RAG pipeline</div>
          <div style="color:{TEXT2};font-size:13px;margin-top:4px">
            The system will generate test questions, run them through your pipeline,
            and score the results using 5 RAGAS metrics simultaneously.
          </div>
        </div>""")

        with gr.Row():
            with gr.Column(scale=1):
                num_q = gr.Slider(3, 10, value=5, step=1, label="How many test questions to generate")
                eval_btn = gr.Button("Run evaluation", variant="primary")

            with gr.Column(scale=2):
                log_panel = gr.HTML()
                health_html = gr.HTML()
                radar_plot = gr.Plot(label="Metric radar chart")
                metrics_html = gr.HTML()
                per_q_html = gr.HTML()

        eval_btn.click(
            fn=run_evaluation,
            inputs=[pipeline_id_state, file_path_state, num_q],
            outputs=[log_panel, health_html, radar_plot, metrics_html, per_q_html],
        )
