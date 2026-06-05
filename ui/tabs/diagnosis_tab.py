"""Tab 3 — Diagnose"""
from __future__ import annotations
import gradio as gr
from api.endpoints.evaluate import _evaluation_store
from api.endpoints.diagnose import _diagnosis_store
from core.models import DiagnosisReport

BG1, BG2, BORDER = "#161b22", "#21262d", "#30363d"
TEXT0, TEXT1, TEXT2 = "#e6edf3", "#c9d1d9", "#8b949e"
ACCENT, SUCCESS, DANGER, WARNING = "#58a6ff", "#3fb950", "#f85149", "#d29922"


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


def _flag(b):
    return (
        f'<span style="color:{DANGER}">✗ detected</span>' if b
        else f'<span style="color:{SUCCESS}">✓ ok</span>'
    )


def _risk_color(v):
    return {"Low": SUCCESS, "Medium": WARNING, "High": DANGER}.get(v, TEXT2)


def run_diagnosis(pipeline_id):
    if not pipeline_id:
        yield _log(["⚠  No pipeline found. Upload a document first."]), ""
        return

    evaluation = _evaluation_store.get(pipeline_id)
    if not evaluation:
        yield _log(["⚠  No evaluation results found. Run the Evaluate tab first."]), ""
        return

    log = []
    log.append("→  Starting multi-layer pipeline diagnosis ...")
    yield _log(log), ""

    log.append("→  Retrieval Inspector: checking chunk quality, recall and ranking ...")
    yield _log(log), ""
    try:
        from agents.retrieval_inspector import run_retrieval_inspector
        chunking, retrieval, ranking, ctx_util = run_retrieval_inspector(evaluation)
        log.append("✓  Retrieval Inspector done")
        log.append(f"   Chunk size issue   : {'yes' if chunking.chunk_size_issue else 'no'}")
        log.append(f"   Overlap issue      : {'yes' if chunking.overlap_issue else 'no'}")
        log.append(f"   Missing chunks     : {'yes' if retrieval.missing_chunks else 'no'}")
        log.append(f"   Low context recall : {'yes' if retrieval.low_recall else 'no'}")
        log.append(f"   Ranking problem    : {'yes' if ranking.relevant_chunks_ranked_low else 'no'}")
        log.append(f"   LLM ignored context: {'yes' if ctx_util.llm_ignored_context else 'no'}")
        if chunking.details:
            log.append(f"   Note: {chunking.details[:120]}")
        if retrieval.details:
            log.append(f"   Note: {retrieval.details[:120]}")
        yield _log(log), ""
    except Exception as e:
        log.append(f"✗  Retrieval Inspector failed: {e}")
        yield _log(log), ""
        return

    log.append("→  Hallucination Detector: checking each answer claim against context ...")
    yield _log(log), ""
    try:
        from agents.hallucination_detector import run_hallucination_detector
        halluc = run_hallucination_detector(evaluation.per_query_results)
        log.append("✓  Hallucination Detector done")
        log.append(f"   Overall risk       : {halluc.risk.value}")
        log.append(f"   Hallucination rate : {halluc.hallucination_rate:.1%} of claims unsupported")
        log.append(f"   Unsupported claims : {len(halluc.unsupported_claims)} found")
        for c in halluc.unsupported_claims[:3]:
            log.append(f"   ⚠  '{c[:90]}{'...' if len(c) > 90 else ''}'")
        yield _log(log), ""
    except Exception as e:
        log.append(f"✗  Hallucination Detector failed: {e}")
        yield _log(log), ""
        return

    report = DiagnosisReport(
        pipeline_id=pipeline_id,
        chunking=chunking,
        retrieval=retrieval,
        ranking=ranking,
        context_utilization=ctx_util,
        hallucination=halluc,
    )
    _diagnosis_store[pipeline_id] = report
    rc = _risk_color(halluc.risk.value)

    log.append("")
    log.append("✓  Diagnosis complete. Go to the Optimize tab to auto-fix these issues.")
    yield _log(log), _diag_html(report, rc)


def _diag_html(r: DiagnosisReport, rc: str) -> str:
    ch, ret, rank, cu, h = r.chunking, r.retrieval, r.ranking, r.context_utilization, r.hallucination

    def card(title, rows_html):
        return (
            f'<div style="background:{BG2};border:1px solid {BORDER};border-radius:8px;padding:14px">'
            f'<div style="color:{TEXT2};font-size:11px;text-transform:uppercase;'
            f'letter-spacing:0.8px;font-weight:600;margin-bottom:10px">{title}</div>'
            f'{rows_html}</div>'
        )

    def row(label, flagged):
        return (
            f'<div style="display:flex;justify-content:space-between;'
            f'padding:4px 0;border-bottom:1px solid {BG1}">'
            f'<span style="color:{TEXT1};font-size:13px">{label}</span>'
            f'<span>{_flag(flagged)}</span></div>'
        )

    return f"""
<div style="background:{BG1};border:1px solid {BORDER};border-radius:10px;padding:22px">
  <div style="color:{TEXT0};font-size:15px;font-weight:700;margin-bottom:18px">
    Diagnosis Report
  </div>

  <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px">
    {card("Chunking",
        row("Chunk size too small or large", ch.chunk_size_issue) +
        row("Overlap causes duplicate context", ch.overlap_issue) +
        (f'<div style="color:{TEXT2};font-size:12px;margin-top:6px">{ch.details}</div>' if ch.details else "")
    )}
    {card("Retrieval",
        row("Needed chunks not retrieved (missing)", ret.missing_chunks) +
        row("Wrong chunks retrieved", ret.wrong_chunks) +
        row("Context recall is low", ret.low_recall) +
        (f'<div style="color:{TEXT2};font-size:12px;margin-top:6px">{ret.details}</div>' if ret.details else "")
    )}
    {card("Ranking",
        row("Relevant chunks ranked too low in results", rank.relevant_chunks_ranked_low) +
        (f'<div style="color:{TEXT2};font-size:12px;margin-top:6px">{rank.details}</div>' if rank.details else "")
    )}
    {card("Context Utilization",
        row("LLM ignored the retrieved context", cu.llm_ignored_context) +
        (f'<div style="color:{TEXT2};font-size:12px;margin-top:6px">{cu.details}</div>' if cu.details else "")
    )}
  </div>

  <div style="background:{BG2};border:2px solid {rc};border-radius:8px;padding:16px">
    <div style="color:{TEXT2};font-size:11px;text-transform:uppercase;
                letter-spacing:0.8px;font-weight:600;margin-bottom:12px">
      Hallucination Detection
    </div>
    <div style="display:flex;gap:32px;align-items:center;flex-wrap:wrap">
      <div>
        <span style="font-size:30px;font-weight:800;color:{rc}">{h.risk.value}</span>
        <span style="color:{TEXT2}"> risk</span>
      </div>
      <div style="color:{TEXT2}">
        Rate: <b style="color:{rc}">{h.hallucination_rate:.1%}</b> of answer claims
        are not supported by retrieved context
      </div>
      <div style="color:{TEXT2}">
        Unsupported claims found: <b style="color:{DANGER}">{len(h.unsupported_claims)}</b>
      </div>
    </div>
    {"<div style='margin-top:12px'>" +
      "".join(f'<div style="color:{DANGER};font-size:12px;padding:3px 0;'
              f'border-bottom:1px solid {BG1}">⚠  {c}</div>'
              for c in h.unsupported_claims[:5]) +
     "</div>" if h.unsupported_claims else ""}
  </div>
</div>"""


def build_diagnosis_tab(pipeline_id_state):
    with gr.Tab("Diagnose"):
        gr.HTML(f"""
        <div style="padding:16px 0 10px">
          <div style="color:{TEXT0};font-size:17px;font-weight:700">
            Diagnose pipeline failures
          </div>
          <div style="color:{TEXT2};font-size:13px;margin-top:4px">
            Checks 5 layers: chunking quality, retrieval gaps, ranking order,
            context utilization, and hallucination in generated answers.
          </div>
        </div>""")

        diag_btn = gr.Button("Run diagnosis", variant="primary")

        with gr.Row():
            with gr.Column(scale=1):
                log_panel = gr.HTML()
            with gr.Column(scale=2):
                diag_html = gr.HTML()

        diag_btn.click(
            fn=run_diagnosis,
            inputs=[pipeline_id_state],
            outputs=[log_panel, diag_html],
        )
