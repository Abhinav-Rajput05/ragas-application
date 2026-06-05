"""
Tab 4 — Optimize

Architecture for speed + correctness:
- RAG answer collection runs in parallel threads (safe — no async loops)
- RAGAS scoring runs OUTSIDE threads (avoids asyncio event loop conflict)
- 3 configs run in a batch: collect answers in parallel, then score all 3 sequentially
- Live log line per config as it finishes
"""
from __future__ import annotations
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product as iproduct

import gradio as gr

from api.endpoints.evaluate import _evaluation_store
from api.endpoints.optimize import _optimization_store, _cost_accuracy_store
from ui.components import build_before_after_chart, build_cost_accuracy_chart

BG1, BG2, BORDER = "#161b22", "#21262d", "#30363d"
TEXT0, TEXT1, TEXT2 = "#e6edf3", "#c9d1d9", "#8b949e"
ACCENT, SUCCESS, DANGER, WARNING = "#58a6ff", "#3fb950", "#f85149", "#d29922"

COST_IN, COST_OUT = 0.001, 0.002
BATCH_SIZE = 3


def _log(lines: list[str]) -> str:
    items = "".join(
        f'<div style="padding:3px 0;border-bottom:1px solid {BG2};'
        f'color:{TEXT1};font-size:13px">{l}</div>'
        for l in lines
    )
    return (
        f'<div style="background:{BG1};border:1px solid {BORDER};'
        f'border-radius:8px;padding:14px;font-family:monospace">{items}</div>'
    )


def _collect_answers_for_config(args: tuple) -> tuple:
    """
    Thread worker: build pipeline + collect RAG answers only.
    NO RAGAS inside the thread (avoids asyncio conflicts).
    Returns (cid, new_config, per_query_results) or (cid, None, None) on error.
    """
    (cid, chunk_size, overlap, top_k, emb,
     pid, doc_id, raw_pages, dataset) = args

    from services.rag_service import build_pipeline_for_config
    from services.evaluation_service import collect_rag_outputs_parallel

    try:
        cfg = build_pipeline_for_config(
            doc_id=doc_id, original_pipeline_id=pid,
            chunk_size=chunk_size, chunk_overlap=overlap,
            top_k=top_k, embedding_model=emb,
            raw_pages=raw_pages,
        )
        pq = collect_rag_outputs_parallel(
            cfg.pipeline_id, dataset.questions, max_workers=3
        )
        return cid, cfg, pq
    except Exception as ex:
        from utils.logger import logger
        logger.warning(f"Config {cid} answer collection failed: {ex}")
        return cid, None, None


def run_optimization(pipeline_id, file_path):
    if not pipeline_id:
        yield _log(["⚠  No pipeline. Upload and evaluate first."]), "", None, None
        return
    evaluation = _evaluation_store.get(pipeline_id)
    if not evaluation:
        yield _log(["⚠  No evaluation results. Run the Evaluate tab first."]), "", None, None
        return
    if not file_path:
        yield _log(["⚠  File path missing. Re-upload the document."]), "", None, None
        return

    from core.config import get_settings
    from services.rag_service import get_raw_pages
    from services.evaluation_service import run_ragas_evaluation
    from agents.evaluation_agent import generate_eval_dataset
    from agents.report_generator import build_before_after_comparison
    from core.models import OptimizationConfig, CostAccuracyPoint, OptimizationResult, CostAccuracyReport

    s = get_settings()
    baseline = evaluation.health_score.score
    base_cfg = evaluation.config
    log: list[str] = []

    log.append("→  Loading document pages ...")
    yield _log(log), "", None, None
    try:
        raw_pages = get_raw_pages(file_path)
        doc_text = " ".join(p.page_content for p in raw_pages)
        log.append(f"✓  {len(raw_pages)} page(s) loaded")
        yield _log(log), "", None, None
    except Exception as e:
        log.append(f"✗  Failed: {e}")
        yield _log(log), "", None, None
        return

    log.append("→  Generating 3 evaluation questions (shared across all configs) ...")
    yield _log(log), "", None, None
    try:
        dataset = generate_eval_dataset(pipeline_id, doc_text, num_questions=3)
        log.append("✓  3 questions ready")
        yield _log(log), "", None, None
    except Exception as e:
        log.append(f"✗  Failed: {e}")
        yield _log(log), "", None, None
        return

    grid = list(iproduct(
        s.opt_chunk_sizes,
        s.opt_chunk_overlaps[:2],
        s.opt_top_k_values[:2],
        s.opt_embedding_models[:2],
    ))[:s.max_optimization_configs]
    total = len(grid)

    jobs: list[tuple] = []
    job_meta: dict[str, dict] = {}
    for i, (cs, ov, k, em) in enumerate(grid):
        cid = f"cfg_{i+1:02d}"
        jobs.append((cid, cs, ov, k, em, pipeline_id, base_cfg.doc_id, raw_pages, dataset))
        job_meta[cid] = {"chunk_size": cs, "overlap": ov, "top_k": k,
                         "model": em.split("/")[-1]}

    log.append(f"→  Starting grid search — {total} configs, batch size {BATCH_SIZE}")
    log.append("   Step 1: collect answers in parallel threads")
    log.append("   Step 2: run RAGAS scoring (outside threads, avoids asyncio issues)")
    yield _log(log), "", None, None

    done_configs: list[OptimizationConfig] = []
    cost_pts: list[CostAccuracyPoint] = []
    done_count = 0
    best_score = 0.0

    for batch_start in range(0, total, BATCH_SIZE):
        batch = jobs[batch_start: batch_start + BATCH_SIZE]
        batch_label = f"batch {batch_start // BATCH_SIZE + 1}/{-(-total // BATCH_SIZE)}"

        log.append(f"→  {batch_label}: collecting answers for {len(batch)} configs in parallel ...")
        yield _log(log), "", None, None

        answered: list[tuple] = []
        with ThreadPoolExecutor(max_workers=len(batch)) as pool:
            futs = {pool.submit(_collect_answers_for_config, j): j[0] for j in batch}
            for fut in as_completed(futs):
                cid, cfg, pq = fut.result()
                answered.append((cid, cfg, pq))

        for cid, cfg, pq in answered:
            done_count += 1
            meta = job_meta[cid]

            if cfg is None or pq is None:
                log.append(
                    f"   {cid}  chunk={meta['chunk_size']} k={meta['top_k']} "
                    f"model={meta['model']}  →  ✗ failed  [{done_count}/{total}]"
                )
                yield _log(log), "", None, None
                continue

            t0 = time.time()
            try:
                res = run_ragas_evaluation(cfg, dataset.questions, pq)
                elapsed = time.time() - t0
                score = res.health_score.score
                cost = round(
                    (meta["chunk_size"] * meta["top_k"] * 3 / 1000) * COST_IN +
                    (200 * 3 / 1000) * COST_OUT, 4
                )
                best_score = max(best_score, score)

                done_configs.append(OptimizationConfig(
                    config_id=cid,
                    chunk_size=meta["chunk_size"],
                    chunk_overlap=meta["overlap"],
                    top_k=meta["top_k"],
                    embedding_model=cfg.embedding_model,
                    health_score=score,
                    metrics=res.aggregate_metrics,
                    token_cost=cost,
                    latency_ms=round(elapsed * 1000, 1),
                ))
                cost_pts.append(CostAccuracyPoint(
                    config_id=cid,
                    token_cost=cost,
                    latency_ms=round(elapsed * 1000, 1),
                    health_score=score,
                    embedding_model=cfg.embedding_model,
                    chunk_size=meta["chunk_size"],
                    top_k=meta["top_k"],
                ))
                log.append(
                    f"   {cid}  chunk={meta['chunk_size']} k={meta['top_k']} "
                    f"model={meta['model']}  →  score {score:.1f}  "
                    f"({elapsed:.0f}s)  [{done_count}/{total} done, best: {best_score:.1f}]"
                )
            except Exception as ex:
                log.append(
                    f"   {cid}  chunk={meta['chunk_size']} k={meta['top_k']} "
                    f"model={meta['model']}  →  ✗ RAGAS failed: {str(ex)[:60]}  "
                    f"[{done_count}/{total}]"
                )
            yield _log(log), "", None, None

    if not done_configs:
        log.append("✗  All configs failed. Check terminal logs for details.")
        yield _log(log), "", None, None
        return

    for p in cost_pts:
        p.is_pareto_optimal = not any(
            q.config_id != p.config_id
            and q.token_cost <= p.token_cost
            and q.health_score >= p.health_score
            and (q.token_cost < p.token_cost or q.health_score > p.health_score)
            for q in cost_pts
        )

    pareto = [p for p in cost_pts if p.is_pareto_optimal]
    rec = max(pareto, key=lambda p: p.health_score) if pareto else cost_pts[0]
    best = max(done_configs, key=lambda c: c.health_score)
    delta = round(best.health_score - baseline, 1)
    sign = "+" if delta >= 0 else ""
    dcol = SUCCESS if delta >= 0 else DANGER

    cost_report = CostAccuracyReport(
        pipeline_id=pipeline_id, points=cost_pts,
        recommended_config_id=rec.config_id,
        recommendation_reason=(
            f"Best accuracy ({rec.health_score:.1f}/100) at ${rec.token_cost:.4f}/run"
        ),
    )
    opt_result = OptimizationResult(
        pipeline_id=pipeline_id, tested_configs=done_configs,
        best_config=best, baseline_health_score=baseline,
        optimized_health_score=best.health_score, improvement=delta,
    )
    _optimization_store[pipeline_id] = opt_result
    _cost_accuracy_store[pipeline_id] = cost_report

    log.append("")
    log.append(f"✓  Grid search complete — {len(done_configs)}/{total} configs succeeded")
    log.append(f"   Baseline score  : {baseline:.1f}/100")
    log.append(f"   Best score      : {best.health_score:.1f}/100  ({sign}{delta} points)")
    log.append(f"   Best chunk size : {best.chunk_size} tokens")
    log.append(f"   Best top-K      : {best.top_k}")
    log.append(f"   Best embedding  : {best.embedding_model.split('/')[-1]}")
    log.append("")
    log.append("   Go to the Report tab to generate the AI Prescription Sheet.")
    yield _log(log), "", None, None

    comparison = build_before_after_comparison(
        pipeline_id, evaluation, best.metrics, best.health_score
    )
    ba = build_before_after_chart(comparison)
    ca = build_cost_accuracy_chart(cost_report)

    summary = f"""
<div style="background:{BG1};border:1px solid {BORDER};border-radius:10px;padding:22px">
  <div style="color:{TEXT0};font-size:15px;font-weight:700;margin-bottom:18px">
    Optimization Summary
  </div>
  <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:18px">
    <div style="background:{BG2};border-radius:8px;padding:14px;text-align:center">
      <div style="color:{TEXT2};font-size:11px;margin-bottom:4px">Before</div>
      <div style="color:{DANGER};font-size:28px;font-weight:800">{baseline:.1f}</div>
      <div style="color:{TEXT2};font-size:11px">/ 100</div>
    </div>
    <div style="background:{BG2};border-radius:8px;padding:14px;text-align:center">
      <div style="color:{TEXT2};font-size:11px;margin-bottom:4px">After</div>
      <div style="color:{SUCCESS};font-size:28px;font-weight:800">{best.health_score:.1f}</div>
      <div style="color:{TEXT2};font-size:11px">/ 100</div>
    </div>
    <div style="background:{BG2};border-radius:8px;padding:14px;text-align:center">
      <div style="color:{TEXT2};font-size:11px;margin-bottom:4px">Improvement</div>
      <div style="color:{dcol};font-size:28px;font-weight:800">{sign}{delta:.1f}</div>
      <div style="color:{TEXT2};font-size:11px">points</div>
    </div>
  </div>
  <div style="background:{BG2};border:1px solid {ACCENT}44;border-radius:8px;padding:14px">
    <div style="color:{ACCENT};font-size:12px;font-weight:700;margin-bottom:10px">
      Recommended configuration
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;
                color:{TEXT2};font-size:13px">
      <div>Chunk size: <b style="color:{TEXT1}">{best.chunk_size} tokens</b></div>
      <div>Overlap:    <b style="color:{TEXT1}">{best.chunk_overlap} tokens</b></div>
      <div>Top-K:      <b style="color:{TEXT1}">{best.top_k}</b></div>
      <div>Embedding:  <b style="color:{TEXT1}">{best.embedding_model.split('/')[-1]}</b></div>
      <div>Cost/run:   <b style="color:{TEXT1}">${best.token_cost:.4f}</b></div>
      <div>Latency:    <b style="color:{TEXT1}">{best.latency_ms:.0f} ms</b></div>
    </div>
  </div>
  <div style="margin-top:8px;color:{TEXT2};font-size:12px">
    Configs tested: {len(done_configs)}/{total}
  </div>
</div>"""

    yield _log(log), summary, ba, ca


def build_optimization_tab(pipeline_id_state, file_path_state):
    with gr.Tab("Optimize"):
        gr.HTML(f"""
        <div style="padding:16px 0 10px">
          <div style="color:{TEXT0};font-size:17px;font-weight:700">
            Auto-optimize the pipeline
          </div>
          <div style="color:{TEXT2};font-size:13px;margin-top:4px">
            Tests multiple chunk sizes, embedding models, and retrieval settings.
            Answers are collected in parallel. Scoring runs per batch.
            Shows cost vs accuracy trade-off on a Pareto chart.
          </div>
        </div>""")

        opt_btn = gr.Button("Run optimization", variant="primary")

        with gr.Row():
            with gr.Column(scale=1):
                log_panel = gr.HTML()
                summary_html = gr.HTML()
            with gr.Column(scale=2):
                ba_chart = gr.Plot(label="Before vs After metrics")
                ca_chart = gr.Plot(label="Cost vs Accuracy — Pareto frontier")

        opt_btn.click(
            fn=run_optimization,
            inputs=[pipeline_id_state, file_path_state],
            outputs=[log_panel, summary_html, ba_chart, ca_chart],
        )
