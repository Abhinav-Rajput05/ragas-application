"""Tab 1 — Upload"""
from __future__ import annotations
import os
import uuid
from pathlib import Path

import gradio as gr

from core.exceptions import DocumentProcessingError
from services.rag_service import ingest_document

UPLOAD_DIR = "./data/uploads"
ALLOWED = {".pdf", ".txt", ".docx"}

BG1, BG2, BORDER = "#161b22", "#21262d", "#30363d"
TEXT0, TEXT1, TEXT2 = "#e6edf3", "#c9d1d9", "#8b949e"
SUCCESS = "#3fb950"


def _log_html(lines: list[str]) -> str:
    items = "".join(
        f'<div style="padding:3px 0;border-bottom:1px solid {BG2};color:{TEXT1};font-size:13px">{line}</div>'
        for line in lines
    )
    return (
        f'<div style="background:{BG1};border:1px solid {BORDER};border-radius:8px;'
        f'padding:14px;font-family:monospace">{items}</div>'
    )


def handle_upload(file_obj):
    if file_obj is None:
        yield _log_html(["⚠  No file selected. Please choose a PDF, TXT or DOCX file."]), ""
        return

    file_path = file_obj if isinstance(file_obj, str) else file_obj.name
    ext = Path(file_path).suffix.lower()

    if ext not in ALLOWED:
        yield _log_html([
            f"✗  File type '{ext}' is not supported.",
            "   Accepted formats: .pdf .txt .docx",
        ]), ""
        return

    filename = Path(file_path).name
    log: list[str] = [f"→  Reading '{filename}' ..."]
    yield _log_html(log), ""

    try:
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        dest = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex[:8]}_{filename}")
        with open(file_path, "rb") as source, open(dest, "wb") as dest_file:
            dest_file.write(source.read())

        log.append(f"✓  File saved to upload directory")
        log.append(f"→  Splitting document into chunks ...")
        yield _log_html(log), ""

        log.append(f"→  Generating embeddings (this takes ~20–40 s) ...")
        yield _log_html(log), ""

        uploaded_doc, pipeline_config = ingest_document(dest)

        log.append(
            f"✓  Chunked into {uploaded_doc.chunk_count} pieces "
            f"({uploaded_doc.page_count} page{'s' if uploaded_doc.page_count != 1 else ''})"
        )
        log.append(f"→  Indexing vectors into ChromaDB ...")
        yield _log_html(log), ""

        log.append(f"✓  Indexed with model '{pipeline_config.embedding_model}'")
        log.append(f"✓  Pipeline ready — ID: {pipeline_config.pipeline_id}")
        log.append("")
        log.append("   You can now go to the Evaluate tab.")
        yield _log_html(log), _result_html(uploaded_doc, pipeline_config)

        return pipeline_config.pipeline_id, uploaded_doc.doc_id, dest

    except DocumentProcessingError as exc:
        log.append(f"✗  Processing failed: {exc}")
        yield _log_html(log), ""
    except Exception as exc:
        log.append(f"✗  Unexpected error: {exc}")
        yield _log_html(log), ""


def _result_html(doc, cfg) -> str:
    rows = [
        ("File", doc.filename),
        ("Type", doc.file_type.upper()),
        ("Pages", str(doc.page_count)),
        ("Chunks created", str(doc.chunk_count)),
        ("Chunk size", f"{cfg.chunk_size} tokens"),
        ("Overlap", f"{cfg.chunk_overlap} tokens"),
        ("Embedding model", cfg.embedding_model),
        ("Pipeline ID", cfg.pipeline_id),
    ]
    trs = "".join(
        f'<tr style="border-bottom:1px solid {BG2}\">'
        f'<td style="padding:7px 10px;color:{TEXT2};width:160px\">{key}</td>'
        f'<td style="padding:7px 10px;color:{TEXT1};font-family:monospace\">{value}</td>'
        f'</tr>'
        for key, value in rows
    )
    return (
        f'<div style="background:{BG1};border:1px solid {BORDER};border-radius:8px;padding:20px\">'
        f'<div style="color:{SUCCESS};font-weight:700;margin-bottom:12px;font-size:14px\">'
        f'Document ingested successfully</div>'
        f'<table style="width:100%;border-collapse:collapse\">{trs}</table>'
        f'</div>'
    )


def handle_upload_states(file_obj):
    if file_obj is None:
        return None, None, None

    file_path = file_obj if isinstance(file_obj, str) else file_obj.name
    if Path(file_path).suffix.lower() not in ALLOWED:
        return None, None, None

    try:
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        dest = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex[:8]}_{Path(file_path).name}")
        with open(file_path, "rb") as source, open(dest, "wb") as dest_file:
            dest_file.write(source.read())

        _, cfg = ingest_document(dest)
        return cfg.pipeline_id, cfg.doc_id, dest
    except Exception:
        return None, None, None


def build_upload_tab(pipeline_id_state, doc_id_state, file_path_state):
    with gr.Tab("Upload"):
        gr.HTML(f"""
        <div style="padding:16px 0 10px">
          <div style="color:{TEXT0};font-size:17px;font-weight:700">Upload a document</div>
          <div style="color:{TEXT2};font-size:13px;margin-top:4px">
            Accepts PDF, TXT, DOCX. The system will chunk the text, embed it,
            and build a searchable RAG pipeline automatically.
          </div>
        </div>""")

        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(
                    label="Choose file  (PDF / TXT / DOCX)",
                    file_types=[".pdf", ".txt", ".docx"],
                    type="filepath",
                )
                upload_btn = gr.Button("Ingest document", variant="primary")

            with gr.Column(scale=2):
                log_box = gr.HTML(value="")
                result_box = gr.HTML(value="")

        upload_btn.click(
            fn=handle_upload,
            inputs=[file_input],
            outputs=[log_box, result_box],
        )
        upload_btn.click(
            fn=handle_upload_states,
            inputs=[file_input],
            outputs=[pipeline_id_state, doc_id_state, file_path_state],
        )
