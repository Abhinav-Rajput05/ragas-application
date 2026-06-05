"""
RAG Doctor — Gradio Application
"""

import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"
os.makedirs("./data/uploads", exist_ok = True)
os.makedirs("./data/chroma", exist_ok = True)
os.makedirs("./logs", exist_ok = True)

import gradio as gr
from ui.tabs.upload_tab import build_upload_tab
from ui.tabs.evaluation_tab import build_evaluation_tab
from ui.tabs.diagnosis_tab import build_diagnosis_tab
from ui.tabs.optimization_tab import build_optimization_tab
from ui.tabs.report_tab import build_report_tab

CSS = """
/* ── Base ── */
*, *::before, *::after { box-sizing: border-box; }
body, .gradio-container {
    background: #0f1117 !important;
    color: #c9d1d9 !important;
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
}

/* ── Tab bar ── */
.gradio-container .tabs > .tab-nav {
    background: #161b22 !important;
    border-bottom: 1px solid #30363d !important;
    gap: 4px !important;
    padding: 6px 12px 0 !important;
}
.gradio-container .tabs > .tab-nav button {
    background: transparent !important;
    color: #8b949e !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    padding: 8px 16px !important;
    transition: all 0.15s !important;
}
.gradio-container .tabs > .tab-nav button:hover {
    color: #c9d1d9 !important;
}
.gradio-container .tabs > .tab-nav button.selected {
    color: #58a6ff !important;
    border-bottom: 2px solid #58a6ff !important;
    background: transparent !important;
    font-weight: 600 !important;
}

/* ── Buttons ── */
.gradio-container button.primary {
    background: #1f6feb !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 10px 20px !important;
    font-size: 13px !important;
    transition: background 0.15s !important;
}
.gradio-container button.primary:hover {
    background: #388bfd !important;
}
.gradio-container button.secondary {
    background: #21262d !important;
    color: #c9d1d9 !important;
    border: 1px solid #30363d !important;
    border-radius: 6px !important;
}

/* ── Blocks / Cards ── */
.gradio-container .block {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
}

/* ── Inputs ── */
.gradio-container input,
.gradio-container textarea,
.gradio-container select {
    background: #0d1117 !important;
    color: #c9d1d9 !important;
    border: 1px solid #30363d !important;
    border-radius: 6px !important;
}
.gradio-container input:focus,
.gradio-container textarea:focus {
    border-color: #58a6ff !important;
    outline: none !important;
}

/* ── Labels ── */
.gradio-container label span,
.gradio-container .label-wrap span {
    color: #8b949e !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}

/* ── Markdown ── */
.gradio-container .markdown-body { color: #c9d1d9 !important; }
.gradio-container .markdown-body h1,
.gradio-container .markdown-body h2,
.gradio-container .markdown-body h3 { color: #58a6ff !important; font-weight: 600 !important; }
.gradio-container .markdown-body code {
    background: #0d1117 !important;
    color: #79c0ff !important;
    border: 1px solid #30363d !important;
    border-radius: 4px !important;
    padding: 1px 5px !important;
}
.gradio-container .markdown-body table th {
    background: #21262d !important;
    color: #8b949e !important;
    border: 1px solid #30363d !important;
}
.gradio-container .markdown-body table td {
    color: #c9d1d9 !important;
    border: 1px solid #21262d !important;
}

/* ── Slider ── */
.gradio-container input[type=range] { accent-color: #58a6ff !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #484f58; }

footer { display: none !important; }
"""

HEADER = """
<div style="
    background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 20px 28px;
    margin-bottom: 4px;
    display: flex;
    align-items: center;
    gap: 20px;
">
    <div style="
        width: 48px; height: 48px;
        background: #1f6feb;
        border-radius: 10px;
        display: flex; align-items: center; justify-content: center;
        font-size: 22px; font-weight: 900; color: white; font-family: monospace;
        flex-shrink: 0;
    ">Rx</div>
    <div>
        <div style="font-size: 20px; font-weight: 700; color: #e6edf3;">RAG Doctor</div>
        <div style="font-size: 12px; color: #6e7681; margin-top: 2px;">
            Evaluate · Diagnose · Optimize your RAG pipeline automatically
        </div>
    </div>
    <div style="margin-left: auto; display: flex; gap: 28px; text-align: center;">
        <div>
            <div style="font-size: 18px; font-weight: 700; color: #58a6ff;">5</div>
            <div style="font-size: 10px; color: #6e7681; margin-top: 2px;">RAGAS metrics</div>
        </div>
        <div>
            <div style="font-size: 18px; font-weight: 700; color: #58a6ff;">5</div>
            <div style="font-size: 10px; color: #6e7681; margin-top: 2px;">AI agents</div>
        </div>
        <div>
            <div style="font-size: 18px; font-weight: 700; color: #58a6ff;">Auto</div>
            <div style="font-size: 10px; color: #6e7681; margin-top: 2px;">Optimization</div>
        </div>
        <div>
            <div style="font-size: 18px; font-weight: 700; color: #58a6ff;">Rx</div>
            <div style="font-size: 10px; color: #6e7681; margin-top: 2px;">Prescription</div>
        </div>
    </div>
</div>
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="RAG Doctor",
        css=CSS,
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.blue,
            neutral_hue=gr.themes.colors.slate,
        ),
    ) as app:
        gr.HTML(HEADER)

        pipeline_id_state = gr.State(value=None)
        doc_id_state = gr.State(value=None)
        file_path_state = gr.State(value=None)

        with gr.Tabs():
            build_upload_tab(pipeline_id_state, doc_id_state, file_path_state)
            build_evaluation_tab(pipeline_id_state, file_path_state)
            build_diagnosis_tab(pipeline_id_state)
            build_optimization_tab(pipeline_id_state, file_path_state)
            build_report_tab(pipeline_id_state)

        return app


if __name__ == "__main__":
    app = build_app()
    app.queue()
    app.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        show_error=True,
        inbrowser=True,
    )
