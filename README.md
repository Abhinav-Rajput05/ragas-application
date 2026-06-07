# 🩺 RAG Doctor

> An Agentic AI System for Automated Evaluation, Diagnosis, and Optimization of Retrieval-Augmented Generation (RAG) Pipelines.

## 📌 Overview

RAG Doctor is an AI-powered platform designed to evaluate, diagnose, and optimize Retrieval-Augmented Generation (RAG) systems automatically.

Unlike traditional RAG applications that focus only on answering questions from documents, RAG Doctor acts as an **AI Engineer for RAG pipelines**, helping developers understand:

- Why a RAG system is failing
- Where retrieval bottlenecks exist
- Whether hallucinations are occurring
- How pipeline performance can be improved
- Whether the system is production-ready

The platform leverages Agentic AI workflows, automated evaluation, and optimization strategies to improve the quality and reliability of RAG systems.

---

## 🚀 Problem Statement

Modern RAG systems often suffer from:

- Poor retrieval quality
- Missing relevant context
- Hallucinated responses
- Low answer relevance
- Inefficient chunking strategies
- Lack of evaluation visibility

Developers typically rely on manual testing, which is time-consuming and inconsistent.

RAG Doctor automates this entire process.

---

## 🎯 Objectives

The platform aims to:

✅ Automatically evaluate RAG pipelines

✅ Detect retrieval and generation failures

✅ Identify root causes of poor performance

✅ Recommend optimization strategies

✅ Compare performance before and after improvements

✅ Generate deployment-readiness reports

---

# 🏗 System Architecture

```text
Document Upload
        │
        ▼
Document Processing
        │
        ▼
Chunking & Embedding
        │
        ▼
Vector Database (ChromaDB)
        │
        ▼
RAG Pipeline
        │
        ▼
Evaluation Agent
        │
        ▼
Diagnosis Agents
        │
        ▼
Optimization Agent
        │
        ▼
Report Generator Agent
```

---

# ✨ Features

## 📄 Document Upload

Supported Formats:

- PDF
- DOCX
- TXT

The system automatically extracts and prepares documents for indexing.

---

## 🤖 Automatic RAG Pipeline Creation

No manual setup required.

The platform automatically:

- Chunks documents
- Generates embeddings
- Stores vectors
- Configures retrieval
- Creates QA workflows

---

## 📝 Automatic Test Dataset Generation

AI-generated evaluation dataset including:

- Questions
- Ground Truth Answers

Example:

```text
Question:
What is Retrieval-Augmented Generation?

Ground Truth:
RAG combines retrieval with language generation
to improve factual accuracy.
```

---

## 📊 RAG Evaluation

The system evaluates performance using:

### RAGAS Metrics

- Faithfulness
- Answer Relevancy
- Context Recall

---

## 🩺 RAG Health Score

Generates an overall health score.

Example:

```text
Health Score: 87/100
Grade: A
Status: Production Ready
```

---

## 🔍 Multi-Level Diagnosis

### Retrieval Diagnosis

Detects:

- Missing chunks
- Incorrect retrieval
- Low recall

### Chunking Diagnosis

Detects:

- Chunk size issues
- Poor overlap settings

### Ranking Diagnosis

Detects:

- Relevant chunks ranked too low

### Context Utilization Diagnosis

Detects:

- Retrieved context ignored by model

### Hallucination Diagnosis

Detects:

- Unsupported claims
- Fabricated information

---

## 🚨 Hallucination Detector Agent

Analyzes:

- Question
- Retrieved Context
- Generated Answer

Outputs:

```text
Hallucination Risk:
Low / Medium / High
```

---

## 🔎 Retrieval Inspector Agent

Evaluates:

- Retrieval quality
- Context relevance
- Recall effectiveness

Outputs:

```text
Retrieval Score
Root Cause
Recommended Fix
```

---

## ⚙️ Auto-Optimization Agent

Automatically experiments with:

### Chunk Size

- 200
- 400
- 600
- 800

### Chunk Overlap

- 20
- 50
- 100

### Top-K Retrieval

- 3
- 5
- 10

### Embedding Models

- MiniLM
- BGE Small
- BGE Base

Outputs:

```text
Best Configuration
Performance Gain
Expected Improvement
```

---

## 📈 Before vs After Comparison

Displays performance improvements.

Example:

```text
Before: 68

After: 89

Improvement: +21
```

---

## 📋 AI Prescription Report

Doctor-style recommendations.

Example:

```text
Diagnosis:
Low Context Recall

Root Cause:
Chunk Size Too Small

Prescription:
Increase Chunk Size from 200 → 600

Expected Improvement:
+15% Recall

Priority:
High
```

---

## 🚦 Production Readiness Assessment

Final deployment assessment.

Categories:

- Not Ready
- Needs Optimization
- Production Ready

Based on:

- Health Score
- Hallucination Rate
- Retrieval Quality
- Faithfulness

---

# 🧠 Multi-Agent Architecture

### Agent 1 – Evaluation Agent

Responsible for:

- Question Generation
- Evaluation Execution

---

### Agent 2 – Retrieval Inspector Agent

Responsible for:

- Retrieval Analysis
- Bottleneck Detection

---

### Agent 3 – Hallucination Detector Agent

Responsible for:

- Unsupported Claim Detection
- Hallucination Analysis

---

### Agent 4 – Optimization Agent

Responsible for:

- Configuration Search
- Pipeline Improvement

---

### Agent 5 – Report Generator Agent

Responsible for:

- Final Report Generation
- Recommendations

---

# 🛠 Tech Stack

| Component | Technology |
|------------|------------|
| Frontend | Gradio |
| Backend | FastAPI |
| RAG Framework | LangChain |
| Vector Database | ChromaDB |
| Embeddings | Sentence Transformers |
| Evaluation | RAGAS |
| Agent Framework | LangGraph |
| LLM | Navigate Labs Nexus API |
| Visualization | Plotly |
| Deployment | Hugging Face |

---

# 📂 Project Structure

```text
RAG-Doctor/
│
├── api/
├── agents/
├── core/
├── services/
├── ui/
│   └── tabs/
│
├── data/
│   ├── uploads/
│   └── chroma/
│
├── logs/
│
├── main.py
├── gradio_app.py
├── requirements.txt
└── README.md
```

---

# ⚡ Installation

### Clone Repository

```bash
git clone https://github.com/your-username/rag-doctor.git

cd rag-doctor
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows

```bash
venv\Scripts\activate
```

Linux / Mac

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 Environment Variables

Create a `.env` file:

```env
NEXUS_API_KEY=your_api_key

NEXUS_BASE_URL=your_api_url

NEXUS_MODEL=gpt-4.1-nano

CHROMA_PERSIST_DIR=./data/chroma
```

---

# ▶️ Running the Backend

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API Documentation:

```text
http://localhost:8000/docs
```

---

# 🎨 Running the Frontend

```bash
python gradio_app.py
```

Access:

```text
http://localhost:7861
```

---

# 📊 Future Improvements

- Support multiple vector databases
- Advanced retrieval benchmarking
- Multi-document comparison
- Enterprise dashboard
- Cost-performance optimization
- LLM benchmarking support
- CI/CD integration

---

# 👥 Team

### Abhinav Rajput
AI/ML Engineer | Backend Developer

### Urvashi Pandey
AI/ML Developer

---

# 🙏 Acknowledgements

Developed during the AI Bootcamp organized by:

- Navigate Labs
- Acropolis Group of Institutions

Special thanks to:

- Dr. Namrata Tapaswi Ma'am
- Rishi Acharya Sir

for their guidance and support throughout the project.

---

## ⭐ RAG Doctor

**"Don't just build RAG systems. Diagnose, optimize, and make them production-ready."**
