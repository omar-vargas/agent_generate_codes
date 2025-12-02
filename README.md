Perfecto — aquí tienes el **README.md completo**, limpio, organizado y listo para copiar y pegar directamente en tu repositorio.

---

# 📘 **README – QualiCode: AI-Assisted Qualitative Coding System**

## 🇬🇧 **English Version**

*(Spanish version included below)*

---

# **QualiCode: An AI-Assisted System for Iterative Qualitative Coding**

QualiCode is a FastAPI-based system that integrates **Large Language Models (LLMs)**, **semantic embeddings**, and **agent-based workflows (LangGraph)** to support and partially automate qualitative data analysis.
It assists researchers in **generating, refining, validating, labeling, clustering, and organizing qualitative codes** using an iterative Human-in-the-Loop (HITL) workflow.

This system was developed as part of a master's thesis evaluating the impact of AI agents on qualitative coding workflows.

---

## 🚀 **Core Features**

### ✔️ **1. Initial Code Generation**

* Generates qualitative codes from interviews, transcripts, or synthetic datasets.
* Supports:

  * Zero-shot generation (no hypotheses)
  * Hypothesis-guided generation (deductive)

### ✔️ **2. Iterative Code Refinement**

* Users can approve or reject codes.
* System produces new codes based on **feedback**.
* State is maintained using **LangGraph checkpointers**.

### ✔️ **3. Code Validation**

* The system justifies each code using textual evidence extracted from the dataset.
* Produces JSON objects:

  ```json
  {"codigo": "...", "justificacion": "...", "ejemplos": "..."}
  ```

### ✔️ **4. Segment-Level Labeling**

* Assigns approved codes to text paragraphs.
* Supports pagination and retrieves all codes used.

### ✔️ **5. Thematic Family Induction**

Two methods available:

#### **A. Traditional LLM-Based Grouping**

* The LLM clusters the codes into families based exclusively on semantic reasoning.

#### **B. Embedding-Based Clustering**

* Generates SentenceTransformer embeddings.
* Performs:

  * K-Means
  * DBSCAN
  * Automatic cluster selection using silhouette score
* The LLM names thematic families **based on cluster contents**.

### ✔️ **6. Cross-Method Comparison**

* Compares LLM-based vs. clustering-based families.
* Generates:

  * Tables
  * Graphics
  * PDF reports
  * JSON/HTML dashboards

### ✔️ **7. Export Tools**

* CSV export of coded segments
* Charts of families and metrics
* Complete dashboards (HTML/PDF)
* Family comparison reports

---

## 🧠 **Technologies Used**

| Component     | Technology                                  |
| ------------- | ------------------------------------------- |
| API Framework | FastAPI                                     |
| AI Models     | Azure OpenAI (GPT models)                   |
| State Machine | LangGraph                                   |
| Embeddings    | SentenceTransformer ALL-MiniLM-L6-v2        |
| Clustering    | K-Means, DBSCAN, Silhouette                 |
| Visualization | Matplotlib, Plotly                          |
| Export        | CSV, PDF (ReportLab), HTML dashboards       |
| Storage       | JSON session states + TXT consolidated docs |

---

## 🚧 **Project Structure**

```
/src
  ├── main.py                # Main FastAPI application
  ├── prompts/               # System prompts for agents
  ├── clustering/            # Embedding & clustering utilities
  ├── export/                # CSV, PDF, HTML exports
  ├── families/              # LLM + cluster family induction
  ├── sessions/              # JSON state per user session
  └── archivos_consolidados/ # User uploaded datasets

README.md
requirements.txt
```

---

## 🔧 **Environment Variables**

Create a `.env` file:

```
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-10-21
OPENAI_API_TYPE=azure
```

---

## ▶️ **Running the System**

### **1. Install dependencies**

```bash
pip install -r requirements.txt
```

### **2. Run FastAPI**

```bash
uvicorn main:app --reload --port 8000
```

### **3. Open the interactive API**

```
http://localhost:8000/docs
```

---

## 🧪 **Key Endpoints**

### **Code Generation**

```
POST /generar/
```

### **Code Validation**

```
POST /justificacion/
```

### **Labeling**

```
POST /etiquetar/
```

### **Thematic Families**

```
POST /familias/
```

### **Clustering + LLM Families**

```
POST /clustering_codigos/
POST /comparar_familias/
```

### **Exports**

```
GET /exportar_csv/{session_id}
GET /exportar_familias/{session_id}
GET /exportar_dashboard/{session_id}
```

---

## 📊 **Metrics Implemented**

### **ISC — Intra-code Semantic Coherence**

Mean similarity of codes to centroid.

### **RI — Redundancy Index**

Proportion of highly similar code pairs.

### **RR — Reduction Ratio**

Measures structural expansion or reduction in refinement.

### **SD — Semantic Drift**

Centroid shift between code sets.

### **FCS — Family Coherence Score**

Average intra-family similarity.

### **FSS — Family Separation Score**

Average inter-family centroid distance.

### **ESD — Embedding-Symbolic Divergence**

Agreement between LLM vs cluster family assignment.

---

## 🧩 **Human-in-the-Loop Workflow**

QualiCode preserves human control:

1. Researcher uploads corpus
2. System generates codes
3. Researcher approves/rejects
4. System proposes refined codes
5. Researcher approves
6. Families induced (two methods)
7. Researcher compares and selects final structure

This balances **automation** with **interpretive transparency**.

---

## 🔬 **Research Context**

This system was developed as part of a master’s thesis on:

* AI-assisted qualitative analysis
* Multi-agent LLM workflows
* Iterative stateful coding
* Semantic clustering
* Comparison of human vs automated thematic structures

---

# 🇪🇸 **Versión en Español**

---

# **QualiCode: Sistema de Codificación Cualitativa Asistido por IA**

*(Versión resumida en español – preguntar si deseas la versión completa traducida.)*

QualiCode es un sistema basado en FastAPI que combina modelos de lenguaje, embeddings semánticos y agentes iterativos para asistir el análisis cualitativo.

Incluye generación de códigos, refinamiento, etiquetado, creación de familias y comparación entre métodos tradicionales y clustering.

---

Si quieres **la versión completa en español**, dímelo y la genero en el mismo formato.

---

## ✔️ Fin del README.md

Si deseas que genere:

* El archivo **README.md** nuevamente como archivo descargable
* Un archivo ZIP completo
* Una versión corta
* Una versión para artículo académico
* Una versión para GitHub Pages

…solo dímelo.
