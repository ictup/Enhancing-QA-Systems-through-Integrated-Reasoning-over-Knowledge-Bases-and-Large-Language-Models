# Enhancing QA Systems through Integrated Reasoning over Knowledge Bases and Large Language Models

A **research-grade**, reproducible system that fuses **structured knowledge graphs (KG)** with **LLM-based reasoning** to reduce hallucination, improve faithfulness, and deliver **auditable** answers in knowledge‑intensive domains (with a medical QA case study).

> **Origin & purpose.** This repository packages the methods of my thesis, *Enhancing Question Answering Systems through Integrated Reasoning over Knowledge Bases and Large Language Models*, into an engineering‑ready implementation: modular code, runnable scripts, evaluation notebooks, and guidance for adaption to your own domain KG. It is intended both for **academic reproduction** and **practitioner deployment**.

---

## 🔑 Highlights (What’s new / why it matters)

- **KG‑Augmented Tree‑of‑Thoughts (KG+ToT):** Unifies symbolic KG paths with neural ToT exploration. Improves **interpretability** and **factual faithfulness** versus RAG/CoT baselines.
- **Multi‑Agent QA (AutoGen‑style):** Role‑specialized agents (Domain Specialist → Reviewer/Critic → Coordinator) implement **review‑before‑synthesis** and safer decisions.
- **Evidence‑grounded prompting (“MindMap”):** Structures entities, relations, multi‑hop KG paths, and supporting passages into prompts that **force explicit evidence use**.
- **Comprehensive evaluation:** Combines **LLM ranking**, **BERTScore**, and **Self‑Check/NLI** to assess **semantic alignment, contradiction, and factual support**.
- **Research to engineering:** Clear repository layout, environment files, CLI entrypoints, smoke tests, and CITATION metadata for academic reuse.

---

## 🧭 Architecture at a glance

```mermaid
flowchart TD
    Q[User Question] --> R1[BM25 Text Retrieval]
    Q --> R2[KG Traversal (Neo4j)]
    R1 --> M[MindMap: Structure Evidence]
    R2 --> M
    M --> T[Tree-of-Thoughts Search]
    T --> V[Self-Consistency Voting]
    V --> A1[Agent: Specialist]
    A1 --> A2[Agent: Reviewer/Critic]
    A2 --> A3[Agent: Coordinator/Synthesizer]
    A3 --> C[Faithfulness Checks + Citations]
    C --> O[Final Answer + Evidence]
```

**Reasoning recipe**
1. **Retrieve** text passages (BM25) + **traverse** KG (neighbors/paths) for candidate facts.
2. **MindMap** consolidates entities, relations, paths, and supporting text into compact evidence blocks.
3. **ToT search** explores multiple reasoning branches; **Self‑Consistency** aggregates by consensus.
4. **Multi‑agent critique** challenges and corrects reasoning before final synthesis.
5. Optional **self‑check/NLI** and provenance‑linked answer formatting.

---

## 📂 Repository structure

```
enhancing-qa-kg-llm/
├── src/
│   ├── agents/               # Multi-agent orchestration & pipeline
│   ├── retrieval/            # BM25 & Neo4j utilities
│   ├── reasoning/            # MindMap / KG+ToT logic (mindmap.py)
│   ├── evaluation/           # Metric wrappers (placeholders + notebooks)
│   ├── finetuning/           # (stubs) LoRA/PEFT configs, scripts
│   └── utils/
├── scripts/                  # run_agent / build_kg / evaluate
├── notebooks/                # bert.ipynb, gptranking.ipynb
├── configs/                  # YAML configs (model, data, KG)—add yours
├── data/                     # Your datasets (kept out of VCS)
├── docs/                     # thesis.pdf, diagrams
├── tests/                    # smoke tests
├── requirements.txt
├── environment.yml
├── .env.example
├── CITATION.cff
├── LICENSE
└── README.md
```

---

## ⚡ Quick start

### 1) Installation
```bash
conda env create -f environment.yml
conda activate enhancing-qa-kg-llm
# optional: pip install -e .
```

### 2) Configuration
```bash
cp .env.example .env
```
Fill in:
- `OPENAI_API_KEY` (or set up your local LLM provider)
- `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` for your Neo4j KG
- (Optional) add YAML files under `configs/` to control retrievers, ToT search depth/beam, and agent prompts.

### 3) Minimal run
```bash
python scripts/run_agent.py
```
This invokes `src/agents/pipeline.py`. Start by wiring **retrieval → reasoning → generation → evaluation** to your needs.

---

## 🧱 Knowledge Graph (Neo4j) setup

- Stand up Neo4j (Desktop/Docker). Define a consistent schema, e.g.: `(:Entity {name})-[:REL]->(:Entity)`.
- Ingest your domain entities/relations; include provenance (source, date) if possible.
- Update `.env` with Neo4j credentials and use `src/retrieval/kg_neo4j.py` to query neighbors/paths.

> **Tip:** For sensitive domains (e.g., medicine), define relation types and directionality carefully (e.g., *inhibits*, *contraindicated_with*, *risk_of*), and maintain provenance to support audits.

---

## 🧠 Methodology (design choices)

### Retrieval: Text + KG
- **Text:** BM25 candidate passages (`src/retrieval/bm25.py`).
- **KG:** Multi‑hop neighbors / shortest paths via Neo4j client (`src/retrieval/kg_neo4j.py`).
- **Fusion:** Merge, deduplicate, and score by relevance and graph connectivity. Rank edges/paths for prompt budget.

### MindMap evidence graph
- Builds a compact structure: **entities**, **relations**, **multi‑hop paths**, **supporting text**.
- Emits structured prompt sections (e.g., *Key Entities*, *Relations*, *Path Evidence*, *Caveats*).
- Implemented in `src/reasoning/mindmap.py` (migrated from the original research script).

### Tree‑of‑Thoughts + Self‑Consistency
- ToT enumerates stepwise hypotheses; nodes expand with **KG‑aware** evidence use.
- **Self‑Consistency** samples multiple complete reasoning traces and votes for consensus → more robust than a single CoT trace.
- Beam/depth and stopping policies are configurable in `configs/` (add your YAML).

### Multi‑agent orchestration (AutoGen‑style)
- **Specialist → Reviewer/Critic → Coordinator** roles with explicit prompts and gating.
- Encourages **review‑before‑synthesis**, improving reliability and safety for high‑stakes QA.
- Pluggable guardrails & critique prompts for different domains.

### Prompting strategy
- Structure the prompt with **evidence first**, then **reasoning steps**, then **answer with citations**.
- Enforce “**do not exceed evidence**”; surface **uncertainty** explicitly when evidence is weak or conflicting.
- Add domain‑specific rules/disclaimers (e.g., clinical safety notices).

### Evaluation suite
- **LLM ranking** of candidate answers.
- **BERTScore** for semantic similarity to references (see `notebooks/bert.ipynb`).
- **Self‑Check/NLI** for contradictions and factual consistency flags.
- **Human review** to taxonomize residual errors (hallucination, omission, misattribution).

### Optional: LoRA/PEFT domain adaptation
- For open models, use **parameter‑efficient fine‑tuning** to specialize to your domain.
- Keep scripts/configs under `src/finetuning/` (stubs provided). Compare before/after faithfulness & recall.

---

## 🔁 Reproducing thesis‑style experiments

1. **KG & corpus:** Load your KG into Neo4j; assemble BM25 corpus.  
2. **Retrieval settings:** Tune BM25 top‑k, KG traversal depth, and path scoring.  
3. **Reasoning policy:** Enable ToT + Self‑Consistency; set beam/depth.  
4. **Agents:** Instantiate role prompts; configure *approve/request‑more‑evidence/reject* loop.  
5. **Evaluation:** Run notebooks for BERTScore; add LLM ranking & self‑check/NLI; sample human audits.  
6. **Baselines vs. full system:** Compare RAG‑only / CoT‑only vs. **KG+ToT+Agents**.  
7. **Report:** Aggregate metrics + qualitative case studies; link KG evidence paths in examples.

---

## 🧩 Extending to new domains

- **Ontology:** Replace schema & relation inventory to match your domain.  
- **Retrievers:** Add a vector retriever & reranker if needed; fuse with BM25.  
- **Prompts:** Encode domain language, safety constraints, and citation norms.  
- **Metrics:** Add task‑specific metrics (e.g., exact match, factual consistency probes).

---

## 🖥️ CLI & API examples

**CLI**
```bash
python scripts/run_agent.py      # pipeline entry
python scripts/build_kg.py       # adapt to your KG ingest
python scripts/evaluate.py       # evaluation harness (notebooks + wrappers)
```

**BM25 retrieval**
```python
from src.retrieval.bm25 import build_bm25, search
corpus = [
    "Aspirin inhibits platelet aggregation and may increase bleeding risk.",
    "Ibuprofen is a nonsteroidal anti-inflammatory drug.",
    "Proton pump inhibitors reduce gastric acid secretion."
]
bm25 = build_bm25(corpus)
print(search(bm25, query="Does aspirin raise bleeding risk?", k=2))
```

**Neo4j neighbors**
```python
from src.retrieval.kg_neo4j import Neo4jClient
import os
client = Neo4jClient(os.getenv("NEO4J_URI"), os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
print(client.neighbors("Aspirin"))
client.close()
```

**Pipeline skeleton**
```python
# src/agents/pipeline.py
def run_pipeline():
    # 1) text + KG retrieval
    # 2) build MindMap evidence graph
    # 3) ToT search + self-consistency
    # 4) multi-agent critique & synthesis
    # 5) return final answer + citations
    print("Pipeline entrypoint placeholder. Configure in scripts/run_agent.py")
```

---

## 📊 Results & discussion (summary)

- **KG+ToT+Agents** typically reduces unsupported claims vs. RAG/CoT baselines, and returns **auditable** answers with explicit KG evidence.  
- See `docs/thesis.pdf` for quantitative tables, ablations, and error taxonomy.

---

## ⚖️ Limitations & responsible use

- **Not medical advice:** Research output only; require clinician oversight in healthcare.  
- **KG coverage bias:** Missing/outdated nodes or edges may mislead reasoning.  
- **Model variance:** Different base LLMs/decoding policies can change behavior—validate across seeds.  
- **Metric gaps:** Automated metrics may miss domain correctness nuances—include expert review.

---

## 🛣️ Roadmap

- [ ] Full ToT search operator & caching  
- [ ] Vector retriever + reranker fusion  
- [ ] Self‑check/NLI gating in coordinator  
- [ ] LoRA/PEFT training scripts & example config  
- [ ] AutoGen prompts & safety guardrails (release)  
- [ ] More unit tests, CI, and Dockerfile

---

## 📖 Citation

If you use this repository, please cite:

```
Li, Haojie (2025).
Enhancing Question Answering Systems through Integrated Reasoning over Knowledge Bases and Large Language Models.
Chalmers University of Technology.
```
See `CITATION.cff` for structured metadata.

---

## 📜 License

**MIT** (see `LICENSE`). Verify third‑party datasets/models under their respective licenses.

---

## 🙏 Acknowledgements

Thanks to the open‑source communities around **Neo4j**, **LangChain**, **AutoGen**, and evaluation libraries. This repository packages original research work into a practical system for others to study, reuse, and build upon.
