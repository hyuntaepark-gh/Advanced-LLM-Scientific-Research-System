# 🔬 Advanced LLM Scientific Research System: Hybrid Retrieval, RAG & Evidence-Grounded Literature Analysis

![Python](https://img.shields.io/badge/Python-Research%20Pipelines-3776AB?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-Scientific%20Literature-00897B)
![RAG](https://img.shields.io/badge/RAG-Evidence--Grounded-6A1B9A)
![Search](https://img.shields.io/badge/Retrieval-BM25%20%7C%20Dense%20%7C%20Hybrid-1565C0)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)
![SciFact](https://img.shields.io/badge/Benchmark-BEIR%20SciFact-455A64)
![Research](https://img.shields.io/badge/Research-Curated%20Literature%20Study-546E7A)

This project develops an **AI-assisted scientific literature research system** that combines information retrieval, natural language processing, retrieval-augmented generation (RAG), and evidence-aware analysis to support the discovery, comparison, and critical evaluation of research papers.

Rather than treating a generated summary as scientific evidence, the system separates **document retrieval, source-grounded synthesis, claim verification, and independent evaluation**. Its modular architecture supports lexical search, optional neural retrieval, topic analysis, research-paper comparison, citation analysis, and reproducible benchmark workflows.

The accompanying research case study examines **10 identifiable arXiv papers on time-series forecasting architectures**, comparing their methodological assumptions and design choices. One numerical comparison is sourced to a table in the original PatchTST paper; these are **author-reported figures, not independently reproduced model results**.

The repository contains an executable offline demonstration, a curated real-paper analysis, and an implementation for the full BEIR SciFact retrieval benchmark. **Official SciFact test measurements, official QASPER evaluation, paid LLM experiments, and independent human factuality scores have not yet been completed.**

The primary objective is to show **how a scientific research workflow can be engineered, evaluated, and audited without confusing retrieved text or generated claims with independently verified evidence**.

---

## 🔍 Project Overview

### Objectives

- Build a modular research-paper discovery system using lexical, semantic, and hybrid retrieval.
- Compare ranking strategies such as BM25, TF-IDF, title-weighted BM25, and reciprocal rank fusion (RRF).
- Support optional scientific embedding models including SPECTER2, BGE-M3, and Qwen3 embeddings.
- Explore topic modeling, clustering, methodological trends, and paper-to-paper comparisons.
- Connect retrieved papers to evidence-aware answers and citation audits.
- Identify unsupported claims, missing evidence, contradictory findings, and potentially misleading comparisons.
- Prepare reproducible evaluation pipelines for BEIR SciFact and exploratory QASPER processing.
- Maintain a transparent separation between software demonstrations, published evidence, and independently measured results.

### Research Questions

1. How can lexical and semantic retrieval be combined to identify scientific papers relevant to a research question?
2. How do retrieval methods differ in ranking behavior, and how should these differences be evaluated against relevance judgments?
3. Can topic models and scientific-text features help organize a collection of research papers without overstating the meaning of unsupervised clusters?
4. How do forecasting architectures differ in their treatment of temporal dependencies, variable interactions, decomposition, and prediction horizons?
5. How can a RAG system link a literature synthesis to specific retrieved documents and source passages?
6. What checks are needed to distinguish a valid citation identifier from a genuinely supported scientific claim?
7. How should retrieval effectiveness, answer correctness, citation support, latency, and statistical uncertainty be evaluated independently?
8. What evidence is necessary before presenting a research assistant as an empirically validated system?

### Datasets & Evidence Sources

| Source | Scope | Use in this project | Evidence status |
|---|---|---|---|
| Synthetic literature corpus | 20 artificial papers; 10 synthetic search queries | Offline functionality testing and notebook demonstrations | **Synthetic; not scientific evidence** |
| Curated arXiv literature | 10 identifiable forecasting papers, 2017–2024 | Architecture comparison, method coding, descriptive figures | **Real papers; curated narrative study** |
| PatchTST source paper | Page 2, Table 1 (Traffic, horizon 96) | One published numerical comparison | **Author-reported; not reproduced** |
| BEIR SciFact | 5,183 documents; 300 judged test claims | Official retrieval benchmark runner | **Implemented; official evaluation not executed** |
| QASPER | External scientific-paper QA dataset | Optional ingestion and exploratory answer processing | **Official scoring not completed** |
| Independent claim reviews | Human-labeled claim/evidence records | Designed factuality validation workflow | **Annotations not collected** |

**Primary sources:** [Curated arXiv records](data/literature/curated_real_papers.jsonl) · [PatchTST paper](https://arxiv.org/abs/2211.14730) · [BEIR benchmark](https://github.com/beir-cellar/beir) · [SciFact source dataset](https://allenai.org/data/scifact)

The 10-paper selection is **purposeful rather than systematic**. Its composition and publication-year counts do not estimate broader field-wide research trends.

---

## 🧠 Methodology

### 1. Scientific Literature Retrieval

The search pipeline ranks papers using document titles, abstracts, and available metadata.

| Method | Purpose | Availability |
|---|---|---|
| BM25 | Term-based relevance ranking | Core implementation |
| TF-IDF | Sparse vector-space baseline | Core implementation |
| Title-weighted BM25 | Emphasize terms appearing in article titles | Core implementation |
| Dense retrieval | Semantic matching through text embeddings | Optional model-dependent execution |
| Hybrid search / RRF | Combine complementary rankings | Core research workflow |
| Reranking and HyDE | Explore query expansion and second-stage ranking | Optional workflow |
| SPECTER2 / BGE-M3 / Qwen3 | Specialist embedding-model comparisons | Adapters; external weights required |

The official SciFact evaluation runner is configured to compare the **three lexical baselines** on the complete verified test corpus. Optional neural models are not represented as officially benchmarked in the supplied release.

### 2. Corpus Preparation & Data Quality

The research workflow supports:

- Document and metadata validation.
- Duplicate detection and DOI-aware deduplication for live arXiv collection.
- Publication-year and method-frequency summaries.
- Title, abstract, and document-field analysis.
- Structured records for datasets, models, metrics, and citation links.
- Explicit distinction between abstracts, PDF-derived passages, and unsupported metadata.

These checks improve traceability but do not themselves establish the factual accuracy of a paper's findings.

### 3. Topic Modeling & Scientific Text Intelligence

The integrated notebook includes exploratory analyses using:

- K-Means clustering and cluster diagnostics.
- Latent Dirichlet Allocation (LDA).
- Non-negative Matrix Factorization (NMF).
- Topic-model agreement and qualitative interpretation.
- Named-method, dataset, and metric term extraction.
- Method co-occurrence and publication-year visualizations.

Topic labels and extracted technical terms are treated as **exploratory descriptors**, not validated scientific taxonomies or proof that a model was actually evaluated in a paper.

### 4. Real-Paper Comparative Research

The curated literature analysis compares 10 forecasting architectures:

| Paper | Year | Method family | Main methodological emphasis |
|---|---:|---|---|
| [DeepAR](https://arxiv.org/abs/1704.04110) | 2017 | Recurrent | Probabilistic autoregressive forecasting across related series |
| [Temporal Fusion Transformer](https://arxiv.org/abs/1912.09363) | 2019 | Hybrid recurrent-attention | Variable selection, gating, and multi-horizon forecasting |
| [Informer](https://arxiv.org/abs/2012.07436) | 2020 | Transformer | ProbSparse attention and efficient long-sequence processing |
| [Autoformer](https://arxiv.org/abs/2106.13008) | 2021 | Transformer | Series decomposition and autocorrelation |
| [FEDformer](https://arxiv.org/abs/2201.12740) | 2022 | Transformer | Frequency-domain modeling and decomposition |
| [DLinear / LTSF-Linear](https://arxiv.org/abs/2205.13504) | 2022 | Linear | Simple forecasting baselines and complexity comparisons |
| [TimesNet](https://arxiv.org/abs/2210.02186) | 2022 | Convolution-oriented | Period-aware, two-dimensional time-series representation |
| [PatchTST](https://arxiv.org/abs/2211.14730) | 2022 | Transformer | Patch tokens and channel-independent processing |
| [iTransformer](https://arxiv.org/abs/2310.06625) | 2023 | Transformer | Variable-wise tokens and cross-variable attention |
| [TimeMixer](https://arxiv.org/abs/2405.14616) | 2024 | MLP | Multiscale decomposition and feature mixing |

The analysis considers **what each architecture assumes about the data**, not simply whether it uses a Transformer. Paper-level descriptions are based primarily on the publicly accessible abstracts; only the identified PatchTST table provides manually checked numerical PDF evidence in this release.

### 5. RAG & Evidence-Grounded Research Synthesis

The project includes a modular research-answering workflow with:

- Retrieval of candidate papers and, when available, full-text passages.
- Source-linked answer generation and structured research summaries.
- Citation ID checks and evidence-span review.
- Abstention or uncertainty reporting where evidence is insufficient.
- Optional LLM providers and prompt-comparison configurations.
- Optional NLI-based claim screening, subject to independent verification.

A correctly formatted citation **does not prove** that the cited source supports the associated claim. External LLM reasoning and independent human claim verification are not presented as completed evaluations.

### 6. Citation Networks, Contradictions & Research Gaps

The pipeline supports exploratory bibliometric and research-gap analysis, including:

- Citation-network construction from supported metadata.
- Research method and topic co-occurrence.
- Potentially conflicting statements and evidence gaps.
- Comparisons of model assumptions and reported experimental settings.
- Sensitivity to source coverage and missing passages.

Citation relationships are only as reliable as their provenance. Synthetic network examples are not evidence about the actual scientific citation graph.

### 7. Official Retrieval Evaluation Protocol

The BEIR SciFact evaluator is designed to:

1. Download or accept the official dataset archive.
2. Validate the upstream archive checksum and reject incomplete or mismatched inputs.
3. Confirm the full corpus, query records, and held-out test relevance labels.
4. Rank the same documents with BM25, TF-IDF, and title-weighted BM25.
5. Report nDCG@10, Recall@10, Precision@10, and MRR@10.
6. Produce machine-readable metrics, comparative statistics, and an execution-status report.

**Current official evaluation status: `BLOCKED_NOT_RUN`.** No official SciFact model scores are included in the project ZIP. The code has been tested with offline fixtures, but those tests are not a substitute for the official dataset.

### 8. Robustness, Reliability & Research Auditing

The project also provides facilities for:

- Controlled retrieval and prompting comparisons.
- Paired statistical analysis and uncertainty reporting when real observations exist.
- Claim/evidence review queues and annotation templates.
- Distinguishing factuality, answer quality, retrieval quality, and citation formatting.
- Research release checks that prevent synthetic scores from being labeled empirical findings.

The strict scientific release audit is separate from basic code tests and deliberately remains incomplete until its required real-world evidence is available.

---

## 📊 Key Insights

### 1. Forecasting Architectures Follow Different Design Assumptions

The curated papers do not form a simple progression from recurrent networks to Transformers. DeepAR emphasizes probabilistic recurrent learning; TFT combines recurrence with attention and variable selection; decomposition-based methods, patch-based representations, variable-wise tokens, and multiscale MLPs introduce different assumptions about temporal structure.

**Takeaway:** A meaningful architectural comparison must consider the forecasting task, prediction horizon, covariates, dataset, and computational budget—not just model family.

### 2. Model Complexity Is Not a Reliable Proxy for Performance

The inclusion of DLinear and TimeMixer alongside attention-heavy architectures demonstrates why **simple and non-Transformer baselines** are important in forecasting research. The curated abstracts motivate this comparison, but do not constitute a controlled leaderboard.

**Takeaway:** Performance claims require harmonized test conditions and reproduced evaluation protocols.

### 3. One Source-Verified Published Comparison Illustrates the Evidence Standard

The original [PatchTST paper](https://arxiv.org/pdf/2211.14730) reports the following multivariate **Traffic** forecasting results at **prediction horizon 96** (PDF page 2, Table 1):

| Model | Author-reported MSE ↓ | Provenance |
|---|---:|---|
| Supervised PatchTST (patched) | **0.367** | PatchTST source table |
| DLinear | 0.410 | Same source table |
| FEDformer | 0.597 | Same source table |

These values are taken from **one published paper**, not separately trained models in this repository. Differences in training protocol, context length, and other settings must be considered before making causal claims about architectural superiority.

### 4. Retrieval Scores Require Real Relevance Judgments

The offline notebook runs on 20 synthetic records and 10 synthetic queries, allowing retrieval and ranking logic to be tested. Those demonstration metrics describe only the artificial fixture.

The curated real-paper pilot also creates retrieval rankings, but it **does not include independently labeled relevance judgments**. Therefore its ranked results cannot be treated as measured retrieval accuracy.

**Takeaway:** Official SciFact results should be reported only after the full dataset, ground-truth qrels, and measured run artifacts have been validated.

### 5. Scientific Claims Need Stronger Evidence Than Citation Formatting

The pipeline distinguishes valid document references, relevant retrieved passages, and genuinely supported factual claims. A citation can point to an existing paper while still failing to substantiate the statement that cites it.

**Takeaway:** Evidence-grounded research assistants require independent passage-level verification and, for substantive scientific factuality claims, external review.

### 6. Reproducibility Includes Reporting What Has Not Been Run

The release separates the status of three fundamentally different activities:

| Activity | Current result |
|---|---|
| Software pipeline and integrated notebook | **Offline execution validated** |
| Curated real-paper literature study | **Completed within the stated limited scope** |
| Full official SciFact retrieval test | **Not executed on original data** |
| Official QASPER answer/evidence evaluation | **Not completed** |
| External LLM factuality and independent human review | **Not completed** |

The original v14 verification reported **82 passing offline tests and 60 successfully executed notebook code cells**. These are engineering checks, **not** measured scientific model accuracy.

---

## 📈 Data Visualization

The following figures are included in the project and linked using paths relative to this README. The first three use the **curated real-paper literature study**. The final two illustrate **synthetic exploratory analysis**.

### 1. Publication-Year Coverage of the Selected Papers

![Publication-year coverage for the 10 curated research papers](reports/generated/real_paper_pilot/publication_year_coverage.png)

Shows the years represented among the 10 purposefully selected forecasting papers. **This is not an estimate of publication growth in the entire research field.**

### 2. Architecture Family Distribution

![Architecture families among curated forecasting papers](reports/generated/real_paper_pilot/method_families.png)

Summarizes the practical model-family labels assigned to the selected papers. Hybrid models and methodological overlap limit the interpretation of a single category per paper.

### 3. Published PatchTST Traffic Forecasting Comparison

![Published Traffic horizon 96 MSE comparison from PatchTST paper](reports/generated/real_paper_pilot/published_patchtst_table1_mse.png)

Visualizes values reported by the **PatchTST authors**, as sourced from their original Table 1. It is not a newly trained model comparison.

### 4. Exploratory Topic-Model Comparison — Synthetic Demonstration

![Synthetic topic modeling comparison](reports/generated/figures/06_topic_model_comparison.png)

Shows how exploratory topic extraction can be displayed in the software demonstration. **Synthetic demo; not real-paper topic validation.**

### 5. Method and Paper Comparison Matrix — Synthetic Demonstration

![Synthetic paper-method matrix](reports/generated/figures/05_paper_method_matrix.png)

Illustrates the notebook's method-comparison visualization. **Synthetic demo; not a verified scientific methods census.**

Additional CSV outputs and research figures are stored in [`reports/generated/real_paper_pilot/`](reports/generated/real_paper_pilot/) and [`reports/generated/figures/`](reports/generated/figures/).

---

## 🛠 Tech Stack

**Programming & Analysis**
- Python, JupyterLab, NumPy, Pandas, SciPy.

**Information Retrieval & NLP**
- BM25, TF-IDF, title-weighted BM25, reciprocal rank fusion.
- Scikit-learn for classical ML, clustering, and topic analysis.
- Optional transformer and scientific embedding integrations (SPECTER2, BGE-M3, Qwen3, SciBERT, SPLADE, ColBERT).

**LLM & Research Workflows**
- Modular RAG and LLM-provider integration.
- Optional LiteLLM, LangGraph, and RAGAS adapters.
- Claim, citation, and evidence-review components.

**Visualization & Networks**
- Matplotlib, NetworkX.
- Optional Streamlit application and Plotly.

**Testing & Reproducibility**
- Pytest, environment verification, checksum validation, experiment manifests, execution-status files, and research release auditing.

**Important:** Optional libraries or model adapters indicate integration points, not proof that those external checkpoints or services were used in a completed evaluation.

---

## 🗂️ Repository Structure

```text
Advanced-LLM-Scientific-Research-System/
│
├── README.md
├── pyproject.toml
├── requirements.txt
├── PROJECT_STATUS.md
├── VALIDATION_REPORT.md
│
├── notebooks/
│   └── 00_Complete_Scientific_Research_Analysis.ipynb
│
├── src/
│   └── research_intelligence/
│       ├── retrieval.py
│       ├── model_adapters.py
│       ├── scientific_analysis.py
│       ├── evidence_research.py
│       ├── official_scifact.py
│       ├── qasper_real.py
│       ├── llm_verification.py
│       └── ...
│
├── scripts/
│   ├── verify_environment.py
│   ├── run_curated_literature_case_study.py
│   ├── run_official_evaluation.py
│   ├── execute_real_research.py
│   └── check_submission.py
│
├── data/
│   ├── demo/                      # Synthetic demonstration records
│   ├── literature/                # Curated real-paper references and source table
│   └── annotations/               # Templates; not completed human judgments
│
├── reports/
│   ├── SCIENTIFIC_CRITICAL_REVIEW_v14.md
│   ├── generated/
│   │   ├── real_paper_pilot/      # Curated study: tables, figures and report
│   │   └── figures/               # Synthetic software-demo figures
│   └── official_evaluation/
│       └── scifact/
│           └── EXECUTION_STATUS.json
│
├── app/
│   └── streamlit_app.py
├── configs/
├── docs/
├── tests/
└── .github/workflows/ci.yml
```

The repository keeps the analysis notebook separate from reusable Python modules. Generated source PDFs, downloaded benchmark corpora, API credentials, large model weights, and local caches are not intended for version control.

---

## 🚀 How to Run

### 1. Clone or Download the Full Repository

Download the complete project, including `src/`, `scripts/`, `notebooks/`, `data/`, and `pyproject.toml`. **Opening only a copied notebook can select the wrong local Python package version.**

### 2. Create a Python Environment

**Windows PowerShell (Python 3.11 recommended):**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[quality,notebooks,pdf]"
```

### 3. Verify the Installation and Run Tests

```powershell
.\.venv\Scripts\python.exe scripts/verify_environment.py
.\.venv\Scripts\python.exe -m pytest -q
```

The environment check detects common missing-file and module-version problems. Automated tests verify software behavior but do not substitute for real-data model validation.

### 4. Run the Curated Real-Paper Literature Study

```powershell
.\.venv\Scripts\python.exe scripts/run_curated_literature_case_study.py
```

This regenerates the literature comparison tables and real-paper pilot figures from the included curated records.

### 5. Open the Integrated Jupyter Notebook

```powershell
.\.venv\Scripts\python.exe -m jupyter lab notebooks/00_Complete_Scientific_Research_Analysis.ipynb
```

Select **Kernel → Restart Kernel and Run All Cells**. The default run uses the synthetic offline demonstration and the included curated real-paper pilot; it does not automatically download external datasets or call paid LLM APIs.

### 6. Run the Official SciFact Retrieval Benchmark

On a machine with Internet access, execute:

```powershell
.\.venv\Scripts\python.exe scripts/run_official_evaluation.py --download-scifact
```

The runner validates the source dataset before evaluating the three lexical baselines. Alternatively, provide an existing original archive:

```powershell
.\.venv\Scripts\python.exe scripts/run_official_evaluation.py --archive "C:\path\to\scifact.zip"
```

Inspect `reports/official_evaluation/scifact/EXECUTION_STATUS.json`. A completed official experiment must explicitly report **`COMPLETED_FULL_OFFICIAL_SCIFACT_TEST`** and include its measurement artifact. The supplied release instead records **`BLOCKED_NOT_RUN`**.

**Dataset:** [BEIR SciFact archive](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip). For the complete protocol, see [`docs/OFFICIAL_BENCHMARK_PROTOCOL.md`](docs/OFFICIAL_BENCHMARK_PROTOCOL.md).

### 7. Review Research Results and Release Status

The most useful reports are:

- [Curated real-paper case study](reports/generated/real_paper_pilot/REAL_PAPER_CASE_STUDY.md).
- [Scientific critical review](reports/SCIENTIFIC_CRITICAL_REVIEW_v14.md).
- [Software validation report](VALIDATION_REPORT.md).
- [Project and benchmark status](PROJECT_STATUS.md).

For the strict scientific-evidence audit, use:

```powershell
.\.venv\Scripts\python.exe scripts/check_submission.py --strict
```

This audit may **correctly fail** while official benchmarks, external LLM evaluation, or independent human evidence are missing. Do not override the failure simply to present a project as empirically validated.

**Windows troubleshooting:** [`docs/RUN_OFFICIAL_EVALUATION_WINDOWS_KO.md`](docs/RUN_OFFICIAL_EVALUATION_WINDOWS_KO.md).

---

## ⚠️ Limitations

- **Curated, non-systematic sample:** The 10-paper review is purposefully selected and does not support population-level claims about literature trends.
- **Abstract-focused evidence:** Method descriptions largely derive from public abstracts rather than a complete, independently extracted full-text corpus.
- **Published vs. reproduced results:** The PatchTST numerical example is author-reported; the forecasting architectures were not independently retrained here.
- **Cross-paper comparability:** Forecasting performance cannot be fairly compared without controlling datasets, horizons, metrics, input windows, splits, and training budgets.
- **Synthetic evaluation:** Offline retrieval scores and topic metrics describe artificial test fixtures, not real-paper search accuracy.
- **Unfinished official benchmark:** BEIR SciFact full-test metrics are not present in this release despite the implemented evaluator.
- **Unfinished answer evaluation:** Official QASPER answer/evidence scoring and independent human claim-level reviews have not been completed.
- **Unexecuted LLM comparisons:** Optional LLM providers, agents, rerankers, and scientific embedding adapters require models, credentials, resources, and real observations.
- **Citation validity:** A valid source identifier or supporting-looking passage does not independently prove a scientific claim.
- **Topic and network reliability:** Unsupervised topics and metadata-derived citation networks require interpretation and provenance checks.
- **External data access:** Live arXiv collection, source-PDF processing, and some benchmarks require downloads and appropriate data-use permissions.
- **Release readiness:** Passing code tests is not equivalent to satisfying strict empirical research evidence gates.

Future work includes a completed official SciFact test, controlled retrieval-model ablations on a frozen corpus, independent QASPER scoring, claim/evidence annotation, reproducible LLM inference logs, and fully documented uncertainty estimates.

---

## 📌 Conclusion

This project demonstrates an integrated approach to scientific literature search and evidence-aware analysis through **retrieval engineering, natural language processing, research synthesis, and reproducibility controls**.

The completed curated study shows that modern time-series forecasting papers use markedly different inductive biases, including probabilistic recurrence, decomposition, patch-based tokenization, variable-wise attention, linear modeling, and multiscale mixing. The source-verified PatchTST example illustrates how **published numerical claims should be attributed and qualified rather than presented as independently reproduced experiments**.

On the software side, the project combines searchable scientific records, exploratory text analysis, modular RAG workflows, evidence auditing, and a checksum-controlled official benchmark runner. It also makes the status of incomplete evaluations explicit.

Overall, the project emphasizes that **discovering relevant papers, generating plausible summaries, and scientifically validating research claims are different tasks**. Transparent evidence provenance and measured evaluation are essential for a trustworthy AI-assisted research workflow.

**Portfolio scope:** A validated scientific-research software prototype with a bounded, real-literature case study—not a completed end-to-end empirical benchmark of every optional retrieval and LLM model.

---

## 📚 References & Data Sources

1. **BEIR — A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models**  
   https://github.com/beir-cellar/beir

2. **SciFact — Verifying Scientific Claims**  
   https://allenai.org/data/scifact

3. **QASPER — A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers**  
   https://allenai.org/data/qasper

4. **DeepAR — Probabilistic Forecasting with Autoregressive Recurrent Networks**  
   https://arxiv.org/abs/1704.04110

5. **Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting**  
   https://arxiv.org/abs/1912.09363

6. **Informer — Beyond Efficient Transformer for Long Sequence Time-Series Forecasting**  
   https://arxiv.org/abs/2012.07436

7. **Autoformer — Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting**  
   https://arxiv.org/abs/2106.13008

8. **FEDformer — Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting**  
   https://arxiv.org/abs/2201.12740

9. **Are Transformers Effective for Time Series Forecasting? (LTSF-Linear / DLinear)**  
   https://arxiv.org/abs/2205.13504

10. **TimesNet — Temporal 2D-Variation Modeling for General Time Series Analysis**  
    https://arxiv.org/abs/2210.02186

11. **A Time Series is Worth 64 Words: Long-term Forecasting with Transformers (PatchTST)**  
    https://arxiv.org/abs/2211.14730

12. **iTransformer — Inverted Transformers Are Effective for Time Series Forecasting**  
    https://arxiv.org/abs/2310.06625

13. **TimeMixer — Decomposable Multiscale Mixing for Time Series Forecasting**  
    https://arxiv.org/abs/2405.14616

14. **Project Methodology, Contribution Attribution, and Reproducibility**  
    [`docs/ORIGIN_AND_CONTRIBUTIONS.md`](docs/ORIGIN_AND_CONTRIBUTIONS.md) · [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)

---

*Research scope: Modular scientific-literature analysis software, synthetic functionality demonstrations, and a curated real-paper case study. Official retrieval/QA benchmarks and independent LLM factuality evaluations must be executed and validated separately before making empirical performance claims.*
