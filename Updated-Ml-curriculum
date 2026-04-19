# Machine Learning Curriculum 2026
## Advanced Edition — From Beginner to Production-Ready ML Engineer

> **Covering:** Classical ML • Deep Learning • LLMs & Generative AI • MLOps • AI Agents • Responsible AI

---

## Curriculum Overview

| Phase | Duration | Core Focus | Key Output |
|:------|:---------|:-----------|:-----------|
| Phase 0 | 2–4 weeks | Math & Python Foundations | NumPy/Pandas notebook portfolio |
| Phase 1 | 6–8 weeks | Classical Machine Learning | End-to-end Scikit-learn project |
| Phase 2 | 8–10 weeks | Deep Learning & Neural Networks | Image/text classifier deployed on HuggingFace |
| Phase 3 | 8–12 weeks | LLMs, GenAI & Prompt Engineering | Fine-tuned LLM + RAG application |
| Phase 4 | 6–8 weeks | MLOps & Production Deployment | Full ML pipeline with CI/CD |
| Phase 5 | Ongoing | AI Agents, Research & Specialisation | Published paper or Kaggle top 10% |

---

## Phase 0 — Foundations (Weeks 1–4)

Before any algorithm makes sense, you need fluency in three areas: Python programming, linear algebra, and probability/statistics. Do not skip or rush this phase — weak foundations are the single largest source of ML practitioner failure.

### Mathematics

- **Linear Algebra** — vectors, matrices, dot products, eigendecomposition, SVD
- **Calculus** — gradients, partial derivatives, the chain rule (backpropagation is just chain rule)
- **Probability & Statistics** — distributions, Bayes' theorem, expectation, variance, MLE
- **Information Theory** — entropy, KL divergence (essential for modern generative models)

### Python & Data Ecosystem

- Python proficiency: list comprehensions, generators, OOP, type hints
- NumPy: vectorised operations, broadcasting — avoid loops
- Pandas: DataFrame manipulation, groupby, merging, time-series
- Matplotlib / Seaborn / Plotly: exploratory data visualisation
- Jupyter / VS Code: reproducible notebook workflows

### Phase 0 Resources

| Resource | Type | Level | Why It Matters in 2026 |
|:---------|:-----|:------|:-----------------------|
| 3Blue1Brown — Essence of Linear Algebra (YouTube) | Video | Beginner | Best visual intuition for linear algebra available anywhere |
| Gilbert Strang — Introduction to Linear Algebra, 6th Ed. | Book | Beginner | The definitive undergraduate text; MIT OCW lectures free |
| StatQuest with Josh Starmer (YouTube) | Video | Beginner | Clear, patient statistics and ML explainers; regularly updated |
| fast.ai Practical Deep Learning — Lesson 0 (setup) | Course | Beginner | Gets your environment production-ready from day one |
| Python for Data Analysis — Wes McKinney, 3rd Ed. | Book | Beginner | Written by the Pandas creator; authoritative reference |

> 💡 **Phase 0 Project:** Build a NumPy-only implementation of linear regression and logistic regression from scratch. Push to GitHub. This proves you understand the math before you lean on libraries.

---

## Phase 1 — Classical Machine Learning (Weeks 5–12)

Classical ML is still the workhorse of industry. Tree-based models (XGBoost, LightGBM) dominate tabular data competitions. Understanding these methods deeply will set you apart from practitioners who jumped straight to deep learning.

### Supervised Learning

- **Linear & Logistic Regression** — regularisation (L1/L2/ElasticNet), interpreting coefficients
- **Decision Trees** — splitting criteria, pruning, overfitting
- **Ensemble Methods** — Random Forests, Gradient Boosting, XGBoost, LightGBM, CatBoost
- **Support Vector Machines** — kernel trick, margin maximisation
- **k-Nearest Neighbours** — distance metrics, curse of dimensionality
- **Naive Bayes** — text classification, probabilistic interpretation

### Unsupervised Learning

- **Clustering** — k-Means, DBSCAN, Hierarchical Clustering, HDBSCAN
- **Dimensionality Reduction** — PCA, t-SNE, UMAP (essential for visualising embeddings in 2026)
- **Anomaly Detection** — Isolation Forest, One-Class SVM, Autoencoders

### Model Evaluation & Feature Engineering

- Train/validation/test splits — avoiding data leakage
- Cross-validation strategies (k-fold, stratified, time-series splits)
- Metrics: accuracy, precision, recall, F1, ROC-AUC, PR-AUC, RMSE, MAE, R²
- Feature engineering: encoding, imputation, scaling, interaction features
- Hyperparameter tuning: GridSearch, RandomSearch, Optuna (Bayesian optimisation)
- Explainability: SHAP values, LIME, permutation importance — now an industry requirement

### Phase 1 Resources

| Resource | Type | Level | Why It Matters in 2026 |
|:---------|:-----|:------|:-----------------------|
| Andrew Ng — Machine Learning Specialisation (Coursera, 2022 edition) | Course | Beginner–Intermediate | Updated trilogy replaces the original; uses scikit-learn and TensorFlow |
| Aurélien Géron — Hands-On Machine Learning, 3rd Ed. (2023) | Book | Intermediate | Best single book for classical ML + deep learning with scikit-learn/Keras |
| Tan, Steinbach & Kumar — Introduction to Data Mining, 2nd Ed. | Book | Intermediate | Comprehensive survey of the field; strong theoretical grounding |
| Kaggle Learn — Intermediate Machine Learning | Course | Intermediate | Practical notebooks on missing values, pipelines, cross-validation, XGBoost |
| SHAP Documentation & Lundberg et al. (2017) | Paper/Docs | Intermediate | SHAP is now a standard tool; understand it before your first job interview |

> 💡 **Phase 1 Project:** Compete on a Kaggle tabular dataset. Build a full pipeline: EDA → feature engineering → XGBoost → SHAP analysis → write-up. Aim for the top 30%.

---

## Phase 2 — Deep Learning & Neural Networks (Weeks 13–22)

Deep learning is no longer optional. Even if you primarily work on tabular data, you need to understand neural networks to collaborate with modern ML teams, use pretrained embeddings, and build LLM-powered applications.

### Neural Network Fundamentals

- Perceptrons, activation functions (ReLU, GELU, SiLU — the 2026 standards)
- Backpropagation and automatic differentiation (autograd)
- Optimisers: SGD, Adam, AdamW, Lion — and when to use each
- Regularisation: dropout, batch normalisation, layer normalisation, weight decay
- Learning rate schedules: cosine annealing, warmup, cyclical

### Computer Vision

- Convolutional Neural Networks (CNNs): convolution, pooling, receptive field
- Architectures: ResNet, EfficientNet, ConvNeXt
- Vision Transformers (ViT) — the dominant paradigm since 2022
- Transfer learning and fine-tuning with torchvision and timm
- Object detection: YOLO v8/v9, DETR
- Diffusion models for image generation: DDPM, Stable Diffusion architecture (conceptual)

### Natural Language Processing

- Text preprocessing: tokenisation, BPE, WordPiece, SentencePiece
- Word embeddings: Word2Vec, GloVe, fastText — historical but foundational
- Recurrent architectures: LSTMs, GRUs — know them, but they are largely superseded
- Attention mechanism and the Transformer architecture (Vaswani et al., 2017) — **read the paper**
- BERT, RoBERTa — encoder-only models for classification/NER
- GPT family — decoder-only models for generation
- HuggingFace Transformers library — the industry standard toolkit

### Frameworks

- **PyTorch** — primary framework; preferred by research and increasingly by industry
- **TensorFlow / Keras** — still widely deployed in production; worth knowing
- **JAX / Flax** — growing importance in research and Google ecosystem

### Phase 2 Resources

| Resource | Type | Level | Why It Matters in 2026 |
|:---------|:-----|:------|:-----------------------|
| fast.ai — Practical Deep Learning for Coders (2022 edition) | Course | Intermediate | Top-down, code-first; gets you building before explaining theory |
| Andrej Karpathy — Neural Networks: Zero to Hero (YouTube, 2023) | Video | Intermediate | Builds GPT from scratch in pure Python; the best deep learning tutorial made |
| Dive into Deep Learning — d2l.ai (interactive textbook) | Book/Web | Intermediate | Multi-framework; covers everything from CNNs to Transformers with code |
| Goodfellow, Bengio & Courville — Deep Learning (MIT Press) | Book | Advanced | The theoretical bible; chapters 6–9 are essential reading |
| The Annotated Transformer — Harvard NLP | Tutorial | Intermediate | Line-by-line implementation of the original Transformer paper |
| Papers With Code — State of the Art benchmarks | Reference | All levels | Track current SOTA on every task; linked to reproducible code |

> 💡 **Phase 2 Project:** Fine-tune a BERT/RoBERTa model on a classification task. Deploy it as a HuggingFace Space with a Gradio interface. Alternatively, build and train a small ViT on a custom image dataset.

---

## Phase 3 — LLMs, Generative AI & Prompt Engineering (Weeks 23–34)

Large Language Models are the defining technology of this era. By 2026, every ML practitioner is expected to understand how to use, evaluate, adapt, and deploy LLMs responsibly.

### Understanding LLMs

- Transformer architecture in depth: multi-head attention, positional encoding, KV cache
- Pretraining: next-token prediction, masked language modelling
- Scaling laws (Chinchilla, 2022) — why model size and data both matter
- Key models: GPT-4, Claude 3, Llama 3, Mistral, Gemini — understand their differences
- Multimodal LLMs: GPT-4o, Gemini 1.5, LLaVA — text + image as first-class

### Fine-Tuning & Adaptation

- Full fine-tuning vs. parameter-efficient methods
- **LoRA and QLoRA** — the standard approach for fine-tuning on consumer hardware
- Instruction tuning and RLHF (Reinforcement Learning from Human Feedback)
- **DPO (Direct Preference Optimisation)** — the 2024 successor to RLHF
- Dataset preparation: instruction datasets, chat templates, JSONL formatting
- Tools: Axolotl, Unsloth, LLaMA-Factory — know at least one

### Retrieval-Augmented Generation (RAG)

- Why RAG: reducing hallucination, grounding in private/recent data
- Vector databases: ChromaDB, Pinecone, Weaviate, pgvector
- Embedding models: text-embedding-3-large, BGE, E5
- RAG pipeline: chunking strategies, retrieval, reranking, generation
- Advanced RAG: HyDE, RAPTOR, contextual retrieval, graph RAG
- Evaluation: RAGAS framework, faithfulness, answer relevancy

### Prompt Engineering

- Zero-shot, few-shot, and chain-of-thought prompting
- Structured output: JSON mode, function calling, tool use
- System prompts and persona design
- LLM-as-judge evaluation patterns
- **DSPy** — programmatic prompt optimisation (the future of prompt engineering)

### Generative AI Beyond Text

- Diffusion models: DDPM, DDIM, latent diffusion (Stable Diffusion)
- Text-to-image: Stable Diffusion XL, FLUX, Midjourney architecture concepts
- Text-to-video: Sora, RunwayML Gen-3 — understand the architecture even if you cannot run it
- Audio generation: Whisper (ASR), Bark, MusicGen

### Phase 3 Resources

| Resource | Type | Level | Why It Matters in 2026 |
|:---------|:-----|:------|:-----------------------|
| Andrej Karpathy — Let's build GPT (YouTube) | Video | Intermediate | Builds a full GPT in 2 hours; foundational for understanding LLMs |
| HuggingFace NLP Course — huggingface.co/learn | Course | Intermediate | Free; covers Transformers, fine-tuning, RAG, and deployment |
| Sebastian Raschka — Build an LLM from Scratch (2024) | Book | Advanced | The most rigorous modern LLM book; implements GPT-2 from scratch |
| LangChain & LlamaIndex Documentation | Docs | Intermediate | Orchestration frameworks for RAG and agent applications |
| LMSYS Chatbot Arena & Open LLM Leaderboard | Reference | All levels | Track real-world LLM performance; informs model selection decisions |
| Attention is All You Need — Vaswani et al. (2017) | Paper | Advanced | Still required reading; everything in LLMs descends from this paper |

> 💡 **Phase 3 Project:** Build a domain-specific RAG application — ingest research papers, build a ChromaDB vector store, create a Q&A chatbot with citation tracking. Deploy on Streamlit Cloud. Alternatively, fine-tune Llama 3 with LoRA on a custom instruction dataset.

---

## Phase 4 — MLOps & Production Deployment (Weeks 35–42)

A model that is not in production does not add value. MLOps is the engineering discipline of reliably building, deploying, monitoring, and iterating on ML systems. In 2026, this is a core competency, not an optional add-on.

### Experiment Tracking & Reproducibility

- MLflow, Weights & Biases (W&B) — log every experiment
- DVC (Data Version Control) — version your datasets alongside your code
- Hydra — configuration management for ML experiments

### Model Serving

- REST APIs with FastAPI — the industry standard for ML APIs
- Model serialisation: ONNX, TorchScript, safetensors
- Serving frameworks: vLLM (LLM serving), Triton Inference Server, BentoML, Ray Serve
- Containerisation: Docker, Docker Compose
- Cloud deployment: AWS SageMaker, GCP Vertex AI, Azure ML, HuggingFace Endpoints

### Pipelines & Orchestration

- Feature stores: Feast, Tecton — separating feature engineering from training
- Pipeline orchestration: Apache Airflow, Prefect, ZenML
- Continuous training: automated retraining triggers on data drift

### Monitoring & Reliability

- Data drift detection: Evidently AI, Alibi Detect
- Model performance monitoring: latency, throughput, error rates
- A/B testing and shadow deployment strategies
- LLM observability: LangSmith, Phoenix Arize, prompt logging

### Phase 4 Resources

| Resource | Type | Level | Why It Matters in 2026 |
|:---------|:-----|:------|:-----------------------|
| Chip Huyen — Designing Machine Learning Systems (2022) | Book | Advanced | The definitive guide to production ML; covers the full lifecycle |
| Made With ML — madewithml.com | Course | Intermediate | Free MLOps course with end-to-end project using modern tools |
| Full Stack Deep Learning — fullstackdeeplearning.com | Course | Intermediate | Covers deployment, CI/CD, and LLM ops in depth |
| Google — Machine Learning Engineering for Production (Coursera) | Course | Advanced | Four-course specialisation on building production ML pipelines |

> 💡 **Phase 4 Project:** Take any model from Phases 1–3 and build a production-grade pipeline: Dockerised FastAPI endpoint + W&B experiment tracking + GitHub Actions CI/CD + Evidently monitoring dashboard. This is the project that gets you hired.

---

## Phase 5 — AI Agents, Research & Specialisation (Ongoing)

By this phase you are no longer a beginner — you are a practitioner. Phase 5 is about going deep in an area that excites you and contributing original work to the field.

### AI Agents (The 2025–2026 Frontier)

Agentic systems — where LLMs plan, use tools, and take actions in the world — are the dominant research and engineering frontier of 2025–2026.

- ReAct, Reflection, Tree of Thought, and Plan-and-Solve prompting patterns
- Tool use and function calling: web search, code execution, API calls
- Multi-agent frameworks: AutoGen, CrewAI, LangGraph
- Memory systems: short-term (context window), long-term (vector DB), episodic
- Agent evaluation: benchmarks (SWE-bench, GAIA, AgentBench)
- Computer use agents: Claude Computer Use, OpenAI Operator concepts

### Responsible AI & Safety

- Bias and fairness: disparate impact, equalised odds, calibration
- Explainability: SHAP, LIME, attention visualisation, mechanistic interpretability
- **EU AI Act (2024)** compliance requirements for high-risk AI systems
- Red-teaming and adversarial robustness
- Privacy-preserving ML: differential privacy, federated learning
- Carbon footprint of ML: model efficiency, distillation, quantisation

### Specialisation Tracks

#### Track A — Research
- Read 2–3 papers per week on arXiv (cs.LG, cs.AI, cs.CL, stat.ML)
- Reproduce a paper from scratch and publish the code
- Contribute to open-source ML projects (HuggingFace, PyTorch, scikit-learn)
- Submit to workshops at NeurIPS, ICML, ICLR, ACL

#### Track B — Industry Engineering
- Contribute to internal ML platforms: feature stores, model registries, serving infra
- Obtain cloud ML certifications: AWS ML Specialty, GCP Professional ML Engineer
- Build domain expertise: healthcare, finance, climate, robotics

#### Track C — Entrepreneurship
- Identify a specific problem where AI provides 10x improvement
- Prototype with LLM APIs (OpenAI, Anthropic, Gemini)
- Study AI product design: latency, cost, hallucination management, UX

### Advanced & Theoretical References

| Resource | Type | Level | Why It Matters in 2026 |
|:---------|:-----|:------|:-----------------------|
| Hastie, Tibshirani & Friedman — The Elements of Statistical Learning, 2nd Ed. | Book | Advanced | The statistical foundations every ML engineer should eventually read |
| Bishop — Pattern Recognition and Machine Learning (2006) | Book | Advanced | Bayesian perspective on ML; dense but rewarding; free PDF available |
| Sutton & Barto — Reinforcement Learning, 2nd Ed. (free online) | Book | Advanced | The definitive RL text; essential if your track includes agents or robotics |
| Pedro Domingos — The Master Algorithm | Book | All levels | Inspirational popular-science read; explains the five ML tribes brilliantly |
| arXiv Sanity Preserver — arxiv-sanity.com | Reference | Advanced | Karpathy's paper organiser; curate your own reading list |
| Ilya Sutskever's Reading List (30 papers) | Reference | Advanced | The list he gave OpenAI engineers; covers the intellectual foundations of modern AI |

---

## GitHub Contribution Strategy

| Phase | Repo Contribution | Example Repositories |
|:------|:-----------------|:--------------------|
| Phase 0 | Add worked examples or fix documentation typos | numpy/numpy, pandas-dev/pandas |
| Phase 1 | Add a notebook tutorial or fix a bug in ML utilities | scikit-learn, mlxtend, shap |
| Phase 2 | Contribute model card, fix docs, add training script | huggingface/transformers, fastai |
| Phase 3 | Add RAG example, improve prompt templates, add eval | langchain-ai/langchain, run-llama |
| Phase 4 | Add deployment example, write tests, improve CI | mlflow, evidently-ai/evidently |
| Phase 5 | Reproduce a paper, add a benchmark, submit a feature | openai/evals, EleutherAI/lm-eval |

> 💡 **Contribution tip:** Start with documentation improvements — they require no review of complex logic, get merged quickly, and get your name into commit history. Maintainers recognise you when you later submit code PRs.

---

## Final Note

Machine learning in 2026 is vast, fast-moving, and genuinely exciting. No curriculum can be exhaustive. The meta-skill you are building is the ability to learn continuously — to read a new paper, understand its contribution relative to what came before, implement it, and evaluate it honestly.

Prioritise understanding over breadth. One deeply understood algorithm is worth more than a shallow familiarity with ten. Build things. Break things. Write about what you learn. Share your code.

> ✨ *The best time to start was yesterday. The second best time is now. Open your IDE, choose the first resource from Phase 0, and begin. The rest will follow.*

---

*Machine Learning Curriculum 2026 • Advanced Edition*
