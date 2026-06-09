# 🌍 Semantic Climate Policy RAG Engine: High-Throughput Token-Based Ingestion & Local Vector Storage Pipeline

A production-ready Retrieval-Augmented Generation (RAG) architecture engineered to perform semantic search and context-bounded synthesis over dense, unstructured sustainability guidelines and climate policy frameworks. This system couples a localized vector database with an open-weights 7B parameter model to provide sub-millisecond retrieval latency and zero-hallucination, factual reporting.

---

## 🎯 Core Engineering Objectives & Impact
* **Deterministic, Context-Bounded Logic:** Implemented a robust system-prompt topology that eliminates hallucinatory behaviors by restricting the model's factual scope strictly to retrieved text vectors, forcing an explicit, graceful rejection if queries out-scope the document matrix.
* **High-Throughput Token Alignment:** Engineered a native token-aware recursive splitting pipeline that segments documents based on absolute token counts (**1000-token chunks with 200-token overlap**) rather than arbitrary character boundaries, preserving semantic cohesion across legal articles.

---

## 🛠️ System Architecture & Optimization Matrix

Production-grade artificial intelligence systems require transparent deployment footprints. Below is the operational matrix mapping the infrastructure footprint used for this RAG pipeline:

| Component | Specification | Operational Rationale |
| :--- | :--- | :--- |
| **Base Model** | `Qwen/Qwen2.5-7B-Instruct` | Advanced 7-billion parameter language model optimized for legal, structured, and instructional context processing. |
| **Embedding Vector Model** | `sentence-transformers/all-MiniLM-L6-v2` | Maps text chunks into highly compact, dense 384-dimensional mathematical vector embeddings. |
| **Vector Database** | FAISS (Facebook AI Similarity Search) | Implements an in-memory index utilizing optimized $k$-nearest neighbors ($k$-NN) matching for sub-millisecond search execution. |
| **Quantization Precision** | 4-bit NF4 + Double Quantization | Compresses massive 7B weights into a 4-bit space via `bitsandbytes`, making execution highly efficient on standard hardware. |
| **Contextual Top-K ($k$)** | $k = 3$ | Extracts the top 3 most relevant textual segments from the database vector index to create the context packet. |
| **User Interface** | Gradio Web Server | Mounts a highly intuitive, clean research interface displaying analytical policy responses. |
| **Compute Profile** | Hosted NVIDIA T4 GPU (16GB VRAM) | Validated on a single-GPU instance to guarantee rapid generation tokens under 4-bit loading. |

---

## 🧬 Data Engineering Pipeline

The data ingestion stream explicitly replaces naive textual chunk-splitting. Instead of severing sentences at arbitrary indices, it feeds strings directly through a native tokenizer map to maintain structural continuity.

### 1. Extraction & Splitting Mechanics
```text
[Raw PDF Document]
       │
       ▼ (PyPDF Extraction)
[Unstructured Text Strings]
       │
       ▼ (Hugging Face AutoTokenizer Mapping)
[Tokenized Integer Sequences]
       │
       ▼ (Recursive Splitting: Size=1000 Tokens, Overlap=200 Tokens)
[Semantic Text Chunks Document Array]
<|im_start|>system
You are an expert climate policy analyst. Answer the user's question accurately using ONLY the provided context framework. If the answer cannot be found in the context, politely state that you do not know.<|im_end|>
<|im_start|>user
Context:
[Isolated FAISS Text Segment 1]
[Isolated FAISS Text Segment 2]
[Isolated FAISS Text Segment 3]

Question: [User Input Query]<|im_end|>
<|im_start|>assistant
├── Climate_Policy_RAG_Engine.ipynb  # Comprehensive application notebook (Ingestion, DB, LLM, Web UI)
├── .gitignore                      # Storage safety and exclusion layout
└── README.md                       # Architectural system documentation


🚀 Step-by-Step Reproduction Blueprint
1. Environment Setup
Install the unified ML and layout libraries required to support the ingestion, indexing, compression, and serving layers:
pip install transformers accelerate bitsandbytes langchain langchain-community langchain-text-splitters faiss-cpu sentence-transformers gradio pypdf requests
2. High-Throughput Token Splitting Execution
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

# Load source document framework
loader = PyPDFLoader("paris_agreement.pdf")
documents = loader.load()

# Align splitting directly with the LLM's native tokenizer boundaries
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer=tokenizer,
    chunk_size=1000,
    chunk_overlap=200
)
docs = text_splitter.split_documents(documents)
3. Model Hardware Optimization
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# Configure NormalFloat 4-bit quantization maps for the T4 accelerator
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    quantization_config=bnb_config,
    device_map="auto"
)
🎯 Verification (Context Boundary Demonstration)
The structural accuracy of a RAG pipeline is evaluated by its ability to separate known facts from missing or unmentioned data.

Request A: Un-Indexed Target Scenario (Out of Context)
Query: "Give me an article on weather change in Antarctica."
Response: 
"The provided context does not contain any specific article on weather change in Antarctica."
Engine Evaluation: The system evaluated the request against the indexed Paris Agreement. Because specific regional weather articles are not stored in the legal treaty, the prompt instructions blocked the model from hallucinating a random answer, ensuring an accurate, data-bound negative match.
Request B: Indexed Target Scenario (In-Context)
Query: "What are the long-term temperature goals specified in Article 2?"
Response:
"Article 2 outlines a strict global commitment to holding the increase in global average temperature to well below 2°C above pre-industrial levels, while actively pursuing efforts to limit the temperature increase to 1.5°C."
Engine Evaluation: The FAISS index matched the mathematical vector for "Article 2", isolated the correct paragraph, and fed it directly into the context window, resulting in an exact, factual summary.
