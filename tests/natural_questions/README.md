# Natural Questions Retrieval Evaluation

This directory contains scripts for evaluating retrieval performance on the Natural Questions dataset, focusing on AnswerRecall@K metric.

## Features

- ✅ Loads Natural Questions dataset from Hugging Face (`sentence-transformers/natural-questions`)
- ✅ Uses wiki-rag FAISS index for document retrieval
- ✅ **AnswerRecall@K**: Primary retrieval metric
  - Normalizes strings for comparison
  - Checks if any gold answer is substring of any retrieved doc
  - Evaluates for K ∈ {1, 5, 20, 50}
- ✅ Optional reader model: `datarpit/distilbert-base-uncased-finetuned-natural-questions`

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Evaluation (Retrieval-Only)

```bash
python3 retrieval_evaluation.py \
    --top-k 1 5 20 50 \
    --device cuda \
    --no-reader \
    --output ./results/nq_retrieval_results.json
```

### Test with Limited Examples

```bash
python3 retrieval_evaluation.py \
    --top-k 1 5 20 50 \
    --max-examples 100 \
    --device cuda \
    --no-reader \
    --output ./results/nq_test_results.json
```

### Full Evaluation with Reader (Optional)

```bash
python3 retrieval_evaluation.py \
    --top-k 1 5 20 50 \
    --device cuda \
    --reader-model datarpit/distilbert-base-uncased-finetuned-natural-questions \
    --output ./results/nq_full_results.json
```

## Arguments

- `--wiki-rag-index`: Path to wiki-rag FAISS index directory (default: `wiki-rag/wiki_rag_data/wiki_index__top_100000__2025-04-11`)
- `--reader-model`: Reader model for answer extraction (default: `datarpit/distilbert-base-uncased-finetuned-natural-questions`)
- `--no-reader`: Skip loading reader model (retrieval-only evaluation)
- `--top-k`: Top-K values to evaluate (default: `1 5 20 50`)
- `--output`: Path to save results JSON file
- `--max-examples`: Limit number of examples (for testing)
- `--embedding-model`: Embedding model for RAG (default: `BAAI/bge-base-en`)
- `--device`: Device to use (`cuda` or `cpu`, auto-detected if not specified)

## Metrics Explained

### AnswerRecall@K (Primary Metric)

- **Definition**: Check if any gold answer is substring of any retrieved doc
- **Normalization**: Strings are normalized (lowercase, whitespace normalized)
- **Range**: 0.0 to 1.0
- **Interpretation**: Higher is better. Measures if retrieval finds documents containing the answer.

**Evaluation for K ∈ {1, 5, 20, 50}**

This is the main and only required metric for retriever ability.

## Output Format

The results JSON file contains:

```json
{
  "evaluation_results": {
    "total_examples": 100000,
    "total_with_answers": 95000,
    "top_k_values": [1, 5, 20, 50],
    "metrics": {
      "answer_recall@1": {
        "mean": 0.4523,
        "std": 0.4978,
        "total": 95000,
        "correct": 42969
      },
      "answer_recall@5": {
        "mean": 0.7234,
        "std": 0.4476,
        "total": 95000,
        "correct": 68723
      },
      ...
    },
    "timing": {
      "total_time_seconds": 12345.67,
      "avg_retrieval_time_ms": 130.01,
      "throughput_examples_per_second": 7.69
    }
  },
  "detailed_results": [...]
}
```

## Dataset Information

Natural Questions is a question answering dataset:
- **Source**: [sentence-transformers/natural-questions](https://huggingface.co/datasets/sentence-transformers/natural-questions)
- **Size**: ~100k question-answer pairs
- **Format**: `query` (question) and `answer` (answer text)
- **Collection**: Reading the NQ train dataset from embedding-training-data

## Performance Notes

- **Retrieval speed**: ~130-200ms per query (depending on GPU)
- **Throughput**: ~5-8 queries/second
- **Memory**: ~2-4GB GPU memory for embeddings

For RTX-6000:
- Can process full dataset (100k examples) in ~3-5 hours
- Can process 10k examples in ~20-30 minutes

## Reader Model (Optional)

The reader model (`datarpit/distilbert-base-uncased-finetuned-natural-questions`) is loaded but not currently used in the evaluation. It's available for future extension to full QA pipeline evaluation.

For retrieval-only evaluation, use `--no-reader` flag.


