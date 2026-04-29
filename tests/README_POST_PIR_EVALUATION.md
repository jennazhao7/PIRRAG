# Post-PIR Retrieval Evaluation

This directory contains evaluation scripts for assessing retrieval performance **after** PIR (Private Information Retrieval) has retrieved actual documents.

## Overview

These scripts evaluate the **actual documents** returned by PIR, not just centroid selection. They are designed to run **after** the PIR retrieval process has completed.

## Workflow

```
1. FHE Phase (Complete)
   └─> Output: pir_input.json (centroid indices)

2. PIR Phase (Complete)
   └─> Output: PIR-retrieved documents (query_*_documents.json files)

3. Evaluation Phase (This step)
   └─> Input: PIR-retrieved documents
   └─> Output: Evaluation metrics
```

## Scripts

### FEVER: `fever/evaluate_post_pir_retrieval.py`

Evaluates fact verification retrieval performance.

**Metrics:**
- **EvidenceRecall@K**: Percentage of queries where at least one evidence sentence is found
- **EvidenceCoverage@K**: Average fraction of evidence sentences found
- **Precision@K**: Fraction of retrieved documents that contain evidence
- **MRR**: Mean Reciprocal Rank of first relevant document

**Usage:**
```bash
python tests/fever/evaluate_post_pir_retrieval.py \
    --pir-output /path/to/pir/output/directory \
    --pir-input /path/to/fever_pir_input.json \
    --output fever_post_pir_results.json \
    --k-values 1 5 10 20 50 100
```

### HotpotQA: `hotpotqa/evaluate_post_pir_retrieval.py`

Evaluates multi-hop question answering retrieval performance.

**Metrics:**
- **TitleRecall@K (Any)**: Percentage where at least one supporting page title is found
- **TitleRecall@K (All)**: Percentage where all supporting page titles are found
- **Precision@K**: Fraction of retrieved documents that are supporting pages
- **MRR**: Mean Reciprocal Rank of first supporting page

**Usage:**
```bash
python tests/hotpotqa/evaluate_post_pir_retrieval.py \
    --pir-output /path/to/pir/output/directory \
    --pir-input /path/to/hotpot_pir_input.json \
    --output hotpot_post_pir_results.json \
    --k-values 1 5 10 20 50 100
```

### Natural Questions: `natural_questions/evaluate_post_pir_retrieval.py`

Evaluates open-domain question answering retrieval performance.

**Metrics:**
- **AnswerRecall@K**: Percentage where at least one answer string is found
- **AnswerCoverage@K**: Average fraction of answer strings found
- **Precision@K**: Fraction of retrieved documents containing answers
- **MRR**: Mean Reciprocal Rank of first document with answer
- **ExactMatch@K**: Percentage where exact answer string is found

**Usage:**
```bash
python tests/natural_questions/evaluate_post_pir_retrieval.py \
    --pir-output /path/to/pir/output/directory \
    --pir-input /path/to/nq_pir_input.json \
    --output nq_post_pir_results.json \
    --k-values 1 5 10 20 50 100
```

## Input Format

### PIR Output Format

The scripts accept PIR output in one of these formats:

**Option 1: Directory with per-query files**
```
pir_output/
├── query_0000_documents.json
├── query_0001_documents.json
└── ...
```

Each file contains:
```json
{
  "query_idx": 0,
  "top_k_indices": [3218, 2134, 827, ...],
  "top_k_distances": [0.123, 0.456, 0.789, ...],
  "documents": [
    "Document text 1...",
    "Document text 2...",
    ...
  ]
}
```

**Option 2: Single JSON file**
```json
[
  {
    "query_idx": 0,
    "documents": ["...", "..."]
  },
  {
    "query_idx": 1,
    "documents": ["...", "..."]
  }
]
```

### PIR Input Format (Optional)

The `--pir-input` argument is optional but recommended. It provides query metadata (evidence, supporting facts, answers) for evaluation:

- **FEVER**: `fever_pir_input.json` with `evidence` field
- **HotpotQA**: `hotpot_pir_input.json` with `supporting_facts` field
- **NQ**: `nq_pir_input.json` with `short_answer_strings` and `answer` fields

## Output Format

Each script outputs a JSON file with aggregated metrics:

```json
{
  "total_queries": 100,
  "metrics": {
    "k1": {
      "evidence_recall": 0.45,
      "evidence_coverage": 0.23,
      "precision": 0.67,
      "mrr": 0.52
    },
    "k5": {
      ...
    }
  }
}
```

## Key Differences from Pre-PIR Evaluation

| Aspect | Pre-PIR (`evaluate_pir_retrieval.py`) | Post-PIR (`evaluate_post_pir_retrieval.py`) |
|--------|--------------------------------------|----------------------------------------------|
| **Input** | Centroid indices from `pir_input.json` | Actual documents from PIR output |
| **Method** | Simulates document retrieval | Evaluates real PIR-retrieved documents |
| **When to Run** | After FHE, before PIR | After PIR completes |
| **What it Evaluates** | Centroid selection quality | End-to-end PIR retrieval quality |

## Example Workflow

```bash
# 1. After FHE completes, you have:
#    - batched_test/fever_output/fever_pir_input.json

# 2. Run PIR retrieval (this step is done by your PIR system)
#    Output: batched_test/fever_output/pir_documents/query_*_documents.json

# 3. Evaluate post-PIR retrieval:
python tests/fever/evaluate_post_pir_retrieval.py \
    --pir-output batched_test/fever_output/pir_documents \
    --pir-input batched_test/fever_output/fever_pir_input.json \
    --output fever_post_pir_evaluation.json
```

## Notes

- The scripts automatically detect whether the input is a directory or single file
- If `--pir-input` is not provided, evaluation will still work but may have limited accuracy (no ground truth metadata)
- All scripts support custom K values via `--k-values`
- Results are saved as JSON and also printed to console

