# Evaluation

## Classification

Measure accuracy, macro precision/recall/F1, weighted F1, confusion matrix and latency.
Macro F1 is especially important when some support intents are rare.

## Retrieval

The evaluation script contains a small manually curated question-to-source set and
computes Recall@K, Precision@K and MRR. This is an educational smoke benchmark, not a
large production evaluation.

## Generation

A robust project should evaluate relevance, context relevance and faithfulness. Automated
LLM judges can be useful but are not ground truth; a small human-reviewed set is recommended.
No generated score is inserted into the repository unless it has actually been measured.
