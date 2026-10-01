# NLP pipeline

## Classical baseline

```text
Raw text → light preprocessing → TF-IDF → Logistic Regression → intent
```

TF-IDF represents word importance using corpus statistics. Unigrams and bigrams allow
the baseline to capture phrases such as "charged twice".

## Transformer

```text
Raw text → tokenizer → input IDs + attention mask → DistilBERT encoder → classification head → intent
```

The pretrained encoder is fine-tuned on labeled support data. We do not aggressively
remove punctuation/stopwords because the Transformer tokenizer and pretrained encoder
expect natural language structure.

## Comparison

Accuracy alone is insufficient for imbalanced support classes. Macro F1 gives every
class equal weight, while weighted F1 reflects class frequency. Results are written
only after actual evaluation.
