# BERT-family intent classifier

This project uses DistilBERT as a lighter BERT-family model. Tokenization converts
text into model vocabulary IDs and an attention mask marks which positions are real
input rather than padding. Self-attention lets each token use information from other
tokens. A classification head maps the encoder representation to intent classes.

Important parameters:
- `max_length`: maximum token sequence length.
- `batch_size`: examples processed per step; reduce it for CPU/small GPU memory.
- `learning_rate`: size of fine-tuning updates; `2e-5` is a conservative starting point.
- `epochs`: passes over training data.
- `weight_decay`: regularization.
- `warmup_ratio`: gradual learning-rate warm-up.
