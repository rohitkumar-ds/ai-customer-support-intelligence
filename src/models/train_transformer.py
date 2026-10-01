"""Fine-tune a pretrained DistilBERT model for intent classification.

Code flow:
dataset -> stratified split -> tokenizer -> pretrained encoder + classification head
-> validation/early stopping -> held-out test metrics -> checkpoint.

"""
from __future__ import annotations
import argparse, inspect
import numpy as np
from src.config import get_settings
from src.nlp.dataset import SupportDataset
from src.evaluation.metrics import classification_metrics, save_json

def train(source="banking77"):
    """Fine-tune the configured Transformer and return genuine held-out metrics."""
    try:
        import torch
        from datasets import Dataset
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
    except ImportError as exc:
        raise RuntimeError("Install torch, datasets and transformers to train the Transformer.") from exc

    settings=get_settings()
    ds=SupportDataset(source=source,path=str(settings.path("data/raw/tickets.jsonl")))
    df=ds.load()
    train_df,val_df,test_df=ds.split(df)
    labels=sorted(df.intent.unique().tolist())
    label2id={label:i for i,label in enumerate(labels)}
    id2label={i:label for label,i in label2id.items()}

    tokenizer=AutoTokenizer.from_pretrained(settings.models.transformer.model_name)

    def tokenize(batch):
        """Tokenize a batch with truncation and fixed padding for predictable tensor shapes."""
        return tokenizer(batch["text"],truncation=True,padding="max_length",
                          max_length=settings.models.transformer.max_length)

    def make_dataset(frame):
        """Convert a pandas split into a Trainer-compatible dataset."""
        d=Dataset.from_pandas(frame[["text","intent"]],preserve_index=False)
        d=d.map(tokenize,batched=True)
        d=d.map(lambda row: {"labels":label2id[row["intent"]]})
        return d.remove_columns(["text","intent"])

    train_ds,val_ds,test_ds=[make_dataset(frame) for frame in (train_df,val_df,test_df)]

    model=AutoModelForSequenceClassification.from_pretrained(
        settings.models.transformer.model_name,
        num_labels=len(labels),label2id=label2id,id2label=id2label
    )

    def compute_metrics(eval_pred):
        """Return accuracy during validation; full metrics are calculated after training."""
        logits,y=eval_pred
        pred=np.argmax(logits,axis=-1)
        return {"accuracy":float(np.mean(pred==y))}

    out=str(settings.path(settings.models.transformer.artifact_path))
    ta_params=inspect.signature(TrainingArguments).parameters
    strategy_param="eval_strategy" if "eval_strategy" in ta_params else "evaluation_strategy"
    args=TrainingArguments(
        output_dir=out,learning_rate=settings.models.transformer.learning_rate,
        per_device_train_batch_size=settings.models.transformer.batch_size,
        per_device_eval_batch_size=settings.models.transformer.batch_size,
        num_train_epochs=settings.models.transformer.epochs,
        weight_decay=settings.models.transformer.weight_decay,
        warmup_steps=max(
            1,
            round(
                (
                    max(1, len(train_ds) // settings.models.transformer.batch_size)
                    * settings.models.transformer.epochs
                )
                * settings.models.transformer.warmup_ratio
            )
        ),
        save_strategy="epoch",load_best_model_at_end=True,
        metric_for_best_model="eval_loss",report_to="none",
        fp16=torch.cuda.is_available(),save_total_limit=2,
        **{strategy_param:"epoch"}
    )

    trainer_kwargs={
        "model":model,"args":args,"train_dataset":train_ds,"eval_dataset":val_ds,
        "compute_metrics":compute_metrics,
        "callbacks":[EarlyStoppingCallback(
            early_stopping_patience=settings.models.transformer.patience)]
    }
    if "processing_class" in inspect.signature(Trainer).parameters:
        trainer_kwargs["processing_class"]=tokenizer
    else:
        trainer_kwargs["tokenizer"]=tokenizer

    trainer=Trainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(out)
    tokenizer.save_pretrained(out)

    pred=trainer.predict(test_ds).predictions.argmax(axis=-1)
    result=classification_metrics(
        test_df.intent.tolist(),
        [id2label[int(x)] for x in pred],
        labels
    )
    save_json(
        {"model_metadata":{"model_type":"distilbert_intent_classifier",
                           "model_name":settings.models.transformer.model_name},
         "test_samples":len(test_df),"global_metrics":result},
        str(settings.path("artifacts/evaluation_reports/transformer_model_report.json"))
    )
    return result

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",choices=["banking77","local"],default="banking77")
    args=parser.parse_args()
    print(train(args.source))
