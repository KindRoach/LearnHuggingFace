import evaluate
import numpy as np
from datasets import load_dataset
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

checkpoint = "bert-base-uncased"
model = BertForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
tokenizer = BertTokenizer.from_pretrained(checkpoint)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"])


raw_datasets = load_dataset("glue", "mrpc")
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

metric = evaluate.load("glue", "mrpc")


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


training_args = TrainingArguments(
    output_dir="test-trainer",
    evaluation_strategy="epoch",
    per_device_train_batch_size=32,
    num_train_epochs=10
)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
