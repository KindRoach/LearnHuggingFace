from itertools import chain

import datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

raw_dataset = datasets.load_dataset("wikitext", "wikitext-103-v1")
non_empty_dataset = raw_dataset.filter(lambda e: len(e["text"]) > 0)
tokenized_dataset = non_empty_dataset.map(lambda e: tokenizer(e["text"]), batched=True)

chunk_size = 128
# chunk_size = tokenizer.model_max_length
eos_id = tokenizer.eos_token_id


def chunk_example(example):
    input_ids = [ids + [eos_id] for ids in example["input_ids"]]
    concat_ids = list(chain.from_iterable(input_ids))
    chunk_ids = [concat_ids[i:i + chunk_size] for i in range(0, len(concat_ids), chunk_size)]
    return {"input_ids": chunk_ids}


old_cols = tokenized_dataset["train"].column_names
chunked_dataset = tokenized_dataset.map(
    chunk_example, batched=True,
    remove_columns=old_cols  # As we change the size of batch, we need remove all old columns.
)

config = AutoConfig.from_pretrained(model_name)
model = AutoModelForCausalLM.from_config(config)
tokenizer.pad_token = tokenizer.eos_token  # DataCollator require pad token
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

args = TrainingArguments(
    output_dir=model_name,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=1,
    fp16=True
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=chunked_dataset["train"],
    eval_dataset=chunked_dataset["validation"],
)

trainer.train()
