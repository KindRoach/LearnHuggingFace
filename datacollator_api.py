from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, DataCollatorWithPadding

checkpoint = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(checkpoint)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
raw_datasets = load_dataset("glue", "mrpc")
tokenized_datasets = raw_datasets.map(
    lambda e: tokenizer(e["sentence1"], e["sentence2"]),
    remove_columns=["sentence1", "sentence2", "idx"],
    batched=True
)

dataloader = DataLoader(tokenized_datasets["train"], batch_size=32, collate_fn=data_collator)

for batch in tqdm(dataloader):
    pass
