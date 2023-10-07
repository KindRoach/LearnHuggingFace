from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

model_inputs = tokenizer(
    ["Hi!", "I'm robt."],
    ["Hello!", "Are you kidding?"],
    padding=True,
    return_tensors="pt"
)

model = BertModel.from_pretrained("bert-base-uncased")
model(**model_inputs)
