from datasets import load_dataset
from tqdm import tqdm

url = "https://github.com/crux82/squad-it/raw/master/"
data_files = {
    "train": url + "SQuAD_it-train.json.gz",
    "test": url + "SQuAD_it-test.json.gz",
}

squad_it_dataset = load_dataset("json", data_files=data_files, field="data", streaming=True)

for example in tqdm(squad_it_dataset["train"]):
    pass

pass
