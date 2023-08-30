from torch.utils.data import Dataset
import pandas as pd

class AbstractiveSummarizationDataset(Dataset):
    
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)

    def __getitem__(self, idx):
        doc, sum_ = self.df.iloc[idx]
        return doc + " <sep> " + sum_ + " <eos>"
    
    def __len__(self):
        return len(self.df.index)
    

class TokenizeCollate:

    def __init__(self, tokenizer_obj):
        self.tokenizer = tokenizer_obj

    def __call__(self, x):
        tokenized = self.tokenizer(x, return_tensors="pt", padding=True).input_ids
        return tokenized[:, :-1], tokenized[:, 1:]