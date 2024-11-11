from datasets import load_dataset
from torch.utils.data import Dataset
import torch
from transformers import BertTokenizer

class QuestionPairsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': label}

def load_data():
    dataset = load_dataset("AlekseyKorshuk/quora-question-pairs")
    df = dataset['train'].to_pandas()  
    df=df.dropna()
    df=df.sample(100000)
    df = df.reset_index(drop=True)
    texts = df['question1'] + " [SEP] " + df['question2']
    labels = df['is_duplicate'].astype(int)
    return texts, labels