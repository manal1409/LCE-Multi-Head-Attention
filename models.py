import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch.nn as nn
import numpy as np

class TraditionalModels:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    def tfidf_cosine_similarity(self, train_texts, test_texts, train_labels):
        vectorizer = TfidfVectorizer()
        train_vectors = vectorizer.fit_transform(train_texts)
        test_vectors = vectorizer.transform(test_texts)

        similarity_matrix = cosine_similarity(test_vectors, train_vectors)

        most_similar_indices = similarity_matrix.argmax(axis=1)
        
        train_labels_list = list(train_labels)
        y_pred = [train_labels_list[idx] for idx in most_similar_indices]
        
        return y_pred
    def word2vec_cosine_similarity(self, train_texts, test_texts, train_labels, model_path='GoogleNews-vectors-negative300.bin'):
        # Loading the pre-trained Word2Vec model
        model = KeyedVectors.load_word2vec_format(model_path, binary=True)
        
        # Convert train and test texts into vectors
        train_vectors = [self.text_to_vector(text, model) for text in train_texts]
        test_vectors = [self.text_to_vector(text, model) for text in test_texts]

        similarity_matrix = cosine_similarity(test_vectors, train_vectors)

        most_similar_indices = similarity_matrix.argmax(axis=1)
        train_labels_list = list(train_labels)
        y_pred = [train_labels_list[idx] for idx in most_similar_indices]
        
        return y_pred

    def text_to_vector(self, text, model):
        return np.mean([model[word] for word in text.split() if word in model] or [np.zeros(model.vector_size)], axis=0)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = list(labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)  # Ensuring labels are integer tensors
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)  

    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

#fine tuning the model
class TransformerModel:
    def __init__(self, model_name='bert-base-uncased'): 
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    def fine_tune_transformer(self, train_texts, train_labels, test_texts, test_labels):
        train_texts = list(train_texts)
        test_texts = list(test_texts)
        train_labels = list(train_labels)  
        test_labels = list(test_labels)  
        
        # Tokenize data
        train_encodings = self.tokenizer(train_texts, truncation=True, padding=True, max_length=128)
        test_encodings = self.tokenizer(test_texts, truncation=True, padding=True, max_length=128)

        # Convert to torch Dataset
        train_dataset = CustomDataset(train_encodings, train_labels)
        test_dataset = CustomDataset(test_encodings, test_labels)

        # Training arguments
        training_args = TrainingArguments(
            output_dir='./results',
            eval_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=16,
            num_train_epochs=3,
            weight_decay=0.01
        )

        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics 
        )

        # Training and evaluation
        trainer.train()
        metrics = trainer.evaluate()
        return metrics

class ProposedModel(nn.Module):
    def __init__(self, embedding_dim=768, num_heads=8):
        super(ProposedModel, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embedding_dim, num_heads)
        self.fc = nn.Linear(embedding_dim, 1)
    
    def forward(self, x):
        attn_output, _ = self.multihead_attn(x, x, x)
        x = self.fc(attn_output.mean(dim=1))
        return torch.sigmoid(x)
        
# Custom Dataset for Transformer Fine-tuning
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
