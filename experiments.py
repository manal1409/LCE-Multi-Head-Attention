from models import TraditionalModels, TransformerModel, ProposedModel
from metrics import calculate_metrics
import torch
from sklearn.model_selection import train_test_split

def run_tfidf_experiment(train_texts, test_texts, train_labels, test_labels):
    model = TraditionalModels()
    y_pred = model.tfidf_cosine_similarity(train_texts, test_texts, train_labels)
    return calculate_metrics(test_labels, y_pred)

def run_word2vec_experiment(train_texts, test_texts, train_labels, test_labels, model_path):
    model = TraditionalModels()
    y_pred = model.word2vec_cosine_similarity(train_texts, test_texts, train_labels, model_path=model_path)
    return calculate_metrics(test_labels, y_pred)


def run_transformer_experiment(train_texts, train_labels, test_texts, test_labels, model_name='bert-base-uncased'):
    model = TransformerModel(model_name) 
    metrics = model.fine_tune_transformer(train_texts, train_labels, test_texts, test_labels)
    return metrics
    
def run_proposed_model_experiment(train_loader, test_loader, embedding_dim=768, num_heads=8):
    model = ProposedModel(embedding_dim, num_heads)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(5):  
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(batch['input_ids'])
            loss = criterion(outputs, batch['labels'].float())
            loss.backward()
            optimizer.step()
    
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            outputs = model(batch['input_ids'])
            all_preds.extend(outputs.round().cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())

    return calculate_metrics(all_labels, all_preds)