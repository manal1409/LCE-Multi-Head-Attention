import experiments as exp
from dataset import load_data
from sklearn.model_selection import train_test_split

def main():
    # Load QQP dataset from Hugging Face
    qqp_texts, qqp_labels = load_data()

    # Split QQP dataset into train/test sets
    qqp_train_texts, qqp_test_texts, qqp_train_labels, qqp_test_labels = train_test_split(qqp_texts, qqp_labels, test_size=0.2, random_state=42)

    # Run TF-IDF Experiment on QQP
    print("\nRunning TF-IDF Experiment on QQP Dataset:")
    tfidf_results_qqp = exp.run_tfidf_experiment(qqp_train_texts, qqp_test_texts, qqp_train_labels, qqp_test_labels)
    print("TF-IDF Results on QQP:", tfidf_results_qqp)

    # Run Word2Vec Experiment on QQP
    print("\nRunning Word2Vec Experiment on QQP Dataset:")
    word2vec_results_qqp = exp.run_word2vec_experiment(qqp_train_texts, qqp_test_texts, qqp_train_labels, qqp_test_labels, model_path='GoogleNews-vectors-negative300.bin')
    print("Word2Vec Results on QQP:", word2vec_results_qqp)

    # Run Transformer Experiment on QQP
    print("\nRunning Transformer Experiment on QQP Dataset:")
    transformer_results_qqp = exp.run_transformer_experiment(qqp_train_texts, qqp_train_labels, qqp_test_texts, qqp_test_labels)
    print("Transformer Results on QQP:", transformer_results_qqp)

if __name__ == "__main__":
    main()