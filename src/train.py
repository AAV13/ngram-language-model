#Script to train and save models
import argparse
from .preprocess import tokenize_file, build_vocabulary, replace_unknowns
from .language_model import NgramLM

def main():
    parser = argparse.ArgumentParser(description="Train an N-gram language model.")
    parser.add_argument('--train-file', type=str, required=True, help='Path to training data.')
    parser.add_argument('--model-file', type=str, required=True, help='Path to save the trained model.')
    parser.add_argument('--n', type=int, required=True, help='Order of the N-gram model.')
    parser.add_argument('--vocab-threshold', type=int, default=1, help='Frequency threshold for vocabulary.')
    
    args = parser.parse_args()

    #1. Pre-process the data
    train_sents_tokenized = tokenize_file(args.train_file, args.n)
    vocabulary = build_vocabulary(train_sents_tokenized, args.vocab_threshold)
    train_sents_processed = replace_unknowns(train_sents_tokenized, vocabulary)

    #2. Initialize and train the model
    model = NgramLM(n=args.n, vocabulary=vocabulary)
    model.train(train_sents_processed)

    #3. Save the model
    model.save(args.model_file)
    print(f"Model successfully trained and saved to {args.model_file}")

if __name__ == '__main__':
    main()