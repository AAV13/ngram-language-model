#Script to evaluate models and get perplexity
import argparse
import math
from .language_model import NgramLM
from .preprocess import tokenize_file, replace_unknowns

#Calculates the perplexity of a model on a given dataset.
def calculate_perplexity(model, tokenized_sentences, smoothing, lambdas, alpha):

    total_log_prob = 0
    total_tokens = 0 #M in the formula, excluding <s> tokens

    for sentence in tokenized_sentences:
        #Perplexity calculation does not include <s> tokens in the count
        total_tokens += len(sentence) - (model.n - 1)
        log_prob = model.get_sentence_log_prob(sentence, smoothing, lambdas, alpha)
        
        if log_prob == -math.inf:
            return math.inf #If any sentence has zero probability, perplexity is infinite
        
        total_log_prob += log_prob

    #Check for empty input files
    if total_tokens == 0:
        print("Warning: Input file is empty or contains no tokens.")
        return float('nan') #Return Not a Number for an undefined result

    #Perplexity formula: PP = exp(-1/M * sum(log2(P)))
    #log2(P) is used, so we use 2 as the base for exp.
    cross_entropy = -total_log_prob / total_tokens
    perplexity = 2 ** cross_entropy
    return perplexity

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained N-gram language model.")
    parser.add_argument('--model-file', type=str, required=True, help='Path to the trained model file.')
    parser.add_argument('--test-file', type=str, required=True, help='Path to the test data.')
    parser.add_argument('--smoothing', type=str, required=True, choices=['mle', 'add1', 'interpolation', 'backoff'])
    #Lambdas as a comma-separated string, e.g., "0.2,0.3,0.5"
    parser.add_argument('--lambdas', type=str, default="0.2,0.3,0.5", help='Lambda weights for interpolation.')
    parser.add_argument('--alpha', type=float, default=0.4, help='Alpha for stupid backoff.')
    args = parser.parse_args()

    #Load the model
    model = NgramLM.load(args.model_file)
    
    #Process test data using the *model's* vocabulary
    test_sents_tokenized = tokenize_file(args.test_file, model.n)
    test_sents_processed = replace_unknowns(test_sents_tokenized, model.vocabulary)

    lambdas = tuple(map(float, args.lambdas.split(','))) if args.smoothing == 'interpolation' else None

    #Calculate perplexity
    pp = calculate_perplexity(model, test_sents_processed, args.smoothing, lambdas, args.alpha)
    print(f"Model: {args.model_file}")
    print(f"Test Set: {args.test_file}")
    print(f"Smoothing: {args.smoothing}")
    if args.smoothing == 'interpolation':
        print(f"Lambdas: {lambdas}")
    if args.smoothing == 'backoff':
        print(f"Alpha: {args.alpha}")
    print("-" * 20)
    print(f"Perplexity: {pp:.4f}")

if __name__ == '__main__':
    main()