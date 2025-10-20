#Script for text generation
import argparse
import numpy as np
from .language_model import NgramLM
from .preprocess import START_TOKEN, END_TOKEN

#Generates a single sentence using the language model.
def generate_sentence(model, smoothing, max_length=20, lambdas=(0.2, 0.3, 0.5), alpha=0.4):
    #Start the sentence with N-1 start tokens
    context = (START_TOKEN,) * (model.n - 1)
    generated_words = []
    
    for _ in range(max_length):
        words, probs = model.get_next_word_dist(context, smoothing, lambdas, alpha)
        
        #Sample the next word from the probability distribution
        next_word = np.random.choice(words, p=probs)
        
        if next_word == END_TOKEN:
            break
        
        generated_words.append(next_word)
        
        #Update context
        if model.n > 1:
            context = context[1:] + (next_word,)

    return " ".join(generated_words)


def main():
    parser = argparse.ArgumentParser(description="Generate text with a trained N-gram model.")
    parser.add_argument('--model-file', type=str, required=True, help='Path to the trained model file.')
    parser.add_argument('--num-sentences', type=int, default=5, help='Number of sentences to generate.')
    parser.add_argument('--max-length', type=int, default=25, help='Maximum length of a sentence.')
    #Best lambdas
    parser.add_argument('--smoothing', type=str, required=True, choices=['add1', 'interpolation', 'backoff'])
    parser.add_argument('--lambdas', type=str, default="0.2,0.3,0.5", help='Lambda weights for interpolation.')
    parser.add_argument('--alpha', type=float, default=0.4, help='Alpha for stupid backoff.')

    args = parser.parse_args()

    model = NgramLM.load(args.model_file)
    lambdas = tuple(map(float, args.lambdas.split(','))) if args.smoothing == 'interpolation' else None
    
    print(f"--- Generating {args.num_sentences} sentences with '{args.smoothing}' smoothing ---")
    for i in range(args.num_sentences):
        sentence = generate_sentence(
            model, 
            args.smoothing,
            args.max_length,
            lambdas,
            args.alpha
        )
        print(f"{i+1}: {sentence}")


if __name__ == '__main__':
    main()