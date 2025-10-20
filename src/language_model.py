#Core classes for N-gram models
import math
import pickle
from collections import defaultdict, Counter
from .preprocess import UNKNOWN_TOKEN

#An N-gram Language Model that supports MLE, Add-1, Interpolation, and Backoff.
class NgramLM:
    #Initializes the N-gram Language Model.
    def __init__(self, n, vocabulary):
        '''Args:
            n (int): The order of the N-gram model.
            vocabulary (set): The vocabulary set.'''
        self.n = n
        self.vocabulary = vocabulary
        self.vocab_size = len(vocabulary)
        
        #Data structures for counts
        #self.counts[k] will store k-gram counts.
        #e.g., self.counts[2][('the', 'cat')] = 10
        self.counts = defaultdict(Counter)
        self.total_tokens = 0

    #Trains the model by counting N-grams from the provided sentences.
    def train(self, tokenized_sentences):
        
        
        print(f"Training {self.n}-gram model...")
        for sentence in tokenized_sentences:
            self.total_tokens += len(sentence)
            #Generate n-grams for all orders from 1 to n
            for k in range(1, self.n + 1):
                for i in range(len(sentence) - k + 1):
                    ngram = tuple(sentence[i:i+k])
                    self.counts[k][ngram] += 1
        print("Training complete.")

    #Calculates Maximum Likelihood Estimation probability
    def _get_mle_prob(self, ngram):
        context_len = len(ngram) - 1
        if context_len == 0:  #Unigram
            return self.counts[1][ngram] / self.total_tokens

        context = ngram[:-1]
        context_count = self.counts[context_len][context]
        
        if context_count == 0:
            return 0.0 #No context seen, so probability is 0
            
        ngram_count = self.counts[len(ngram)][ngram]
        return ngram_count / context_count

    #Calculates Add-1 (Laplace) smoothed probability
    def _get_add1_prob(self, ngram):  
        context_len = len(ngram) - 1
        ngram_count = self.counts[len(ngram)][ngram]
        
        if context_len == 0: #Unigram
            return (ngram_count + 1) / (self.total_tokens + self.vocab_size)

        context = ngram[:-1]
        context_count = self.counts[context_len][context]
        
        return (ngram_count + 1) / (context_count + self.vocab_size)

    def _get_interpolated_prob(self, ngram, lambdas):
        #Calculates linearly interpolated probability
        #Ensure ngram has length n for this implementation
        if len(ngram) != self.n:
            #This case can be handled more robustly, but for perplexity calculation
            #on full sentences, we'll mostly see n-grams of length n.
            return self._get_mle_prob(ngram)

        trigram_prob = self._get_mle_prob(ngram)
        bigram_prob = self._get_mle_prob(ngram[1:])
        unigram_prob = self._get_mle_prob(ngram[2:])
        
        return (lambdas[2] * trigram_prob + 
                lambdas[1] * bigram_prob + 
                lambdas[0] * unigram_prob)

    def _get_backoff_prob(self, ngram, alpha):
        #Calculates Stupid Backoff score (not a true probability)
        k = len(ngram)
        if k == 1:
            #Base case: unigram MLE, or a small epsilon if count is 0
            return self._get_mle_prob(ngram) or 1e-10

        context_len = k - 1
        context = ngram[:-1]
        
        ngram_count = self.counts[k][ngram]
        context_count = self.counts[context_len][context]

        if ngram_count > 0 and context_count > 0:
            return ngram_count / context_count
        else:
            return alpha * self._get_backoff_prob(ngram[1:], alpha)

    #Calculates the total log probability of a sentence.
    def get_sentence_log_prob(self, sentence, smoothing, lambdas=(0.2, 0.3, 0.5), alpha=0.4):
        """
        Args:
            sentence (list): A tokenized sentence.
            smoothing (str): 'mle', 'add1', 'interpolation', or 'backoff'.
            lambdas (tuple): Interpolation weights for (unigram, bigram, trigram).
            alpha (float): Backoff factor.
            
        Returns:
            float: The total log2 probability of the sentence.
        """
        total_log_prob = 0.0
        #We start predicting from the first word, using the <s> tokens as context.
        for i in range(self.n - 1, len(sentence)):
            ngram = tuple(sentence[i - self.n + 1 : i + 1])
            
            prob = 0.0
            if smoothing == 'mle':
                prob = self._get_mle_prob(ngram)
            elif smoothing == 'add1':
                prob = self._get_add1_prob(ngram)
            elif smoothing == 'interpolation':
                prob = self._get_interpolated_prob(ngram, lambdas)
            elif smoothing == 'backoff':
                prob = self._get_backoff_prob(ngram, alpha)
            
            if prob == 0.0:
                return -math.inf #Zero probability makes the whole sentence impossible

            total_log_prob += math.log2(prob)
            
        return total_log_prob

    def get_next_word_dist(self, context, smoothing, lambdas=(0.2, 0.3, 0.5), alpha=0.4):
        '''
        Computes the probability distribution for the next word given a context.
        Used for text generation.
        
        Args:
            context (tuple): The preceding n-1 words.
            smoothing, lambdas, alpha: Model parameters.
            
        Returns:
            tuple: (list of words, list of probabilities)
        '''
        words = list(self.vocabulary)
        probs = []
        for word in words:
            ngram = context + (word,)
            prob = 0.0
            if smoothing == 'mle':
                prob = self._get_mle_prob(ngram)
            elif smoothing == 'add1':
                prob = self._get_add1_prob(ngram)
            elif smoothing == 'interpolation':
                prob = self._get_interpolated_prob(ngram, lambdas)
            elif smoothing == 'backoff':
                prob = self._get_backoff_prob(ngram, alpha)
            probs.append(prob)
        
        #Normalize probabilities for backoff, as it's not a true distribution
        if smoothing == 'backoff':
            total_prob = sum(probs)
            if total_prob > 0:
                probs = [p / total_prob for p in probs]
            else: #Handle case where all probs are zero
                probs = [1.0 / len(words)] * len(words)

        return words, probs

    #Saves the model to a file using pickle
    def save(self, filepath):
        print(f"Saving model to {filepath}")
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath):
        #Loads a model from a file
        print(f"Loading model from {filepath}")
        with open(filepath, 'rb') as f:
            return pickle.load(f)