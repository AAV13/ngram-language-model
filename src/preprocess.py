#For data loading and tokenization
import os
from collections import Counter

#Define special tokens
START_TOKEN = "<s>"
END_TOKEN = "</s>"
UNKNOWN_TOKEN = "<unk>"

def tokenize_line(line, n):
    '''
    Tokenizes a single line of text, converts to lowercase,
    and adds start/end tokens.
    
    Args:
        line (str): A string representing a sentence.
        n (int): The order of the N-gram model, used to determine
                 the number of start tokens.
    
    Returns:
        list: A list of tokens for the sentence.
    '''
    line = line.strip().lower()
    tokens = line.split()
    #For an N-gram model, we need N-1 start tokens
    start_tokens = [START_TOKEN] * (n - 1) if n > 1 else []
    return start_tokens + tokens + [END_TOKEN]

def tokenize_file(filepath, n):
    '''
    Tokenizes all lines in a given file.
    
    Args:
        filepath (str): Path to the text file.
        n (int): The N-gram order.
    
    Returns:
        list: A list of tokenized sentences (each sentence is a list of tokens).
    '''
    print(f"Tokenizing file: {filepath}")
    tokenized_sentences = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            tokenized_sentences.append(tokenize_line(line, n))
    return tokenized_sentences

def build_vocabulary(tokenized_sentences, threshold=1):
    """
    Builds a vocabulary from tokenized training data, replacing rare words
    with an unknown token.
    
    Args:
        tokenized_sentences (list): A list of tokenized sentences.
        threshold (int): The frequency threshold below which words are
                         replaced by UNKNOWN_TOKEN.
    
    Returns:
        set: The final vocabulary.
    """
    print("Building vocabulary...")
    word_counts = Counter()
    for sentence in tokenized_sentences:
        word_counts.update(sentence)

    #The vocabulary includes all words that meet the frequency threshold,
    #plus the special tokens. Note that <s> and </s> are not replaced.
    vocabulary = {word for word, count in word_counts.items() if count > threshold}
    vocabulary.add(UNKNOWN_TOKEN)
    vocabulary.add(END_TOKEN)
    #Start token is added separately as it shouldn't be counted for <unk>
    if START_TOKEN in word_counts:
        vocabulary.add(START_TOKEN)
        
    print(f"Vocabulary size: {len(vocabulary)}")
    return vocabulary

def replace_unknowns(tokenized_sentences, vocabulary):
    '''
    Replaces out-of-vocabulary words in a dataset with the UNKNOWN_TOKEN.
    
    Args:
        tokenized_sentences (list): The dataset to process.
        vocabulary (set): The vocabulary built from the training data.
    
    Returns:
        list: The processed dataset with OOV words replaced.
    '''
    print("Replacing unknown words...")
    processed_sentences = []
    for sentence in tokenized_sentences:
        processed_sentence = [
            word if word in vocabulary else UNKNOWN_TOKEN for word in sentence
        ]
        processed_sentences.append(processed_sentence)
    return processed_sentences