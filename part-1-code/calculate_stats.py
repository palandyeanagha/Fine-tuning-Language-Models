#!/usr/bin/env python3
"""Calculate data statistics before and after T5 tokenization"""

from transformers import T5TokenizerFast
import numpy as np
import re
from collections import Counter

tokenizer = T5TokenizerFast.from_pretrained('t5-small')

# Read files
with open('data/train.nl', 'r') as f:
    train_nl = [line.strip() for line in f.readlines()]
with open('data/dev.nl', 'r') as f:
    dev_nl = [line.strip() for line in f.readlines()]
with open('data/train.sql', 'r') as f:
    train_sql = [line.strip() for line in f.readlines()]
with open('data/dev.sql', 'r') as f:
    dev_sql = [line.strip() for line in f.readlines()]

print("=" * 60)
print("BEFORE PREPROCESSING (word-level)")
print("=" * 60)

def get_word_stats(texts, is_sql=False):
    lengths = []
    all_words = []
    for text in texts:
        if is_sql:
            words = re.findall(r'\w+|[^\w\s]', text)
        else:
            words = text.lower().split()
        lengths.append(len(words))
        all_words.extend(words)
    mean_len = np.mean(lengths) if lengths else 0
    vocab_size = len(set(all_words))
    return mean_len, vocab_size

train_nl_mean, train_nl_vocab = get_word_stats(train_nl, False)
dev_nl_mean, dev_nl_vocab = get_word_stats(dev_nl, False)
train_sql_mean, train_sql_vocab = get_word_stats(train_sql, True)
dev_sql_mean, dev_sql_vocab = get_word_stats(dev_sql, True)

print(f'Train examples: {len(train_nl)}')
print(f'Dev examples: {len(dev_nl)}')
print(f'Train NL mean length: {train_nl_mean:.2f}')
print(f'Dev NL mean length: {dev_nl_mean:.2f}')
print(f'Train SQL mean length: {train_sql_mean:.2f}')
print(f'Dev SQL mean length: {dev_sql_mean:.2f}')
print(f'Train NL vocab: {train_nl_vocab}')
print(f'Dev NL vocab: {dev_nl_vocab}')
print(f'Train SQL vocab: {train_sql_vocab}')
print(f'Dev SQL vocab: {dev_sql_vocab}')

print("\n" + "=" * 60)
print("AFTER T5 TOKENIZATION")
print("=" * 60)

def get_tokenized_stats(texts, tokenizer):
    lengths = []
    all_tokens = set()
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        lengths.append(len(tokens))
        all_tokens.update(tokens)
    mean_len = np.mean(lengths) if lengths else 0
    vocab_size = len(all_tokens)
    return mean_len, vocab_size

train_nl_tok_mean, train_nl_tok_vocab = get_tokenized_stats(train_nl, tokenizer)
dev_nl_tok_mean, dev_nl_tok_vocab = get_tokenized_stats(dev_nl, tokenizer)
train_sql_tok_mean, train_sql_tok_vocab = get_tokenized_stats(train_sql, tokenizer)
dev_sql_tok_mean, dev_sql_tok_vocab = get_tokenized_stats(dev_sql, tokenizer)

print(f'Train NL mean token length: {train_nl_tok_mean:.2f}')
print(f'Dev NL mean token length: {dev_nl_tok_mean:.2f}')
print(f'Train SQL mean token length: {train_sql_tok_mean:.2f}')
print(f'Dev SQL mean token length: {dev_sql_tok_mean:.2f}')
print(f'Train NL unique tokens: {train_nl_tok_vocab}')
print(f'Dev NL unique tokens: {dev_nl_tok_vocab}')
print(f'Train SQL unique tokens: {train_sql_tok_vocab}')
print(f'Dev SQL unique tokens: {dev_sql_tok_vocab}')