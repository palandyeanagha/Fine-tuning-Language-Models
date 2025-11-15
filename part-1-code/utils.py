import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example

# acc: 92.028
# def custom_transform(example):
#     text = example["text"]
#     words = word_tokenize(text)
#     new_words = []
#     detok = TreebankWordDetokenizer()

#     for w in words:
#         # 20% chance to replace
#         if random.random() < 0.2:
#             syns = wordnet.synsets(w)
#             if syns:
#                 lemmas = syns[0].lemmas()
#                 if lemmas:
#                     synonym = lemmas[0].name().replace('_', ' ')
#                     if synonym.lower() != w.lower():
#                         new_words.append(synonym)
#                         continue
#         new_words.append(w)

#     example["text"] = detok.detokenize(new_words)
#     return example

# Helper: replace some words with synonyms
def synonym_replace(text, prob=0.4):
    words = word_tokenize(text)
    new_words = []
    for word in words:
        if random.random() < prob:
            synsets = wordnet.synsets(word)
            if synsets:
                # Get all lemma names, filter out the original word
                lemmas = [lemma.name() for syn in synsets for lemma in syn.lemmas() if lemma.name().lower() != word.lower()]
                if lemmas:
                    new_word = random.choice(lemmas).replace("_", " ")
                    new_words.append(new_word)
                    continue
        new_words.append(word)
    return TreebankWordDetokenizer().detokenize(new_words)

# Helper: introduce typos (keyboard-neighbor swaps)
KEYBOARD_NEIGHBORS = {
    'a': 'qwsz', 'b': 'vghn', 'c': 'xdfv', 'd': 'serfcx', 'e': 'wsdr', 'f': 'drtgvc',
    'g': 'ftyhbv', 'h': 'gyujnb', 'i': 'ujko', 'j': 'huikmn', 'k': 'jiolm', 'l': 'kop',
    'm': 'njk', 'n': 'bhjm', 'o': 'iklp', 'p': 'ol', 'q': 'wa', 'r': 'edft', 's': 'awedz',
    't': 'rfgy', 'u': 'yhji', 'v': 'cfgb', 'w': 'qase', 'x': 'zsdc', 'y': 'tghu', 'z': 'asx'
}

def introduce_typos(text, char_prob=0.15):
    words = word_tokenize(text)
    new_words = []
    for word in words:
        new_word = ""
        for c in word:
            if c.lower() in KEYBOARD_NEIGHBORS and random.random() < char_prob:
                new_word += random.choice(KEYBOARD_NEIGHBORS[c.lower()])
            else:
                new_word += c
        new_words.append(new_word)
    return TreebankWordDetokenizer().detokenize(new_words)

# Main custom transform
def custom_transform(example):
    text = example["text"]
    # Apply synonym replacement (~40% words)
    text = synonym_replace(text, prob=0.4)
    # Apply typos (~15% of characters)
    text = introduce_typos(text, char_prob=0.15)
    example["text"] = text
    return example

### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


# def custom_transform(example):
#     ################################
#     ##### YOUR CODE BEGINGS HERE ###

#     # Design and implement the transformation as mentioned in pdf
#     # You are free to implement any transformation but the comments at the top roughly describe
#     # how you could implement two of them --- synonym replacement and typos.

#     # You should update example["text"] using your transformation
#     text = example["text"]
#     words = word_tokenize(text)
#     new_words = []
#     detok = TreebankWordDetokenizer()

#     for w in words:
#         # 20% chance to replace
#         if random.random() < 0.2:
#             syns = wordnet.synsets(w)
#             if syns:
#                 lemmas = syns[0].lemmas()
#                 if lemmas:
#                     synonym = lemmas[0].name().replace('_', ' ')
#                     if synonym.lower() != w.lower():
#                         new_words.append(synonym)
#                         continue
#         new_words.append(w)

#     example["text"] = detok.detokenize(new_words)
#     #return example

#     #raise NotImplementedError

#     ##### YOUR CODE ENDS HERE ######

#     return example

