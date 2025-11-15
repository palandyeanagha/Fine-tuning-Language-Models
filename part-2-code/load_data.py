import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        self.split = split
        self.tokenizer = T5TokenizerFast.from_pretrained('t5-small')
        self.data = self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        nl_path = os.path.join(data_folder, f'{split}.nl')
        sql_path = os.path.join(data_folder, f'{split}.sql')
        
        nl_lines = load_lines(nl_path)
        
        if split == 'test':
            # Test set doesn't have SQL queries
            return [(nl, None) for nl in nl_lines]
        else:
            sql_lines = load_lines(sql_path)
            return list(zip(nl_lines, sql_lines))
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        nl_query, sql_query = self.data[idx]
        
        # Tokenize encoder input (natural language query)
        encoder_inputs = self.tokenizer(
            nl_query,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        encoder_ids = encoder_inputs['input_ids'].squeeze(0)
        encoder_mask = encoder_inputs['attention_mask'].squeeze(0)
        
        if self.split == 'test':
            # For test set, we don't have targets
            return {
                'encoder_ids': encoder_ids,
                'encoder_mask': encoder_mask,
                'nl_query': nl_query
            }
        else:
            # Tokenize decoder input and targets
            # For T5, decoder input is the target sequence shifted right
            decoder_inputs = self.tokenizer(
                sql_query,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            decoder_ids = decoder_inputs['input_ids'].squeeze(0)
            
            # Decoder input: shift right by one position, prepend pad token
            # This is for teacher forcing during training
            # We need to ensure same length as targets (max_length)
            decoder_input = torch.cat([
                torch.tensor([self.tokenizer.pad_token_id]),
                decoder_ids[:-1]
            ])
            # Ensure same length (should already be max_length, but just in case)
            if len(decoder_input) < len(decoder_ids):
                # Pad to match
                padding = torch.full((len(decoder_ids) - len(decoder_input),), self.tokenizer.pad_token_id)
                decoder_input = torch.cat([decoder_input, padding])
            elif len(decoder_input) > len(decoder_ids):
                # Truncate (shouldn't happen)
                decoder_input = decoder_input[:len(decoder_ids)]
            
            # Decoder targets: the actual SQL tokens (what we predict)
            decoder_targets = decoder_ids
            
            # Initial decoder input (just the pad token for generation)
            initial_decoder_input = torch.tensor([self.tokenizer.pad_token_id])
            
            return {
                'encoder_ids': encoder_ids,
                'encoder_mask': encoder_mask,
                'decoder_input': decoder_input,
                'decoder_targets': decoder_targets,
                'initial_decoder_input': initial_decoder_input,
                'nl_query': nl_query,
                'sql_query': sql_query
            }

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_ids = [item['encoder_ids'] for item in batch]
    encoder_mask = [item['encoder_mask'] for item in batch]
    decoder_input = [item['decoder_input'] for item in batch]
    decoder_targets = [item['decoder_targets'] for item in batch]
    initial_decoder_input = [item['initial_decoder_input'] for item in batch]
    
    # Stack tensors (they're already padded to max_length in __getitem__)
    encoder_ids = torch.stack(encoder_ids)
    encoder_mask = torch.stack(encoder_mask)
    decoder_input = torch.stack(decoder_input)
    decoder_targets = torch.stack(decoder_targets)
    initial_decoder_input = torch.stack(initial_decoder_input)
    
    return encoder_ids, encoder_mask, decoder_input, decoder_targets, initial_decoder_input

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_ids = [item['encoder_ids'] for item in batch]
    encoder_mask = [item['encoder_mask'] for item in batch]
    initial_decoder_input = [torch.tensor([PAD_IDX]) for _ in batch]  # pad token as start
    nl_queries = [item['nl_query'] for item in batch]
    
    # Stack tensors
    encoder_ids = torch.stack(encoder_ids)
    encoder_mask = torch.stack(encoder_mask)
    initial_decoder_input = torch.stack(initial_decoder_input)
    
    return encoder_ids, encoder_mask, initial_decoder_input, nl_queries

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    # TODO
    return train_x, train_y, dev_x, dev_y, test_x