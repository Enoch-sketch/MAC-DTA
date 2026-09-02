"""
Vocabulary building and loading utilities for PMHGT-DTA
"""
import pickle
import os
from collections import Counter


class WordVocab:
    """
    Vocabulary class for encoding SMILES and protein sequences
    """
    def __init__(self, texts=None, max_size=None, min_freq=1):
        self.pad_index = 0
        self.unk_index = 1
        self.eos_index = 2
        self.sos_index = 3
        self.mask_index = 4
        
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.eos_token = "<eos>"
        self.sos_token = "<sos>"
        self.mask_token = "<mask>"
        
        self.stoi = {}
        self.itos = {}
        
        if texts is not None:
            self._build_vocab(texts, max_size, min_freq)
    
    def _build_vocab(self, texts, max_size, min_freq):
        """Build vocabulary from texts"""
        counter = Counter()
        for text in texts:
            counter.update(list(text))
        
        # Special tokens
        self.stoi = {
            self.pad_token: self.pad_index,
            self.unk_token: self.unk_index,
            self.eos_token: self.eos_index,
            self.sos_token: self.sos_index,
            self.mask_token: self.mask_index,
        }
        
        # Add regular tokens
        idx = 5
        for token, freq in counter.most_common(max_size):
            if freq >= min_freq:
                self.stoi[token] = idx
                idx += 1
        
        # Build reverse mapping
        self.itos = {v: k for k, v in self.stoi.items()}
    
    def __len__(self):
        return len(self.stoi)
    
    def save_vocab(self, vocab_path):
        """Save vocabulary to file"""
        with open(vocab_path, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load_vocab(vocab_path):
        """Load vocabulary from file"""
        with open(vocab_path, 'rb') as f:
            return pickle.load(f)


def build_smiles_vocab(smiles_list, save_path=None):
    """
    Build vocabulary for SMILES strings
    """
    # Common SMILES tokens (single and double character)
    tokens = set()
    for smiles in smiles_list:
        i = 0
        while i < len(smiles):
            # Check for two-character tokens first
            if i + 1 < len(smiles):
                two_char = smiles[i:i+2]
                if two_char in ['Cl', 'Br', 'Si', 'Se', 'Na', 'Li', 'Mg', 'Ca', 'Fe', 'Zn']:
                    tokens.add(two_char)
                    i += 2
                    continue
            tokens.add(smiles[i])
            i += 1
    
    vocab = WordVocab()
    vocab.stoi = {
        vocab.pad_token: vocab.pad_index,
        vocab.unk_token: vocab.unk_index,
        vocab.eos_token: vocab.eos_index,
        vocab.sos_token: vocab.sos_index,
        vocab.mask_token: vocab.mask_index,
    }
    
    idx = 5
    for token in sorted(tokens):
        vocab.stoi[token] = idx
        idx += 1
    
    vocab.itos = {v: k for k, v in vocab.stoi.items()}
    
    if save_path:
        vocab.save_vocab(save_path)
    
    return vocab


def build_protein_vocab(sequences, save_path=None):
    """
    Build vocabulary for protein sequences
    """
    # Standard amino acids
    amino_acids = set('ACDEFGHIKLMNPQRSTVWY')
    
    # Add any additional characters found in sequences
    for seq in sequences:
        for char in seq:
            amino_acids.add(char)
    
    vocab = WordVocab()
    vocab.stoi = {
        vocab.pad_token: vocab.pad_index,
        vocab.unk_token: vocab.unk_index,
        vocab.eos_token: vocab.eos_index,
        vocab.sos_token: vocab.sos_index,
        vocab.mask_token: vocab.mask_index,
    }
    
    idx = 5
    for aa in sorted(amino_acids):
        vocab.stoi[aa] = idx
        idx += 1
    
    vocab.itos = {v: k for k, v in vocab.stoi.items()}
    
    if save_path:
        vocab.save_vocab(save_path)
    
    return vocab


if __name__ == "__main__":
    import pandas as pd
    import os
    
    # Create Vocab directory
    os.makedirs('Vocab', exist_ok=True)
    
    # Build vocabularies from davis dataset
    if os.path.exists('davis_processed.csv'):
        df = pd.read_csv('davis_processed.csv')
    elif os.path.exists('davis_processed/davis_processed.csv'):
        df = pd.read_csv('davis_processed/davis_processed.csv')
    else:
        print("Could not find davis_processed.csv")
        exit(1)
    
    # Build SMILES vocabulary
    smiles_list = df['compound_iso_smiles'].unique().tolist()
    smiles_vocab = build_smiles_vocab(smiles_list, 'Vocab/smiles_vocab.pkl')
    print(f"SMILES vocabulary size: {len(smiles_vocab)}")
    
    # Build protein vocabulary
    sequences = df['target_sequence'].unique().tolist()
    protein_vocab = build_protein_vocab(sequences, 'Vocab/protein_vocab.pkl')
    print(f"Protein vocabulary size: {len(protein_vocab)}")
    
    print("Vocabularies saved to Vocab/")

