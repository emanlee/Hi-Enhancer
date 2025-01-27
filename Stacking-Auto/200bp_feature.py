import os
import json
import logging
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Dict, List
import torch
import transformers
from torch.utils.data import Dataset

# Configure logging
logging.basicConfig(level=logging.INFO)

# Function to generate fragments using a sliding window
def sliding_window(sequence, window_size=200, step_size=50):
    for i in range(0, len(sequence) - window_size + 1, step_size):
        yield sequence[i:i + window_size], i

def read_and_split_sequences_from_csv(file_path, window_size=200, step_size=50, output_path='./split_sequences.csv'):
    """
    Read the CSV file, generate sequence fragments using a sliding window, and save them to a new CSV file.

    Parameters:
    file_path (str): Path to the input CSV file.
    window_size (int): Window size, default is 200bp.
    step_size (int): Sliding step size, default is 50bp.
    output_path (str): Path to save the split sequences.
    """
    df = pd.read_csv(file_path)
    sequences = df['sequence'].tolist()

    all_fragments = []
    for idx, sequence in enumerate(sequences):
        for fragment, start in sliding_window(sequence, window_size, step_size):
            all_fragments.append([fragment, idx, start])

    # Save the generated fragments as CSV
    fragment_df = pd.DataFrame(all_fragments, columns=['fragment', 'sequence_idx', 'start'])

    # Only keep the first column and rename it to 'x0'
    fragment_df = fragment_df[['fragment']]  # Keep only the 'fragment' column
    fragment_df = fragment_df.rename(columns={'fragment': 'x0'})  # Rename to 'x0'

    # Final file will only contain the 'x0' column
    fragment_df.to_csv(output_path, index=False)
    logging.info(f"Saved split sequences to {output_path}")

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="zhihan1996/DNABERT-2-117M")

@dataclass
class DataArguments:
    data_path: str = field(default='./')  # Default to current directory
    kmer: int = field(default=-1)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    output_dir: str = field(default="./output")

def generate_kmer_str(sequence: str, k: int) -> str:
    return " ".join([sequence[i:i+k] for i in range(len(sequence) - k + 1)])

def load_or_generate_kmer(data_path: str, texts: List[str], k: int) -> List[str]:
    kmer_path = data_path.replace(".csv", f"_{k}mer.json")
    if os.path.exists(kmer_path):
        logging.warning(f"Loading k-mer from {kmer_path}...")
        with open(kmer_path, "r") as f:
            kmer = json.load(f)
    else:
        logging.warning(f"Generating k-mer...")
        kmer = [generate_kmer_str(text, k) for text in texts]
        with open(kmer_path, "w") as f:
            logging.warning(f"Saving k-mer to {kmer_path}...")
            json.dump(kmer, f)
    return kmer

class SupervisedDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, kmer: int = -1):
        super(SupervisedDataset, self).__init__()
        df = pd.read_csv(data_path)
        texts = df['x0'].tolist()  # Changed to read the 'x0' column
        labels = [0] * len(texts)  # Can customize label processing logic

        if kmer != -1:
            logging.warning(f"Using {kmer}-mer as input...")
            texts = load_or_generate_kmer(data_path, texts, kmer)

        output = tokenizer(
            texts,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        self.input_ids = output["input_ids"]
        self.attention_mask = output["attention_mask"]
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

def save_features_to_csv(dataset: SupervisedDataset, output_path='./features.csv'):
    # Only keep the 'input_ids' column
    features = {
        "input_ids": dataset.input_ids.numpy().tolist(),  # Keep 'input_ids'
    }
    
    print(f"Input IDs length: {len(features['input_ids'])}")
    
    # Create a DataFrame and rename the column
    df = pd.DataFrame(features)
    df = df.rename(columns={'input_ids': 'x0'})  # Rename to 'x0'
    
    # Save to CSV file
    df.to_csv(output_path, index=False)
    logging.info(f"Saved features to {output_path}")


def extract_features():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=512,
        use_fast=True,
        trust_remote_code=True,
    )

    # Load the split sequence fragment dataset
    split_sequences_path = os.path.join(data_args.data_path, "split_sequences.csv")
    
    # Ensure the file exists
    if not os.path.exists(split_sequences_path):
        logging.error(f"Split sequences file not found: {split_sequences_path}")
        return

    dataset = SupervisedDataset(data_path=split_sequences_path, tokenizer=tokenizer, kmer=data_args.kmer)

    # Extract features and save to CSV file
    features_output_path = './features.csv'  # Output to current folder
    save_features_to_csv(dataset, features_output_path)

if __name__ == "__main__":
    # First, split the sequences
    input_csv_path = './dev.csv'  # Input file path
    split_sequences_output_path = './split_sequences.csv'  # Output file path (current directory)
    read_and_split_sequences_from_csv(input_csv_path, window_size=200, step_size=50, output_path=split_sequences_output_path)

    # Extract features
    extract_features()
