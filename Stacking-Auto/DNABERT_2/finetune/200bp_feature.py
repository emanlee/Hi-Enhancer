import os
import csv
import json
import logging
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Dict, List
import torch
import transformers
from torch.utils.data import Dataset

# 配置日志
logging.basicConfig(level=logging.INFO)

# 滑动窗口生成片段的函数
def sliding_window(sequence, window_size=200, step_size=50):
    for i in range(0, len(sequence) - window_size + 1, step_size):
        yield sequence[i:i + window_size], i

def read_and_split_sequences_from_csv(file_path, window_size=200, step_size=50, output_path='split_sequences.csv'):
    """
    读取CSV文件，使用滑动窗口生成序列片段，并保存为新的CSV文件。

    Parameters:
    file_path (str): 输入的CSV文件路径。
    window_size (int): 窗口大小，默认200bp。
    step_size (int): 滑动步长，默认50bp。
    output_path (str): 保存分割序列的文件路径。
    """
    df = pd.read_csv(file_path)
    sequences = df['sequence'].tolist()

    all_fragments = []
    for idx, sequence in enumerate(sequences):
        for fragment, start in sliding_window(sequence, window_size, step_size):
            all_fragments.append([fragment, idx, start])

    # 将生成的片段保存为CSV
    fragment_df = pd.DataFrame(all_fragments, columns=['fragment', 'sequence_idx', 'start'])
    fragment_df.to_csv(output_path, index=False)
    logging.info(f"Saved split sequences to {output_path}")

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="zhihan1996/DNABERT-2-117M")

@dataclass
class DataArguments:
    data_path: str = field(default=None)
    kmer: int = field(default=-1)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    output_dir: str = field(default="output")

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
        texts = df['fragment'].tolist()
        labels = [0] * len(texts)  # 可以自定义标签处理逻辑

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

def save_features_to_csv(dataset: SupervisedDataset, output_path: str):
    features = {
        "input_ids": dataset.input_ids.numpy().tolist(),
        "attention_mask": dataset.attention_mask.numpy().tolist(),
        "labels": dataset.labels
    }
    
    print(f"Input IDs length: {len(features['input_ids'])}")
    print(f"Attention Mask length: {len(features['attention_mask'])}")
    print(f"Labels length: {len(features['labels'])}")
    
    df = pd.DataFrame(features)
    df.to_csv(output_path, index=False)

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

    # 加载分割的序列片段数据集
    split_sequences_path = os.path.join(data_args.data_path, "split_sequences.csv")
    
    # 确保文件存在
    if not os.path.exists(split_sequences_path):
        logging.error(f"Split sequences file not found: {split_sequences_path}")
        return

    dataset = SupervisedDataset(data_path=split_sequences_path, tokenizer=tokenizer, kmer=data_args.kmer)

    # 提取特征并保存为CSV文件
    features_output_path = os.path.join(training_args.output_dir, "features.csv")
    save_features_to_csv(dataset, features_output_path)

if __name__ == "__main__":
    # 先进行序列分割
    input_csv_path = 'dev.csv'  # 输入文件路径
    split_sequences_output_path = './sample_data/split_sequences.csv'  # 确保这里是你想要保存的路径
    read_and_split_sequences_from_csv(input_csv_path, window_size=200, step_size=50, output_path=split_sequences_output_path)

    # 提取特征
    extract_features()
