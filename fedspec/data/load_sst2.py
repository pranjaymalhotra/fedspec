"""
Load SST-2 dataset from GLUE benchmark.
Provides tokenized dataset ready for training.
"""
import torch
from typing import Dict, Tuple
from datasets import load_dataset
from transformers import AutoTokenizer


def load_sst2(
    model_name: str = "bert-base-uncased",
    max_length: int = 128,
    seed: int = 42
) -> Tuple[Dict, Dict, AutoTokenizer]:
    """
    Load and tokenize SST-2 dataset.
    
    SST-2 (Stanford Sentiment Treebank) is a binary sentiment classification task.
    Labels: 0 = negative, 1 = positive
    
    Args:
        model_name: HuggingFace model name for tokenizer
        max_length: Maximum sequence length for tokenization
        seed: Random seed for shuffling
    
    Returns:
        Tuple of (train_dataset, val_dataset, tokenizer)
        - train_dataset: Dict with 'input_ids', 'attention_mask', 'labels'
        - val_dataset: Dict with 'input_ids', 'attention_mask', 'labels'
        - tokenizer: The tokenizer instance
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load SST-2 from GLUE
    dataset = load_dataset("glue", "sst2")
    
    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples["sentence"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors=None  # Return lists, will convert to tensors later
        )
    
    # Tokenize datasets
    train_data = dataset["train"].shuffle(seed=seed)
    val_data = dataset["validation"]
    
    train_tokenized = train_data.map(
        tokenize_function,
        batched=True,
        remove_columns=["sentence", "idx"]
    )
    val_tokenized = val_data.map(
        tokenize_function,
        batched=True,
        remove_columns=["sentence", "idx"]
    )
    
    # Convert to tensors
    train_dataset = {
        "input_ids": torch.tensor(train_tokenized["input_ids"], dtype=torch.long),
        "attention_mask": torch.tensor(train_tokenized["attention_mask"], dtype=torch.long),
        "labels": torch.tensor(train_tokenized["label"], dtype=torch.long)
    }
    
    val_dataset = {
        "input_ids": torch.tensor(val_tokenized["input_ids"], dtype=torch.long),
        "attention_mask": torch.tensor(val_tokenized["attention_mask"], dtype=torch.long),
        "labels": torch.tensor(val_tokenized["label"], dtype=torch.long)
    }
    
    # Shapes:
    # train_dataset["input_ids"]: (67349, 128) for SST-2 train
    # train_dataset["attention_mask"]: (67349, 128)
    # train_dataset["labels"]: (67349,)
    # val_dataset["input_ids"]: (872, 128) for SST-2 validation
    
    return train_dataset, val_dataset, tokenizer


def get_dataloader(
    dataset: Dict,
    batch_size: int,
    shuffle: bool = True,
    seed: int = 42
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader from a tokenized dataset dict.
    
    Args:
        dataset: Dict with 'input_ids', 'attention_mask', 'labels'
        batch_size: Batch size
        shuffle: Whether to shuffle
        seed: Random seed for shuffling
    
    Returns:
        DataLoader instance
    """
    # Create TensorDataset
    tensor_dataset = torch.utils.data.TensorDataset(
        dataset["input_ids"],
        dataset["attention_mask"],
        dataset["labels"]
    )
    
    # Create generator for reproducibility
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    dataloader = torch.utils.data.DataLoader(
        tensor_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        drop_last=False
    )
    
    return dataloader
