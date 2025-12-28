"""
Federated data splitting utilities.
Implements IID and non-IID (Dirichlet) splits.
"""
import torch
import numpy as np
from typing import Dict, List


def iid_split(
    dataset: Dict,
    num_clients: int,
    seed: int = 42
) -> List[Dict]:
    """
    Split dataset into IID partitions for federated learning.
    
    Each client receives approximately equal number of samples,
    with same label distribution as the original dataset.
    
    Args:
        dataset: Dict with 'input_ids', 'attention_mask', 'labels'
        num_clients: Number of clients to split data among
        seed: Random seed for reproducibility
    
    Returns:
        List of client datasets, each a Dict with same keys as input
    """
    np.random.seed(seed)
    
    num_samples = dataset["input_ids"].shape[0]
    indices = np.random.permutation(num_samples)
    
    # Split indices into approximately equal parts
    # np.array_split handles uneven divisions
    client_indices = np.array_split(indices, num_clients)
    
    client_datasets = []
    for i, client_idx in enumerate(client_indices):
        client_idx = torch.tensor(client_idx, dtype=torch.long)
        client_data = {
            "input_ids": dataset["input_ids"][client_idx],
            "attention_mask": dataset["attention_mask"][client_idx],
            "labels": dataset["labels"][client_idx]
        }
        client_datasets.append(client_data)
        # Shape: client_data["input_ids"]: (num_samples_client_i, max_length)
    
    return client_datasets


def dirichlet_split(
    dataset: Dict,
    num_clients: int,
    alpha: float = 0.5,
    seed: int = 42
) -> List[Dict]:
    """
    Split dataset using Dirichlet distribution for non-IID federated learning.
    
    Lower alpha = more heterogeneous (non-IID)
    Higher alpha = more homogeneous (closer to IID)
    
    alpha = 0.1: highly non-IID
    alpha = 0.5: moderately non-IID
    alpha = 1.0: slightly non-IID
    alpha -> inf: IID
    
    For each class, samples are distributed among clients according to
    Dirichlet(alpha, ..., alpha) distribution.
    
    Args:
        dataset: Dict with 'input_ids', 'attention_mask', 'labels'
        num_clients: Number of clients to split data among
        alpha: Dirichlet concentration parameter
        seed: Random seed for reproducibility
    
    Returns:
        List of client datasets, each a Dict with same keys as input
    """
    np.random.seed(seed)
    
    labels = dataset["labels"].numpy()
    num_samples = len(labels)
    num_classes = len(np.unique(labels))  # For SST-2: 2 classes
    
    # Initialize client indices
    client_indices = [[] for _ in range(num_clients)]
    
    # For each class, distribute samples among clients using Dirichlet
    for class_idx in range(num_classes):
        # Get indices of samples belonging to this class
        class_sample_indices = np.where(labels == class_idx)[0]
        np.random.shuffle(class_sample_indices)
        
        # Sample Dirichlet distribution for this class
        # proportions[i] = fraction of class samples going to client i
        proportions = np.random.dirichlet([alpha] * num_clients)
        
        # Convert proportions to sample counts
        # Ensure sum equals number of class samples
        proportions = proportions / proportions.sum()
        proportions = (proportions * len(class_sample_indices)).astype(int)
        
        # Handle rounding errors by adding remainder to random clients
        remainder = len(class_sample_indices) - proportions.sum()
        if remainder > 0:
            for i in np.random.choice(num_clients, remainder, replace=False):
                proportions[i] += 1
        elif remainder < 0:
            # Remove from clients with most samples
            for i in np.argsort(proportions)[::-1][:abs(remainder)]:
                proportions[i] -= 1
        
        # Assign samples to clients based on proportions
        start_idx = 0
        for client_id in range(num_clients):
            end_idx = start_idx + proportions[client_id]
            client_indices[client_id].extend(
                class_sample_indices[start_idx:end_idx].tolist()
            )
            start_idx = end_idx
    
    # Create client datasets
    client_datasets = []
    for client_id in range(num_clients):
        if len(client_indices[client_id]) == 0:
            # Handle edge case: client has no data
            # This can happen with very low alpha and many clients
            client_data = {
                "input_ids": torch.zeros((0, dataset["input_ids"].shape[1]), dtype=torch.long),
                "attention_mask": torch.zeros((0, dataset["attention_mask"].shape[1]), dtype=torch.long),
                "labels": torch.zeros((0,), dtype=torch.long)
            }
        else:
            idx = torch.tensor(client_indices[client_id], dtype=torch.long)
            client_data = {
                "input_ids": dataset["input_ids"][idx],
                "attention_mask": dataset["attention_mask"][idx],
                "labels": dataset["labels"][idx]
            }
        client_datasets.append(client_data)
    
    return client_datasets


def compute_label_distribution(dataset: Dict) -> Dict[int, int]:
    """
    Compute label distribution for a dataset.
    
    Args:
        dataset: Dict with 'labels' key
    
    Returns:
        Dict mapping label to count
    """
    labels = dataset["labels"].numpy()
    unique, counts = np.unique(labels, return_counts=True)
    return {int(label): int(count) for label, count in zip(unique, counts)}


def compute_heterogeneity_score(client_datasets: List[Dict]) -> float:
    """
    Compute heterogeneity score across clients.
    
    Uses Jensen-Shannon divergence between client label distributions
    and the global uniform distribution.
    
    Higher score = more heterogeneous
    
    Args:
        client_datasets: List of client datasets
    
    Returns:
        Heterogeneity score in [0, 1]
    """
    from scipy.stats import entropy
    
    # Get all unique labels
    all_labels = set()
    for client_data in client_datasets:
        labels = client_data["labels"].numpy()
        all_labels.update(labels.tolist())
    all_labels = sorted(all_labels)
    num_classes = len(all_labels)
    
    if num_classes == 0:
        return 0.0
    
    # Compute label distribution for each client
    client_distributions = []
    for client_data in client_datasets:
        dist = np.zeros(num_classes)
        if client_data["labels"].shape[0] > 0:
            labels = client_data["labels"].numpy()
            for i, label in enumerate(all_labels):
                dist[i] = np.sum(labels == label)
            dist = dist / dist.sum() if dist.sum() > 0 else dist
        client_distributions.append(dist)
    
    # Compute average distribution
    avg_distribution = np.mean(client_distributions, axis=0)
    
    # Uniform distribution
    uniform_distribution = np.ones(num_classes) / num_classes
    
    # Jensen-Shannon divergence between average and uniform
    # JSD = 0.5 * KL(P || M) + 0.5 * KL(Q || M), where M = 0.5 * (P + Q)
    m = 0.5 * (avg_distribution + uniform_distribution)
    jsd = 0.5 * entropy(avg_distribution, m) + 0.5 * entropy(uniform_distribution, m)
    
    # Normalize to [0, 1] by dividing by max possible JSD (log(2))
    max_jsd = np.log(2)
    heterogeneity = jsd / max_jsd
    
    return float(heterogeneity)
