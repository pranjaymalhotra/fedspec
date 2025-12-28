"""
Plot Frobenius gap comparison between FedSpec and FedAvg.
"""
import os
import sys
import argparse
import csv
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def load_metrics_csv(filepath: str) -> Dict[str, List]:
    """
    Load metrics from CSV file.
    
    Args:
        filepath: Path to CSV file
    
    Returns:
        Dict with column names as keys and lists as values
    """
    data = {}
    
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, value in row.items():
                if key not in data:
                    data[key] = []
                # Try to convert to float, keep as string if fails
                try:
                    data[key].append(float(value))
                except ValueError:
                    data[key].append(value)
    
    return data


def plot_frobenius_gap(
    fedspec_csv: str,
    fedavg_csv: str,
    output_path: str = 'frobenius_gap.pdf',
    title: str = 'Frobenius Gap: FedSpec vs FedAvg'
):
    """
    Plot Frobenius gap comparison.
    
    Args:
        fedspec_csv: Path to FedSpec metrics CSV
        fedavg_csv: Path to FedAvg metrics CSV
        output_path: Output file path
        title: Plot title
    """
    # Load data
    fedspec_data = load_metrics_csv(fedspec_csv)
    fedavg_data = load_metrics_csv(fedavg_csv)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Plot FedSpec
    rounds = fedspec_data.get('round', list(range(1, len(fedspec_data['frobenius_gap']) + 1)))
    ax.plot(
        rounds,
        fedspec_data['frobenius_gap'],
        'b-o',
        label='FedSpec',
        linewidth=2,
        markersize=6
    )
    
    # Plot FedAvg
    ax.plot(
        rounds,
        fedavg_data['frobenius_gap'],
        'r--s',
        label='FedAvg',
        linewidth=2,
        markersize=6
    )
    
    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Frobenius Gap', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Use log scale if values span multiple orders of magnitude
    gap_range = max(fedspec_data['frobenius_gap'] + fedavg_data['frobenius_gap']) / \
                (min(fedspec_data['frobenius_gap'] + fedavg_data['frobenius_gap']) + 1e-10)
    if gap_range > 100:
        ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved Frobenius gap plot to {output_path}")
    plt.close()


def plot_frobenius_gap_multiple_alpha(
    csv_files: Dict[str, str],
    output_path: str = 'frobenius_gap_alpha.pdf',
    method: str = 'fedspec'
):
    """
    Plot Frobenius gap for different Dirichlet alpha values.
    
    Args:
        csv_files: Dict mapping alpha values to CSV file paths
        output_path: Output file path
        method: 'fedspec' or 'fedavg'
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']
    
    for i, (alpha, filepath) in enumerate(sorted(csv_files.items())):
        data = load_metrics_csv(filepath)
        rounds = data.get('round', list(range(1, len(data['frobenius_gap']) + 1)))
        
        ax.plot(
            rounds,
            data['frobenius_gap'],
            color=colors[i % len(colors)],
            marker=markers[i % len(markers)],
            label=f'alpha = {alpha}',
            linewidth=2,
            markersize=5
        )
    
    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Frobenius Gap', fontsize=12)
    ax.set_title(f'{method.upper()} Frobenius Gap vs Dirichlet Alpha', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    plt.close()


def plot_rank_evolution(
    fedspec_csv: str,
    output_path: str = 'rank_evolution.pdf'
):
    """
    Plot adaptive rank evolution for FedSpec.
    
    Args:
        fedspec_csv: Path to FedSpec metrics CSV
        output_path: Output file path
    """
    data = load_metrics_csv(fedspec_csv)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    rounds = data.get('round', list(range(1, len(data['rank']) + 1)))
    
    ax.step(
        rounds,
        data['rank'],
        'b-',
        where='mid',
        linewidth=2,
        label='Adaptive Rank'
    )
    ax.scatter(rounds, data['rank'], color='blue', s=50, zorder=5)
    
    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('LoRA Rank', fontsize=12)
    ax.set_title('FedSpec Adaptive Rank Evolution', fontsize=14)
    ax.set_ylim(0, max(data['rank']) + 2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved rank evolution plot to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot Frobenius gap')
    parser.add_argument('--fedspec', type=str, required=True,
                       help='Path to FedSpec metrics CSV')
    parser.add_argument('--fedavg', type=str, required=True,
                       help='Path to FedAvg metrics CSV')
    parser.add_argument('--output', type=str, default='frobenius_gap.pdf',
                       help='Output file path')
    parser.add_argument('--title', type=str, 
                       default='Frobenius Gap: FedSpec vs FedAvg',
                       help='Plot title')
    
    args = parser.parse_args()
    
    plot_frobenius_gap(
        fedspec_csv=args.fedspec,
        fedavg_csv=args.fedavg,
        output_path=args.output,
        title=args.title
    )


if __name__ == '__main__':
    main()
