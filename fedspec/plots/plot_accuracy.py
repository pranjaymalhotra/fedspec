"""
Plot accuracy comparison between FedSpec, FedAvg, and Centralized.
"""
import os
import sys
import argparse
import csv
from typing import Dict, List, Optional

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
                try:
                    data[key].append(float(value))
                except ValueError:
                    data[key].append(value)
    
    return data


def plot_accuracy_comparison(
    fedspec_csv: str,
    fedavg_csv: str,
    centralized_csv: Optional[str] = None,
    output_path: str = 'accuracy_comparison.pdf',
    title: str = 'Validation Accuracy Comparison'
):
    """
    Plot accuracy comparison between methods.
    
    Args:
        fedspec_csv: Path to FedSpec metrics CSV
        fedavg_csv: Path to FedAvg metrics CSV
        centralized_csv: Optional path to centralized metrics CSV
        output_path: Output file path
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Load and plot FedSpec
    fedspec_data = load_metrics_csv(fedspec_csv)
    rounds = fedspec_data.get('round', list(range(1, len(fedspec_data['accuracy']) + 1)))
    ax.plot(
        rounds,
        fedspec_data['accuracy'],
        'b-o',
        label='FedSpec',
        linewidth=2,
        markersize=6
    )
    
    # Load and plot FedAvg
    fedavg_data = load_metrics_csv(fedavg_csv)
    ax.plot(
        rounds,
        fedavg_data['accuracy'],
        'r--s',
        label='FedAvg',
        linewidth=2,
        markersize=6
    )
    
    # Load and plot centralized (if provided)
    if centralized_csv is not None and os.path.exists(centralized_csv):
        centralized_data = load_metrics_csv(centralized_csv)
        
        # Centralized has epochs, not rounds
        # Plot as horizontal line (final accuracy) or epochs scaled to rounds
        final_acc = centralized_data['val_accuracy'][-1]
        ax.axhline(
            y=final_acc,
            color='g',
            linestyle=':',
            linewidth=2,
            label=f'Centralized ({final_acc:.3f})'
        )
    
    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Validation Accuracy', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Set y-axis limits
    all_acc = fedspec_data['accuracy'] + fedavg_data['accuracy']
    ax.set_ylim(min(all_acc) - 0.05, 1.0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved accuracy plot to {output_path}")
    plt.close()


def plot_accuracy_vs_communication(
    fedspec_csv: str,
    fedavg_csv: str,
    output_path: str = 'accuracy_vs_comm.pdf'
):
    """
    Plot accuracy vs cumulative communication cost.
    
    Args:
        fedspec_csv: Path to FedSpec metrics CSV
        fedavg_csv: Path to FedAvg metrics CSV
        output_path: Output file path
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # FedSpec
    fedspec_data = load_metrics_csv(fedspec_csv)
    fedspec_cum_bytes = np.cumsum(fedspec_data['communication_bytes'])
    fedspec_cum_mb = fedspec_cum_bytes / (1024 * 1024)  # Convert to MB
    
    ax.plot(
        fedspec_cum_mb,
        fedspec_data['accuracy'],
        'b-o',
        label='FedSpec',
        linewidth=2,
        markersize=6
    )
    
    # FedAvg
    fedavg_data = load_metrics_csv(fedavg_csv)
    fedavg_cum_bytes = np.cumsum(fedavg_data['communication_bytes'])
    fedavg_cum_mb = fedavg_cum_bytes / (1024 * 1024)
    
    ax.plot(
        fedavg_cum_mb,
        fedavg_data['accuracy'],
        'r--s',
        label='FedAvg',
        linewidth=2,
        markersize=6
    )
    
    ax.set_xlabel('Cumulative Communication (MB)', fontsize=12)
    ax.set_ylabel('Validation Accuracy', fontsize=12)
    ax.set_title('Accuracy vs Communication Cost', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved accuracy vs communication plot to {output_path}")
    plt.close()


def plot_all_metrics(
    fedspec_csv: str,
    fedavg_csv: str,
    centralized_csv: Optional[str] = None,
    output_path: str = 'all_metrics.pdf'
):
    """
    Create a 2x2 subplot with all relevant metrics.
    
    Args:
        fedspec_csv: Path to FedSpec metrics CSV
        fedavg_csv: Path to FedAvg metrics CSV
        centralized_csv: Optional path to centralized metrics CSV
        output_path: Output file path
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Load data
    fedspec_data = load_metrics_csv(fedspec_csv)
    fedavg_data = load_metrics_csv(fedavg_csv)
    rounds = fedspec_data.get('round', list(range(1, len(fedspec_data['accuracy']) + 1)))
    
    # 1. Accuracy vs Rounds (top-left)
    ax1 = axes[0, 0]
    ax1.plot(rounds, fedspec_data['accuracy'], 'b-o', label='FedSpec', linewidth=2, markersize=5)
    ax1.plot(rounds, fedavg_data['accuracy'], 'r--s', label='FedAvg', linewidth=2, markersize=5)
    
    if centralized_csv is not None and os.path.exists(centralized_csv):
        centralized_data = load_metrics_csv(centralized_csv)
        final_acc = centralized_data['val_accuracy'][-1]
        ax1.axhline(y=final_acc, color='g', linestyle=':', linewidth=2, 
                   label=f'Centralized ({final_acc:.3f})')
    
    ax1.set_xlabel('Round', fontsize=11)
    ax1.set_ylabel('Validation Accuracy', fontsize=11)
    ax1.set_title('(a) Accuracy vs Rounds', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. Frobenius Gap vs Rounds (top-right)
    ax2 = axes[0, 1]
    ax2.plot(rounds, fedspec_data['frobenius_gap'], 'b-o', label='FedSpec', linewidth=2, markersize=5)
    ax2.plot(rounds, fedavg_data['frobenius_gap'], 'r--s', label='FedAvg', linewidth=2, markersize=5)
    ax2.set_xlabel('Round', fontsize=11)
    ax2.set_ylabel('Frobenius Gap', fontsize=11)
    ax2.set_title('(b) Frobenius Gap vs Rounds', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 3. Rank vs Rounds (bottom-left)
    ax3 = axes[1, 0]
    ax3.step(rounds, fedspec_data['rank'], 'b-', where='mid', linewidth=2, label='FedSpec')
    ax3.scatter(rounds, fedspec_data['rank'], color='blue', s=40, zorder=5)
    ax3.step(rounds, fedavg_data['rank'], 'r--', where='mid', linewidth=2, label='FedAvg')
    ax3.set_xlabel('Round', fontsize=11)
    ax3.set_ylabel('LoRA Rank', fontsize=11)
    ax3.set_title('(c) Rank vs Rounds', fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, max(max(fedspec_data['rank']), max(fedavg_data['rank'])) + 2)
    
    # 4. Accuracy vs Communication (bottom-right)
    ax4 = axes[1, 1]
    fedspec_cum_mb = np.cumsum(fedspec_data['communication_bytes']) / (1024 * 1024)
    fedavg_cum_mb = np.cumsum(fedavg_data['communication_bytes']) / (1024 * 1024)
    
    ax4.plot(fedspec_cum_mb, fedspec_data['accuracy'], 'b-o', label='FedSpec', linewidth=2, markersize=5)
    ax4.plot(fedavg_cum_mb, fedavg_data['accuracy'], 'r--s', label='FedAvg', linewidth=2, markersize=5)
    ax4.set_xlabel('Cumulative Communication (MB)', fontsize=11)
    ax4.set_ylabel('Validation Accuracy', fontsize=11)
    ax4.set_title('(d) Accuracy vs Communication', fontsize=12)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined metrics plot to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot accuracy comparison')
    parser.add_argument('--fedspec', type=str, required=True,
                       help='Path to FedSpec metrics CSV')
    parser.add_argument('--fedavg', type=str, required=True,
                       help='Path to FedAvg metrics CSV')
    parser.add_argument('--centralized', type=str, default=None,
                       help='Path to centralized metrics CSV')
    parser.add_argument('--output', type=str, default='accuracy_comparison.pdf',
                       help='Output file path')
    parser.add_argument('--all', action='store_true',
                       help='Create combined plot with all metrics')
    
    args = parser.parse_args()
    
    if args.all:
        plot_all_metrics(
            fedspec_csv=args.fedspec,
            fedavg_csv=args.fedavg,
            centralized_csv=args.centralized,
            output_path=args.output.replace('.pdf', '_all.pdf')
        )
    else:
        plot_accuracy_comparison(
            fedspec_csv=args.fedspec,
            fedavg_csv=args.fedavg,
            centralized_csv=args.centralized,
            output_path=args.output
        )


if __name__ == '__main__':
    main()
