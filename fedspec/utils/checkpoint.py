"""
Checkpoint management for resumable experiments.
Supports saving/loading experiment state to enable resume after interruption.
"""
import os
import json
import pickle
import torch
from typing import Dict, Any, Optional
from datetime import datetime


class CheckpointManager:
    """Manages experiment checkpoints for resumability."""
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
        """
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
    def save_experiment_state(
        self,
        experiment_id: str,
        state: Dict[str, Any],
        model_state: Optional[Dict] = None
    ):
        """
        Save experiment state to checkpoint.
        
        Args:
            experiment_id: Unique experiment identifier
            state: Experiment state (round, client, metrics, etc.)
            model_state: Model state dict (optional)
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{experiment_id}.json")
        
        # Add timestamp
        state['timestamp'] = datetime.now().isoformat()
        state['experiment_id'] = experiment_id
        
        # Save JSON state
        with open(checkpoint_path, 'w') as f:
            json.dump(state, f, indent=2)
        
        # Save model state if provided
        if model_state is not None:
            model_path = os.path.join(self.checkpoint_dir, f"{experiment_id}_model.pt")
            torch.save(model_state, model_path)
        
        print(f"✓ Checkpoint saved: {experiment_id}")
    
    def load_experiment_state(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """
        Load experiment state from checkpoint.
        
        Args:
            experiment_id: Unique experiment identifier
            
        Returns:
            Experiment state dict or None if not found
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{experiment_id}.json")
        
        if not os.path.exists(checkpoint_path):
            return None
        
        with open(checkpoint_path, 'r') as f:
            state = json.load(f)
        
        print(f"✓ Checkpoint loaded: {experiment_id}")
        return state
    
    def load_model_state(self, experiment_id: str) -> Optional[Dict]:
        """
        Load model state from checkpoint.
        
        Args:
            experiment_id: Unique experiment identifier
            
        Returns:
            Model state dict or None if not found
        """
        model_path = os.path.join(self.checkpoint_dir, f"{experiment_id}_model.pt")
        
        if not os.path.exists(model_path):
            return None
        
        return torch.load(model_path)
    
    def checkpoint_exists(self, experiment_id: str) -> bool:
        """Check if checkpoint exists for experiment."""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{experiment_id}.json")
        return os.path.exists(checkpoint_path)
    
    def list_checkpoints(self):
        """List all available checkpoints."""
        checkpoints = []
        for file in os.listdir(self.checkpoint_dir):
            if file.endswith('.json'):
                checkpoint_id = file.replace('.json', '')
                checkpoint_path = os.path.join(self.checkpoint_dir, file)
                
                with open(checkpoint_path, 'r') as f:
                    state = json.load(f)
                
                checkpoints.append({
                    'id': checkpoint_id,
                    'timestamp': state.get('timestamp', 'unknown'),
                    'progress': state.get('progress', 'unknown')
                })
        
        return checkpoints
    
    def delete_checkpoint(self, experiment_id: str):
        """Delete checkpoint files."""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{experiment_id}.json")
        model_path = os.path.join(self.checkpoint_dir, f"{experiment_id}_model.pt")
        
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        if os.path.exists(model_path):
            os.remove(model_path)
        
        print(f"✓ Checkpoint deleted: {experiment_id}")


def get_experiment_id(method: str, alpha: float, split_type: str) -> str:
    """Generate unique experiment ID."""
    return f"{method}_{split_type}_alpha{alpha}"


def should_resume(checkpoint_manager: CheckpointManager, experiment_id: str) -> bool:
    """Check if experiment should be resumed from checkpoint."""
    if checkpoint_manager.checkpoint_exists(experiment_id):
        print(f"\n{'='*70}")
        print(f"Checkpoint found for: {experiment_id}")
        state = checkpoint_manager.load_experiment_state(experiment_id)
        if state:
            print(f"Last saved: {state.get('timestamp', 'unknown')}")
            print(f"Progress: Round {state.get('current_round', 0)}/{state.get('total_rounds', '?')}")
            print(f"{'='*70}")
            return True
    return False
