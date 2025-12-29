"""
Distributed experiment runner for Mac + Windows setup.
Allows parallel execution across multiple machines with automatic work distribution.
"""
import os
import json
import socket
from typing import List, Dict, Any
from datetime import datetime


class WorkDistributor:
    """Distributes experiments across multiple machines."""
    
    def __init__(self, work_dir: str = "distributed_work"):
        """
        Initialize work distributor.
        
        Args:
            work_dir: Directory for work coordination
        """
        self.work_dir = work_dir
        os.makedirs(work_dir, exist_ok=True)
        self.machine_id = self._get_machine_id()
        
    def _get_machine_id(self) -> str:
        """Get unique machine identifier."""
        hostname = socket.gethostname()
        # Simplify to 'mac' or 'windows'
        if 'MacBook' in hostname or 'darwin' in hostname.lower():
            return 'mac'
        else:
            return 'windows'
    
    def create_work_items(
        self,
        alphas: List[float],
        methods: List[str],
        split_types: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Create list of work items (experiments).
        
        Args:
            alphas: List of alpha values
            methods: List of methods
            split_types: List of split types
            
        Returns:
            List of work items
        """
        work_items = []
        item_id = 0
        
        for alpha in alphas:
            for method in methods:
                for split_type in split_types:
                    work_items.append({
                        'item_id': item_id,
                        'alpha': alpha,
                        'method': method,
                        'split_type': split_type,
                        'status': 'pending',
                        'assigned_to': None,
                        'started_at': None,
                        'completed_at': None
                    })
                    item_id += 1
        
        return work_items
    
    def save_work_queue(self, work_items: List[Dict[str, Any]]):
        """Save work queue to shared location."""
        queue_path = os.path.join(self.work_dir, 'work_queue.json')
        with open(queue_path, 'w') as f:
            json.dump(work_items, f, indent=2)
    
    def load_work_queue(self) -> List[Dict[str, Any]]:
        """Load work queue from shared location."""
        queue_path = os.path.join(self.work_dir, 'work_queue.json')
        if not os.path.exists(queue_path):
            return []
        
        with open(queue_path, 'r') as f:
            return json.load(f)
    
    def claim_work_item(self) -> Dict[str, Any]:
        """
        Claim next available work item for this machine.
        
        Returns:
            Work item dict or None if no work available
        """
        work_items = self.load_work_queue()
        
        # Find first pending item
        for item in work_items:
            if item['status'] == 'pending':
                item['status'] = 'in_progress'
                item['assigned_to'] = self.machine_id
                item['started_at'] = datetime.now().isoformat()
                self.save_work_queue(work_items)
                return item
        
        return None
    
    def mark_completed(self, item_id: int, success: bool = True):
        """Mark work item as completed."""
        work_items = self.load_work_queue()
        
        for item in work_items:
            if item['item_id'] == item_id:
                item['status'] = 'completed' if success else 'failed'
                item['completed_at'] = datetime.now().isoformat()
                break
        
        self.save_work_queue(work_items)
    
    def mark_failed(self, item_id: int):
        """Mark work item as failed (can be retried)."""
        work_items = self.load_work_queue()
        
        for item in work_items:
            if item['item_id'] == item_id:
                item['status'] = 'pending'  # Reset to pending for retry
                item['assigned_to'] = None
                item['started_at'] = None
                break
        
        self.save_work_queue(work_items)
    
    def get_progress_summary(self) -> Dict[str, int]:
        """Get summary of work progress."""
        work_items = self.load_work_queue()
        
        summary = {
            'total': len(work_items),
            'pending': sum(1 for item in work_items if item['status'] == 'pending'),
            'in_progress': sum(1 for item in work_items if item['status'] == 'in_progress'),
            'completed': sum(1 for item in work_items if item['status'] == 'completed'),
            'failed': sum(1 for item in work_items if item['status'] == 'failed')
        }
        
        # Count by machine
        summary['mac_working'] = sum(1 for item in work_items 
                                     if item['status'] == 'in_progress' and item['assigned_to'] == 'mac')
        summary['windows_working'] = sum(1 for item in work_items 
                                         if item['status'] == 'in_progress' and item['assigned_to'] == 'windows')
        
        return summary
    
    def print_progress(self):
        """Print progress summary."""
        summary = self.get_progress_summary()
        
        print("\n" + "="*70)
        print("Distributed Work Progress")
        print("="*70)
        print(f"Total items: {summary['total']}")
        print(f"Completed:   {summary['completed']} ({summary['completed']/summary['total']*100:.1f}%)")
        print(f"In progress: {summary['in_progress']}")
        print(f"  - Mac:     {summary['mac_working']}")
        print(f"  - Windows: {summary['windows_working']}")
        print(f"Pending:     {summary['pending']}")
        print(f"Failed:      {summary['failed']}")
        print("="*70 + "\n")
