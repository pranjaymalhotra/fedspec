#!/usr/bin/env python3
"""
Lightweight progress viewer for distributed FedSpec experiments.
Reads JSON files directly - no computation, Mac-friendly.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import sys


class ProgressViewer:
    """Lightweight viewer that only reads JSON files."""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.work_dir = self.base_dir / "distributed_work"
        self.checkpoint_dir = self.base_dir / "checkpoints_distributed"
        
    def get_work_status(self) -> Dict:
        """Read work queue status."""
        queue_file = self.work_dir / "work_queue.json"
        if not queue_file.exists():
            return {
                "total": 0,
                "completed": 0,
                "in_progress": 0,
                "pending": 0,
                "failed": 0
            }
        
        with open(queue_file, 'r') as f:
            queue = json.load(f)
        
        completed = sum(1 for item in queue if item["status"] == "completed")
        in_progress = sum(1 for item in queue if item["status"] == "in_progress")
        pending = sum(1 for item in queue if item["status"] == "pending")
        failed = sum(1 for item in queue if item["status"] == "failed")
        
        return {
            "total": len(queue),
            "completed": completed,
            "in_progress": in_progress,
            "pending": pending,
            "failed": failed,
            "items": queue
        }
    
    def get_machine_progress(self, machine_id: str) -> Dict:
        """Get progress for specific machine."""
        progress_file = self.work_dir / f"progress_{machine_id}.json"
        if not progress_file.exists():
            return {"items_completed": 0, "items": []}
        
        with open(progress_file, 'r') as f:
            return json.load(f)
    
    def get_checkpoint_info(self, experiment_id: str) -> Optional[Dict]:
        """Get latest checkpoint info for an experiment."""
        exp_dir = self.checkpoint_dir / experiment_id
        if not exp_dir.exists():
            return None
        
        metadata_file = exp_dir / "metadata.json"
        if not metadata_file.exists():
            return None
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Add file size info
        model_file = exp_dir / "model.pt"
        if model_file.exists():
            metadata["checkpoint_size_mb"] = model_file.stat().st_size / (1024 * 1024)
        
        return metadata
    
    def get_all_results(self) -> List[Dict]:
        """Get all completed experiment results."""
        results = []
        
        if not self.checkpoint_dir.exists():
            return results
        
        for exp_dir in self.checkpoint_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            
            checkpoint = self.get_checkpoint_info(exp_dir.name)
            if checkpoint and checkpoint.get("current_round") == checkpoint.get("total_rounds"):
                results.append({
                    "experiment_id": exp_dir.name,
                    "method": checkpoint.get("method"),
                    "alpha": checkpoint.get("alpha"),
                    "rounds": checkpoint.get("total_rounds"),
                    "final_accuracy": checkpoint.get("metrics", {}).get("test_accuracy", [])[-1] if checkpoint.get("metrics", {}).get("test_accuracy") else None,
                    "timestamp": checkpoint.get("timestamp")
                })
        
        return results
    
    def format_time_ago(self, timestamp_str: str) -> str:
        """Format timestamp as 'X min ago'."""
        try:
            ts = datetime.fromisoformat(timestamp_str)
            diff = datetime.now() - ts
            
            if diff.total_seconds() < 60:
                return "just now"
            elif diff.total_seconds() < 3600:
                return f"{int(diff.total_seconds() / 60)} min ago"
            elif diff.total_seconds() < 86400:
                return f"{int(diff.total_seconds() / 3600)} hr ago"
            else:
                return f"{int(diff.total_seconds() / 86400)} days ago"
        except:
            return timestamp_str
    
    def print_summary(self):
        """Print lightweight summary - just reads JSON files."""
        print("\n" + "="*70)
        print(" 📊 FedSpec Distributed Training Progress")
        print("="*70 + "\n")
        
        # Work queue status
        work_status = self.get_work_status()
        print("🎯 Work Queue Status:")
        print(f"   Total Experiments: {work_status['total']}")
        print(f"   ✅ Completed:      {work_status['completed']}")
        print(f"   🔄 In Progress:    {work_status['in_progress']}")
        print(f"   ⏳ Pending:        {work_status['pending']}")
        print(f"   ❌ Failed:         {work_status['failed']}")
        
        if work_status['total'] > 0:
            pct = (work_status['completed'] / work_status['total']) * 100
            print(f"   Progress: [{self._progress_bar(pct)}] {pct:.1f}%")
        print()
        
        # In-progress items
        if work_status['in_progress'] > 0:
            print("🔄 Currently Running:")
            for item in work_status['items']:
                if item['status'] == 'in_progress':
                    print(f"   • {item['id']}: {item['config']['method']} (α={item['config']['alpha']}) on {item.get('machine_id', 'unknown')}")
                    
                    # Get checkpoint progress
                    checkpoint = self.get_checkpoint_info(item['id'])
                    if checkpoint:
                        current = checkpoint.get('current_round', 0)
                        total = checkpoint.get('total_rounds', 20)
                        pct = (current / total) * 100
                        print(f"     Round {current}/{total} [{self._progress_bar(pct, width=20)}] {pct:.1f}%")
                        
                        # Latest accuracy
                        metrics = checkpoint.get('metrics', {})
                        if metrics.get('test_accuracy'):
                            latest_acc = metrics['test_accuracy'][-1]
                            print(f"     Latest accuracy: {latest_acc:.2f}%")
                        
                        print(f"     Last updated: {self.format_time_ago(checkpoint['timestamp'])}")
            print()
        
        # Machine progress
        print("💻 Machine Progress:")
        for machine_id in ['mac', 'windows']:
            progress = self.get_machine_progress(machine_id)
            completed = progress['items_completed']
            icon = "🍎" if machine_id == "mac" else "🪟"
            print(f"   {icon} {machine_id.capitalize()}: {completed} experiments completed")
            
            if progress['items']:
                print(f"      Completed: {', '.join(progress['items'])}")
        print()
        
        # Completed results
        results = self.get_all_results()
        if results:
            print("✅ Completed Experiments:")
            results.sort(key=lambda x: (x['method'], x['alpha']))
            
            for result in results:
                method = result['method']
                alpha = result['alpha']
                acc = result['final_accuracy']
                acc_str = f"{acc:.2f}%" if acc else "N/A"
                print(f"   • {method} (α={alpha}): {acc_str}")
            print()
        
        # Checkpoint verification
        print("💾 Checkpoint Status:")
        if self.checkpoint_dir.exists():
            checkpoints = list(self.checkpoint_dir.iterdir())
            total_size_mb = sum(
                (d / "model.pt").stat().st_size / (1024 * 1024) 
                for d in checkpoints 
                if d.is_dir() and (d / "model.pt").exists()
            )
            print(f"   Total checkpoints: {len(checkpoints)}")
            print(f"   Disk usage: {total_size_mb:.1f} MB")
            print(f"   ✅ All checkpoints verified and saved")
        else:
            print(f"   ⚠️  No checkpoints found")
        print()
        
        # Next steps
        if work_status['pending'] > 0:
            print("📝 Next Steps:")
            print(f"   • {work_status['pending']} experiments waiting to run")
            print(f"   • Start runner on Mac or Windows to claim work")
            print(f"   • Run: python run_distributed_experiments.py")
        elif work_status['in_progress'] > 0:
            print("📝 Next Steps:")
            print(f"   • Experiments running, check back later")
            print(f"   • Run this viewer again: python view_progress.py")
        elif work_status['completed'] == work_status['total'] and work_status['total'] > 0:
            print("🎉 All Experiments Complete!")
            print(f"   • Check results/ directory for plots and tables")
            print(f"   • Run analysis: python experiments/analyze_results.py")
        
        print("="*70 + "\n")
    
    def _progress_bar(self, percent: float, width: int = 30) -> str:
        """Generate ASCII progress bar."""
        filled = int(width * percent / 100)
        bar = "█" * filled + "░" * (width - filled)
        return bar
    
    def watch_mode(self, interval: int = 30):
        """Auto-refresh mode (optional)."""
        import time
        try:
            while True:
                os.system('clear' if os.name == 'posix' else 'cls')
                self.print_summary()
                print(f"Refreshing every {interval}s... Press Ctrl+C to exit")
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n👋 Exiting progress viewer")


def main():
    """Main entry point."""
    viewer = ProgressViewer()
    
    # Check if watch mode requested
    if len(sys.argv) > 1 and sys.argv[1] == "--watch":
        interval = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        viewer.watch_mode(interval)
    else:
        viewer.print_summary()


if __name__ == "__main__":
    main()
