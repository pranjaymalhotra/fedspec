#!/usr/bin/env python3
"""
Checkpoint verification tool - ensures all checkpoints are valid.
Lightweight script that checks file integrity without loading models.
"""

import json
from pathlib import Path
from datetime import datetime
import hashlib


def verify_checkpoint_directory(checkpoint_dir: Path) -> dict:
    """Verify all checkpoints in a directory."""
    results = {
        "total_experiments": 0,
        "valid_checkpoints": 0,
        "invalid_checkpoints": 0,
        "missing_files": 0,
        "corrupted_metadata": 0,
        "details": []
    }
    
    if not checkpoint_dir.exists():
        print(f"⚠️  Checkpoint directory not found: {checkpoint_dir}")
        return results
    
    for exp_dir in checkpoint_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        
        results["total_experiments"] += 1
        exp_name = exp_dir.name
        
        # Check for required files
        metadata_file = exp_dir / "metadata.json"
        model_file = exp_dir / "model.pt"
        
        status = "✅ Valid"
        issues = []
        
        if not metadata_file.exists():
            status = "❌ Invalid"
            issues.append("Missing metadata.json")
            results["missing_files"] += 1
        else:
            # Verify metadata can be read
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # Check required fields
                required = ["experiment_id", "method", "alpha", "current_round", "total_rounds", "timestamp"]
                for field in required:
                    if field not in metadata:
                        issues.append(f"Missing field: {field}")
                
            except json.JSONDecodeError:
                status = "❌ Invalid"
                issues.append("Corrupted metadata.json")
                results["corrupted_metadata"] += 1
        
        if not model_file.exists():
            status = "❌ Invalid"
            issues.append("Missing model.pt")
            results["missing_files"] += 1
        else:
            # Check file size is reasonable (> 1KB)
            size_kb = model_file.stat().st_size / 1024
            if size_kb < 1:
                status = "❌ Invalid"
                issues.append("model.pt too small (likely corrupted)")
        
        if status == "✅ Valid":
            results["valid_checkpoints"] += 1
        else:
            results["invalid_checkpoints"] += 1
        
        results["details"].append({
            "experiment": exp_name,
            "status": status,
            "issues": issues
        })
    
    return results


def print_verification_report(results: dict):
    """Print human-readable verification report."""
    print("\n" + "="*70)
    print(" 💾 Checkpoint Verification Report")
    print("="*70 + "\n")
    
    print(f"Total Experiments: {results['total_experiments']}")
    print(f"✅ Valid:          {results['valid_checkpoints']}")
    print(f"❌ Invalid:        {results['invalid_checkpoints']}")
    
    if results['missing_files'] > 0:
        print(f"   Missing files:  {results['missing_files']}")
    if results['corrupted_metadata'] > 0:
        print(f"   Corrupted:      {results['corrupted_metadata']}")
    
    print()
    
    # Show details
    if results['details']:
        print("Details:")
        for detail in results['details']:
            status_icon = detail['status']
            exp = detail['experiment']
            print(f"  {status_icon} {exp}")
            
            if detail['issues']:
                for issue in detail['issues']:
                    print(f"      • {issue}")
        print()
    
    # Overall status
    if results['invalid_checkpoints'] == 0 and results['total_experiments'] > 0:
        print("🎉 All checkpoints verified successfully!")
    elif results['total_experiments'] == 0:
        print("⚠️  No checkpoints found")
    else:
        print(f"⚠️  Found {results['invalid_checkpoints']} invalid checkpoint(s)")
    
    print("="*70 + "\n")


def main():
    """Main entry point."""
    # Check both checkpoint directories
    base_dir = Path(".")
    
    print("\nChecking checkpoint directories...")
    
    # Quick experiments
    quick_dir = base_dir / "checkpoints_quick"
    if quick_dir.exists():
        print(f"\n📁 Quick Experiments: {quick_dir}")
        results = verify_checkpoint_directory(quick_dir)
        print_verification_report(results)
    
    # Distributed experiments
    dist_dir = base_dir / "checkpoints_distributed"
    if dist_dir.exists():
        print(f"\n📁 Distributed Experiments: {dist_dir}")
        results = verify_checkpoint_directory(dist_dir)
        print_verification_report(results)
    
    # Paper experiments
    paper_dir = base_dir / "checkpoints"
    if paper_dir.exists():
        print(f"\n📁 Paper Experiments: {paper_dir}")
        results = verify_checkpoint_directory(paper_dir)
        print_verification_report(results)
    
    if not any([quick_dir.exists(), dist_dir.exists(), paper_dir.exists()]):
        print("\n⚠️  No checkpoint directories found")
        print("   Run experiments first to generate checkpoints")


if __name__ == "__main__":
    main()
