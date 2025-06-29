#!/usr/bin/env python3
"""
Training Progress Monitor for NeurIPS Open Polymer Prediction 2025
"""

import time
import os
import subprocess
import sys
from datetime import datetime, timedelta

def get_process_info():
    """Get information about the training process."""
    try:
        result = subprocess.run(
            ['ps', 'aux'], 
            capture_output=True, 
            text=True, 
            check=True
        )
        
        for line in result.stdout.split('\n'):
            if 'train_advanced.py' in line and 'python' in line:
                parts = line.split()
                if len(parts) >= 11:
                    pid = parts[1]
                    cpu_percent = parts[2]
                    memory_percent = parts[3]
                    elapsed_time = parts[9]
                    return {
                        'pid': pid,
                        'cpu': cpu_percent,
                        'memory': memory_percent,
                        'elapsed': elapsed_time,
                        'running': True
                    }
        
        return {'running': False}
    
    except Exception as e:
        return {'running': False, 'error': str(e)}

def check_output_files():
    """Check what output files have been created."""
    output_dir = 'output'
    models_dir = 'models'
    
    files_status = {
        'baseline_complete': os.path.exists(os.path.join(output_dir, 'submission_rf.csv')),
        'transformer_complete': os.path.exists(os.path.join(output_dir, 'submission_transformer.csv')),
        'gnn_complete': os.path.exists(os.path.join(output_dir, 'submission_gnn.csv')),
        'ensemble_complete': os.path.exists(os.path.join(output_dir, 'submission_ensemble.csv')),
        'performance_summary': os.path.exists(os.path.join(output_dir, 'advanced_performance_summary.png')),
        'models_saved': len([f for f in os.listdir(models_dir) if f.endswith('.pkl')]) if os.path.exists(models_dir) else 0
    }
    
    return files_status

def estimate_completion():
    """Estimate completion time based on current progress."""
    # Rough estimates based on model complexity
    estimated_times = {
        'transformer_cv': 15,  # 15 minutes for 5-fold CV
        'gnn_cv': 10,         # 10 minutes for 5-fold CV  
        'ensemble': 2,        # 2 minutes for ensemble
        'finalization': 3     # 3 minutes for saving and summary
    }
    
    files_status = check_output_files()
    remaining_time = 0
    
    if not files_status['transformer_complete']:
        remaining_time += estimated_times['transformer_cv']
    
    if not files_status['gnn_complete']:
        remaining_time += estimated_times['gnn_cv']
    
    if not files_status['ensemble_complete']:
        remaining_time += estimated_times['ensemble']
    
    if not files_status['performance_summary']:
        remaining_time += estimated_times['finalization']
    
    return remaining_time

def format_time(minutes):
    """Format time in a readable way."""
    if minutes < 1:
        return "< 1 minute"
    elif minutes < 60:
        return f"{int(minutes)} minutes"
    else:
        hours = int(minutes // 60)
        mins = int(minutes % 60)
        return f"{hours}h {mins}m"

def main():
    """Monitor training progress."""
    print("🔍 NeurIPS Polymer Prediction Training Monitor")
    print("=" * 50)
    
    while True:
        try:
            # Clear screen
            os.system('clear' if os.name == 'posix' else 'cls')
            
            print("🔍 NeurIPS Polymer Prediction Training Monitor")
            print("=" * 50)
            print(f"📅 Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print()
            
            # Check process status
            process_info = get_process_info()
            
            if process_info['running']:
                print("✅ Training Status: RUNNING")
                print(f"🆔 Process ID: {process_info['pid']}")
                print(f"💻 CPU Usage: {process_info['cpu']}%")
                print(f"🧠 Memory Usage: {process_info['memory']}%")
                print(f"⏱️  Elapsed Time: {process_info['elapsed']}")
            else:
                print("❌ Training Status: NOT RUNNING")
                if 'error' in process_info:
                    print(f"Error: {process_info['error']}")
            
            print()
            
            # Check file completion status
            files_status = check_output_files()
            print("📁 Progress Status:")
            print(f"   Baseline Model: {'✅' if files_status['baseline_complete'] else '⏳'}")
            print(f"   Transformer Model: {'✅' if files_status['transformer_complete'] else '⏳'}")
            print(f"   GNN Model: {'✅' if files_status['gnn_complete'] else '⏳'}")
            print(f"   Ensemble Model: {'✅' if files_status['ensemble_complete'] else '⏳'}")
            print(f"   Performance Summary: {'✅' if files_status['performance_summary'] else '⏳'}")
            print(f"   Models Saved: {files_status['models_saved']} files")
            
            print()
            
            # Estimate completion time
            if process_info['running']:
                remaining_minutes = estimate_completion()
                if remaining_minutes > 0:
                    estimated_completion = datetime.now() + timedelta(minutes=remaining_minutes)
                    print("⏰ Time Estimates:")
                    print(f"   Estimated remaining: {format_time(remaining_minutes)}")
                    print(f"   Estimated completion: {estimated_completion.strftime('%H:%M:%S')}")
                else:
                    print("🎉 Training appears to be in final stages!")
            
            print()
            print("Press Ctrl+C to exit monitor")
            print("=" * 50)
            
            # Check if training is complete
            if not process_info['running']:
                if files_status['performance_summary']:
                    print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
                    print("📊 Check the output/ directory for results")
                    break
                else:
                    print("⚠️  Training stopped but may not be complete")
                    print("Check for any error messages in the terminal")
                    break
            
            # Wait before next update
            time.sleep(10)
            
        except KeyboardInterrupt:
            print("\n👋 Monitor stopped by user")
            break
        except Exception as e:
            print(f"❌ Monitor error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main() 