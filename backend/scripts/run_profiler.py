"""
CLI Script to run the TinyML Performance Profiler.
Outputs flash size (KB), RAM usage (KB), inference latency (ms), and MACs.
"""

import sys
import os

# Add parent directory to path so app modules can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.profiler import ModelProfiler

def main():
    profiler = ModelProfiler("models/autoencoder.tflite")
    profile = profiler.get_full_profile()

    print("\n" + "=" * 60)
    print("      🏭 TinyML Edge Model Performance Profiler Report")
    print("=" * 60)
    print(f" Target Hardware        : {profile['hardware_target']}")
    print(f" Quantization Scheme    : {profile['quantization']}")
    print(f" Flash Size             : {profile['model_size_kb']} KB")
    print(f" Flash Budget (< 256 KB): {'✅ PASSED' if profile['within_size_budget'] else '❌ EXCEEDED'}")
    print(f" Peak RAM Consumption   : {profile['peak_ram_kb']} KB")
    print(f" Avg Inference Latency  : {profile['avg_latency_ms']} ms")
    print(f" Multiply-Accumulate Ops: {profile['mac_operations_m']} M MACs")
    print("=" * 60)
    print(f" Overall Deployment Status: {profile['status']}\n")

if __name__ == "__main__":
    main()
