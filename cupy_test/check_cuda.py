#!/usr/bin/env python3

import os
import subprocess

def check_cuda_installation():
    """Check CUDA installation and paths"""
    
    print("=== CUDA INSTALLATION CHECK ===")
    
    # Check CUDA version
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("NVCC found:")
            print(result.stdout.strip())
        else:
            print("NVCC not found in PATH")
    except FileNotFoundError:
        print("NVCC not found - CUDA might not be installed")
    
    # Check nvidia-smi
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("\nnvidia-smi found:")
            # Extract just the driver/CUDA version line
            lines = result.stdout.split('\n')
            for line in lines:
                if 'CUDA Version:' in line:
                    print(line.strip())
                    break
        else:
            print("nvidia-smi failed")
    except FileNotFoundError:
        print("nvidia-smi not found")
    
    # Check environment variables
    print(f"\nEnvironment variables:")
    cuda_home = os.environ.get('CUDA_HOME', 'Not set')
    cuda_path = os.environ.get('CUDA_PATH', 'Not set')
    ld_library_path = os.environ.get('LD_LIBRARY_PATH', 'Not set')
    
    print(f"CUDA_HOME: {cuda_home}")
    print(f"CUDA_PATH: {cuda_path}")
    print(f"LD_LIBRARY_PATH: {ld_library_path}")
    
    # Common CUDA locations
    print(f"\nChecking common CUDA locations:")
    common_paths = [
        '/usr/local/cuda',
        '/usr/local/cuda-12',
        '/usr/local/cuda-11', 
        '/opt/cuda',
        '/usr/lib/x86_64-linux-gnu'
    ]
    
    for path in common_paths:
        if os.path.exists(path):
            print(f"  {path}: EXISTS")
            # Check for libnvrtc.so
            nvrtc_paths = [
                f"{path}/lib64/libnvrtc.so",
                f"{path}/lib64/libnvrtc.so.12",
                f"{path}/lib/x86_64-linux-gnu/libnvrtc.so.12"
            ]
            for nvrtc_path in nvrtc_paths:
                if os.path.exists(nvrtc_path):
                    print(f"    Found: {nvrtc_path}")
        else:
            print(f"  {path}: NOT FOUND")
    
    # Search for libnvrtc files
    print(f"\nSearching for libnvrtc files:")
    try:
        result = subprocess.run(['find', '/usr', '/opt', '-name', 'libnvrtc*', '2>/dev/null'], 
                              capture_output=True, text=True, shell=True)
        if result.stdout.strip():
            print("Found libnvrtc files:")
            for line in result.stdout.strip().split('\n'):
                print(f"  {line}")
        else:
            print("No libnvrtc files found")
    except:
        print("Could not search for libnvrtc files")
    
    # Try importing cupy without compiling kernels
    print(f"\n=== CUPY CHECK ===")
    try:
        import cupy as cp
        print(f"CuPy imported successfully")
        print(f"CuPy version: {cp.__version__}")
        
        # Check device
        try:
            device = cp.cuda.Device()
            print(f"CUDA device: {device}")
            print(f"Device name: {device.attributes['Name']}")
            print(f"Compute capability: {device.compute_capability}")
        except Exception as e:
            print(f"Could not get device info: {e}")
            
    except Exception as e:
        print(f"Could not import CuPy: {e}")

if __name__ == "__main__":
    check_cuda_installation()