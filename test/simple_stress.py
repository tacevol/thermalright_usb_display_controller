#!/usr/bin/env python3
"""
Simple, effective CPU stress test that should show up in BTOP
"""

import multiprocessing
import time
import sys

def stress_core(core_id, duration=30):
    """Simple stress test for one core"""
    print(f"Stressing core {core_id} for {duration} seconds...")
    
    # Set CPU affinity if possible
    try:
        import psutil
        p = psutil.Process()
        p.cpu_affinity([core_id])
        print(f"Bound to core {core_id}")
    except Exception as e:
        print(f"Could not bind to core {core_id}: {e}")
    
    # Simple but effective stress
    start_time = time.time()
    counter = 0
    
    while time.time() - start_time < duration:
        # Simple arithmetic loop - this should definitely show up in BTOP
        for i in range(100000):
            counter += i
            if counter > 1000000:
                counter = 0
        
        # Very small sleep to prevent overheating
        time.sleep(0.001)
    
    print(f"Core {core_id} completed {counter} operations")

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 simple_stress.py <num_cores> [duration]")
        print("Example: python3 simple_stress.py 4 30")
        return
    
    num_cores = int(sys.argv[1])
    duration = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    
    print(f"Starting stress test on {num_cores} cores for {duration} seconds")
    print("This should show up in BTOP as high CPU usage")
    print("Press Ctrl+C to stop early")
    
    processes = []
    
    try:
        # Start stress processes
        for i in range(num_cores):
            p = multiprocessing.Process(target=stress_core, args=(i, duration))
            p.start()
            processes.append(p)
        
        # Wait for completion or interruption
        for p in processes:
            p.join()
            
    except KeyboardInterrupt:
        print("\nStopping stress tests...")
        for p in processes:
            p.terminate()
            p.join()
    
    print("Stress test completed!")

if __name__ == "__main__":
    main()
