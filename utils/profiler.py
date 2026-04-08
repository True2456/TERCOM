import time
from contextlib import contextmanager
import numpy as np

class Profiler:
    def __init__(self):
        self.stats = {}
        self.window_size = 50
        
    @contextmanager
    def track(self, name):
        start = time.perf_counter()
        yield
        end = time.perf_counter()
        
        duration = (end - start) * 1000 # ms
        if name not in self.stats:
            self.stats[name] = []
            
        self.stats[name].append(duration)
        if len(self.stats[name]) > self.window_size:
            self.stats[name].pop(0)

    def report(self):
        print("\n--- PERFORMANCE REPORT (ms) ---")
        print(f"{'Task':<20} | {'PC (Avg)':<10} | {'Pi Zero 2 (Est.)':<10}")
        print("-" * 50)
        
        # Pi Scaling Factor (Approximation: Pi Zero 2 W vs RTX 3060 PC CPU)
        # pi_factor = 15.0 
        # But for FFT/NumPy on Pi vs 3060, it's more like 20x for CPU-bound ops
        pi_factor = 20.0 
        
        total_pi = 0
        for name, durations in self.stats.items():
            avg_pc = np.mean(durations)
            est_pi = avg_pc * pi_factor
            
            # Special case for GPU tasks (Simulation) which don't run on Pi
            if "Simulate" in name:
                print(f"{name:<20} | {avg_pc:10.2f} | N/A (Simulator)")
                continue
                
            total_pi += est_pi
            print(f"{name:<20} | {avg_pc:10.2f} | {est_pi:10.2f}")
            
        print("-" * 50)
        print(f"{'TOTAL TRN LATENCY':<20} | {'-':<10} | {total_pi:10.2f} ms")
        
        rate = 1000.0 / total_pi if total_pi > 0 else 0
        print(f"{'EST. MATCH RATE':<20} | {'-':<10} | {rate:10.2f} Hz")
        print("-" * 50)

if __name__ == "__main__":
    p = Profiler()
    with p.track("Test"):
        time.sleep(0.01)
    p.report()
