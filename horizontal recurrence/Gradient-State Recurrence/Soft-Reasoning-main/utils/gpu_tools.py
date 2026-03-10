import torch
import threading
import time
import atexit
import matplotlib.pyplot as plt

class GPUMonitor:
    def __init__(self, llm, interval=1.0):
        self.llm = llm
        self.interval = interval
        self.running = False
        self.records = []
        self.thread = threading.Thread(target=self.monitor, daemon=True)

        self.total_gpu_blocks = self.get_total_gpu_blocks()

    def get_total_gpu_blocks(self):
        return self.llm.llm_engine.scheduler[0].block_manager.num_total_gpu_blocks

    def get_used_gpu_blocks(self):
        free_blocks = self.llm.llm_engine.scheduler[0].block_manager.get_num_free_gpu_blocks()
        return self.total_gpu_blocks - free_blocks

    def monitor(self):
        while self.running:
            timestamp = time.time()
            allocated = torch.cuda.memory_allocated() / (1024**3)  # GiB
            reserved = torch.cuda.memory_reserved() / (1024**3)    # GiB
            used_blocks = self.get_used_gpu_blocks()
            self.records.append((timestamp, allocated, reserved, used_blocks))
            time.sleep(self.interval)

    def start(self):
        if not self.running:
            self.running = True
            self.thread.start()
            print("[GPUMonitor] Monitoring started.")

    def stop(self):
        if self.running:
            self.running = False
            self.thread.join()
            print("[GPUMonitor] Monitoring stopped.")

    def report(self):
        if not self.records:
            print("[GPUMonitor] No data recorded.")
            return

        _, allocated, reserved, used_blocks = zip(*self.records)

        peak_allocated = max(allocated)
        avg_allocated = sum(allocated) / len(allocated)
        peak_reserved = max(reserved)
        avg_reserved = sum(reserved) / len(reserved)
        peak_blocks = max(used_blocks)
        avg_blocks = sum(used_blocks) / len(used_blocks)
        block_size_bytes  = self.llm.llm_engine.model_executor.driver_worker.get_cache_block_size_bytes()

        print(f"[GPUMonitor] GPU Memory & Blocks Report:\n"
              f"- Peak Allocated: {peak_allocated:.2f} GiB\n"
              f"- Average Allocated: {avg_allocated:.2f} GiB\n"
              f"- Peak Reserved: {peak_reserved:.2f} GiB\n"
              f"- Average Reserved: {avg_reserved:.2f} GiB\n"
              f"- Peak GPU Blocks Used: {peak_blocks}\n"
              f"- Average GPU Blocks Used: {avg_blocks:.2f}\n"
              f"- Block Size: {block_size_bytes} bytes"
              )

