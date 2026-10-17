"""Starts the registry and the four agents as separate processes on their own ports (Ctrl+C stops all)."""
import subprocess
import sys
import time

SERVICES = [("registry", 8100), ("diagnosis", 8101), ("parts", 8102), ("scheduling", 8103), ("coordinator", 8110)]


def main():
    processes = []
    try:
        for module, port in SERVICES:
            processes.append(subprocess.Popen([sys.executable, "-m", f"workshop_a2a.{module}"]))
            print(f"started {module} on port {port}", flush=True)
            time.sleep(1)
        print("All services running. Run 'python ask.py' in a second terminal.", flush=True)
        while all(p.poll() is None for p in processes):
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for p in processes:
            p.terminate()


if __name__ == "__main__":
    main()
