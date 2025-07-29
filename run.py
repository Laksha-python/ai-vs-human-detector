# run.py
import os
import subprocess

print("🔍 Launching screen capture detection...")
subprocess.run(["python", os.path.abspath("detect_realtime.py")])
