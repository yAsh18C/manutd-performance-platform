"""
Run this once to download and process all data.
On Streamlit Cloud this runs automatically via startup.
"""
import os
import subprocess
import sys

def setup():
    print("Setting up data pipeline...")
    scripts = [
        'ingestion/fetch_data.py',
        'transformation/transform.py',
        'analysis/module1_xg.py',
        'analysis/module2_pass_network.py',
        'analysis/module3_pressing.py',
        'analysis/module4_recruitment.py',
        'analysis/module5_spatial.py',
    ]
    for script in scripts:
        print(f"Running {script}...")
        result = subprocess.run([sys.executable, '-W', 'ignore', script])
        if result.returncode != 0:
            print(f"Warning: {script} had errors")
    print("Setup complete. Run: streamlit run app.py")

if __name__ == '__main__':
    setup()
