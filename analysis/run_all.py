"""
========================================================
STEP 3: Run All Analytical Modules
Executes all 5 analytical modules in sequence.
========================================================
"""
import subprocess
import sys
import os

modules = [
    'analysis/module1_xg.py',
    'analysis/module2_pass_network.py',
    'analysis/module3_pressing.py',
    'analysis/module4_recruitment.py',
    'analysis/module5_spatial.py',
]

print("="*55)
print("  MANUTD PLATFORM — Running All Modules")
print("="*55)

base = os.path.dirname(os.path.dirname(__file__))
for module in modules:
    print(f"\nRunning {module}...")
    result = subprocess.run([sys.executable, '-W', 'ignore', os.path.join(base, module)], 
                          capture_output=False)
    if result.returncode != 0:
        print(f"  ERROR in {module}")

print("\n[DONE] Run: python dashboards/build_dashboard.py")
