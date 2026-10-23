"""
Quick test: (re)creates the synthetic data and runs both scripts in quick mode.
Results go to results_quick/ (the full results in results/ are not touched).

    python quick_test.py

Takes about 1 minute on a normal PC and ends with "QUICK TEST PASSED".
"""
import os, sys, subprocess, time

here = os.path.dirname(os.path.abspath(__file__))
os.chdir(here)
py = sys.executable
t0 = time.time()

steps = [
    [py, "make_synthetic_data.py"],
    [py, "01_synthetic_test.py", "--quick", "--repeats", "2"],
    [py, "02_section_inversion.py", "--quick"],
]
for cmd in steps:
    print("\n>>>", " ".join(cmd[1:]))
    if subprocess.run(cmd).returncode != 0:
        print("QUICK TEST FAILED at", cmd[1])
        sys.exit(1)

need = ["results_quick/synthetic_test/synthetic_test_summary.csv",
        "results_quick/section_inversion/inverted_impedance_section.npy",
        "results_quick/section_inversion/well_comparison_metrics.csv",
        "results_quick/section_inversion/section_vs_true_model_metrics.csv",
        "results_quick/section_inversion/fig06_inverted_impedance_section.png"]
miss = [f for f in need if not os.path.exists(f)]
if miss:
    print("QUICK TEST FAILED, missing:", miss)
    sys.exit(1)

import pandas as pd
sec = pd.read_csv("results_quick/section_inversion/section_vs_true_model_metrics.csv", index_col=0)
print("\n", sec)
if sec.loc["HPSO", "CC_section"] <= sec.loc["Initial model", "CC_section"]:
    print("QUICK TEST FAILED: HPSO did not improve on the initial model")
    sys.exit(1)
print(f"\nQUICK TEST PASSED  ({time.time()-t0:.0f} s)")
