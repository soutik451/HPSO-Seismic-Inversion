"""
Settings for the HPSO post-stack impedance inversion.
By default everything runs on the synthetic data set in synthetic_data/.
Edit DATA_DIR / file names / column names to use your own data.
"""
import os

# ---------------- data -----------------
# synthetic data set created by make_synthetic_data.py (replace with your own files)
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "synthetic_data")
SEISMIC_FILE = "synthetic_seismic.txt"     # (n_samples, n_traces) text matrix, one column per CDP
WELL_FILE = "synthetic_well.csv"           # well file (csv or xlsx)
WELL_IMP_COL = "True_Impedance"            # true (high resolution) acoustic impedance at the well
WELL_SEIS_COL = "Well_Tied_Seismic"        # well-tied seismic trace at the well (used for case B)
TRUE_SECTION_FILE = "synthetic_true_impedance_section.npy"  # only known for synthetic data (optional)
COMPARE_COLS = {}                          # optional: other impedance results at the well {name: column}

# ---------------- geometry -------------
T0_MS = 1700.0          # time of the first sample
DT = 0.002              # sample interval (s)
N_SAMPLES = 201         # 1700-2100 ms
WELL_TRACE = 50         # column index of the well (middle trace)
CDP_FIRST = 1001        # CDP number of the first column

# ---------------- initial model --------
LFM_CUTOFF_HZ = 10.0    # low-pass cut-off of the well impedance
DTW_MAX_SHIFT = 6       # max shift (samples) between neighbouring traces when tracking structure
LFM_LATERAL_SMOOTH = 3  # running mean over traces (1 = none)

# ---------------- wavelet --------------
WAVELET_LEN = 61        # samples (120 ms)
RICKER_FREQ = 30.0      # Hz, wavelet of the synthetic test

# ---------------- inversion ------------
BOUND = 0.45            # search space: ln Z = ln Z0 +/- BOUND (about -36 % / +57 %)
LAMBDA = 0.5            # weight of the initial-model constraint (tuned on independent synthetic models)
PSO = dict(n_particles=50, n_iter=100, w_start=0.9, w_end=0.4, c1=1.5, c2=1.5,
           vmax_frac=0.2, init_std=0.15, band=0.32)
QNM_ITER = 300
PSO_ONLY_ITER = 300     # iterations of the stand-alone PSO used for comparison
SEED = 42

# ---------------- synthetic test -------
NOISE_LEVELS = [0.0, 0.1, 0.2, 0.3]   # Gaussian noise std as a fraction of the synthetic std
N_REPEATS = 5                         # noise realisations / PSO seeds per noise level

# evaluation in the seismic band as well (inversion can not recover frequencies above it)
EVAL_BAND_HZ = 80.0

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# ---------------- section inversion options ----
EDGE_TAPER = 10                       # samples at each end of the window with reduced data weight
WAVELET_SMOOTH_HZ = 3.0               # smoothing of the average amplitude spectrum (statistical wavelet)
RICKER_SCAN_HZ = list(range(15, 51, 5))
# weight of the initial-model constraint for the section inversion: chosen with the discrepancy principle
# (final misfit = noise level estimated from the well tie, 1 - CC_tie^2)
LAMBDA_GRID = [0.5, 1, 2, 4, 8, 16, 32, 64, 128]
