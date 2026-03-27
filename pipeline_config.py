import warnings
import pandas as pd

# Paths and high-level run switches.
DATA_FOLDER = r"/Users/dinahang/Desktop/AAE25_FYP Subject_DATA/QAR_DATA_copy"
OUTPUT_FOLDER = r"/Users/dinahang/Desktop/AAE25_FYP Subject_DATA/output"

SCENARIO_KEYWORDS = ['vcrosswind', 'clear', 'lowvis_day', 'lowvis_night', 'crosswind']

# Flight/environment constants used in feature engineering.
RUNWAY_TRUE_HEADING_DEG = 71.0
RUNWAY_ELEVATION_FT = 28.0
QNH_HPA = 1022.0
HPA_TO_FEET = 27.0

N_SECONDS_BEFORE_TD = 15.0
FALLBACK_DT_SECONDS = 1.0
HARD_LANDING_FPM_THRESHOLD = -450.0

CTRL_MAX_ABS = 16383.0

DO_PERM_IMPORTANCE = True
PERM_N_REPEATS = 10

# Candidate column names used to map varying raw CSV schemas.
THROTTLE_CANDIDATES = [
    "throttle", "throttleCmd", "throttleLever", "thrLever", "thr", "thrustLevers",
    "throttle1", "throttle2", "thrLever1", "thrLever2",
    "eng1ThrLever", "eng2ThrLever",
]
AOA_CANDIDATES = ["angleOfAttackAOA", "aoa", "angleOfAttack", "alpha"]
BARO_ALT_CANDIDATES = ["alt_baro", "altBaro", "baroAlt", "alt_baro_feet", "alt_baro_ft", "pressureAlt", "pressAlt"]
BANK_CANDIDATES = ["bank", "bank_raw", "bankDeg", "roll", "rollDeg", "roll_raw"]
PITCH_CANDIDATES = ["pit", "pitch", "pitchDeg", "pitch_raw"]


def configure_runtime() -> None:
    # Keep notebook/terminal output clean and easier to scan.
    warnings.filterwarnings("ignore", category=UserWarning)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 200)
