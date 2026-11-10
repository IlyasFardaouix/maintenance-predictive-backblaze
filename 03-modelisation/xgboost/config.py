# Hyperparamètres XGBoost v3 — protocole commun
WINDOW         = 30
CAP            = 125.0
ALERT_RUL      = 30
MAX_GAP        = 7
N_HEALTHY_MULT = 2
SEED           = 42
THRESHOLD_RUL  = 54.75

XGB_PARAMS = {
    'objective':        'reg:squarederror',
    'tree_method':      'hist',
    'device':           'cuda',
    'max_depth':        6,
    'learning_rate':    0.05,
    'subsample':        0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 10,
    'seed':             SEED,
}
NUM_BOOST_ROUND       = 500
EARLY_STOPPING_ROUNDS = 20
BEST_ITERATION        = 66