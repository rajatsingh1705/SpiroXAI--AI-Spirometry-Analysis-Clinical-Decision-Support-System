
import sys, os
sys.path.append('c:/Users/NIHAR/projects/Minor Project/backend')
from ml_pipeline import get_predictor, CLASS_NAMES
import numpy as np

p = get_predictor()
raw = {
    'Sex': 1, 'Age': 65, 'Weight': 85, 'Height': 170, 'BMI': 29.4,
    'Baseline_PEF_Ls': 2.0, 'Baseline_FEF2575_Ls': 0.4,
    'Baseline_Extrapolated_Volume': 0.3, 'Baseline_Forced_Expiratory_Time': 12.0,
    'Baseline_Number_Acceptable_Curves': 3,
    'Race_Black': 0, 'Race_Mexican American': 0, 'Race_Other hispanic': 0,
    'Race_Other race, including multi-racial': 0, 'Race_White': 1
}
res = p.predict(raw)
print(f"PRED: {CLASS_NAMES[res['predicted_class']]}")
print(f"BARS: {res['probabilities']}")
