# coding: utf-8
import sys, os
sys.path.append('c:/Users/NIHAR/projects/Minor Project/backend')
from ml_pipeline import get_predictor
p = get_predictor()

def check(name, sex, age, wt, ht, pef, fef, ev, fet, crv):
    bmi = wt / ((ht/100)**2)
    inp = {
        "Sex": sex, "Age": age, "Weight": wt, "Height": ht, "BMI": bmi,
        "Baseline_PEF_Ls": pef, "Baseline_FEF2575_Ls": fef,
        "Baseline_Extrapolated_Volume": ev, "Baseline_Forced_Expiratory_Time": fet,
        "Baseline_Number_Acceptable_Curves": crv,
        "Race_Black": 0, "Race_Mexican American": 0, "Race_Other hispanic": 0,
        "Race_Other race, including multi-racial": 0, "Race_White": 1
    }
    # Predict directly
    res = p.predict(inp)
    
    print(f"\n--- {name} ---")
    print(f"Inputs: PEF={pef}, FEF={fef}, FET={fet}, Vol={ev}")
    print(f"Prediction: {res['prediction']} (Confidence: {res['confidence_pct']})")
    for l, pr in zip(res['probability_distribution']['labels'], res['probability_distribution']['probabilities']):
        print(f"  {l}: {pr*100:.1f}%")

print("Running checks...")

check("Normal", 1, 45, 70, 170, 8.5, 4.2, 0.10, 4.0, 3)

# Try very severe obstruction
check("Sev Obstruction 1", 1, 65, 85, 170, 2.0, 0.4, 0.30, 12.0, 3)
check("Sev Obstruction 2", 1, 65, 85, 170, 1.0, 0.1, 0.50, 15.0, 3)

# Try very severe restriction
check("Sev Restriction 1", 0, 50, 60, 150, 3.0, 3.0, 0.05, 1.5, 3)
check("Sev Restriction 2", 0, 50, 60, 150, 1.5, 1.5, 0.02, 0.8, 3)
