"""
Input validation for spirometry prediction requests.
Returns structured errors: [{"field": "...", "message": "..."}]
"""

import math
from typing import List, Dict, Any

# Clinical validation ranges for each input field
FIELD_RULES = {
    "Age":    {"min": 5,   "max": 120,  "integer": True,  "label": "Age"},
    "Sex":    {"min": 0,   "max": 1,    "integer": True,  "label": "Sex"},
    "Weight": {"min": 10,  "max": 300,  "integer": False, "label": "Weight (kg)"},
    "Height": {"min": 50,  "max": 250,  "integer": False, "label": "Height (cm)"},
    "BMI":    {"min": 10,  "max": 70,   "integer": False, "label": "BMI"},
    "Baseline_PEF_Ls":                    {"min": 0.0, "max": 20.0, "integer": False, "label": "PEF (L/s)"},
    "Baseline_FEF2575_Ls":                {"min": 0.0, "max": 15.0, "integer": False, "label": "FEF 25-75% (L/s)"},
    "Baseline_Extrapolated_Volume":       {"min": 0.0, "max": 5.0,  "integer": False, "label": "Extrapolated Volume (L)"},
    "Baseline_Forced_Expiratory_Time":    {"min": 0.0, "max": 30.0, "integer": False, "label": "Forced Expiratory Time (s)"},
    "Baseline_Number_Acceptable_Curves":  {"min": 0,   "max": 20,   "integer": True,  "label": "Acceptable Curves"},
}

RACE_FIELDS = [
    "Race_Black",
    "Race_Mexican American",
    "Race_Other hispanic",
    "Race_Other race, including multi-racial",
    "Race_White",
]


def validate_prediction_input(data: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Validate all prediction input fields.
    Returns list of error dicts: [{"field": "field_name", "message": "..."}]
    Empty list means all inputs are valid.
    """
    errors = []

    for field_name, rules in FIELD_RULES.items():
        value = data.get(field_name)
        label = rules["label"]

        # Check presence
        if value is None:
            errors.append({"field": field_name, "message": f"{label} is required"})
            continue

        # Check type
        try:
            num_val = float(value)
        except (ValueError, TypeError):
            errors.append({"field": field_name, "message": f"{label} must be a number"})
            continue

        # Check NaN / Inf
        if math.isnan(num_val) or math.isinf(num_val):
            errors.append({"field": field_name, "message": f"{label} must be a valid number (not NaN or Inf)"})
            continue

        # Check negative
        if num_val < 0:
            errors.append({"field": field_name, "message": f"{label} cannot be negative"})
            continue

        # Check range
        if num_val < rules["min"] or num_val > rules["max"]:
            errors.append({
                "field": field_name,
                "message": f"{label} must be between {rules['min']} and {rules['max']}"
            })
            continue

        # Check integer constraint
        if rules.get("integer") and num_val != int(num_val):
            errors.append({"field": field_name, "message": f"{label} must be a whole number"})

    # Validate that at least one race field is set
    race_sum = 0
    for rf in RACE_FIELDS:
        val = data.get(rf, 0)
        try:
            race_sum += float(val)
        except (ValueError, TypeError):
            pass

    if race_sum == 0:
        errors.append({"field": "Race", "message": "Race/ethnicity is required"})

    return errors
