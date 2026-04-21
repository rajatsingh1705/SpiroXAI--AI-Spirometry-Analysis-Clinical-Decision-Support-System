import pandas as pd
df = pd.read_csv('train_raw.csv')
mask = (df['Sex'] == 'Male') & (df['Age'] >= 20) & (df['Age'] <= 24) & (df['Height'] >= 174) & (df['Height'] <= 180) & (df['Weight'] >= 60) & (df['Weight'] <= 70) & (df['Race'].str.contains('Other race', na=False))
cols = ['Age','Sex','Race','Weight','Height','BMI','Baseline_PEF_Ls','Baseline_FEF2575_Ls','Baseline_Extrapolated_Volume_L','Baseline_Forced_Expiratory_Time_s','Baseline_Number_Acceptable_Curves','Disease_Label']
results = df[mask][cols]
results.index = results.index + 2
for idx, row in results.iterrows():
    print(f"\n--- Row {idx} ---")
    for col in cols:
        print(f"  {col}: {row[col]}")
