import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]  # project root
csv_path = BASE_DIR / "prediction_logs.csv"

df = pd.read_csv(csv_path)

report = Report(metrics=[DataDriftPreset()])
mid = len(df) // 2

result = report.run(reference_data=df.iloc[:mid], current_data=df.iloc[mid:])

result.save_html(str(BASE_DIR / "evidently_report.html"))
print("✅ Evidently report generated: evidently_report.html")
