from sqis.preprocessing import wildppg_init as wppg
import os
from sqis.utils.run_pipeline import run_pipeline

current_directory = os.path.dirname(os.path.abspath(__file__))
raw_path_relative = os.path.join(current_directory, '../data/raw')
data_path = os.path.abspath(raw_path_relative)

if not os.path.exists(data_path):
    raise FileNotFoundError(f"File path {data_path} does not exist")
else:
    df = run_pipeline(data_path)
    df.head(5)