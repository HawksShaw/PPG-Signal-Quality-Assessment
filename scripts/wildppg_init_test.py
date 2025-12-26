from sqis.preprocessing import wildppg_init as wppg
import os

current_directory = os.path.dirname(os.path.abspath(__file__))
raw_path_relative = os.path.join(current_directory, '../data/raw')
data_path = os.path.abspath(raw_path_relative)

if not os.path.exists(data_path):
    raise FileNotFoundError(f"File path {data_path} does not exist")
else:
    loaded_stream = wppg.wildppg_stream(data_path)

    for i, stream in enumerate(loaded_stream):
        print(f"Window {i} loaded for subject: {stream['metadata']['subject_id']}")
        if i >=5: break