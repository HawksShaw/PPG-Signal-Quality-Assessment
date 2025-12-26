from sqis.preprocessing import wildppg_init as wppg

loaded_stream = wppg.wildppg_stream('../data/raw')

for window in loaded_stream:
    print("hi")