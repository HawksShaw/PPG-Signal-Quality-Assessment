import heartpy as hp

def peaks_heartpy(signal, fs):
    try:
        working_data, measures = hp.process(signal, sample_rate=fs, clean_rr=True)
        return working_data['peaklist']
    except Exception as e:
        print(f"Error while analyzing systolic peaks: {e}")
        return []