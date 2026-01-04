import numpy as np

def signal_window(signal, window_start, window_length, fs):

    # --- Feed an entire signal to cut into windows --- 
    window_samples = fs*window_length
    get_window_signal = signal[:window_samples]
    window_time = np.linspace(window_start, window_start + window_length, window_samples)

    return window_signal, window_time



# --- USE THIS FOR DOCKER STREAMING --- 

# 1. STOP & CLEAN UP
# Write-Host "🛑 Cleaning up..."
# docker stop ppg-assessment 2>$null
# docker rm ppg-assessment 2>$null

# # Remove contents of the data folder (keep the folder itself)
# if (Test-Path .\object_store_segments) { 
#     Remove-Item -Path .\object_store_segments\* -Recurse -Force -ErrorAction SilentlyContinue 
# }

# # Delete the old database file
# Remove-Item -Path .\quality_assessment.db -Force -ErrorAction SilentlyContinue

# # 2. CREATE FRESH FILES
# Write-Host "✨ Creating fresh environment..."
# New-Item -Path . -Name "quality_assessment.db" -ItemType "file" | Out-Null
# if (-not (Test-Path .\object_store_segments)) { 
#     New-Item -Path . -Name "object_store_segments" -ItemType "directory" | Out-Null 
# }

# # 3. RUN WITH CORRECT MAPPING
# Write-Host "🐳 Starting Service..."
# docker run --name ppg-assessment -p 8000:8000 `
#   -v ${PWD}/object_store_segments:/app/object_store `
#   -v ${PWD}/quality_assessment.db:/app/quality_assessment.db `
#   ppg-assessment