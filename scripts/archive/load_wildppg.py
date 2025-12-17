import os
import urllib.request
import scipy.io

# ---- Set data file name and Hugging Face URL ----
datafile = 'WildPPG.mat'
hf_url = 'https://huggingface.co/datasets/eth-siplab/WildPPG/resolve/main/WildPPG.mat'

# ---- Check if file exists; if not, download ----
if not os.path.isfile(datafile):
    print(f'Data file {datafile} not found.')
    print('Downloading from Hugging Face...')
    urllib.request.urlretrieve(hf_url, datafile)
    print('Download complete!')
else:
    print(f'Found data file: {datafile}')

# ---- Load the data ----
data = scipy.io.loadmat(datafile)