from window_estimation import *
from prepare_wildppg import *
import matplotlib.pyplot as plt
import numpy as np

window_signal, window_time = signal_cutoff(signal, 0, 10)
