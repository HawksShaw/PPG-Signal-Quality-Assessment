import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('./data/raw/PPG_Dataset.csv')

y = data.iloc[:, 0]
x = np.linspace(0, 1, len(y))

y_window, x_window = y[:240], x[:240]

plt.figure()
plt.subplot(2,1,1)
plt.plot(x_window, y_window)
plt.subplot(2,1,2)
plt.plot(x, y)
plt.show()

