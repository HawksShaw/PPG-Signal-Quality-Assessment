import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sqlite3
import json

db_path = "quality_assessment.db"

def load_data():
    connect = sqlite3.connect(db_path)
    