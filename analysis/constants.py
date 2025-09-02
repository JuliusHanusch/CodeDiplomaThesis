import sqlite3
import pandas as pd 
import altair as alt
from pathlib import Path
from sklearn.linear_model import LinearRegression
import numpy as np
import json
import sys


db_path = Path("AION_4.db")

output_folder_fig = Path("./cache/figures/")
output_folder_tex = Path("./cache/tex/")
output_folder_fig.mkdir(parents=True, exist_ok=True)
output_folder_tex.mkdir(parents=True, exist_ok=True)

alt.renderers.enable("browser")