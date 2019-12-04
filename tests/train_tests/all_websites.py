from pathlib import Path
import re
import math
from random import randrange
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

rootdir = Path('data/result/') 
file_list = [f for f in rootdir.glob('**/*') if f.is_file()]

site_index = {}
for i in range(67):
    site = file_list[i]
    site_index[site] = i


def what_site(n, diff):
    # need to first create a unique list of each site
    # going to have 25 rows analyzed for each of the 67 files
    # 67*25 = 1675
    dataset = np.zeros(shape = (1675, n))

    for file in file_list:
        site = site_index[file]
        f = open(file, "r")
        a = 0
        for row in f.readlines():
            if a == 25:
                break
            a += 1
        


what_site(200, 0.1)