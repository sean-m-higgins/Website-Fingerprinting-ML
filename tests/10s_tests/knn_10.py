from pathlib import Path
import re
import numpy as np

x = np.empty(shape = (15000, 1100)) # initialize numpy array
rootdir = Path('data/10s_website1/') 
file_list = [f for f in rootdir.glob('**/*') if f.is_file()]

count_row = 0
for file in file_list:
    next_file = open(file, "r")
    for row in next_file.readlines():
        a = re.split(" ", str(row))
        if len(a) < 1: # Does not count blank lines at end of the file
	        continue
        count_row += 1
        for i in range(0, 1100):
            x[count_row, i] = a[i+1]
    print(file)
    
print(x)