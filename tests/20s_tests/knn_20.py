from pathlib import Path
import re
import math
import numpy as np

# Min length is 1059
# Min row is 15001
x = np.empty(shape = (15000, 1000, 2)) # initialize numpy array
rootdir = Path('data/20s_website1/') 
file_list = [f for f in rootdir.glob('**/*') if f.is_file()]

count_row = 0
counter = 0
for file in file_list:
    next_file = open(file, "r")
    counter+=1
    for row in next_file.readlines():
        a = re.split(" ", str(row))
        if len(a) < 1: # Does not count blank lines at end of the file
	        continue
        for i in range(0, 1000):
            b = re.split(":", str(a[i+2]))
            x[count_row, i, 0] = float(b[0])
            x[count_row, i, 1] = float(b[1])
        count_row += 1
        if count_row >= 15000:
            break
    # print(str(math.floor((counter / 116)*100)) + "%")

print(x)
