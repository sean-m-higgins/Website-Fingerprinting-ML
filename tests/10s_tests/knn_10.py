from pathlib import Path
import re
import math
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Min length is 1059
# Min row is 15001
x = np.empty(shape = (15000, 1100, 4)) # initialize numpy array
rootdir = Path('data/10s_website1/') 
file_list = [f for f in rootdir.glob('**/*') if f.is_file()]

count_row = 0
counter = 0
website_index = 0
for file in file_list:
    next_file = open(file, "r")
    counter+=1
    for row in next_file.readlines():
        a = re.split(" ", str(row))
        if len(a) < 1: # Does not count blank lines at end of the file
	        continue
        c = re.split(":", a[2])
        time_start = float(c[0])
        for i in range(0, 1100):
            x[count_row, i, 0] = website_index
            b = re.split(":", str(a[i+2]))
            x[count_row, i, 1] = float(b[0]) - time_start
            if float(b[1]) > 0:
                x[count_row, i, 2] = 1 # positive packetsize
            else:
                x[count_row, i, 2] = 0 # negative packetsize
            x[count_row, i, 3] = float(b[1])
        count_row += 1
        if count_row >= 15000:
            break
    website_index += 1
    print(str(math.floor((counter / 116)*100)) + "%")
    '''
        Each row has data in form 
            website_num     time_from_start     pos_neg     actual_packetsize
                0                   1              2               3
    '''
    
print(x)


'''
    Now to do actual knn
    Need to think of how we will split the data and create X and create y

neigh = KNeighborsClassifier(n_neighbors=10)
X = 'a' # Need to initialize X
y = 'b' # Need to initialize Y
neigh.fit(X, y)
print(neigh.predict([[1.1]]))
print(neigh.predict_proba([[0.9]]))
'''