# Plan here is to take our data sets and put them NumPy arrays

from pathlib import Path
import re
import numpy as np

rootdir = Path('data/10s_website1/') # change here to get the info for your dataset
file_list = [f for f in rootdir.glob('**/*') if f.is_file()]
# print(len(file_list))

# for f in file_list:
#     print(f)


# For the numpy array, we might need to make it multi-dimensional
# We need to decide if we want to have an entire array for all websites or 
# to create on array per site and remove it after to save on storage

# Issue with numpy is that we need to have a certain size I will iterate through the 
# Currently it says the minimum length of a line is 1123 entires
# additionally we 15064 total rows

# I believe a good size for a numpy array is...
# 	- 15000 for the 1st dimension (which is each row)
# 	- 1100 for the second dimension (which takes the first 1100 time/packet information)

# Can do this by initializing x = np.empty(shape = (15000, 1100))


all_files = []
min = 99999
num_tot_files = 0
num_tot_rows = 0
min_file = ""
min_row = 0
for file in file_list:
	num_tot_files += 1
	new_file = []
	next_file = open(file, "r")
	count_row = 0
	for row in next_file.readlines():
		a = re.split(" ", str(row))
		if len(a) < 1: # Does not count blank lines at end of the file
			continue
		num_tot_rows += 1
		if min > len(a) and len(a) > 1:
			min = len(a)
			min_file = file
			min_row = count_row
		new_file.append(a)
		count_row += 1
	next_file.close()
	all_files.append(new_file)
	# print(all_files)
	# break

print(min)
print(num_tot_files)
print(num_tot_rows)
# print(min_file)
# print(min_row)
# print(all_files)
print('complete')
