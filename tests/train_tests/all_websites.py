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


def what_site(l, n, diff):
    # need to first create a unique list of each site
    # going to have 25 rows analyzed for each of the 67 files
    # 67*25 = 1675
    dataset = np.zeros(shape = ((67*l), n))
    arr_packet = np.zeros(shape = ((67*l), n))
    arr_magnitude = np.zeros(shape = ((67*l), n))
    a = 0
    for file in file_list:
        site = site_index[file]
        f = open(file, "r")
        counter = 0
        for row in f.readlines():
            split = re.split(" ", str(row))
            if counter >= l or len(split) < 2:
                break
            origin = re.split(":", str(split[2]))
            start_time = float(origin[0])
            ten_sec_reached = True
            count = 0
            prev_time = start_time
            count_seconds = 0
            while ten_sec_reached:
                info = re.split(":", str(split[count+2]))
                if len(info)<2:
                    break
                time = float(info[0])
                if (len(split) <= count + 3) or ((time - start_time) >= 10):
                    break
                if time - prev_time < diff:
                    count += 1
                    continue
                gap = (math.floor((time-prev_time) / diff)) - 1
                while gap > 0:
                    arr_packet[counter + (a*l), count_seconds] = 0
                    arr_magnitude[counter + (a*l), count_seconds] = 0
                    gap-=1
                    count_seconds+=1
                    prev_time += diff
                    if count_seconds >= n:
                        break
                if count_seconds >= n:
                    break
                packet = float(info[1])
                arr_packet[counter + (a*l), count_seconds] = packet
                if packet > 0:
                    arr_magnitude[counter + (a*l), count_seconds] = 1
                elif packet < 0:
                    arr_magnitude[counter + (a*l), count_seconds] = -1
                prev_time += diff
                count_seconds +=1
                if count_seconds >= n:
                    break
                count += 1
            counter += 1
        a+=1

        y = np.zeros(shape=((l*2), 1))
        for i in range((l*2)):
            if i < l:
                y[i, 0] = 1
            elif i>=l and i<(l*2):
                y[i, 0] = 0
                
        return arr_packet, arr_magnitude, y


a,b = what_site(25, 200, 0.1)
print(b)