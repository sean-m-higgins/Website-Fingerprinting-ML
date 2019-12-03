from pathlib import Path
import re
import math
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# Paramters for definition are as follows
#   input = string to get to file directory (i.e. "data/result/facebook.com")
#   l = number of rows youwant to look through, starting bias
#   n = number of time differences you want to check
#   dif = the time difference needed to record that packet
def not_website(input, l, n, diff):
    f = open(input, "r")
    arr_packet = np.zeros(shape = (l, n))
    arr_magnitude = np.zeros(shape = (l, n))
    counter = 0
    for row in f:
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
                arr_packet[counter, count_seconds] = 0
                arr_magnitude[counter, count_seconds] = 0
                gap-=1
                count_seconds+=1
                if count_seconds >= n:
                    break
            if count_seconds >= n:
                break
            packet = float(info[1])
            arr_packet[counter, count_seconds] = packet
            if packet > 0:
                arr_magnitude[counter, count_seconds] = 1
            elif packet < 0:
                arr_magnitude[counter, count_seconds] = -1
            prev_time = time
            count_seconds +=1
            count += 1
        counter += 1
    return arr_packet, arr_magnitude


website = "data/result/facebook.com"

p, m = not_website(website, 30, 200, 0.1)
# print(p)
# print(m)