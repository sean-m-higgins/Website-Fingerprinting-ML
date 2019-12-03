from pathlib import Path
import re
import math
from random import randrange
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
            prev_time += diff
            count_seconds +=1
            if count_seconds >= n:
                break
            count += 1
        counter += 1
    return arr_packet, arr_magnitude


def create_train_test(web, l, n, diff):
    train_packet = np.zeros(shape = (l, n))
    train_magnitude = np.zeros(shape = (l, n))
    
    rootdir = Path('data/result/') 
    file_list = [f for f in rootdir.glob('**/*') if f.is_file()]

    for i in range(67):
        if file_list[i] == website:
            file_list.remove(i)
    
    for i in range(l):
        random_site = randrange(66)
        con = 0
        fil = file_list[random_site - 1]
        view_file = open(fil, "r")
        for row in view_file.readlines():
            con+=1
        random_row = randrange(con)
        # print(view_file)
        # Now I need to iterate through this row and create a row in the numpy array
        line = ""
        cont = 0
        view_file = open(fil, "r")
        for row in view_file.readlines():
            if cont == random_row:
                line = row
                # print(line)
                break
            cont+=1
        split = re.split(" ", str(line))
        if len(split) < 2:
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
                train_packet[i, count_seconds] = 0
                train_magnitude[i, count_seconds] = 0
                gap-=1
                count_seconds+=1
                if count_seconds >= n:
                    break
            if count_seconds >= n:
                break
            packet = float(info[1])
            train_packet[i, count_seconds] = packet
            if packet > 0:
                train_magnitude[i, count_seconds] = 1
            elif packet < 0:
                train_magnitude[i, count_seconds] = -1
            prev_time += diff
            count_seconds +=1
            if count_seconds >= n:
                break
            count += 1
    return train_packet, train_magnitude


website = "data/result/facebook.com"

p, m = not_website(website, 30, 200, 0.1)
train_p, train_m = create_train_test(website, 30, 200, 0.1)
# print(train_p)
# print(p)

y = np.zeros(shape=(60, 1))
for i in range(60):
    if i < 30:
        y[i, 0] = 1
    elif i>=30 and i<60:
        y[i, 0] = 0
# print(y)

x_p = np.concatenate((p, train_p), axis=0)
x_m = np.concatenate((m, train_m), axis=0)
ds_packet = np.concatenate((y, x_p), axis=1)
ds_magnitude = np.concatenate((y, x_m), axis=1)
# print(ds_packet)
# print(ds_magnitude)

''' 
    Need to then split the test and train into 30-70 and then we are free to do testing

    I will combine the two functions for simplification once completed
'''


