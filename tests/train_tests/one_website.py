from pathlib import Path
import re
import math
from random import randrange
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

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
                prev_time += diff
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
    # return arr_packet, arr_magnitude

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
                prev_time += diff
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
    # return train_packet, train_magnitude

    y = np.zeros(shape=(60, 1))
    for i in range(60):
        if i < 30:
            y[i, 0] = 1
        elif i>=30 and i<60:
            y[i, 0] = 0

    x_p = np.concatenate((arr_packet, train_packet), axis=0)
    x_m = np.concatenate((arr_magnitude, train_magnitude), axis=0)
    # ds_packet = np.concatenate((y, x_p), axis=1)
    # ds_magnitude = np.concatenate((y, x_m), axis=1)

    return x_p, x_m, y


website = "data/result/facebook.com"

x_packet, x_magnitude, y_vals = not_website(website, 30, 200, 0.1)

X_p_train, X_p_test, y_p_train, y_p_test = train_test_split(x_packet, y_vals, test_size=0.2, random_state=0)
X_m_train, X_m_test, y_m_train, y_m_test = train_test_split(x_magnitude, y_vals, test_size=0.2, random_state=0)

X_p_dev, X_p_test, y_p_dev, y_p_test = train_test_split(X_m_test, y_m_test, test_size=0.5, random_state=0)
X_m_dev, X_m_test, y_m_dev, y_m_test = train_test_split(X_p_test, y_p_test, test_size=0.5, random_state=0)



# Data is split 80 / 20
'''
split_horizontally_idx = int(ds_packet.shape[0]* 0.8)
train_packet = ds_packet[:split_horizontally_idx , :]
test_packet = ds_packet[split_horizontally_idx: , :]

split_horizontally_idx = int(ds_magnitude.shape[0]* 0.8)
train_magnitude = ds_magnitude[:split_horizontally_idx , :]
test_magnitude = ds_magnitude[split_horizontally_idx: , :]
'''

'''
    Now use this data for data learning
    train_packet and test_packet for actual packet size
    train_magnitude and test_magnitude for 1, 0, or -1
'''


