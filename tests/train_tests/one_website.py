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
            prev_time = time
            count_seconds +=1
            count += 1
        counter += 1
    return arr_packet, arr_magnitude


# Now I need to create my x and y variables to run a successful KNN test
# Also I need to think through this a lot since we want it to guess the time the second website starts
# I think we should create a generic test now, and create binary classifier
# Basically following same procedure above except the row is now generic amongst all files
# But some lines need to contain lines from given website

def create_train_test(web, l, n, diff):
    # Copy paste as above basically except make it a random row in a random website
    train_packet = np.zeros(shape = (l, n))
    train_magnitude = np.zeros(shape = (l, n))
    # test_packet = np.zeros(shape = (l, n))
    # test_magnitude = np.zeros(shape = (l, n))

    rootdir = Path('data/result/') 
    file_list = [f for f in rootdir.glob('**/*') if f.is_file()]
    
    for i in range(l):
        random_site = randrange(67)
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
            continue
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
            prev_time = time
            count_seconds +=1
            count += 1
    return train_packet, train_magnitude


''' 
Plan
    Concatenate the non_website with website and see if it can tell if it is facebook or not
'''


website = "data/result/facebook.com"

p, m = not_website(website, 30, 200, 0.1)
train_p, train_m = create_train_test(website, 30, 200, 0.01)
print(train_p[0])
print(p)
# print(m)
