from pathlib import Path
import re
import math
from random import randrange
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
import time

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

    y = np.zeros(shape=((l*2), 1))
    for i in range((l*2)):
        if i < l:
            y[i, 0] = 1
        elif i>=l and i<(l*2):
            y[i, 0] = 0

    x_p = np.concatenate((arr_packet, train_packet), axis=0)
    x_m = np.concatenate((arr_magnitude, train_magnitude), axis=0)
    # ds_packet = np.concatenate((y, x_p), axis=1)
    # ds_magnitude = np.concatenate((y, x_m), axis=1)

    return x_p, x_m, y

def split(train, y, i, n):
	t_n = len(train)
	new_arr = np.asarray(train)
	new_y = np.asarray(y)
	if i == 1:
		low_index = 0
		high_index = int(t_n / n)
	elif i == n:
		low_index = int(((i-1)/n)*t_n)
		high_index = t_n
	else:
		low_index = int(((i-1)/n)*t_n)
		high_index = int((i/n)*t_n)
	X_train = np.concatenate((new_arr[:low_index],new_arr[high_index:]))
	X_test = new_arr[low_index:high_index]
	y_train = np.concatenate((new_y[:low_index],new_y[high_index:]))
	y_test = new_y[low_index:high_index]
	return X_train, X_test, y_train, y_test


start_start = time.time()

website = "data/result/facebook.com"

x_packet, x_magnitude, y_vals = not_website(website, 50, 200, 0.1)

# Standardize the features
sc = StandardScaler()
sc.fit(x_packet)

sc1 = StandardScaler()
sc1.fit(x_magnitude)

X_p_train, X_p_test, y_p_train, y_p_test = train_test_split(x_packet, y_vals, test_size=0.2, random_state=0)
X_m_train, X_m_test, y_m_train, y_m_test = train_test_split(x_magnitude, y_vals, test_size=0.2, random_state=0)

X_p_dev, X_p_test, y_p_dev, y_p_test = train_test_split(X_m_test, y_m_test, test_size=0.5, random_state=0)
X_m_dev, X_m_test, y_m_dev, y_m_test = train_test_split(X_p_test, y_p_test, test_size=0.5, random_state=0)

y_m_train_raveled = y_m_train.ravel()
y_p_train_raveled = y_p_train.ravel()

#-----------------------------------------------------------------------
# GRID SEARCH Logistic Regression:
start = time.time()

grid_C = [0.01, 1.0, 10, 100]
grid_penalty = ['l1', 'l2']

top_p_accuracy = 0
top_p_C = 0.0
top_p_penalty = []
top_m_accuracy = 0
top_m_C = 0.0
top_m_penalty = []

for penalty in grid_penalty:
	for C in grid_C:
		#Logistic Regression on packet size - dev
		clf = LogisticRegression(random_state=0, solver='liblinear', penalty=penalty, C=C).fit(X_p_train, y_p_train_raveled)
		p_dev_eval = accuracy_score(y_p_dev, clf.predict(X_p_dev)) * 100
		print("Accuracy of log reg on packet size - dev: ", p_dev_eval, "%")
		if p_dev_eval > top_p_accuracy:
			top_p_accuracy = p_dev_eval
			top_p_C = C
			top_p_penalty = penalty
		#Logistic Regression on magnitude - dev
		clf2 = LogisticRegression(random_state=0, solver='liblinear', penalty=penalty, C=C).fit(X_m_train, y_m_train_raveled)
		m_dev_eval = accuracy_score(y_m_dev, clf2.predict(X_m_dev)) * 100
		print("Accuracy of log reg on magnitude - dev: ", m_dev_eval, "%")
		if m_dev_eval > top_m_accuracy:
			top_m_accuracy = m_dev_eval
			top_m_C = C
			top_m_penalty = penalty

print('Top Model Accuracy p: ' + str(top_p_accuracy) + ' C: ' + str(top_p_C) + ' penalty: ' + str(top_p_penalty))
print('Top Model Accuracy m: ' + str(top_m_accuracy) + ' C: ' + str(top_m_C) + ' penalty: ' + str(top_m_penalty))

end = time.time()
print(end - start)

#-----------------------------------------------------------------------
# GRID SEARCH KNN:
start = time.time()

grid_neighbors = [3, 5, 7, 21]

top_p_accuracy = 0
top_p_neighbor = 0.0
top_m_accuracy = 0
top_m_neighbor = 0.0

for neighbor in grid_neighbors:	
	# KNN of 5 on packet_size - dev
	neigh1 = KNeighborsClassifier(n_neighbors=neighbor).fit(X_p_train, y_p_train_raveled)
	p_dev_ev = accuracy_score(y_p_dev, neigh1.predict(X_p_dev)) * 100
	print("Accuracy of k nearest neighbors on packet size - dev: ", p_dev_ev, "%")
	if p_dev_ev > top_p_accuracy:
		top_p_accuracy = p_dev_ev
		top_p_neighbor = neighbor
	# KNN of 5 on magnitude - dev
	neigh3 = KNeighborsClassifier(n_neighbors=neighbor).fit(X_m_train, y_m_train_raveled)
	m_dev_ev = accuracy_score(y_m_dev, neigh3.predict(X_m_dev)) * 100
	print("Accuracy of k nearest neighbors on magnitude - dev: ", m_dev_ev, "%")
	if m_dev_ev > top_m_accuracy:
		top_m_accuracy = m_dev_ev
		top_m_neighbor = neighbor

print('Top Model Accuracy p: ' + str(top_p_accuracy) + ' neighbors: ' + str(top_p_neighbor))
print('Top Model Accuracy m: ' + str(top_m_accuracy) + ' neighbors: ' + str(top_m_neighbor))

end = time.time()
print(end - start)

#-----------------------------------------------------------------------
# GRID SEARCH PERCEPTRON:
start = time.time()

grid_penalty = ['l1', 'l2']

top_p_accuracy = 0
top_p_penalty = []
top_m_accuracy = 0
top_m_penalty = []

for penalty in grid_penalty:	
	# Perceptron on packet_size - dev
	perc1 = Perceptron(penalty=penalty, random_state = 0).fit(X_p_train, y_p_train_raveled)
	p_dev_eva = accuracy_score(y_p_dev, perc1.predict(X_p_dev)) * 100
	print("Accuracy of perceptron on packet size - dev: ", p_dev_eva, "%")
	if p_dev_eva > top_p_accuracy:
		top_p_accuracy = p_dev_eva
		top_p_penalty = penalty
	# Perceptron on magnitude - dev
	perc3 = Perceptron(penalty=penalty, random_state = 0).fit(X_m_train, y_m_train_raveled)
	m_dev_eva = accuracy_score(y_m_dev, perc3.predict(X_m_dev)) * 100
	print("Accuracy of perceptron on magnitude - dev: ", m_dev_eva, "%")
	if m_dev_eva > top_m_accuracy:
		top_m_accuracy = m_dev_eva
		top_m_penalty = penalty

print('Top Model Accuracy p: ' + str(top_p_accuracy) + ' C: ' + str(top_p_penalty))
print('Top Model Accuracy m: ' + str(top_m_accuracy) + ' C: ' + str(top_m_penalty))

end = time.time()
print(end - start)

#-----------------------------------------------------------------------
# GRID SEARCH SVM:
start = time.time()

grid_C = [0.01, 1.0, 10, 100]

top_p_accuracy = 0
top_p_C = 0.0
top_m_accuracy = 0
top_m_C = 0.0

for C in grid_C:
	# SVM on packet_size - dev
	svm1 = SVC(gamma='auto', C=C).fit(X_p_train, y_p_train_raveled)
	p_dev_e = accuracy_score(y_p_dev, svm1.predict(X_p_dev)) * 100
	print("Accuracy of SVM on packet size - dev: ", p_dev_e, "%")
	if p_dev_e > top_p_accuracy:
		top_p_accuracy = p_dev_e
		top_p_C = C
	# SVM on magnitude - dev
	svm3 = SVC(gamma='auto', C=C).fit(X_m_train, y_m_train_raveled)
	m_dev_e = accuracy_score(y_m_dev, svm3.predict(X_m_dev)) * 100
	print("Accuracy of SVM on magnitude - dev: ", m_dev_e, "%")
	if m_dev_e > top_m_accuracy:
		top_m_accuracy = m_dev_e
		top_m_C = C

print('Top Model Accuracy p: ' + str(top_p_accuracy) + ' C: ' + str(top_p_C))
print('Top Model Accuracy m: ' + str(top_m_accuracy) + ' C: ' + str(top_m_C))

end = time.time()
print(end - start)

#-----------------------------------------------------------------------
# N-FOLD CROSS VALIDATION:

# n = 3
# acc_scores = []

# for i in range(n):
# 	X_train_val, X_test_val, y_train_val, y_test_val = split(X_p_train, y_p_train_raveled, i+1, n)
# 	svc = svm.SVC(kernel='linear', C=1.0).fit(X_train_val, y_train_val)
# 	predictions = svc.predict(X_test_val)
# 	cur_accuracy_score = accuracy_score(y_test_val, predictions)
# 	print("Accuracy Score " + str(i) + ": " + str(cur_accuracy_score))
# 	acc_scores.append(cur_accuracy_score)

# print('N-Fold Cross Validation Average Accuracy Score: ' + str(sum(acc_scores)/len(acc_scores)))
end = time.time()
print("Total time: " + str(end - start_start))

