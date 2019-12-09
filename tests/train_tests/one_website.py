from pathlib import Path
import re
import math
from random import randrange
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC

# Paramters for definition are as follows
#   input = string to get to file directory (i.e. "data/result/facebook.com")
#   num_rows = number of rows you want to look through, starting bias
#   num_columns = number of time differences you want to check (number of columns)
#   dif = the time difference needed to record that packet
def not_website(input, num_rows, num_columns, diff):
    file = open(input, "r")
    given_website_packets = np.zeros(shape = (num_rows, num_columns))
    given_website_magnitudes = np.zeros(shape = (num_rows, num_columns))
    row_index = 0
    # go through each row in file or until num_rows threshold
    for row in file:
        split = re.split(" ", str(row))
        if row_index >= num_rows or len(split) < 2:
            break
        origin = re.split(":", str(split[2]))
        start_time = float(origin[0])
        prev_time = start_time
        count = 2
        column_index = 0
        # get each item in the current row and check the time difference to determine if adding to the array
        while column_index <= num_columns:
            time_and_packet = re.split(":", str(split[count]))
            if len(time_and_packet) < 2:
                break
            time = float(time_and_packet[0])
            if (len(split) <= count+1) or ((time - start_time) >= 10):
                break
            if time - prev_time < diff:
                count += 1
                continue
            gap = (math.floor((time-prev_time) / diff)) - 1
            while gap > 0:
                given_website_packets[row_index, column_index] = 0
                given_website_magnitudes[row_index, column_index] = 0
                gap -= 1
                column_index += 1
                prev_time += diff  
                if column_index >= num_columns:
                    break
            if column_index >= num_columns:
                break
            packet = float(time_and_packet[1])
            given_website_packets[row_index, column_index] = packet
            if packet > 0:
                given_website_magnitudes[row_index, column_index] = 1
            elif packet < 0:
                given_website_magnitudes[row_index, column_index] = -1
            prev_time += diff   
            count += 1
            column_index +=1
        row_index += 1

    all_other_packets = np.zeros(shape = (num_rows, num_columns))
    all_other_magnitudes = np.zeros(shape = (num_rows, num_columns))
    
    rootdir = Path('data/result/') 
    file_list = [f for f in rootdir.glob('**/*') if f.is_file()]

    # remove current website from possible files
    for i in range(67):
        if file_list[i] == website:
            file_list.remove(i)
    
    # go through and get a random row in a random file until num_rows threshold
    for i in range(num_rows):
        random_site = randrange(66)
        new_file = file_list[random_site - 1]
        view_file = open(new_file, "r")
        new_file_length = 0
        for row in view_file.readlines():
            new_file_length += 1
        random_row = randrange(new_file_length)
        view_file = open(new_file, "r")
        line = ""
        new_file_row_count = 0
        for row in view_file.readlines():
            if new_file_row_count == random_row:
                line = row
                break
            new_file_row_count += 1
        split = re.split(" ", str(line))
        if len(split) < 2:
            break
        origin = re.split(":", str(split[2]))
        start_time = float(origin[0])
        prev_time = start_time
        count = 2
        column_index = 0
        # get each item in the current row and check the time difference to determine if adding to the array
        while column_index <= num_columns:
            time_and_packet = re.split(":", str(split[count]))
            if len(time_and_packet) < 2:
                break
            time = float(time_and_packet[0])
            if (len(split) <= count+1) or ((time - start_time) >= 10):  
                break
            if time - prev_time < diff:
                count += 1
                continue
            gap = (math.floor((time-prev_time) / diff)) - 1
            while gap > 0:
                all_other_packets[i, column_index] = 0
                all_other_magnitudes[i, column_index] = 0
                gap -= 1
                column_index += 1
                prev_time += diff  
                if column_index >= num_columns:
                    break
            if column_index >= num_columns:
                break
            packet = float(time_and_packet[1])
            all_other_packets[i, column_index] = packet
            if packet > 0:
                all_other_magnitudes[i, column_index] = 1
            elif packet < 0:
                all_other_magnitudes[i, column_index] = -1
            prev_time += diff  
            count += 1
            column_index +=1

    y = np.concatenate((np.zeros(shape=((num_rows), 1)), np.ones(shape=((num_rows), 1))))

    x_p = np.concatenate((given_website_packets, all_other_packets))
    x_m = np.concatenate((given_website_magnitudes, all_other_magnitudes))

    return x_p, x_m, y


website = "data/result/facebook.com"
file = open(website, "r")
website_length = 0
for row in file:
    website_length += 1
# print(website_length) -- 1011

x_packet, x_magnitude, y_vals = not_website(website, int(website_length/2), 250, 0.075)

x_new = np.add(x_packet, x_magnitude)

# Standardize the features
sc = StandardScaler()
sc.fit(x_packet)

sc1 = StandardScaler()
sc1.fit(x_magnitude)

X_p_train, X_p_test, y_p_train, y_p_test = train_test_split(x_packet, y_vals, test_size=0.2, random_state=0)
X_m_train, X_m_test, y_m_train, y_m_test = train_test_split(x_magnitude, y_vals, test_size=0.2, random_state=0)

X_p_dev, X_p_test, y_p_dev, y_p_test = train_test_split(X_p_test, y_p_test, test_size=0.5, random_state=0)
X_m_dev, X_m_test, y_m_dev, y_m_test = train_test_split(X_m_test, y_m_test, test_size=0.5, random_state=0)

y_m_train = y_m_train.ravel()
y_p_train = y_p_train.ravel()

#-----------------------------------------------------------------------
#Logistic Regression on packet size - dev
clf = LogisticRegression(random_state=0, solver='liblinear').fit(X_p_train, y_p_train)
p_dev_eval = accuracy_score(y_p_dev, clf.predict(X_p_dev)) * 100
print("Accuracy of log reg on packet size - dev: ", p_dev_eval, "%")

#Logistic Regression on packet size - test
clf1 = LogisticRegression(random_state=0, solver='liblinear').fit(X_p_train, y_p_train)
p_test_eval = accuracy_score(y_p_test, clf1.predict(X_p_test)) * 100
print("Accuracy of log reg on packet size - test: ", p_test_eval, "%")

#Logistic Regression on magnitude - dev
clf2 = LogisticRegression(random_state=0, solver='liblinear').fit(X_m_train, y_m_train)
m_dev_eval = accuracy_score(y_m_dev, clf2.predict(X_m_dev)) * 100
print("Accuracy of log reg on magnitude - dev: ", m_dev_eval, "%")

#Logistic Regression on magnitude - test
clf3 = LogisticRegression(random_state=0, solver='liblinear').fit(X_m_train, y_m_train)
m_dev_test = accuracy_score(y_m_test, clf3.predict(X_m_test)) * 100
print("Accuracy of log reg on magnitude - test: ", m_dev_test, "%")

print("")

#-----------------------------------------------------------------------
# KNN of 5 on packet_size - dev
neigh1 = KNeighborsClassifier(n_neighbors=5).fit(X_p_train, y_p_train)
p_dev_ev = accuracy_score(y_p_dev, neigh1.predict(X_p_dev)) * 100
print("Accuracy of k nearest neighbors on packet size - dev: ", p_dev_ev, "%")

# KNN of 5 on packet_size - test
neigh2 = KNeighborsClassifier(n_neighbors=5).fit(X_p_train, y_p_train)
p_test_ev = accuracy_score(y_p_test, neigh2.predict(X_p_test)) * 100
print("Accuracy of k nearest neighbors on packet size - test: ", p_test_ev, "%")

# KNN of 5 on magnitude - dev
neigh3 = KNeighborsClassifier(n_neighbors=5).fit(X_m_train, y_m_train)
m_dev_ev = accuracy_score(y_m_dev, neigh3.predict(X_m_dev)) * 100
print("Accuracy of k nearest neighbors on magnitude - dev: ", m_dev_ev, "%")

# KNN of 5 on magnitude - test
neigh4 = KNeighborsClassifier(n_neighbors=5).fit(X_m_train, y_m_train)
m_test_ev = accuracy_score(y_m_test, neigh4.predict(X_m_test)) * 100
print("Accuracy of k nearest neighbors on magnitude - test: ", m_test_ev, "%")

print("")

#-----------------------------------------------------------------------
# Perceptron on packet_size - dev
perc1 = Perceptron(random_state = 0).fit(X_p_train, y_p_train)
p_dev_eva = accuracy_score(y_p_dev, perc1.predict(X_p_dev)) * 100
print("Accuracy of perceptron on packet size - dev: ", p_dev_eva, "%")

# Perceptron on packet_size - test
perc2 = Perceptron(random_state = 0).fit(X_p_train, y_p_train)
p_test_eva = accuracy_score(y_p_test, perc2.predict(X_p_test)) * 100
print("Accuracy of perceptron on packet size - test: ", p_test_eva, "%")

# Perceptron on magnitude - dev
perc3 = Perceptron(random_state = 0).fit(X_m_train, y_m_train)
m_dev_eva = accuracy_score(y_m_dev, perc3.predict(X_m_dev)) * 100
print("Accuracy of perceptron on magnitude - dev: ", m_dev_eva, "%")

# Perceptron on magnitude - test
perc4 = Perceptron(random_state = 0).fit(X_m_train, y_m_train)
m_test_eva = accuracy_score(y_m_test, perc4.predict(X_m_test)) * 100
print("Accuracy of perceptron on magnitude - test: ", m_test_eva, "%")

print("")

#-----------------------------------------------------------------------
# SVM on packet_size - dev
svm1 = SVC(gamma='auto').fit(X_p_train, y_p_train)
p_dev_e = accuracy_score(y_p_dev, svm1.predict(X_p_dev)) * 100
print("Accuracy of SVM on packet size - dev: ", p_dev_e, "%")

# SVM on packet_size - test
svm2 = SVC(gamma='auto').fit(X_p_train, y_p_train)
p_test_e = accuracy_score(y_p_test, svm2.predict(X_p_test)) * 100
print("Accuracy of SVM on packet size - test: ", p_test_e, "%")

# SVM on magnitude - dev
svm3 = SVC(gamma='auto').fit(X_m_train, y_m_train)
m_dev_e = accuracy_score(y_m_dev, svm3.predict(X_m_dev)) * 100
print("Accuracy of SVM on magnitude - dev: ", m_dev_e, "%")

# SVM on magnitude - test
svm4 = SVC(gamma='auto').fit(X_m_train, y_m_train)
m_test_e = accuracy_score(y_m_test, svm4.predict(X_m_test)) * 100
print("Accuracy of SVM on magnitude - test: ", m_test_e, "%")

