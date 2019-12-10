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

rootdir = Path('data/10s_website1/') 
file_list = [f for f in rootdir.glob('**/*') if f.is_file()]

site_index = {}
for i in range(116):
    site = file_list[i]
    site_index[site] = i


def what_site(l, n, diff):
    dataset = np.zeros(shape = ((116*l), n))
    arr_packet = np.zeros(shape = ((116*l), n))
    arr_magnitude = np.zeros(shape = ((116*l), n))
    # print(arr_magnitude.shape)
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
            index = counter + (a*l)
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
                    arr_packet[index, count_seconds] = 0
                    arr_magnitude[index, count_seconds] = 0
                    gap-=1
                    count_seconds+=1
                    prev_time += diff
                    if count_seconds >= n:
                        break
                if count_seconds >= n:
                    break
                packet = float(info[1])
                arr_packet[index, count_seconds] = packet
                if packet > 0:
                    arr_magnitude[index, count_seconds] = 1
                elif packet < 0:
                    arr_magnitude[index, count_seconds] = -1
                prev_time += diff
                count_seconds +=1
                if count_seconds >= n:
                    break
                count += 1
            counter += 1
        a+=1

    y = np.zeros(shape=((l*116), 1))
    a = 0
    count = -1
    for i in range((l*116)):
        count+=1
        if count/25 < 1:
            y[i, 0] = a
            continue
        elif count/25 == 1:
            count = -1
            a+=1
            
    return arr_packet, arr_magnitude, y


x_packet, x_magnitude, y_vals = what_site(25, 200, 0.1)
# print(y_vals)

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
