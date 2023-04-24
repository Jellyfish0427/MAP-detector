import numpy as np
import pandas as pd
import csv
import random
from scipy import stats
from scipy import integrate
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

training_data = []
testing_data = []

feature = []
prior = []
delta = 1e-6

## 1.split ##

# read csv
df = pd.read_csv("Wine.csv", sep=",", header=None)
df_values = df.values
#value shape : (484, 14)

# count the number of 3 classes
class_num = [0, 0, 0] 
for i in range(1, df_values.shape[0]): 
    class_num[int(df_values[i][0])] += 1
#class_num : [175, 205, 103]

# shuffle + split
random.shuffle(df_values[1:class_num[0]+1]) #shuffle class 0
random.shuffle(df_values[class_num[0]+1:class_num[0]+class_num[1]+1]) #shuffle class 1
random.shuffle(df_values[class_num[0]+class_num[1]+1:df_values.shape[0]]) #shuffle class 2

# split testing data
testing_data.append(df_values[1:21])
testing_data.append(df_values[class_num[0]+1:class_num[0]+21])
testing_data.append(df_values[class_num[0]+class_num[1]+1:class_num[0]+class_num[1]+21])
testing_data = np.array(testing_data)
testing_data= np.reshape(testing_data, (-1, 14)) #(60,14)

# split training data
training_data.append(df_values[21:class_num[0]+1])
training_data.append(df_values[class_num[0]+21:class_num[0]+class_num[1]+1])
training_data.append(df_values[class_num[0]+class_num[1]+21:df_values.shape[0]])
training_data = np.array(training_data)

temp = np.concatenate((training_data[0], training_data[1]))#different sizes, use np.concatenate
training_data = np.concatenate((temp, training_data[2]))

# dataframe
train_df = pd.DataFrame(training_data)
test_df = pd.DataFrame(testing_data)

# save data
train_df.to_csv('train.csv')
test_df.to_csv('test.csv')

## 2.MAP ##

train_class_num = [class_num[0]-20, class_num[1]-20, class_num[2]-20]

# 13 features
feature_0 = np.zeros(shape=[13, train_class_num[0]]) #(13, 155)
feature_1 = np.zeros(shape=[13, train_class_num[1]]) #(13, 185)
feature_2 = np.zeros(shape=[13, train_class_num[2]]) #(13, 83)

feature.append(feature_0)
feature.append(feature_1)
feature.append(feature_2)

for idx in range(0, train_class_num[0]):
    for f_idx in range(13):
        feature_0[f_idx][idx] = train_df.values[idx][f_idx+1] #+1:skip class

for idx in range(0, train_class_num[1]):
    for f_idx in range(13):
        feature_1[f_idx][idx] = train_df.values[idx+train_class_num[0]][f_idx+1]
        
for idx in range(0, train_class_num[2]):
    for f_idx in range(13):
        feature_2[f_idx][idx] = train_df.values[idx+train_class_num[0]+train_class_num[1]][f_idx+1]       

feature.append(feature_0)
feature.append(feature_1)
feature.append(feature_2)

distribution = []
# calculate likelihood distribution
for label_idx in range(3): #label
    temp = []
    for f_idx in range(13): #feature
        mean = np.mean(feature[label_idx][f_idx])
        std = np.std(feature[label_idx][f_idx])
        #print(mean,std)
        temp.append(stats.norm(mean, std))
    distribution.append(temp)
        
prior = [train_class_num[0]/sum(train_class_num), train_class_num[1]/sum(train_class_num), train_class_num[2]/sum(train_class_num)] #prior prob.
#P(c): class proir probability

train_data = train_df.values.astype(float)
test_data = test_df.values.astype(float)
np.random.shuffle(test_data)

#P(c|X) = likelihood * P(c)
correct = 0
for data_idx in range(60):
    posts = [1., 1., 1.] #P(c|X): posterior probability
    for label_idx in range(3): #label
        post = 1.* prior[label_idx] 
        for f_idx in range(13): #feature            
            likelihood = integrate.quad(distribution[label_idx][f_idx].pdf, test_data[data_idx][f_idx+1], test_data[data_idx][f_idx+1]+delta)
            post = post * likelihood[0]
        posts[label_idx] = post
    #print(posts)
    label = np.argmax(posts)
    #print(label,int(test_data[data_idx][0]))
    
    if label == int(test_data[data_idx][0]):
        correct += 1

print('accuracy : ', correct/len(test_data)) 

## 3.PCA ##

# training data
PCA_2D = PCA(n_components=2)
markers = ['s','x','o']
wines = ['wine_0','wine_1','wine_2']
labels = [0.,1.,2.]
fig = plt.figure(figsize=(12,12))

x_train = np.delete(train_data, 0, 1)
y_train = train_data[:,0]
x_2D = PCA_2D.fit(x_train).transform(x_train) 

for c, i, target_name, m in zip('rgb', labels, wines, markers):
    plt.scatter(x_2D[y_train==i, 0], x_2D[y_train==i, 1], c=c, label=target_name, marker=m) 
plt.xlabel('PCA-feature-1')
plt.ylabel('PCA-feature-2')
plt.legend(wines ,loc='upper right')
plt.savefig('PCA_train.png')
plt.clf()

# testing data
x_test = np.delete(test_data, 0, 1)
y_test = test_data[:,0]
x_2D = PCA_2D.fit(x_test).transform(x_test) 

for c, i, target_name, m in zip('rgb', labels, wines, markers):
    plt.scatter(x_2D[y_test==i, 0], x_2D[y_test==i, 1], c=c, label=target_name, marker=m) 
plt.xlabel('PCA-feature-1')
plt.ylabel('PCA-feature-2')
plt.legend(wines ,loc='upper right')
plt.savefig('PCA_test.png')
plt.clf()

plt.show()
