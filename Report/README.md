# MAP detector

## 1. Split Wine.csv into training data and testing data

### (1) Read Wine.csv 
```js
df = pd.read_csv("Wine.csv", sep=",", header=None)
df_values = df.values
``` 

### (2) Count the number of each class
```js
class_num = [0, 0, 0] 
for i in range(1, df_values.shape[0]): 
    class_num[int(df_values[i][0])] += 1
``` 
Label 0 has 175 instances, label 1 has 205 instances, and label 2 has 103 instances. 

### (3) Shuffle the dataset and split it into training data and testing data.
Split testing data 
```js 
testing_data.append(df_values[1:21])
testing_data.append(df_values[class_num[0]+1:class_num[0]+21])
testing_data.append(df_values[class_num[0]+class_num[1]+1:class_num[0]+class_num[1]+21])
testing_data = np.array(testing_data)
testing_data= np.reshape(testing_data, (-1, 14)) #(60,14)
``` 

Split training data
```js
training_data.append(df_values[21:class_num[0]+1])
training_data.append(df_values[class_num[0]+21:class_num[0]+class_num[1]+1])
training_data.append(df_values[class_num[0]+class_num[1]+21:df_values.shape[0]])
training_data = np.array(training_data)
``` 
```js
temp = np.concatenate((training_data[0], training_data[1]))#different sizes, use np.concatenate
training_data = np.concatenate((temp, training_data[2]))
``` 
Shuffle the dataset of 3 classes respectively. Select the top 20 instances of each class as testing data and the others as training data. 

### (4) Save as train.csv and test.csv
Dataframe
```js
train_df = pd.DataFrame(training_data)
test_df = pd.DataFrame(testing_data)
``` 

Save data
```js
train_df.to_csv('train.csv')
test_df.to_csv('test.csv')
``` 
## 2.	Calculate the accuracy rate of the MAP detector.
![截圖 2023-04-25 下午4 30 04](https://user-images.githubusercontent.com/128220508/234219951-ae58f65c-87b6-4f7d-9ef2-ccfff69bd06d.png)  
Thus, the posterior probability P(c│X) is proportional to P(X|c) * P(c). 

### (1) Count the number of each class 
```js
train_class_num = [class_num[0]-20, class_num[1]-20, class_num[2]-20]
``` 
Each class selects 20 instances for testing data, so train_class_num = [155, 185, 83] means that label 0 has 155 instances, label 1 has 185 instances, and label 2 has 83 instances in training data. 
### (2) Split the training data based on labels and features.
13 features 
```js
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
``` 

### (3) Calculate the likelihood distribution of each feature. 
```js
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
``` 
Calculate the mean and standard deviation with ```numpy.mean``` and ```numpy.std```, then created the Gaussian distribution with ```scipy.stats.norm``` 

### (4)	Calculate the prior probabilities
P(c): class proir probability
```js
prior = [train_class_num[0]/sum(train_class_num), train_class_num[1]/sum(train_class_num), train_class_num[2]/sum(train_class_num)]
```
The prior probabilities are (the number of labels) / (the total number of training data). prior = [0.3664, 0.4373, 0.1962]. 

### (5) Shuffle the testing data 
```js
np.random.shuffle(test_data)
``` 
Although I have shuffled the dataset already, I shuffle the label this time. 

### (6) Calculate the posterior probability 
P(c|X) = likelihood * P(c) 
```js
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
```
The likelihood function is to calculate and judge it based on the category with high probability. I calculate the probability of likelihood by integration of the probability density function of the distribution. The integrating interval has to be a very small value, so I set the delta as 1e-6. 

### (7) Accuracy
```js
print('accuracy : ', correct/len(test_data)) 
```
Accuracy is the number of corrections divided by the number of testing data (60). And this is the screenshot of the result.
#### Accuracy: 0.9833333

## 3. Plot the visualized result of testing data
### (1) Delete the labels of the training and testing data
```js
x_train = np.delete(train_data, 0, 1)
y_train = train_data[:,0]
``` 
```js
x_test = np.delete(test_data, 0, 1)
y_test = test_data[:,0]
``` 

### (2) Plot the visualized result of training data and testing data
Training data
```js
PCA_2D = PCA(n_components=2)
markers = ['s','x','o']
wines = ['wine_0','wine_1','wine_2']
labels = [0.,1.,2.]
fig = plt.figure(figsize=(12,12))

x_2D = PCA_2D.fit(x_train).transform(x_train) 

for c, i, target_name, m in zip('rgb', labels, wines, markers):
    plt.scatter(x_2D[y_train==i, 0], x_2D[y_train==i, 1], c=c, label=target_name, marker=m) 
plt.xlabel('PCA-feature-1')
plt.ylabel('PCA-feature-2')
plt.legend(wines ,loc='upper right')
plt.savefig('PCA_train.png')
plt.clf()
``` 

Testing data 
```js
x_2D = PCA_2D.fit(x_test).transform(x_test) 

for c, i, target_name, m in zip('rgb', labels, wines, markers):
    plt.scatter(x_2D[y_test==i, 0], x_2D[y_test==i, 1], c=c, label=target_name, marker=m) 
plt.xlabel('PCA-feature-1')
plt.ylabel('PCA-feature-2')
plt.legend(wines ,loc='upper right')
plt.savefig('PCA_test.png')
plt.clf()

plt.show()
```
![截圖 2023-04-25 下午5 01 06](https://user-images.githubusercontent.com/128220508/234227834-08dd6d00-d870-4f53-948c-9f2169ab31d5.png)
![截圖 2023-04-25 下午5 02 00](https://user-images.githubusercontent.com/128220508/234228052-4399f13a-ce1a-4155-9138-c22f0909760d.png)


I use the PCA function to reduce the dimensions. It is an effective way to reduce calculations in high-dimensional data. PCA is to make the principal features of the singular vector the selection standard. The left side of the figure above is the PCA of the training data and the right side is the PCA of the testing data. It can see that our three classes distinguish 3 blocks through dimension reduction. Red is class 0, green is class 1, blue is class 2.

