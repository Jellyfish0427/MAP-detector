## Introduction
Implement the MAP probability of the classifier for 60 instances of 3 types of wines and 13 different features.

## Information of each feature
1. Alcohol 
2. Malic acid 
3. Ash 
4. Alcalinity of ash 
5. Magnesium 
6. Total phenols 
7. Flavanoids 
8. Non Flavonoid phenols 
9. Proanthocyanins 
10. Color intensity 
11. Hue
12. OD280/OD315 of diluted wines 
13. Proline 
All the features are independent and the distribution of them is Gaussian distribution. 

## Implement 
1. Split Wine.csv into training data and testing data. Randomly select 20 instances of each category as testing data. Save the training dataset as train.csv and testing dataset as test.csv. (423 instances for training and 60 instances for testing.) 
2. To evaluate the posterior probabilities, I need to learn likelihood functions and prior distribution from the training dataset. Then, I calculate the accuracy rate of the MAP detector by comparing to the label of each instance in the test data. 
3. Plot the visualized result of testing data.
