# MAP detector

## 1. Split Wine.csv into training data and testing data

### (1) Read Wine.csv 
```js
df = pd.read_csv("Wine.csv", sep=",", header=None)
df_values = df.values
``` 

### (2) Count the number of each class
