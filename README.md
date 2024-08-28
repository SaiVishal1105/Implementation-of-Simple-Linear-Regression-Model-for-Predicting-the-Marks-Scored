# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas. 

## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: SAI VISHAL D
RegisterNumber: 212223230180 

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
```
```
dataset=pd.read_csv('/content/student_scores.csv')
print(dataset.head())
print(dataset.tail())
```
![image](https://github.com/user-attachments/assets/def9f64a-c259-493a-bc7e-22cb07521192)
```
dataset.info()
```
![image](https://github.com/user-attachments/assets/3c8f2e56-0ccd-4e28-a91d-cda3b14d983d)
```
X=dataset.iloc[:,:-1].values
print(X)
Y=dataset.iloc[:,-1].values
print(Y)
```
![image](https://github.com/user-attachments/assets/c486c0d6-731a-4c11-88fe-7ea732bb40ff)
```
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)
```
```
X_train.shape
```
![image](https://github.com/user-attachments/assets/22f634a0-6063-47ec-a603-cd1896b7fcf9)
```
X_test.shape
```
![image](https://github.com/user-attachments/assets/0e51d1a1-7b9e-4c1d-b619-e06d3ea6a789)
```
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train, Y_train)
```
![image](https://github.com/user-attachments/assets/c79a51ec-a405-40ad-a06f-0e303297513d)
```
Y_pred=reg.predict(X_test)
print(Y_pred)
print(Y_test)
```
![image](https://github.com/user-attachments/assets/db1e646d-5db0-4e81-8f38-f610d6e8c443)
```
plt.scatter(X_train,Y_train,color="green")
plt.plot(X_train,reg.predict(X_train),color="red")
plt.title('Training set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
```
plt.scatter(X_test,Y_test,color="blue")
plt.plot(X_test,reg.predict(X_test),color="silver")
plt.title('Test set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
## Output:
![image](https://github.com/user-attachments/assets/4be700cb-838e-49d0-b82f-69dfc1fe6685)
![image](https://github.com/user-attachments/assets/0569081f-0f4a-4126-8c5d-0861a15fa508)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
