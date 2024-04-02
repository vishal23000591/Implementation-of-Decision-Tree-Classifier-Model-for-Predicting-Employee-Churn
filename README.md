# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Upload and read the dataset.    
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.


## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Vishal S
RegisterNumber: 212223110063
*/

import pandas as pd
data=pd.read_csv("Employee.csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
![decision tree classifier model](sam.png)
![Screenshot 2024-04-02 085821](https://github.com/vishal23000591/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/147139719/00fc9755-3f08-4365-a9f2-2edd3381b889)
![Screenshot 2024-04-02 085910](https://github.com/vishal23000591/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/147139719/a5585fad-b840-4f27-bae8-ca5c192dfc83)
![Screenshot 2024-04-02 090009](https://github.com/vishal23000591/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/147139719/83fee744-26d9-462c-9109-ebabe6ec945d)
![Screenshot 2024-04-02 090046](https://github.com/vishal23000591/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/147139719/9eaba7a9-70fc-4713-9bec-7d4206a17b43)
![Screenshot 2024-04-02 090135](https://github.com/vishal23000591/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/147139719/da565e26-5882-4499-ac54-df50484997a4)
![Screenshot 2024-04-02 090252](https://github.com/vishal23000591/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/147139719/fd5387de-7ec2-479f-a452-8e5f8eea7b23)
![Screenshot 2024-04-02 090326](https://github.com/vishal23000591/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/147139719/5be1355f-be1f-4140-8d98-81ee8dab6ab7)
![Screenshot 2024-04-02 090345](https://github.com/vishal23000591/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/147139719/2d10e614-7b82-4b2b-80d7-c3321731fbad)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
