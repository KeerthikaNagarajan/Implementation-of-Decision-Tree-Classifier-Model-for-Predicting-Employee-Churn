# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import required packages and read the data file.
2. Use LabelEncoder to convert categorical data into numerical data.
3. Split data into training set and testing set,
4. Predict Y values.
5. Calculate accuracy of the model.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Keerthika N
RegisterNumber: 212221230049
*/
```
```
import pandas as pd
data=pd.read_csv("Employee.csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing  import LabelEncoder
le=LabelEncoder()

data['salary']=le.fit_transform(data['salary'])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
data.head()

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
* data.head()

![1](https://github.com/KeerthikaNagarajan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/93427089/8c4995cd-769b-4a60-8679-5b30b9c5e184)

* data.info()

![2](https://github.com/KeerthikaNagarajan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/93427089/13dcf775-686c-4b02-835e-9cc40745fef8)


* isnull().sum()

![3](https://github.com/KeerthikaNagarajan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/93427089/49616980-73f5-4ee1-a048-567cd4224c22)


* Data value counts

![4](https://github.com/KeerthikaNagarajan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/93427089/780ca202-3198-47c3-aa45-0930dacd9c6f)


* data.head() for salary

![5](https://github.com/KeerthikaNagarajan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/93427089/e2f5c8b8-ea53-4abd-8ea2-2f950a86b367)


* x.head()

![6](https://github.com/KeerthikaNagarajan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/93427089/75874387-a294-4047-8af5-0808408cdb66)


* Accuracy value

![7](https://github.com/KeerthikaNagarajan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/93427089/8fde8d76-320f-49d8-a146-35911b999f65)


* Data precision

![8](https://github.com/KeerthikaNagarajan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/93427089/e0956b4b-31d9-4fcd-b5ca-c18b5b61fafd)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
