# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


#print(x)
#print(y)

#Training the Simple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#print(x_test)
#print(y_test)

#predicting the test set

y_pred = regressor.predict(x_test)


#visualising the training set results

plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train),color = 'blue' )
plt.title('Salary vs Experiance (Training set)')
plt.xlabel('Years of Experiance')
plt.ylabel('Salary')
plt.show()


#visualising the test set results

plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train),color = 'blue' )
plt.title('Salary vs Experiance (Test set)')
plt.xlabel('Years of Experiance')
plt.ylabel('Salary')
plt.show()



#implementing
print(regressor.predict([[13]]))