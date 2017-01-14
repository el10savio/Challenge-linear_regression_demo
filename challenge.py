import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np

#read data
dataframe = pd.read_csv('challenge_dataset.txt', sep=",", header = None)
dataframe.columns = ["Brain","Body"]
x_values = dataframe[['Brain']]
y_values = dataframe[['Body']]

#train model on data
body_reg = linear_model.LinearRegression()
body_reg.fit(x_values, y_values)

#visualize results
plt.scatter(x_values, y_values)
plt.plot(x_values, body_reg.predict(x_values))
plt.show()

#predict values
x_values_test=np.array([6.1101,5.5277,8.5186])
y_values_test=np.array([17.592,9.1302,13.662])
print("Mean Squared Error: %.2f"
% np.mean((body_reg.predict(x_values_test.reshape(-1,1)) - y_values_test.reshape(-1,1)) ** 2))
for i in range(0,len(x_values_test)):
	print("Error of test #%d: %.2f"
	% (i, (body_reg.predict(x_values_test[i] - y_values_test[i]))))
