import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle

from pandas import read_csv
#Lets load the data and sample some
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
data = read_csv('./housing.csv', header=None, delimiter=r"\s+", names=column_names)
print(data.head(5))


# Dimension of the data
print(np.shape(data))

# Let's summarize the data to see the distribution of data
print(data.describe())


import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

fig, axs = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))
index = 0
axs = axs.flatten()
for k,v in data.items():
    sns.boxplot(y=k, data=data, ax=axs[index])
    index += 1
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)

#Columns like CRIM, ZN, RM, B seems to have outliers. Let's see the outliers percentage in every column.
for k, v in data.items():
        q1 = v.quantile(0.25)
        q3 = v.quantile(0.75)
        irq = q3 - q1
        v_col = v[(v <= q1 - 1.5 * irq) | (v >= q3 + 1.5 * irq)]
        perc = np.shape(v_col)[0] * 100.0 / np.shape(data)[0]
        print("Column %s outliers = %.2f%%" % (k, perc))

#Let's remove MEDV outliers (MEDV = 50.0) before plotting more distributions
data = data[~(data['MEDV'] >= 50.0)]
print(np.shape(data))

#Let's see how these features plus MEDV distributions looks like
fig, axs = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))
index = 0
axs = axs.flatten()
for k,v in data.items():
    sns.distplot(v, ax=axs[index])
    index += 1
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)


#Now let's plot the pairwise correlation on data.
plt.figure(figsize=(20, 10))
sns.heatmap(data.corr().abs(),  annot=True)

##From correlation matrix, we see TAX and RAD are highly correlated features. 
# The columns LSTAT, INDUS, RM, TAX, NOX, PTRAIO has a correlation score above 0.5 with MEDV which is a good 
# indication of using as predictors. Let's plot these columns against MEDV.

from sklearn import preprocessing
# Let's scale the columns before plotting them against MEDV
min_max_scaler = preprocessing.MinMaxScaler()
column_sels = ['LSTAT', 'INDUS', 'NOX', 'PTRATIO', 'RM', 'TAX', 'DIS', 'AGE']
x = data.loc[:,column_sels]
y = data['MEDV']
x = pd.DataFrame(data=min_max_scaler.fit_transform(x), columns=column_sels)
fig, axs = plt.subplots(ncols=4, nrows=2, figsize=(20, 10))
index = 0
axs = axs.flatten()
for i, k in enumerate(column_sels):
    sns.regplot(y=y, x=x[k], ax=axs[i])
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)


#So with these analsis, we may try predict MEDV with 'LSTAT', 'INDUS', 'NOX', 'PTRATIO', 'RM', 'TAX', 
# 'DIS', 'AGE' features. Let's try to remove the skewness of the data trough log transformation.
y =  np.log1p(y)
for col in x.columns:
    if np.abs(x[col].skew()) > 0.3:
        x[col] = np.log1p(x[col])

#Let's try Linear, Ridge Regression on dataset first.
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import numpy as np

l_regression = linear_model.LinearRegression()
kf = KFold(n_splits=10)
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
scores = cross_val_score(l_regression, x_scaled, y, cv=kf, scoring='neg_mean_squared_error')
print("MSE: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))

scores_map = {}
scores_map['LinearRegression'] = scores
l_ridge = linear_model.Ridge()
scores = cross_val_score(l_ridge, x_scaled, y, cv=kf, scoring='neg_mean_squared_error')
scores_map['Ridge'] = scores
print("MSE: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))

# Lets try polinomial regression with L2 with degree for the best fit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
#for degree in range(2, 6):
#    model = make_pipeline(PolynomialFeatures(degree=degree), linear_model.Ridge())
#    scores = cross_val_score(model, x_scaled, y, cv=kf, scoring='neg_mean_squared_error')
#    print("MSE: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
model = make_pipeline(PolynomialFeatures(degree=3), linear_model.Ridge())
scores = cross_val_score(model, x_scaled, y, cv=kf, scoring='neg_mean_squared_error')
scores_map['PolyRidge'] = scores
print("MSE: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))


#The Liner Regression with and without L2 regularization does not make significant difference is MSE score. 
# However polynomial regression with degree=3 has a better MSE. Let's try some non prametric regression techniques: 
# SVR with kernal rbf, DecisionTreeRegressor, KNeighborsRegressor etc.
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
#grid_sv = GridSearchCV(svr_rbf, cv=kf, param_grid={"C": [1e0, 1e1, 1e2, 1e3], "gamma": np.logspace(-2, 2, 5)}, scoring='neg_mean_squared_error')
#grid_sv.fit(x_scaled, y)
#print("Best classifier :", grid_sv.best_estimator_)
scores = cross_val_score(svr_rbf, x_scaled, y, cv=kf, scoring='neg_mean_squared_error')
scores_map['SVR'] = scores
print("MSE: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))




# Checking the information and statistics of the data
print("data information:")
data.info()

print("data statistical description:")
print(data.describe())

# Check for any missing values in the data
print("Checking for missing values in the data:")
print(data.isnull().sum())

# Handle missing values (if necessary) by dropping rows with NaN values
print("Dropping rows with missing values...")
data = data.dropna()

# Correlation matrix to understand relationships between features
print("Generating the correlation matrix:")
print(data.corr())

# Visualize pairplot of the data
print("Displaying pairplot of the data:")
sns.pairplot(data)

# Scatter plots for specific variables against the target 'Price'
print("Displaying scatter plot between 'CRIM' and 'Price':")
plt.scatter(data['CRIM'], data['Price'])
plt.xlabel("Crime Rate")
plt.ylabel("Price")
plt.show()

print("Displaying scatter plot between 'RM' and 'Price':")
plt.scatter(data['RM'], data['Price'])
plt.xlabel("RM")
plt.ylabel("Price")
plt.show()

# Regression plots
print("Displaying regression plots for 'RM', 'LSTAT', and 'CHAS':")
sns.regplot(x="RM", y="Price", data=data)
sns.regplot(x="LSTAT", y="Price", data=data)
sns.regplot(x="CHAS", y="Price", data=data)

# Split the data into independent features (X) and dependent feature (y - 'Price')
print("Splitting the data into features (X) and target variable (y):")
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split the data into training and testing sets
print("Performing train-test split (70% training, 30% testing)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature scaling using StandardScaler
print("Applying StandardScaler to scale the features...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Saving the scaler model for future use
print("Saving the scaler model...")
pickle.dump(scaler, open('scaling.pkl', 'wb'))

# Create and train the Linear Regression model
print("Training the Linear Regression model...")
regression = LinearRegression()
regression.fit(X_train, y_train)

# Save the trained regression model
print("Saving the trained regression model...")
pickle.dump(regression, open('regmodel.pkl', 'wb'))

# Predict using the trained model
print("Making predictions on the test set...")
reg_pred = regression.predict(X_test)

# Plot predictions against actual values
print("Displaying scatter plot of actual vs predicted prices:")
plt.scatter(y_test, reg_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.show()

# Calculate residuals (differences between actual and predicted values)
print("Calculating residuals...")
residuals = y_test - reg_pred

# Display the residuals distribution
print("Displaying residuals distribution:")
sns.displot(residuals, kind="kde")

# Plot residuals vs predicted values to check for patterns
print("Displaying residuals vs predicted values scatter plot:")
plt.scatter(reg_pred, residuals)
plt.xlabel("Predicted Prices")
plt.ylabel("Residuals")
plt.show()

# Calculate and print error metrics
print("Calculating error metrics...")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, reg_pred)}")
print(f"Mean Squared Error: {mean_squared_error(y_test, reg_pred)}")
print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_test, reg_pred))}")

# Calculate and print R-squared and Adjusted R-squared scores
score = r2_score(y_test, reg_pred)
print(f"R-squared score: {score}")

adjusted_r2 = 1 - (1 - score) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)
print(f"Adjusted R-squared score: {adjusted_r2}")

# Example of predicting new data using the model
print("Predicting new data with the trained model...")
new_data = scaler.transform(data.iloc[0, :-1].values.reshape(1, -1))
predicted_price = regression.predict(new_data)
print(f"Predicted Price for the first data point: {predicted_price}")

# Load the saved model and predict again
print("Loading the saved model and making a prediction for the same new data...")
pickled_model = pickle.load(open('regmodel.pkl', 'rb'))
print(f"Loaded model's prediction: {pickled_model.predict(new_data)}")