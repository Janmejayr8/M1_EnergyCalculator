import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

hours_slept=[4,5,5.5,6,6.5,7,7.5,8,8.5,9]
energy_scores = [35,42,48,52,58,65,70,78,84,90]

x = [[hours] for hours in hours_slept]
y = energy_scores

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 42)
model = LinearRegression()
model.fit(x_train,y_train)

predictions = model.predict(x_test)
mse = mean_squared_error(y_test, predictions)
print("Predicted energy scores: ", predictions.round(2))
print("Actual energy scores: ",y_test)
print(f"mean squared error: {mse:.2f}")
plt.scatter(x,y,color="steelblue", label = 'Real data')
plt.plot(x,model.predict(x), color = "tomato", label = "Model Prediction")
plt.xlabel("Hours slept")
plt.ylabel("Energy Score")
plt.title("Hours Slept vs. Energy Score")
plt.legend()
plt.show()