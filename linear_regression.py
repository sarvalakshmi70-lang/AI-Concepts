from sklearn.linear_model import LinearRegression

X = [[1], [2], [3], [4], [5]]
y = [40, 45, 50, 60, 65]

model = LinearRegression()

model.fit(X, y)

prediction = model.predict([[6]])

print("Predicted marks:", prediction[0])