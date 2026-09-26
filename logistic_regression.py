from sklearn.linear_model import LogisticRegression

X = [[1], [2], [3], [4], [5]]
y = [0, 0, 0, 1, 1]

model = LogisticRegression()

model.fit(X, y)

prediction = model.predict([[6]])

print("Prediction:", prediction[0])

if prediction[0] == 1:
    print("Result: Pass")
else:
    print("Result: Fail")