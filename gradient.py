import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

w1 = np.random.randn(2, 2)
w2 = np.random.randn(2, 1)

b1 = np.zeros((1, 2))
b2 = np.zeros((1, 1))

learning_rate = 0.1

for epoch in range(10000):

    z1 = np.dot(X, w1) + b1
    a1 = 1 / (1 + np.exp(-z1))

    z2 = np.dot(a1, w2) + b2
    prediction = 1 / (1 + np.exp(-z2))

    loss = np.mean((y - prediction) ** 2)

    error = prediction - y

    gradient_w2 = np.dot(a1.T, error)
    gradient_b2 = np.sum(error, axis=0, keepdims=True)

    error_hidden = np.dot(error, w2.T)
    gradient_hidden = error_hidden * a1 * (1 - a1)

    gradient_w1 = np.dot(X.T, gradient_hidden)
    gradient_b1 = np.sum(gradient_hidden, axis=0, keepdims=True)

    w2 = w2 - learning_rate * gradient_w2
    b2 = b2 - learning_rate * gradient_b2

    w1 = w1 - learning_rate * gradient_w1
    b1 = b1 - learning_rate * gradient_b1

print("Prediction:")
print(prediction)

print("Loss:")
print(loss)