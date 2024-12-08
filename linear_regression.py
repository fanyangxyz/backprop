import numpy as np


class LinearRegression:
    def __init__(self, lr=0.01, num_iters=1000):
        self.lr = lr
        self.num_iters = num_iters
        self.weights = None

    def fit_vectorized(self, X, y):
        n_samples, n_features = X.shape
        x = np.hstack((np.ones((n_samples, 1)), X))
        self.weights = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)

    def fit_gradient_descent(self, X, y):
        n_samples, n_features = X.shape
        x = np.hstack((np.ones((n_samples, 1)), X))
        self.weights = np.zeros(n_features + 1)

        for _ in range(self.num_iters):
            y_predicted = x.dot(self.weights)
            dw = (1 / n_samples) * x.T.dot(y_predicted - y)
            self.weights -= self.lr * dw

    def fit(self, X, y, use_vectorized=False):
        if use_vectorized:
            self.fit_vectorized(X, y)
            return
        self.fit_gradient_descent(X, y)

    def predict(self, X):
        x = np.hstack((np.ones((X.shape[0], 1)), X))
        return x.dot(self.weights)

    def loss(self, X, y):
        y_pred = self.predict(X)
        return np.linalg.norm(y_pred - y)


def generate_data(n_samples, n_features):
    X = np.random.randn(n_samples, n_features)
    w = np.random.randn(n_features)
    b = np.random.randn()
    y = np.dot(X, w) + b
    return X, y


def main():
    X, y = generate_data(100, 2)
    model = LinearRegression()
    # model.fit(X, y, use_vectorized=True)
    model.fit(X, y, use_vectorized=False)
    y_pred = model.predict(X)
    loss = model.loss(X, y)
    print(f"Loss: {loss:.4f}")


if __name__ == "__main__":
    main()
