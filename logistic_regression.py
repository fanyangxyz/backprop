import numpy as np


class LogisticRegression:
    def __init__(self, lr=0.01, num_iters=100):
        self.lr = lr
        self.num_iters = num_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(self.num_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)
            loss = self._loss(y, y_predicted)
            accuracy = self._accuracy(y, y_predicted)
            if epoch % 10 == 0:
                print(f"Epoch: {epoch}, Loss: {loss}, Accuracy: {accuracy}")
            # compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            # update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        return y_predicted

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _loss(self, y_true, y_predicted):
        return -np.mean(y_true * np.log(y_predicted) + (1 - y_true) * np.log(1 - y_predicted))

    def _accuracy(self, y_true, y_predicted):
        y_predicted = np.round(y_predicted)
        return np.mean(y_true == y_predicted)

    def score(self, X, y):
        y_predicted = self.predict(X)
        return self._accuracy(y, y_predicted)


def generate_data(n_samples, n_features):
    X = np.random.randn(n_samples, n_features)
    w = np.random.randn(n_features)
    b = np.random.randn()
    y = np.dot(X, w) + b
    y = (y > 0).astype(int)
    return X, y


def main():
    # Example usage
    lr = LogisticRegression()
    X_train, y_train = generate_data(5000, 10)
    X_test, y_test = generate_data(500, 10)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    accuracy = lr.score(X_test, y_test)
    print(f"Accuracy: {accuracy}")


if __name__ == "__main__":
    main()
