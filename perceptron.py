import numpy as np


class Perceptron:
    def __init__(self, learning_rate=0.01, n_iter=50):
        self.learning_rate = learning_rate
        self.n_iter = n_iter

    def fit(self, X, y):
        self.weights_ = np.zeros(X.shape[1] + 1)
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.learning_rate * (target - self.predict(xi))
                self.weights_[1:] += update * xi
                self.weights_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.weights_[1:]) + self.weights_[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)


# 샘플 데이터셋 (XOR 문제를 풀기 위해 간단한 논리게이트 데이터를 사용합니다)
X = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
y = np.array([1, -1, -1, -1])  # AND 게이트의 결과로 사용

# 퍼셉트론 모델 초기화 및 학습
ppn = Perceptron(learning_rate=0.1, n_iter=10)
ppn.fit(X, y)

# 예측 결과 출력
print("Weights:", ppn.weights_)
print("Errors:", ppn.errors_)
print("Predictions:", ppn.predict(X))
