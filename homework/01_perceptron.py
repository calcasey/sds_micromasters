import numpy as np

def perceptron(x, y, T):
    theta = np.zeros(x.shape[1])
    theta_nought = 0
    mistakes = 0

    for step in range(T):
        print(f'Iteration {step}')
        for i in range(x.shape[0]):
            if y[i] * ((np.dot(x[i], theta)) + theta_nought) <= 0:
                theta = theta + y[i]*x[i]
                theta_nought = theta_nought + y[i]
                print(f'{i}: Prediction incorrect. Theta updated to: {theta}. Theta nought updated to: {theta_nought}')
                mistakes += 1
            else:
                print(i)

    return theta, theta_nought, mistakes

x = np.array([[-4, 2], [-2, 1], [-1, -1], [2, 2], [1, -2]])
y = np.array([[1], [1], [-1], [-1], [-1]])
print(perceptron(x, y, 10))
