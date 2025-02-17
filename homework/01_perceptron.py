import numpy as np

def perceptron(x, y, T, offset=False):
    theta = np.zeros(x.shape[1])
    mistakes = 0
    theta_nought = 0

    for step in range(T):
        print(f'Iteration {step}')
        for i in range(x.shape[0]):
            if y[i] * ((np.dot(x[i], theta)) + theta_nought) <= 0:
                theta = theta + y[i]*x[i]
                print(f'{i}: Prediction incorrect. Theta updated to: {theta}.')
                if offset:
                    theta_nought = theta_nought + y[i]
                    print(f'Theta nought updated to: {theta_nought}')
                mistakes += 1
            else:
                print(i)
    if offset:
        return theta, theta_nought, mistakes
    else:
         return theta, mistakes

x = np.array([[np.cos(np.pi), 0, 0], [0, np.cos(2*np.pi), 0], [0, 0, np.cos(3*np.pi)]])
y = np.array([[1], [1], [1]])
print(perceptron(x, y, 5, offset=False))
