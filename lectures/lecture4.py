import numpy as np

def hinge_loss(x_list, y_list, theta=np.array([0., 1., 2.])):
    total_loss = 0.
    for i, x in enumerate(x_list):
        z = y_list[i] - (np.dot(theta, x))
        if z >= 1.:
            loss = 0.
        else:
            loss = 1. - z
        total_loss += loss
    return total_loss / len(x_list)

def squared_error(x_list, y_list, theta=np.array([0., 1., 2.])):
    total_loss = 0.
    for i, x in enumerate(x_list):
        z = (y_list[i] - np.dot(theta, x))
        loss = (z**2)/2
        total_loss += loss
    return total_loss / len(x_list)

x1 = np.array([1., 0., 1.])
y1 = 2.

x2 = np.array([1., 1., 1.])
y2 = 2.7

x3 = np.array([1., 1., -1.])
y3 = -0.7

x4 = np.array([-1., 1., 1.])
y4 = 2

sample_list = [x1, x2, x3, x4]
label_list = [y1, y2, y3, y4]

print(hinge_loss(sample_list, label_list))
print(squared_error(sample_list, label_list))