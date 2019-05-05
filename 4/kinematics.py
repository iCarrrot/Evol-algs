import numpy as np
import math
import matplotlib.pyplot as plt


def plot_point(point, angle, length, start_angle):
    '''
    point - Tuple (x, y)
    angle - Angle you want your end point at in degrees.
    length - Length of the line you want to plot.

    Will plot the line on a 10 x 10 plot.
    '''

    # unpack the first point
    x, y = point

    # find the end point
    endangle = ((start_angle + 180 - angle)  + 180) % 360 - 180
    endy = length * math.sin(math.radians(endangle)) + y
    endx = length * math.cos(math.radians(endangle)) + x
    plt.plot([x, endx], [y, endy])

    return endx, endy, endangle


def show_points(lengths, angles, start_point=(0, 0), target=None):
    x, y = start_point
    angle = 0

    for i in range(len(lengths)):
        x, y, angle = plot_point((x, y), angles[i], lengths[i], angle)
    plt.scatter([x], [y], c='r')
    if target is not None:
        plt.scatter([target[0]], [target[1]], c='g')

    plt.axis('equal')

    plt.show()


def objective_function(X, lengths, target, ranges=None, penalty=None):
    if penalty is None:
        penalty = np.sum(lengths) * 10000000

    angles = np.zeros(X.shape)
    # angles[:,-1] = 270
    points_x = np.zeros(X.shape)
    points_y = np.zeros(X.shape)
    # scores = np.zeros(X.shape[0])
    pen = 0
    for i in range(X.shape[1]):
        angles[:,i] = ((X[:, i] + 180 - angles[:, i-1])  + 180) % 360 - 180

        points_x[:, i] = (lengths[i] * np.cos(np.radians(angles[:,i]))
                          + points_x[:, i-1])
        points_y[:, i] = (lengths[i] * np.sin(np.radians(angles[:,i]))
                          + points_y[:, i-1])
    if ranges is not None:
        pen = np.sum(np.logical_or(
            X < ranges[0, :],
            X > ranges[1, :]) * (- penalty), axis=1)
    # print(pen)
    # print(X < ranges[0, :])
    # print(X > ranges[1, :])
    # print(pen.shape)
    # print(points_x)
    # print(points_y)
    # print(angles)
    scores = (pen
              - np.sqrt((target[0] - points_x[:, -1])**2 + (target[1] - points_y[:, -1])**2))

    return scores, points_x, points_y