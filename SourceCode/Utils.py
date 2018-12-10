import matplotlib
# matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib import animation

import numpy as np
import time

L = 28
sequences_dir = "Dataset/sequences/"

def get_discrete_line(start, end):
    """Bresenham's Line Algorithm
    Produces a list of tuples from start and end

    >>> points1 = get_line((0, 0), (3, 4))
    >>> points2 = get_line((3, 4), (0, 0))
    >>> assert(set(points1) == set(points2))
    >>> print points1
    [(0, 0), (1, 1), (1, 2), (2, 3), (3, 4)]
    >>> print points2
    [(3, 4), (2, 3), (1, 2), (1, 1), (0, 0)]
    """
    # Setup initial conditions
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1

    # Determine how steep the line is
    is_steep = abs(dy) > abs(dx)

    # Rotate line
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    # Swap start and end points if necessary and store swap state
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True

    # Recalculate differentials
    dx = x2 - x1
    dy = y2 - y1

    # Calculate error
    error = int(dx / 2.0)
    ystep = 1 if y1 < y2 else -1

    # Iterate over bounding box generating points between start and end
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = (y, x) if is_steep else (x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += ystep
            error += dx

    # Reverse the list if the coordinates were swapped
    if swapped:
        points.reverse()
    return points

def show_point_sequence(sequence_points):
    img = np.zeros([L, L])
    fig = plt.figure()
    plt_object = plt.imshow(img, cmap='gist_gray_r', vmin=0, vmax=1)

    def init():
        plt_object.set_data(np.zeros([L, L]))
        return plt_object

    def animate(i):
        counter = 0
        while counter < i:
            x, y = sequence_points[counter][0], sequence_points[counter][1]
            if x >= 0:
                counter += 1
            if counter == len(sequence_points) - 1:
                return plt_object
        x, y = sequence_points[counter][0], sequence_points[counter][1]
        x, y = x - 1, y - 1
        img[y, x] = 1
        plt_object.set_data(img)
        return plt_object

    _ = animation.FuncAnimation(fig, animate, init_func=init, frames=len(sequence_points),
                                interval=50)
    plt.show()


def show_point_file(id=15, label='train'):
    file_name = sequences_dir + "{}img-{}-points.txt".format(label, id)
    sequence_points = open(file_name, 'r').readlines()[1:]
    sequence_points = [tuple(map(int, s[:-1].split(","))) for s in sequence_points]
    show_point_sequence(sequence_points)


def show_positions_mus(trajectory, mu):
    x = trajectory[:, 0]
    y = trajectory[:, 1]
    mu_x = mu[:, 0]
    mu_y = mu[:, 1]
    n = len(x)
    fig, ax = plt.subplots()
    line1, = ax.plot(x[:1], y[:1], 'r')
    line2, = ax.plot(mu_x[:1], mu_y[:1], 'b')

    def animate1(i):
        line1.set_xdata(x[:i + 1])
        line1.set_ydata(y[:i + 1])
        return line1,

    def animate2(i):
        line2.set_xdata(mu_x[:i + 1])
        line2.set_ydata(mu_y[:i + 1])
        return line1,

    plt.legend(["Positions", "Mus"])
    # plt.axis([-15.0, 15.0, -15.0, 15.0])
    plt.axis([0, 28, 0, 28])
    ax.set_autoscale_on(False)

    positions_anim = animation.FuncAnimation(fig, animate1, n)
    mu_anim = animation.FuncAnimation(fig, animate2, n)
    plt.show()


def show_trajectory(trajectory):
    x = trajectory[:, 0]
    y = trajectory[:, 1]
    fig, ax = plt.subplots()
    line1, = ax.plot(x[:1], y[:1], 'r')
    n = len(x)

    def animate1(i):
        line1.set_xdata(x[:i + 1])
        line1.set_ydata(y[:i + 1])
        return line1,

    plt.legend(["trajectory"])
    plt.axis([-15.0, 15.0, -15.0, 15.0])
    ax.set_autoscale_on(False)

    positions_anim = animation.FuncAnimation(fig, animate1, n)
    plt.show()


def show_positions_by_x_z(trajectories_list, x_list, z_list):
    assert len(trajectories_list) == 4 or len(trajectories_list) == 16 or len(trajectories_list) == 9
    if len(trajectories_list) == 4:
        m, n = 2, 2
    elif len(trajectories_list) == 9:
        m, n = 3, 3
    else:
        m, n = 4, 4
    fig, axs = plt.subplots(m, n)
    lines_list = []
    legends = ["x = {}, z = {}".format(x, z) for x, z in zip(x_list, z_list)]
    print(legends)
    for i in range(m):
        for j in range(n):
            index = j + n * i
            line, = axs[i][j].plot(trajectories_list[index][:1, 0], trajectories_list[index][:1, 1],
                                   label=legends[index])
            axs[i][j].legend()
            axs[i][j].axis([-5.0, 5.0, -5.0, 5.0])
            axs[i][j].set_autoscale_on(False)
            lines_list.append(line)

    def animate(time):
        for i in range(m):
            for j in range(n):
                index = j + n * i
                lines_list[index].set_xdata(trajectories_list[index][:time + 1, 0])
                lines_list[index].set_ydata(trajectories_list[index][:time + 1, 1])
        return lines_list

    anim = animation.FuncAnimation(fig, animate, len(trajectories_list[0]))
    plt.show()


def animate_X_X_G(X, X_G, LEN, initial_length=10, max_len=100, save_dir=None):
    assert len(X) == 4 or len(X) == 9 or len(X) == 16 or len(X) == 18
    if len(X) == 4:
        m, n = 2, 2
    elif len(X) == 9:
        m, n = 3, 3
    elif len(X) == 16:
        m, n = 4, 4
    else:
        m, n = 3, 6
    fig, axs = plt.subplots(m, n, figsize=(4 * m, 4 * n))
    img_list = []
    zero = np.zeros_like(X[0, :, :, initial_length:initial_length + 1])
    for i in range(m):
        for j in range(n):
            index = j + n * i
            axs[i, j].set_axis_off()
            img_array = np.concatenate(
                [X_G[index, :, :, initial_length:initial_length + 1], X[index, :, :, initial_length:initial_length + 1],
                 zero],
                axis=2)
            img = axs[i, j].imshow(img_array)
            img_list.append(img)

    def animate(time):
        t = time + initial_length
        for i in range(m):
            for j in range(n):
                index = j + n * i
                if t < LEN[index] + initial_length:
                    img_array = np.concatenate(
                        [X_G[index, :, :, t:t + 1],
                         X[index, :, :, t:t + 1], X[index, :, :, initial_length - 1:initial_length]],
                        axis=2)
                    img_list[index].set_data(img_array)
        return img_list

    anim = animation.FuncAnimation(fig, animate, max_len)
    if save_dir:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, bitrate=1800)
        anim.save(save_dir, writer=writer)
    else:
        plt.show()


def animate_X_G_heat_map(X_G, HM, LEN, likelihood, initial_length=10, interval=200, save_dir=None):
    fig, axs = plt.subplots(2, 1)
    zero = np.zeros_like(X_G[:, :, initial_length:initial_length + 1])
    X_G_image = None
    HM_image = None
    axs[0].set_axis_off()
    axs[1].set_axis_off()
    img_array = np.concatenate(
        [X_G[:, :, initial_length - 1:initial_length], zero, zero], axis=2)
    HM_array = np.concatenate([zero, zero, zero], axis=2)
    X_G_image = axs[0].imshow(img_array)
    print(zero.shape)
    HM_image = axs[1].imshow(HM_array)


    def animate(time):
        t = time + initial_length - 1
        if time < LEN:
            img_array = np.concatenate(
                [X_G[:, :, t:t + 1], zero,
                 zero],
                axis=2)
            X_G_image.set_data(img_array)
            HM_array = np.concatenate([X_G[:, :, t:t + 1], HM[:, :, time:time+1], zero], axis=2)
            HM_image.set_data(HM_array)
            if time > 0:
                axs[1].set_title(likelihood[time - 1])
        return X_G_image, HM_image

    anim = animation.FuncAnimation(fig, animate, LEN, interval=interval)
    if save_dir:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, bitrate=1800)
        anim.save(save_dir, writer=writer)
    else:
        plt.show()


if __name__ == "__main__":
    show_point_file(71, 'train')
