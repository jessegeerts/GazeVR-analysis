import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import load_raw_data
import seaborn as sns


def update_lines(num, data, line):
    line.set_data(data[0:2, :num])
    line.set_3d_properties(data[2, :num])
    return line


def plot_trajectory(filename, color_gradient=False, color_map='winter'):
    """

    :param filename:
    :param color_gradient:
    :param color_map:
    :return:
    """
    # Load head position data
    repo = load_raw_data.load_head_data(filename)
    repo = repo[:20000]
    data = np.array((repo['Value.M41'].values, repo['Value.M42'].values, repo['Value.M43'].values))

    # Create figure
    fig = plt.figure()
    ax = p3.Axes3D(fig)

    if color_gradient is True:
        cm = plt.get_cmap(color_map)
        ax.set_prop_cycle('color',[cm(1.*i/(data.shape[1]-1)) for i in range(data.shape[1]-1)])
        for i in range(data.shape[1]-1):
            ax.plot(data[0,i:i+2],data[2,i:i+2],data[1,i:i+2])
    else:
        ax.plot(data[0, ], data[2, ], data[1, ])

    # Setting the axes properties (note: reverse X and Y axes)
    ax.set_xlim3d([-1.0, 1.0])
    ax.set_xlabel('X')

    ax.set_ylim3d([-1, 1])
    ax.set_ylabel('Z')

    ax.set_zlim3d([0, 2.0])
    ax.set_zlabel('Y')

    ax.set_title('3D trajectory')


def plot_3d_video():
    # Attaching 3D axis to the figure
    fig = plt.figure()
    ax = p3.Axes3D(fig)

    # Reading the data from a CSV file using pandas
    repo = load_raw_data.load_head_data('./Data/head.csv')
    repo = repo[:20000]

    data = np.array((repo['Value.M41'].values, repo['Value.M42'].values, repo['Value.M43'].values))

    line = ax.plot(data[0, 0:1], data[2, 0:1], data[1, 0:1])[0]

    # Setting the axes properties
    ax.set_xlim3d([-2.0, 2.0])
    ax.set_xlabel('X')

    ax.set_ylim3d([-2.0, 2.0])
    ax.set_ylabel('Z')

    ax.set_zlim3d([-2.0, 2.0])
    ax.set_zlabel('Y')

    ax.set_title('3D trajectory')

    animation.FuncAnimation(fig, update_lines, data.shape[1], fargs=(data, line), interval=1, blit=False)

    plt.show()
