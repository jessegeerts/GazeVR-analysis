import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import load_raw_data


def set_3d_axes(labels=['X', 'Y', 'Z'], xlim=[-1.0, 1.0], ylim=[-1, 1], zlim=[0, 2.0]):
    """
    Creates a 3d axis

    :param labels:
    :param xlim:
    :param ylim:
    :param zlim:
    :return: matplotlib figure and axis objects
    """

    # Create figure
    fig = plt.figure()
    ax = p3.Axes3D(fig)

    # Setting the axes properties (note: reverse X and Y axes)
    ax.set_xlim3d(xlim)
    ax.set_xlabel(labels[0])

    ax.set_ylim3d(ylim)
    ax.set_ylabel(labels[1])

    ax.set_zlim3d(zlim)
    ax.set_zlabel(labels[2])

    ax.set_title('3D trajectory')

    return fig, ax


def update_lines(num, data, line, points_drawn_per_frame):
    line.set_data(data[0:2, :num * points_drawn_per_frame])
    line.set_3d_properties(data[2, :num * points_drawn_per_frame])
    return line


def plot_trajectory(df, color_gradient=False, color_map='winter'):
    """

    :param filename:
    :param color_gradient:
    :param color_map:
    :return:
    """

    fig, ax = set_3d_axes()

    # Load head position data
    data = np.array((df['X'].values, df['Y'].values, df['Z'].values))

    if color_gradient is True:
        cm = plt.get_cmap(color_map)
        ax.set_prop_cycle('color', [cm(1. * i / (data.shape[1] - 1)) for i in range(data.shape[1] - 1)])
        for i in range(data.shape[1] - 1):
            ax.plot(data[0, i:i + 2], data[2, i:i + 2], data[1, i:i + 2])
    else:
        ax.plot(data[0, ], data[2, ], data[1, ])


def plot_3d_video(df, points_drawn_per_frame=50, filename='trajectory'):
    """
    Plots a 3d video showing the trajectory of the head position in space.

    Usage:
    data = get_location_dataframe(matrix_dataframe)
    plot_3d_video(df)

    """
    fig, ax = set_3d_axes()
    # Load data into numpy array
    data = np.array((df['X'].values, df['Y'].values, df['Z'].values))
    line = ax.plot(data[0, 0:1], data[2, 0:1], data[1, 0:1])[0]  # Note: Z and Y axes are swapped here

    ani = animation.FuncAnimation(fig,
                                  update_lines,
                                  int(data.shape[1] / points_drawn_per_frame),
                                  fargs=(data, line, points_drawn_per_frame),
                                  interval=1,
                                  blit=False)

    filename = filename + '.mp4' if filename[-4:] != '.mp4' else filename
    ani.save(filename, fps=30, extra_args=['-vcodec', 'libx264'])
