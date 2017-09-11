import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from tqdm import tqdm


class MovementPlanner:
    def __init__(self, movement_variance, xlim=(-10, 10), ylim=(-10, 10), n_points=1000):
        self.movement_var = movement_variance
        self.xlim = xlim
        self.ylim = ylim
        self.x_axis = np.linspace(xlim[0], xlim[1], n_points)
        self.y_axis = np.linspace(ylim[0], ylim[1], n_points)
        self.xs, self.ys = np.meshgrid(self.x_axis, self.y_axis)
        self.grid_area = (self.xs[0, 1] - self.xs[0, 0]) * (self.ys[1, 0] - self.ys[0, 0])

        self.centre_target = (2.5, 0)
        self.centre_penalty = (-2.5, 0)
        self.radius_target = 5
        self.radius_penalty = 5

        self.G0 = 100  # reward value
        self.G1 = -100  # penalty value

    def P_R_given_xy(self, aim_location, target_location, target_radius):
        probability_r = self.gauss2d(self.xs, self.ys, self.movement_var, aim_location) * \
                        self.target(self.xs, self.ys, target_location, target_radius)
        return np.sum(self.grid_area * probability_r)

    def expected_gain_landscape(self, reward, penalty):
        P_R0 = np.zeros(self.xs.shape)
        P_R1 = np.zeros(self.xs.shape)
        for num_x, x in tqdm(enumerate(self.x_axis)):
            for num_y, y in enumerate(self.y_axis):
                P_R0[num_y, num_x] = self.P_R_given_xy((x, y), self.centre_target, self.radius_target)
                P_R1[num_y, num_x] = self.P_R_given_xy((x, y), self.centre_penalty, self.radius_penalty)

        expected_gain = reward * P_R0 + penalty * P_R1
        return expected_gain

    def surf_plot(self, data, ax, elev, azim, vrange=(-100,100)):
        ax.plot_surface(self.xs, self.ys, data, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False,
                        vmin=vrange[0], vmax=vrange[1])
        ax.view_init(elev, azim)

    def heat_map(self, data, ax, plot_targets=False, vmin=None, vmax=None):
        ax.imshow(data, vmin=vmin, vmax=vmax, extent=(self.xs.min(), self.xs.max(), self.ys.min(), self.ys.max()),
                  cmap=cm.coolwarm)
        if plot_targets is True:
            target = plt.Circle(self.centre_target, radius=self.radius_target, color='g', fill=False)
            ax.add_patch(target)
            penalty = plt.Circle(self.centre_penalty, radius=self.radius_penalty, color='r', fill=False)
            ax.add_patch(penalty)

    @staticmethod
    def target(xs, ys, centre, r):
        """

        :param xs:
        :param ys:
        :param centre: Tuple or list containing x,y centre coordinates
        :param r:
        :return:
        """
        return ((xs - centre[0]) ** 2 + (ys - centre[1]) ** 2 < r ** 2).astype(float)

    @staticmethod
    def gauss2d(x, y, var, centre):
        """
        Define a 2-D Gaussian with variance var and mean centre.

        :param x: Scalar or array with x positions
        :param y: Scalar or array with y positions
        :param var: Variance of the distribution
        :param centre: Mean of the distribution
        :return:
        """
        return np.exp(-((x - centre[0]) ** 2 + (y - centre[1]) ** 2) / (2 * var)) / (2 * np.pi * var)

