import numpy as np


def nearest(items, pivot):
    """
    General function for finding the entry in items that is nearest to pivot.

    :param items: Array of numbers or datetime objects
    :param pivot: Single number of datetime object
    :return:
    """
    return min(items, key=lambda x: abs(x - pivot))


def series2mat4(head):
    """
    
    :param head:
    :return:
    """
    return np.array([[head.loc['Value.M11'], head.loc['Value.M21'], head.loc['Value.M31'], head.loc['Value.M41']],
                     [head.loc['Value.M12'], head.loc['Value.M22'], head.loc['Value.M32'], head.loc['Value.M42']],
                     [head.loc['Value.M13'], head.loc['Value.M23'], head.loc['Value.M33'], head.loc['Value.M43']],
                     [head.loc['Value.M14'], head.loc['Value.M24'], head.loc['Value.M34'], head.loc['Value.M44']]])
