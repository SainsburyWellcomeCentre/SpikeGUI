import numpy as np


def make_cmd(n_points=1000000, duration=10, phy=0):
    """

    :param int n_points:
    :param float duration: seconds
    :param phy:
    :return: cmd, cmd_x
    """
    cmd_x = np.linspace(0, duration, n_points)
    max_angle = 180
    freq = 0.1
    # phy = 3/2*np.pi
    cmd = max_angle * np.sin(2*np.pi*freq*cmd_x + phy)
    return cmd, cmd_x
