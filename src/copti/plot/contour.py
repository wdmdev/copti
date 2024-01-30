from typing import Callable, Union

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import matplotlib.cm as cm #colormap


def plot_2d(f:Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]], 
            x1: NDArray[np.float64], x2: NDArray[np.float64],
            save_path:Union[str,None]=None, show:bool=False, 
            color_map:str="RdYlBu_r") -> None:
    """ Plot a 2d function f(x, y) with a contour plot.

    Args:
        f (float, float) -> float: Function to plot.
    """

    X1, X2 = np.meshgrid(x1, x2)
    Z = f(X1, X2)

    fig, ax = plt.subplots()
    cmap = cm.get_cmap(color_map)
    CS = ax.contour(X1, X2, Z, color_map=cmap)
    ax.clabel(CS, inline=True, fontsize=10)
    ax.set_title('Contour Plof of f(x1, x2)')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')

    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()
