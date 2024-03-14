from typing import Callable, Union, List

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import matplotlib.cm as cm #colormap
import matplotlib.patches as mpatches
import matplotlib.lines as mlines



def plot_2d(f:Callable[[NDArray[np.float64]], np.float64], 
            x1: NDArray[np.float64], x2: NDArray[np.float64],
            save_path:Union[str,None]=None, show:bool=False, 
            color_map:str="RdYlBu_r") -> None:
    """ Plot a 2d function f(x, y) with a contour plot.

    Args:
        f (float, float) -> float: Function to plot.
        x1 (np.ndarray): Array of x1 values.
        x2 (np.ndarray): Array of x2 values.
        save_path (str, optional): Path to save the plot. Defaults to None.
        show (bool, optional): Show the plot. Defaults to False.
        color_map (str, optional): Color map for the contour plot. Defaults to "RdYlBu_r".
    """

    X1, X2 = np.meshgrid(x1, x2)
    X = np.array([X1, X2])
    Z = f(X)

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

def plot_2d_constrained(f: Callable[[np.ndarray], float], 
                        x1: NDArray[np.float64], x2: NDArray[np.float64],
                        constraints: List[Callable[[NDArray[np.float64]], float]],
                        solution_path: Union[NDArray[np.float64], None] = None,
                        save_path: Union[str, None] = None, 
                        show: bool = False, 
                        color_map: str = "RdYlBu_r") -> None:
    """Plot a 2D function f(x, y) with a contour plot, the feasible region, and a solution path.

    Args:
    ----------
        f : Callable[[NDArray[np.float64]], float]
            Function to plot.
        x1 : NDArray[np.float64] 
            Array of x1 values.
        x2 : NDArray[np.float64] 
            Array of x2 values.
        constraints : List[Callable[[NDArray[np.float64]], float]]
            List of constraint functions.
        save_path : Union[str, None], optional
            Path to save the plot. Defaults to None.
        show : bool, optional
            Show the plot. Defaults to False.
        color_map : str, optional
            Color map for the contour plot. Defaults to "RdYlBu_r".
        solution_path : NDArray[np.float64], optional
            Array of points representing the solution path. Defaults to None.
    """
    # Create a meshgrid for the x values
    X1, X2 = np.meshgrid(x1, x2)
    Z = np.zeros(X1.shape)

    # Evaluate f(x) over the grid
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            x = np.array([X1[i, j], X2[i, j]])
            Z[i, j] = f(x)

    # Evaluate constraints and find the feasible region
    feasible = np.ones(X1.shape, dtype=bool)
    for constraint in constraints:
        for j in range(X1.shape[0]):
            for k in range(X1.shape[1]):
                x = np.array([X1[j, k], X2[j, k]])
                if constraint(x) < 0:
                    feasible[j, k] = False

    # Plotting
    plt.figure(figsize=(8, 6))
    cmap = cm.get_cmap(color_map)
    contour = plt.contour(X1, X2, Z, levels=np.linspace(Z.min(), Z.max(), 50), cmap=cmap)
    plt.clabel(contour, inline=True, fontsize=8)

    plt.imshow(feasible, extent=(x1.min(), x1.max(), x2.min(), x2.max()), 
               origin='lower', alpha=0.3, cmap='Greys', aspect='auto')

    handles = []
    grey_patch = mpatches.Patch(color='grey', alpha=0.3, label='Feasible Region')
    handles.append(grey_patch)

    # Plot solution path if provided
    if solution_path is not None:
        plt.plot(solution_path[:, 0], solution_path[:, 1], '-', color='green')
        plt.plot(solution_path[0, 0], solution_path[0, 1], 'go', markersize=5)  # Start
        plt.plot(solution_path[-1, 0], solution_path[-1, 1], 'g*', markersize=10)  # End

        # Legend
        line_legend = mlines.Line2D([], [], color='green', markersize=5, label='Solution Path')
        circle_legend = mlines.Line2D([], [], color='green', marker='o', markersize=5, linestyle='None', 
                                      label=f'Initial Point ({solution_path[0, 0]:.2f}, {solution_path[0, 1]:.2f})')
        star_legend = mlines.Line2D([], [], color='green', marker='*', markersize=10, linestyle='None', 
                                    label=f'Solution ({solution_path[-1, 0]:.2f}, {solution_path[-1, 1]:.2f})')

        handles.extend([line_legend, circle_legend, star_legend])

    plt.legend(handles=handles)

    plt.colorbar(contour)
    plt.title(f'Contour Plot of {f.__name__} and Feasible Region')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid(True)

    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()

