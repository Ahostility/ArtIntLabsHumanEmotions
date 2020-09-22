import numpy as np
import cv2
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

class Plot():
    """Comfort set of tools for plots.

    Attributes:\n
    \tfig (Figure): The top level container for all the plot elements (see matplotlib.figure.Figure).
    \tcanvas (FigureCanvasAgg): The canvas the figure renders into. Calls the draw and print fig methods, creates the renderers, etc.
    \tax (Axes/Axes3D): The axes contains most of the figure elements and sets the coordinate system (see matplotlib.axes.Axes).
    """

    def __init__(self, projection: str = '2d', figsize: tuple = None):
        """
        Args:\n
        \tprojection (str): 2D or 3D.
        \tfigsize (tuple): (width, height) of figure size in inches.
        """
        self.fig = Figure(figsize=figsize)
        self.canvas = FigureCanvasAgg(self.fig)
        self.projection = projection.lower()
        if self.projection == '2d':
            self.ax = self.fig.gca()
        elif self.projection == '3d':
            self.ax = Axes3D(self.fig)
        self.clear()

    def rotate_axes(self, elev: int = None, azim: int = None):
        """Set the elevation and azimuth of the axes in degrees (not radians).
        """
        if self.projection == '3d':
            self.ax.view_init(elev, azim)

    def clear(self):
        """Clear the current axes.
        """
        self.ax.clear()
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        if self.projection == '3d':
            self.ax.set_zlabel('Z')

    def imshow(self, winname: str, waitable: bool = False):
        """Same as cv2.imshow().

        Args:\n
        \twinname (str): Window name.
        \twaitable (bool): Wait for escape to close the window or not.
        """
        cv2.imshow(winname, self.to_image())
        if waitable and cv2.waitKey(0) == 27:
            cv2.destroyWindow(winname)

    def imwrite(self, filename: str, params: dict = None):
        """Same as cv2.imwrite().

        Args:\n
        \tfilename (str): Name of the file.
        \tparams (dict): OpenCV writing parameters.
        """
        cv2.imwrite(filename, self.to_image(), params)

    def to_image(self):
        """Converts the inner plot into a BGR image (numpy).
        """
        self.ax.margins(0)
        self.canvas.draw()
        buf = np.frombuffer(self.canvas.tostring_argb(), dtype=np.uint8)
        w, h = self.canvas.get_width_height()
        image = buf.reshape((h, w, 4))[:, :, 1:]
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image
