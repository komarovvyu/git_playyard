from click import style
from typing_extensions import NamedTuple
import numpy as np
import plotly.graph_objects as go

type FloatList = list[float]
# 3D vectors, points, etc.
class XYZ_Point(NamedTuple):
    x: float = 0.
    y: float = 0.
    z: float = 0.
# 3D Trace
#TODO Correct: in SurfaceOfRevolution.init_unit_circle() x,y,z are replaced to np.array!
class XYZ_SegLine(NamedTuple):
    x: FloatList = np.array #TODO change to numpy arrays? In this case it will not work as a source of Plotly go.Scatter
    y: FloatList = np.array
    z: FloatList = np.array
# 3D Surface
class XYZ_Surface(NamedTuple):
    x: list[FloatList] = []
    y: list[FloatList] = []
    z: list[FloatList] = []
# 3D rotation using Euler angles: r' = Rz+omega Rx+chi Rz+phi r
class Euler_Angles(NamedTuple):
    omega: float = 0.
    phi: float = 0.
    chi: float = 0.
    unit: str = 'deg'
# (R) Radius - (L) section Length sequences for surface of revolution description
class RL_Lists(NamedTuple):
    r: FloatList = []
    l: FloatList = []
# Plotly Surface appearance
class PlotlySurfaceStyle(NamedTuple):
    surface = dict(
        color="red",
        transparency=0.5,
    )
    border = dict(
        width = 10,
        color = "blue",
        dash = "solid", # "dash", "dot"
        #transparency = 0.2,
        show_bases = dict(
            start=True,
            end=True
        ),
        style = [] #TODO lines for non-degenerated, markers for degenerated borders
    )

class SurfaceOfRevolution():
    unit_circle = XYZ_SegLine()
    """
    Surface_of_Revolution implements generation of Plotly Surface trace as
      a composite N-sided prism approximation of a Surface of Revolution
      with an arbitrary piecewise-linear generatrix.
    Non-rotated surface propagated in negative-x direction from the origin_point
      with initial profile.l[0] indentation (use profile.l[0] = 0. for
      no-indented surfaces).
    """
    #TODO #001 separate a basic UVSurface class
    def __init__(self, origin_point:XYZ_Point=XYZ_Point(), euler_rot:Euler_Angles=Euler_Angles(),
                 profile:RL_Lists=RL_Lists(), seg_num=50,
                 style:PlotlySurfaceStyle = PlotlySurfaceStyle()):# -> Plotly.Surface ???
        self.origin = origin_point
        self.rotation = euler_rot
        self.profile = profile
        #001 v_, u_sizes and some operations with UV-coordinate array are too abstract for this Class!
        self.v_size = len(profile.l) # v - slow coordinate in uv-surface representation, accords to latitudial (inter-segmental) sections
        self.u_size = seg_num + 1 # u - fast coordinate in uv-surface representation, accords to longitudinal (intra-segmental) sections

        if self.unit_circle == XYZ_SegLine():
            self.init_unit_circle()
        self.init_xyz_stack = self.calc_init_surf() # initialization of (v*u, 3) 2D array of the non-rotated surface

        self.style = style
        self.set_style()

        #self.calc_init_borders() #initialization of self.init_borders.x&y&z list

    def surface_x_section(self, x_index:int):
        section = XYZ_SegLine(
            x=self.init_surf.x[x_index],
            y=self.init_surf.y[x_index],
            z=self.init_surf.z[x_index],
        )
        return section

    def calc_init_borders(self):
        self.init_borders = list()
        if self.style.border['show_bases']['start']:
            self.init_borders.append(self.surface_x_section(1))
        if self.style.border['show_bases']['end']:
            self.init_borders.append(self.surface_x_section(-1))

    def add_to(self, fig):
        #TODO rotate surface: self.rot_surf <- R( self.init_surf )
        rot_xyz_stack = self.init_xyz_stack
        x = rot_xyz_stack[:,0].reshape((self.v_size, self.u_size))
        y = rot_xyz_stack[:,1].reshape((self.v_size, self.u_size))
        z = rot_xyz_stack[:,2].reshape((self.v_size, self.u_size))
        fig.add_trace(go.Surface(
            x=x, y=y, z=z,
            #surfacecolor=self.scolor, colorscale=self.scscale, opacity=self.sopacity,
            #showscale=False,
        ))
        #TODO rotate borders
        #self.rot_borders = self.init_borders
        #for i in range(0, len(self.rot_borders)):
        #    border = self.rot_borders[i]
        #    fig.add_trace(go.Scatter3d(
        #        x=border.x, y=border.y, z=border.z,
        #        mode='lines',
        #        line=dict(
        #            color=self.style.border['color'],
        #            width=self.style.border['width'],
        #            dash=self.style.border['dash'],
        #            # opacity= ?
        #        ),
        #    ))

    def set_style(self):
        self.scolor = np.zeros_like(self.u_size * self.v_size)
        print(self.style.surface)
        self.scscale = [self.style.surface['color'], self.style.surface['color']]
        self.sopacity = 1 - self.style.surface['transparency']

    def init_unit_circle(self):
        #yz unit circle
        _ksi = np.linspace(0., 2*np.pi, self.u_size)
        self.unit_circle = XYZ_SegLine(
            x=np.zeros_like(_ksi),
            y=np.cos(_ksi),
            z=np.sin(_ksi),
        )

    def calc_init_surf(self):
        # return xyz_stack - (u*v, 3)-shaped 2D array
        u, v = self.u_size, self.v_size
        #print(f"generate {u}-angular {v}-segment prism with profile:",
        #      f"{self.profile}",
        #      f"and unit 'parallel' circle:",
        #      f"{self.unit_circle}",
        #      sep="\n")

        # fast index: xyz (point coordinate index),
        # mid. index: u (intra-segment point index),
        # slow index: v (segment index)
        xyz_uv = np.array((self.unit_circle.x + self.origin.x,
                           self.unit_circle.y + self.origin.y,
                           self.unit_circle.z + self.origin.z),
                          dtype=np.float32).T.reshape(1, u, 3).repeat(v, axis=0)
        px = np.array(self.profile.l).cumsum()
        p0 = np.zeros_like(px)
        v_shift = np.array((px, p0, p0), dtype=np.float32).T.reshape(v, 1, 3)
        pr = np.array(self.profile.r)
        p1 = np.ones_like(px)
        v_scale = np.array((p1, pr, pr), dtype=np.float32).T.reshape(v, 1, 3)
        return (xyz_uv * v_scale + v_shift).reshape((-1,3))

if __name__ == "__main__":
    fig = go.Figure()

    profile = RL_Lists(r=[0., 0.8, 1., 0.2, 0.], l=[0., 0.2, 1., 1.8, 2.])
    surf = SurfaceOfRevolution(profile=profile)
    surf.add_to(fig)

    fig.show()