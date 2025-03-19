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
    x: FloatList = [] #TODO change to numpy arrays? In this case it will not work as a source of Plotly go.Scatter
    y: FloatList = []
    z: FloatList = []
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
        color = "red",
        transparency=0.5
    )
    border = dict(
        width = 2,
        color = "blue",
        transparency = 0.2
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
    def __init__(self, origin_point:XYZ_Point=XYZ_Point(), euler_rot:Euler_Angles=Euler_Angles(),
                 profile:RL_Lists=RL_Lists(), seg_num=6,
                 style:PlotlySurfaceStyle = PlotlySurfaceStyle()):# -> Plotly.Surface ???
        self.origin = origin_point
        self.rotation = euler_rot
        self.profile = profile
        self.seg = seg_num
        self.style = style
        if self.unit_circle == XYZ_SegLine():
            self.init_unit_circle()
        self.calc_init_surf()

    def init_unit_circle(self):
        #yz unit circle
        _ksi = np.linspace(0., 2*np.pi, self.seg)
        self.unit_circle = XYZ_SegLine(
            x=np.zeros_like(_ksi),
            y=np.cos(_ksi),
            z=np.sin(_ksi),
        )
        #self.unit_circle.x.append(np.zeros_like(_ksi))
        #self.unit_circle.y.append(np.cos(_ksi))
        #self.unit_circle.z.append(np.sin(_ksi))

    def calc_init_surf(self):
        self.init_surf = XYZ_Surface()
        axis_x = self.origin.x
        for l, r in zip(self.profile.l, self.profile.r):
            axis_x -= l
            #print('arg: ', self.unit_circle.x, ';    ', axis_x)
            #print('np: ', self.unit_circle.x + axis_x)
            #print('list: ', list(self.unit_circle.x + axis_x))
            self.init_surf.x.append(list(self.unit_circle.x + axis_x))
            self.init_surf.y.append(list(self.origin.y + self.unit_circle.y * r))
            self.init_surf.z.append(list(self.origin.z + self.unit_circle.z * r))

if __name__ == "__main__":
    fig = go.Figure()

    profile = RL_Lists(r=[0., 1., 0.], l=[0., 1., 1.])
    surf = SurfaceOfRevolution(profile=profile)
    fig.add_trace(go.Surface(
        x=surf.init_surf.x, y=surf.init_surf.y, z=surf.init_surf.z
    ))

    fig.show()