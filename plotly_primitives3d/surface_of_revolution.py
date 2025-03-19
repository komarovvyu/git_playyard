from typing_extensions import NamedTuple
from win32verstamp import null_byte

type FloatList = list[float]
# 3D vectors, points, etc.
class XYZ_Point(NamedTuple):
    x: float = 0.
    y: float = 0.
    z: float = 0.
# 3D rotation using Euler angles: r' = Rz+omega Rx+chi Rz+phi r
class Euler_Angles(NamedTuple):
    omega: float = 0.
    phi: float = 0.
    chi: float = 0.
    unit: str = 'deg'
# (r) Radius - (l) section Length sequences for surface of revolution description
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

class Surface_of_Revolution():
    """
    Surface_of_Revolution implements generation of Plotly Surface trace as
      a composite N-sided prism approximation of a Surface of Revolution
      with an arbitrary piecewise-linear generatrix.
    Non-rotated surface propagated in negative-x direction from the origin_point
      with initial profile.l[0] indentation (use profile.l[0] = 0. for
      no-indented surfaces).
    """
    def __init__(self, origin_point:XYZ_Point=XYZ_Point(), euler_rot:Euler_Angles=Euler_Angles(),
                 profile:RL_Lists=RL_Lists(), seg_num=20,
                 style:PlotlySurfaceStyle = PlotlySurfaceStyle()):# -> Plotly.Surface ???
        self.origin = origin_point
        self.rotation = euler_rot
        self.profile = profile
        self.seg = seg_num
        self.style = style

        self.init_surf = self.calc_init_surf()

    def calc_init_surf(self):
        #yz unit circle
        _phi = np.linspace(0., 2*np.pi, self.seg)
        _uy = np.cos(_phi)
        _uz = np.sin(_phi)
        #xr profile
        _x = (self.tip['x'],
              self.tip['x'] - self.con['l'],
              self.tip['x'] - self.con['l'],
              self.tip['x'] - self.con['l'] - self.cyl['l'],
              self.tip['x'] - self.con['l'] - self.cyl['l'],
        )
        _r = (0,
              self.con['r'],
              self.cyl['r'],
              self.cyl['r'],
              0,
        )
        #init surface dict
        arrow_surf = dict(
            x=[], y=[], z=[],
        )
        #arrow surface
        for i in range(0, 5):
            arrow_surf['x'].append([_x[i] for _ in range(self.seg)])
            arrow_surf['y'].append(list(self.tip['y'] + _uy * _r[i]))
            arrow_surf['z'].append(list(self.tip['z'] + _uz * _r[i]))




if __name__ == "__main__":
    point_3d = XYZ_Point(x=2.)
    empty_profile = RL_Lists()

    print(point_3d, empty_profile)