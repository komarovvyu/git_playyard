import copy
from typing_extensions import NamedTuple
import numpy as np
import plotly.graph_objects as go

#TODO remove NamedTupple inheritance {GD} -- it is redundant
#TODO make class for 3D vectors stack, particular case -- 1 point {GD}
#TODO ? flat stack > UV stack > UV surface > Revolution surface > ... {GD}

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
# create a class of rotations! :(
Euler_Angles_EXAMPLE = dict(
    omega = 60.,
    phi = 60.,
    chi = 60.,
    unit = 'deg', # or 'rad'
)
def get_rad(euler_angles, angle_name='omega'):
    deg_to_rad_factor = np.pi / 180.
    if (angle_name not in euler_angles.keys()) or (angle_name=='unit'):
        raise Exception(f'Wrong angle name {angle_name} for {euler_angles}.')
    if euler_angles['unit'] == 'rad':
        return euler_angles[angle_name]
    elif euler_angles['unit'] == 'deg':
        return euler_angles[angle_name] * deg_to_rad_factor
    else:
        raise Exception(f'Unsupported units {euler_angles.unit} in {euler_angles}.')
# (R) Radius - (L) section Length sequences for surface of revolution description
class RL_Lists(NamedTuple):
    r: FloatList = []
    l: FloatList = []
# Plotly Surface appearance
class PlotlySurfaceStyle(NamedTuple):
    surface = dict(
        color = "red",
        opacity = 0.5,
        uv_colors = np.array([]),
        colorscale = [0., 0.],
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
    def __init__(self, origin_point:XYZ_Point=XYZ_Point(),
                 profile:RL_Lists=RL_Lists(), seg_num=50,
                 style:PlotlySurfaceStyle = PlotlySurfaceStyle()):# -> Plotly.Surface ???
        self.origin = origin_point
        self.profile = profile
        #001 v_, u_sizes and some operations with UV-coordinate array are too abstract for this Class!
        self.v_size = len(profile.l) # v - slow coordinate in uv-surface representation, accords to latitudial (inter-segmental) sections
        self.u_size = seg_num + 1 # u - fast coordinate in uv-surface representation, accords to longitudinal (intra-segmental) sections

        if self.unit_circle == XYZ_SegLine():
            self.init_unit_circle()
        self.init_surf_xyz_stack = self.calc_init_surf() # initialization of (v*u, 3) 2D array of the non-rotated surface
        self.style = self.set_style(style)
        self.init_borders_xyz_stack = self.calc_init_borders() #initialization of visible borders as list of (xyz)-stacks

        #self.set_rotation()

    def set_rotation(self, euler_angles=Euler_Angles_EXAMPLE):
        # calculate R -- 3D rotation matrix = R(z, omega)*R(x, chi)*R(z, phi) Euler angle rotation
        #  applicable for R * r-column matrix multiplication
        om = get_rad(euler_angles, 'omega')
        co, so = np.cos(om), np.sin(om)
        R_om = np.matrix(([co, -so, 0.],
                          [so, co, 0.],
                          [0., 0., 1.]))
        chi = get_rad(euler_angles, 'chi')
        cc, sc = np.cos(chi), np.sin(chi)
        R_chi = np.matrix(([1., 0., 0.],
                           [0., cc, -sc],
                           [0., sc, cc]))
        phi = get_rad(euler_angles, 'phi')
        cp, sp = np.cos(phi), np.sin(phi)
        R_phi = np.matrix(([cp, -sp, 0.],
                           [sp, cp, 0.],
                           [0., 0., 1.]))
        print(R_om, R_chi, R_phi, '===', R_om * R_chi * R_phi, sep='\n*\n')
        pass  ### !!! continue here!!!

    def surface_x_section(self, v_index:int):
        xyz_uv_stack = self.init_surf_xyz_stack.reshape((self.v_size, self.u_size, 3)) #TODO rearrange v and u indices
        section = xyz_uv_stack[v_index, :, :].reshape((-1, 3))
        return section

    def calc_init_borders(self):
        # return list of stacked xyz triplets as list(np.array[v_size, 3])
        result_borders = list()
        if self.style.border['show_bases']['start']:
            result_borders.append(self.surface_x_section(v_index=0))
        if self.style.border['show_bases']['end']:
            result_borders.append(self.surface_x_section(v_index=-1))

        return result_borders

    def add_to(self, fig, euler_angles=Euler_Angles_EXAMPLE):
        #TODO rotate surface: self.rot_surf <- R( self.init_surf )
        self.set_rotation(euler_angles)
        rot_xyz_stack = self.init_surf_xyz_stack
        x = rot_xyz_stack[:,0].reshape((self.v_size, self.u_size))
        y = rot_xyz_stack[:,1].reshape((self.v_size, self.u_size))
        z = rot_xyz_stack[:,2].reshape((self.v_size, self.u_size))
        fig.add_trace(go.Surface(
            x=x, y=y, z=z,
            surfacecolor=self.style.surface['uv_colors'],
            colorscale=self.style.surface['colorscale'],
            opacity=self.style.surface['opacity'],
            #showscale=False,
        ))
        #TODO rotate borders
        self.rot_borders = self.init_borders_xyz_stack
        for border in self.rot_borders:
            x = border[:, 0]
            y = border[:, 1]
            z = border[:, 2]
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines',
                line=dict(
                    color=self.style.border['color'],
                    width=self.style.border['width'],
                    dash=self.style.border['dash'],
                    # opacity= ?
                ),
            ))

    def set_style(self, style):
        _style = copy.deepcopy(style)
        _style.surface['uv_colors'] = np.zeros(self.u_size * self.v_size)
        _style.surface['colorscale'] = [style.surface['color'], style.surface['color']]
        #TODO get rid of ['<field names>'], adopt to fig.add_trace(go.Surface(...)) usage
        return _style

    def init_unit_circle(self):
        #TODO check actuality of the current unit circle or make it as entity property
        #TODO change data type to np.array xyz-stack for the similarity with borders and surface representation after the appropriate classes realization
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

    profile = RL_Lists(r=[0.1, 0.8, 1., 0.2, 0.1], l=[0., 0.2, 1., 1.8, 2.])
    surf = SurfaceOfRevolution(profile=profile)
    surf.add_to(fig)

    #fig.show()