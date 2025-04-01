"""
Examples of creation, reshape, etc. of multidimensional np.arrays, lists etc.
"""
import numpy as np

def create_np_arrays():
    '''
    Demonstration of np array creation routines
    https://numpy.org/devdocs/reference/routines.array-creation.html
    :return: None
    '''

    # numpy.zeros(shape, dtype=float, order='C', *, like=None)
    # http://numpy.org/devdocs/reference/generated/numpy.zeros.html
    a = np.zeros((2, 3)) # 2 rows (lines) of 3 columns
    b = np.zeros_like(a)
    c = np.zeros(a.shape)
    print(a, b, c, sep="\n")
    # t = np.zeros((2,), dtype=[('x', 'i4'), ('y', 'i4')])  ??? x, y ???

def create_operate_matrix():
    Rz_p90d = np.matrix('0. -1. 0.; 1. 0. 0.; 0. 0. 1.', dtype=np.float32) # 'line 1; line 2; line 3'
    print(f"R = 90 degree rotation around z: \n {Rz_p90d}")
    print(f"R^-1: \n {Rz_p90d.I};", "\n"
          f"R^t: \n {Rz_p90d.T}") # Inverse, Transpose, As is, A1 - flattened as is, H - complex conjugate transpose
    print(f"R*R: \n {Rz_p90d * Rz_p90d}", "\n"
          f"R^2: \n {Rz_p90d ** 2}") # * and ** as matrix operations

    col_vec = np.matrix('1.; 2.; 3.')
    print(f"cv = column vector: \n {col_vec}", "\n",
          f"R * cv: \n {Rz_p90d * col_vec}") # 3x3 (rotation) matrix - column vector multiplication
    row_vec = np.matrix('1. 2. 3.')
    print(f"rv = row vector: \n {row_vec}", "\n",
          f"rv * R: \n {row_vec * Rz_p90d}") # column vector - 3x3 (rotation) matrix multiplication
    print(f"rv * cv: \n {row_vec * col_vec}", "\n",
          f"cv * rv: \n {col_vec * row_vec}") # results [[14.]] and 3x3 matrix

    x = (1.0, 1.1, 1.2, 1.3)
    y = (2.0, 2.1, 2.2, 2.3)
    z = (3.0, 3.1, 3.2, 3.3)
    print(f"vectors collection input: (x, y, z) = {(x, y, z)}")
    vec_stack = np.matrix((x, y, z), dtype=np.float32).T # rows - x,y,z triplets
    print(f"resulted stack of vectors: \n {vec_stack}", "\n",
          f"i=1 (second) vector: \n {vec_stack[1]}")
    rot_stack = np.matvec(Rz_p90d, vec_stack)
    print(f"R - vector stack product : \n {rot_stack}")
    x_rot, y_rot, z_rot = tuple(rot_stack[:, 0].flat), tuple(rot_stack[:, 1].flat), tuple(rot_stack[:, 2].flat)
    print(f"rotated vectors: (x, y, z) = \n",
          f"({x_rot} ,\n {y_rot} ,\n {z_rot})")

    #Ulybin's way
    vec_stack = np.array((x, y, z), dtype=np.float32).T.reshape(-1, 3, 1)
    print(f"resulted stack of vectors: \n {vec_stack}", "\n",
          f"i=1 (second) vector: \n {vec_stack[1,:,0]}")
    rot_stack = np.matmul(Rz_p90d, vec_stack)
    print(f"R - vector stack product : \n {rot_stack}")
    x_rot = tuple(rot_stack[:, 0].ravel())
    y_rot = tuple(rot_stack[:, 1].ravel())
    z_rot = tuple(rot_stack[:, 2].ravel())
    print(f"rotated vectors: (x, y, z) = \n",
          f"({x_rot} ,\n {y_rot} ,\n {z_rot})")
#rotated vectors: (x, y, z) =
# ((matrix([[-2. , -2.1, -2.2, -2.3]], dtype=float32),) ,
# (matrix([[1. , 1.1, 1.2, 1.3]], dtype=float32),) ,
# (matrix([[3. , 3.1, 3.2, 3.3]], dtype=float32),))  #TODO how to ger rid of matrix([[...]]) ?

def vec3d_stack_to_uv_conversion():
    u_size = 5
    v_size = 4
    xyz_stack = np.random.rand(u_size * v_size, 3)
    print("stack:\n", xyz_stack)
    xyz_uv = xyz_stack.reshape((-1, u_size, 3))
    print("uv:\n", xyz_uv)
    #xyz_stack = xyz_uv.reshape((-1, 3))
    #print("new stack:\n", xyz_stack)


    shift_uv = np.array(
        [[[0.5, 0.1, 0.3]],
         [[0.2,0.01, 1.0]],
         [[0.8, 0.9, 0.9]],
         [[0.3, 0.5, 0.0]]],
    )
    print(f"shift_uv shape: {shift_uv.shape}")
    shifted_uv = xyz_uv + shift_uv
    print("shifted uv:\n", shifted_uv)
    #print("x from uv:\n", xyz_uv[:,:,0])#.reshape((-1, u_size)))
    #print("y from uv:\n", xyz_uv[:,:,0])#.reshape((-1, u_size)))
    #print("z from uv:\n", xyz_uv[:,:,0])#.reshape((-1, u_size)))

def matrix_3d_vector_stack_product():
    R = np.matrix("0, -1, 0; 1, 0, 0; 0, 0, 1", dtype=np.float32)
    print(R)




if __name__ == "__main__":
    x = [1,2,3]
    y = [4,5,6]
    z = [7,8,9]
    a = np.array((x, y, y), dtype=np.float32).T.reshape(3,1,3)
    print(a, a.shape, sep="\n")
    b = a.repeat(2, axis=0)
    print(b)

    #p =