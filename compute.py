import numpy as np
from matplotlib import pyplot as plt
import scipy.sparse as sp
import dolfin

def fenics_spy(matrix, show=False, title=""):
    """Plot the sparsity pattern of a Dolfin GenericMatrix
    wrapper around matplotlib.pyplot.spy

    See cos.compute.spy for details on arguments
    """
    rows, cols, values = matrix.data()
    mat = sp.csr_matrix((values, cols, rows))
    spy(mat, show=show, title=title)

def spy(M, show=False, title=""):
    """Plot the sparsity pattern of a numpy sparse matrix
    wrapper around matplotlib.pyplot.spy

    Keyword arguments:
        show: whether to hold the plot (using dolfin.show())
        title: an optional title for the plot
    """
    plt.spy(M, markersize=0.5)
    plt.title(title)
    if show:
        plt.show()

def numpy_matrix(matrix, m, n):
    """Given a Dolfin GenericMatrix, return a copy as a numpy CSR matrix.

    m: number of rows of the resulting matrix
    n: number of columns of the resulting matrix
    """
    if type(matrix) == dolfin.cpp.la.Vector:
        rows, cols, values = sp.find(matrix.array())
        return sp.coo_matrix((values, (rows, cols)), shape=(m, n))
    else:
        rows, cols, values = matrix.data(deepcopy=True)
        return sp.csr_matrix((values, cols, rows))

def stitch(funcs, V_fine, c_w, c_h, N_w, N_h):
    """Stitch the dolfin Functions funcs into a global V_Fine function

    This is used for MsFEM type methods, where funcs are functions defined on the cells of a coarse mesh
    that need to be stitched into a function over the whole domain. The meshes need to be compatible and
    indexed in the same way, and the DOFs need to be at the vertices, which effectively limits it to
    CG1 functions defined over a regular rectangular mesh (built with UnitSquareMesh).

    funcs: a list of Functions defined on the cell mesh
    V_fine: the global function space
    c_w: the width of the cell mesh, as passed to UnitSquareMesh
    c_h: the width of the cell mesh, as passed to UnitSquareMesh
    N_w: the width of the global mesh, as passed to UnitSquareMesh
    N_h: the width of the global mesh, as passed to UnitSquareMesh
    """

    if not funcs:
        return None
    if N_w % c_w != 0:
        raise ValueError("Cell and global meshes have incompatible widths ({0} and {1})".format(c_w, N_w))
    if N_h % c_h != 0:
        raise ValueError("Cell and global meshes have incompatible heights ({0} and {1})".format(c_h, N_h))
    cells_per_side_w = N_w//c_w
    cells_per_side_h = N_h//c_h
    vtd = dolfin.vertex_to_dof_map(funcs[0].function_space())
    vtd_f = dolfin.vertex_to_dof_map(V_fine)
    v_f = dolfin.Function(V_fine)
    v_f_vec = v_f.vector()
    dof_per_vertex = max(1, reduce(lambda s,i:s+max(1, i.num_sub_spaces()), V_fine.split(), 0))

    offset = 0

    idc = np.array([[j*(dof_per_vertex*(N_w+1))+i for i in range(dof_per_vertex*(c_w+1))] for j in range(c_h+1)]).ravel()

    for j in range(cells_per_side_h):
        for i in range(cells_per_side_w):
            vertices_f = vtd_f[idc+offset]
            v_f_vec[vertices_f] = funcs[j*cells_per_side_w+i].vector()[vtd]
            offset += c_w*dof_per_vertex
        offset += ((N_w+1)*(c_h-1)+1)*dof_per_vertex
    return v_f
