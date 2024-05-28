import numpy as np
import polyscope as ps
from wavejets import WavejetComputation

# Example usage
def main():

    # parameters
    k_order = 2
    color_using_phi = (2,0)
    ampli_coef = 1
    file_path = 'examples_mesh/octopus.obj'



    (color_k,color_n) = color_using_phi

    ps.init()

    # Load a 3D model
    wc = WavejetComputation.cls_load_obj(file_path, (lambda x: x*2))
    original_mesh: ps.SurfaceMesh = ps.register_surface_mesh("original mesh", wc.only_coordinates(), wc.only_faces())

    coo = wc.only_coordinates()
    phis = np.ndarray((len(coo), wc.compute_number_of_phi(k_order)), np.complex64)
    new_coo_pos_enhance = np.ndarray((len(coo), 3), np.float64)
    
    real_phi_2_0 = np.ndarray(len(coo), np.float64)
    gaussian = np.ndarray(len(coo), np.float64)
    mean = np.ndarray(len(coo), np.float64)

    # Compute
    for i in range(len(coo)):
        print("Computing phis : ", i/len(coo)*100, "%")
        phis[i], normal, radius = wc.compute_phis(k_order, i)
        real_phi_2_0[i] = phis[i, wc.k_n_to_index(color_k, color_n)].real
        gaussian[i] = wc.gaussian_curvature(phis[i]).real
        mean[i] = wc.mean_curvature(phis[i]).real
        new_coo_pos_enhance[i] = wc.enhance_position(i, normal, ampli_coef, phis[i], radius)

    original_mesh.add_scalar_quantity('real_phi_2_0', real_phi_2_0)
    original_mesh.add_scalar_quantity('gaussian', gaussian)
    original_mesh.add_scalar_quantity('mean', mean)

    wc.set_coordinates(new_coo_pos_enhance)

    new_mesh: ps.SurfaceMesh = ps.register_surface_mesh("new mesh", wc.only_coordinates(), wc.only_faces())

    # Show result
    ps.show()

def test():
    wc = WavejetComputation.cls_load_obj('examples_mesh/octopus.obj', (lambda x: x*2))
    print(len(wc.only_coordinates()))

if __name__ == "__main__":
    main()