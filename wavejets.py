import numpy as np
import polyscope as ps
from collections import deque
from scipy.spatial.distance import euclidean
from typing import List, Callable
from wavefront import WavefrontOBJ
from utils import point_cart_to_pol
from graph import Graph

class WavejetComputation(WavefrontOBJ):
    nodes_get_by_bfs: Callable[[int], int]
    graph: Graph

    def __init__(self, nodes_get_by_bfs: Callable[[int], int], default_mtl='default_mtl'):
        super().__init__(default_mtl)
        self.nodes_get_by_bfs = nodes_get_by_bfs
        self.graph = Graph(self)

    def compute_number_of_phi(self, k_order: int)-> int:
        return self.__class__.cls_number_of_phi(k_order)
    
    @classmethod
    def cls_number_of_phi(cls, k_order: int)-> int:
        # Straightforward version :
        # index = 0
        # for i in range(k_order):
        #     for _ in range(-i, i+1):
        #         index+=1

        # First loop removed :
        # index = 0
        # for i in range(k_order):
        #     index+= i*2 +1

        return (k_order+1)**2
    
    def k_n_to_index(self, k: int, n: int)-> int:
        return self.__class__.cls_k_n_to_index(k,n)
    
    @classmethod
    def cls_k_n_to_index(cls, k: int, n: int)-> int:
        return cls.cls_number_of_phi(k) -1 -(k-n)

    def compute_heightmap(self, k_order: int, p_idx: int)-> np.ndarray:
        # We are looking for a tangent plane with a little offset, so that the heightmap won't be 0 at p
        faces = self.neighbor_faces(p_idx)
        coo = self.only_coordinates()

        orth = np.array([0.,0.,0.], np.float64)
        for face in faces:
            orth += np.cross(face[0]-face[1], face[1]-face[2])
        
        orth = orth / np.linalg.norm(orth)

        vect1 = np.random.rand(3)
        vect1 -= vect1.dot(orth) * orth
        vect2 = np.cross(vect1, orth)

        new_base = np.column_stack([vect1, vect2, orth])

        # We gather and compute the neighbors in the new base
        seen: List[int] = []
        to_treat: deque[int] = deque([p_idx])
        uniques = np.array([])

        points = np.array([])

        min_k = self.compute_number_of_phi(k_order)

        while True: # do loop
            seen, to_treat = self.graph.bfs(
                (self.nodes_get_by_bfs(min_k)-len(uniques)),
                seen,
                to_treat
            )
            
            points = np.array([
                points[i] if i < points.size
                else np.linalg.inv(new_base).dot(coo[seen[i]])
                for i in range(len(seen))
            ])

            uniques = np.unique(points, axis=0)
            # We remove lines containing NaN or Inf
            uniques = uniques[~np.any(np.isnan(uniques) | np.isinf(uniques), axis=1)]

            if (len(uniques) >= self.nodes_get_by_bfs(min_k)):
                break

        p = coo[p_idx]
        compute_dist = lambda point: euclidean(point, p)
        dists = np.apply_along_axis(compute_dist, 1, uniques)

        get_dist = lambda dist_tuple: dist_tuple[0]
        get_point = lambda dist_tuple: dist_tuple[1]
        dist_and_point_list = [(dists[i], uniques[i]) for i in range(len(uniques))]

        return np.array([
            get_point(dist_and_point) for dist_and_point in
                sorted(dist_and_point_list, key=get_dist)[:min_k]
        ])
    
    def compute_b(self, k: int, n: int, r: float, theta: float)-> complex:
        return (r**k) * (np.e ** (1j * n * theta))
    
    def compute_phis(self, k_order: int, p_idx: int)-> np.ndarray:
        heightmap = self.compute_heightmap(k_order, p_idx)

        numb_phi = self.compute_number_of_phi(k_order)

        b_list = np.ndarray((len(heightmap), numb_phi), np.complex64)
        neighbors_height = np.ndarray(len(heightmap), np.float64)
        phis = np.ndarray(numb_phi, np.complex64) # Just for type hinting

        for index_col in range(len(heightmap)):
            neighbors_height[index_col] = heightmap[index_col][2]
            for k in range(k_order+1):
                for n in range(-k, k+1):
                    r, theta = point_cart_to_pol(heightmap[index_col][0:2])
                    b_list[index_col, self.k_n_to_index(k,n)] = self.compute_b(k,n,r,theta)

        phis = np.linalg.lstsq(b_list, neighbors_height, rcond=None)[0]

        return phis
    
    @classmethod
    def cls_load_obj(cls, filename: str, nodes_get_by_bfs: Callable[[int], int], default_mtl='default_mtl', triangulate=False):
        obj = cls(nodes_get_by_bfs, default_mtl)
        obj.load_obj(filename, default_mtl, triangulate)
        return obj
    
    def load_obj(self, filename: str, default_mtl='default_mtl', triangulate=False):
        super().load_obj(filename, default_mtl, triangulate)
        self.graph = Graph(self)
    
    def set_coordinates(self, new_coordinates: np.ndarray)-> None:
        super().set_coordinates(new_coordinates)
        self.graph = Graph(self)

# Example usage
def main():

    # parameters
    k_order = 2
    color_using_phi = (2,0)
    (color_k,color_n) = color_using_phi

    ps.init()

    # Load a 3D model
    wc = WavejetComputation.cls_load_obj('examples_mesh/octopus.obj', (lambda x: x*2))
    original_mesh: ps.SurfaceMesh = ps.register_surface_mesh("original mesh", np.array(wc.only_coordinates()), np.array(wc.only_faces()))

    coo = wc.only_coordinates()
    phis = np.ndarray((len(coo), wc.compute_number_of_phi(k_order)), np.complex64)
    real_phi = np.ndarray((len(coo)), np.int64)

    # Compute phis
    for i in range(len(coo)):
        print("Computing phis : ", i/len(coo)*100, "%")
        phis[i] = wc.compute_phis(k_order, i)
        real_phi[i] = phis[i, wc.k_n_to_index(color_k, color_n)].real

    original_mesh.add_scalar_quantity('real_phi', real_phi)

    # Show result
    ps.show()

def test():
    wc = WavejetComputation.cls_load_obj('examples_mesh/octopus.obj', (lambda x: x*2))
    print(len(wc.only_coordinates()))

if __name__ == "__main__":
    main()