import numpy as np
import polyscope as ps
from collections import deque
from scipy.spatial.distance import euclidean
from typing import List, Callable, Tuple
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
    
    def index_to_k_n(self, index: int)-> Tuple[int, int]:
        return self.__class__.cls_index_to_k_n(index)
    
    @classmethod
    def cls_index_to_k_n(cls, index: int)-> Tuple[int, int]:
        k = int(np.floor(np.sqrt(index)))
        n = int(index - cls.cls_k_n_to_index(k, -k) -k)
        return k,n

    def compute_heightmap(self, k_order: int, p_idx: int)-> Tuple[np.ndarray, np.ndarray, float]:
        # We are looking for a tangent plane
        faces = self.neighbor_faces(p_idx)
        coo = self.only_coordinates()

        orth = np.array([0.,0.,0.], np.float64)
        for face in faces:
            orth += np.cross(face[0]-face[1], face[1]-face[2])
        
        norm = np.linalg.norm(orth)
        if norm == 0: # Yes, it does happen
            return np.array([]), np.array([0., 0., 0.]), 0.
        orth /= np.linalg.norm(orth)

        vect1 = np.random.rand(3)
        vect1 -= vect1.dot(orth) * orth
        vect2 = np.cross(vect1, orth)

        new_base = np.column_stack([vect1, vect2, orth])

        p = np.linalg.inv(new_base).dot(coo[p_idx])
        offset = p[2]
        p[2] = 0.

        # We gather and compute the neighbors in the new base
        seen: List[int] = []
        to_treat: deque[int] = deque([p_idx])
        uniques = np.array([])

        points = np.array([])

        min_k = self.compute_number_of_phi(k_order)

        def use_offset(vect):
            vect[2]-=offset
            return vect

        while True: # do loop
            seen, to_treat = self.graph.bfs(
                (self.nodes_get_by_bfs(min_k)-len(uniques)),
                seen,
                to_treat
            )
            
            points = np.array([
                points[i] if i < points.size
                else use_offset(np.linalg.inv(new_base).dot(coo[seen[i]]))
                for i in range(len(seen))
            ])

            uniques = np.unique(points, axis=0)
            # We remove lines containing NaN or Inf
            uniques = uniques[~np.any(np.isnan(uniques) | np.isinf(uniques), axis=1)]

            if (len(uniques) >= self.nodes_get_by_bfs(min_k)):
                break

        compute_dist = lambda point: euclidean(point, p)
        dists = np.apply_along_axis(compute_dist, 1, uniques)

        get_dist = lambda dist_tuple: dist_tuple[0]
        get_point = lambda dist_tuple: dist_tuple[1]
        dist_and_point_list = [(dists[i], uniques[i]) for i in range(len(uniques))]

        points_kept = sorted(dist_and_point_list, key=get_dist)[:min_k]

        return (np.array([get_point(dist_and_point) for dist_and_point in points_kept]),
            orth,
            get_dist(points_kept[-1]))
    
    def compute_b(self, k: int, n: int, r: float, theta: float)-> complex:
        return (r**k) * (np.e ** (1j * n * theta))
    
    def compute_phis(self, k_order: int, p_idx: int)-> Tuple[np.ndarray, np.ndarray, float]:
        heightmap, normal, radius = self.compute_heightmap(k_order, p_idx)

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

        return phis, normal, radius
    
    def gaussian_curvature(self, phis: np.ndarray)-> np.complex64:
        return self.__class__.cls_gaussian_curvature(phis)
    
    @classmethod
    def cls_gaussian_curvature(cls, phis: np.ndarray)-> np.complex64:
        # Formula depicted in `paper.pdf` uses 2-Wavejet (or Wavejet of order 2)
        if(len(phis) < cls.cls_k_n_to_index(2,2)+1 ):
            raise Exception("Order of provided Wavejet too low. Please provide at least 2-Wavejet")
        a = (4 * phis[cls.cls_k_n_to_index(2,0)]**2 - 16 * phis[cls.cls_k_n_to_index(2,-2)] * phis[cls.cls_k_n_to_index(2,2)]) / (1 + 4 * phis[cls.cls_k_n_to_index(1,-1)] * phis[cls.cls_k_n_to_index(1,1)])**2
        return a
    
    def mean_curvature(self, phis: np.ndarray)-> np.complex64:
        return self.__class__.cls_mean_curvature(phis)
    
    @classmethod
    def cls_mean_curvature(cls, phis: np.ndarray)-> np.complex64:
        # Formula depicted in `paper.pdf` uses 2-Wavejet (or Wavejet of order 2)
        if(len(phis) < cls.cls_k_n_to_index(2,2)+1 ):
            raise Exception("Order of provided Wavejet too low. Please provide at least 2-Wavejet")
        return (2 * phis[cls.cls_k_n_to_index(2,0)] * ( 1 + 4 * phis[cls.cls_k_n_to_index(1,-1)] * phis[cls.cls_k_n_to_index(1,1)]) + 4 * phis[cls.cls_k_n_to_index(2,-2)] * phis[cls.cls_k_n_to_index(1,1)]**2 + 4 * phis[cls.cls_k_n_to_index(2,2)] * phis[cls.cls_k_n_to_index(1,-1)]**2) / (1 + 4 * phis[cls.cls_k_n_to_index(1,-1)] * phis[cls.cls_k_n_to_index(1,1)])**(3/2)
    
    def coef_a0s(self, phis: np.ndarray, s: float)-> float:
        k_max,_= self.index_to_k_n(len(phis)-1)
        result = 0.
        for k in range(2, k_max+1):
            result += (phis[self.k_n_to_index(k,0)] * s**(k+2)) / ( k + 2 )
        return result
    
    def enhance_position(self, p_idx: int, normal: np.ndarray, user_coef: float, phis: np.ndarray, radius_s: float)-> np.ndarray:
        return self.only_coordinates()[p_idx] - (phis[self.k_n_to_index(0,0)] + 2 * np.pi * (user_coef -1) * self.coef_a0s(phis, radius_s)) * normal
    
    def coef_a1s(self, phis: np.ndarray, s: float)-> float:
        k_max,_= self.index_to_k_n(len(phis)-1)
        result = 0.
        for k in range(3, k_max+1):
            result += (phis[self.k_n_to_index(k,1)] * s**(k+2)) / ( k + 2 )
        return result

    def tangent_phis(self, phis: np.ndarray)-> np.ndarray:
        for idx in range(len(phis)):
            k,n = self.index_to_k_n(idx)
            new_phi = complex()
            for j in range(1, k-1):
                for p in range(-(k-j), k-j+1):
                    m = n-p
                    if np.abs(m) > j:
                        continue
                    new_phi += (phis[self.k_n_to_index(k-j, p)] / 2j) * (phis[self.k_n_to_index(j+1, m+1)] * ( m + j + 2 ) + phis[self.k_n_to_index(j+1, m-1)] * ( m - j - 2 ))
            phis[idx] -= new_phi

        return np.array(phis)
    
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