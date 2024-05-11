import numpy as np
from scipy.special import comb, factorial
from scipy.interpolate import griddata
import polyscope as ps
from math import ceil,floor

from typing import Tuple, List

from wavefront import WavefrontOBJ

class WavejetComputation(WavefrontOBJ):
    precision_length: float # Length (and height) of the grid around each point where will be computed the plan
    precision_size: int # Number of values computed along each axis when computing the plan

    def __init__(self, precision_length: float= 3, precision_size: int= 3, default_mtl='default_mtl' ):
        super().__init__(default_mtl)
        if(precision_size % 2 == 0):
            raise(ValueError("precision_size should be an odd number, to keep the origin point in the plan"))

        self.precision_length = precision_length
        self.precision_size = precision_size

    def compute_plan(self, p: Tuple[float, float, float])-> np.ndarray:
        points_with_value = []
        values = []
        for point in self.only_coordinates():
            points_with_value.append((point[0], point[1]))
            values.append(point[2] - p[2])

        x = np.linspace(p[0] - self.precision_length/2, p[0] + self.precision_length/2, self.precision_size)
        y = np.linspace(p[1] - self.precision_length/2, p[1] + self.precision_length/2, self.precision_size)
        X, Y = np.meshgrid(x, y)

        return griddata(points_with_value, values, (X,Y), 'cubic')

    def compute_partial_derivative(self, f: np.ndarray, k: int, j: int)-> np.ndarray:
        derivative: np.ndarray = f
        for _ in range(k-j):
            derivative = np.gradient(derivative, axis=0)
        for _ in range(j):
            derivative = np.gradient(derivative, axis=1)
        return derivative

    def compute_wavejet_at_point(self, k: int, n: int, p: Tuple[float, float, float])-> complex:
        phi_kn: complex = complex()
        plan = self.compute_plan(p)
        for j in range(k+1):
            derivative_at_point = self.compute_partial_derivative(plan, k, j)[int(self.precision_size/2), int(self.precision_size/2)]

            b_kjn = 0
            if (k + n) % 2 != 0:
                for h in range((n-k)//2 + 1):
                    b_kjn += comb(k-j, h) * comb(j, (n-k)//2 - h) * (-1)**h
                
                b_kjn *= (1 / (2**k * 1j**j))

            phi_kn += (1 / (factorial(j) * factorial(k-j))) * b_kjn * derivative_at_point

        return phi_kn
    
    def compute_wavejets(self, k: int, n: int)-> List[complex]:
        return [self.compute_wavejet_at_point(k, n, p) for p in self.only_coordinates()]

# Example usage
if __name__ == "__main__":
    ps.init()

    # Load a 3D model
    wc = WavejetComputation.load_obj('examples_mesh/octopus.obj')
    original_mesh: ps.SurfaceMesh = ps.register_surface_mesh("original mesh", wc.only_coordinates(), wc.only_faces())

    # Compute wavejets
    wavejets: List[complex] = wc.compute_wavejets(2,0)

    realwjs: List[float] = []
    for i in range(len(wavejets)):
        realwjs.append(wavejets[i].real)
    maxwj = np.max(realwjs)
    minwj = np.min(realwjs)
    print(realwjs)
    print(maxwj, minwj)

    # normalized for colorization
    for i in range(len(realwjs)):
        realwjs[i] = (realwjs[i] - minwj) / (maxwj - minwj)
    
    print(realwjs)
    colors = np.array([[realwjs[i], realwjs[i], realwjs[i]] for i in range(len(realwjs))])
    original_mesh.add_color_quantity("red wavejets", colors, enabled=True)

    # Show result
    ps.show()