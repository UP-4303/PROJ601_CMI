import numpy as np
from typing import Tuple, List

def point_cart_to_pol(p: np.ndarray)-> np.ndarray:
    return np.array(cart_to_pol(p[0], p[1]))

def cart_to_pol(x: float, y:float)-> Tuple[float, float]:
    theta = np.arctan2(y, x)
    r = np.hypot(x, y)
    return r, theta

def pol_to_cart(theta: float, r: float)-> Tuple[float, float]:
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y