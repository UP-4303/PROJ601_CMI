from itertools import combinations
from typing import List, Tuple
from collections import deque
import numpy as np

class Graph:
    """
    Really lightweight graph implementation
    Used for variant of breadth-first search on WavefrontOBJ
    Does NOT support weight, as it is only used for BFS
    """

    # Can't use ndarray, as 2nd dimension will be inhomogeneous in size
    adjs: List[List[int]]

    # Type checking can't be used for wavefront as it would cause an import cycle
    def __init__(self, wavefront):
        self.adjs = [[] for _ in range(len(wavefront.only_coordinates()))]

        for face in wavefront.only_faces():
            for idx_1, idx_2 in combinations(face, 2):
                self.adjs[idx_1].append(idx_2)
                self.adjs[idx_2].append(idx_1)

    def bfs(self, seen_count_stop: int, seen: List[int], to_treat: 'deque[int]')-> Tuple[List[int], 'deque[int]']:
        """
        Seen length may be bigger thant seen_count_stop
        This version of BFS does not keep track of previous array as we do not use it for pathfinding
        As we do not intend to scan the whole graph, seen is not a List[bool] of self.adjs.size length, but a List[int] containing all visited idx
        """
        while len(seen) < seen_count_stop:
            if(to_treat == deque([])):
                raise Exception("Not enough points for graph BSF")
            s = to_treat.popleft()

            for v in self.adjs[s]:
                if not v in seen:
                    seen.append(v)
                    to_treat.append(v)
        return seen, to_treat
