# wavefront.py

# Mostly not my code, source :
# https://codimd.math.cnrs.fr/s/TmrOWEmyJ itself based on :
# https://jamesgregson.ca/loadsave-wavefront-obj-files-in-python.html

import numpy as np
from typing import Union, List

class WavefrontOBJ:
    path: Union[None, str]

    def __init__( self, default_mtl='default_mtl' ):
        self.path      = None               # path of loaded object
        self.mtllibs   = []                 # .mtl files references via mtllib
        self.mtls      = [ default_mtl ]    # materials referenced
        self.mtlid     = []                 # indices into self.mtls for each polygon
        self.vertices  = []                 # vertices as an Nx3 or Nx6 array (per vtx colors)
        self.normals   = []                 # normals
        self.texcoords = []                 # texture coordinates
        self.polygons  = []                 # M*Nv*3 array, Nv=# of vertices, stored as\ vid,tid,nid (-1 for N/A)

    def only_coordinates( self )-> np.ndarray:
        V = np.ndarray((len(self.vertices), 3))
        for i in range(len(self.vertices)):
            V[i][0]= self.vertices[i][0]
            V[i][1]= self.vertices[i][1]
            V[i][2]= self.vertices[i][2]
        return V
    
    def only_faces( self )-> List[List[int]]:
        all_faces = []
        for i in range(len(self.polygons)):
            face = []
            for j in range(len(self.polygons[i])):
                face.append(self.polygons[i][j][0])
            all_faces.append(face)
        return all_faces
    
    # Apply on one face
    def face_indices_to_values(self, face: List[int])-> List[np.ndarray]:
        coo = self.only_coordinates()
        return [coo[point] for point in face]
    
    # Apply on a list of faces
    def faces_points_indices_to_values(self, faces: np.ndarray)-> np.ndarray:
        return np.apply_along_axis(self.face_indices_to_values, 0, faces)
    
    def neighbor_faces( self, p_idx: int)-> List[List[np.ndarray]]:
        faces: List = []
        for face in self.only_faces():
            for indice in face:
                if p_idx == indice:
                    faces.append(self.face_indices_to_values(face))
                    break
        return faces

    def set_coordinates( self, new_coordinates: np.ndarray )-> None:
        for i in range(len(new_coordinates)):
            for j in range(len(new_coordinates[i])): # Not tested with more than 3D, don't even know if an obj in more than 3D can be loaded
                self.vertices[i][j] = new_coordinates[i][j]
    
    @classmethod
    def cls_load_obj( cls, filename: str, default_mtl='default_mtl', triangulate=False ):
        obj = cls(default_mtl)
        obj.load_obj(filename, default_mtl, triangulate)
        return obj
    
    def load_obj( self, filename: str, default_mtl='default_mtl', triangulate=False ):
        """
        Reads a .obj file from disk and returns a WavefrontOBJ instance

        Handles only very rudimentary reading and contains no error handling!

        Does not handle:
            - relative indexing
            - subobjects or groups
            - lines, splines, beziers, etc.
        """
        # parses a vertex record as either vid, vid/tid, vid//nid or vid/tid/nid
        # and returns a 3-tuple where unparsed values are replaced with -1
        def parse_vertex( vstr ):
            vals = vstr.split('/')
            vid = int(vals[0])-1
            tid = int(vals[1])-1 if len(vals) > 1 and vals[1] else -1
            nid = int(vals[2])-1 if len(vals) > 2 else -1
            return (vid,tid,nid)

        with open( filename, 'r' ) as objf:
            self.path = filename
            cur_mat = self.mtls.index(default_mtl)
            for line in objf:
                toks = line.split()
                if not toks:
                    continue
                if toks[0] == 'v':
                    self.vertices.append( [ float(v) for v in toks[1:]] )
                elif toks[0] == 'vn':
                    self.normals.append( [ float(v) for v in toks[1:]] )
                elif toks[0] == 'vt':
                    self.texcoords.append( [ float(v) for v in toks[1:]] )
                elif toks[0] == 'f':
                    poly = [ parse_vertex(vstr) for vstr in toks[1:] ]
                    if triangulate:
                        for i in range(2,len(poly)):
                            self.mtlid.append( cur_mat )
                            self.polygons.append( (poly[0], poly[i-1], poly[i] ) )
                    else:
                        self.mtlid.append(cur_mat)
                        self.polygons.append( poly )
                elif toks[0] == 'mtllib':
                    self.mtllibs.append( toks[1] )
                elif toks[0] == 'usemtl':
                    if toks[1] not in self.mtls:
                        self.mtls.append(toks[1])
                    cur_mat = self.mtls.index( toks[1] )
    
    def save_obj( self, filename: str ):
        """
        Saves a WavefrontOBJ object to a file
        Warning: Contains no error checking!
        """
        with open( filename, 'w' ) as ofile:
            for mlib in self.mtllibs:
                ofile.write('mtllib {}\n'.format(mlib))
            for vtx in self.vertices:
                ofile.write('v '+' '.join(['{}'.format(v) for v in vtx])+'\n')
            for tex in self.texcoords:
                ofile.write('vt '+' '.join(['{}'.format(vt) for vt in tex])+'\n')
            for nrm in self.normals:
                ofile.write('vn '+' '.join(['{}'.format(vn) for vn in nrm])+'\n')
            if not self.mtlid:
                self.mtlid = [-1] * len(self.polygons)
            poly_idx = np.argsort( np.array( self.mtlid ) )
            cur_mat = -1
            for pid in poly_idx:
                if self.mtlid[pid] != cur_mat:                
                    cur_mat = self.mtlid[pid]
                    ofile.write('usemtl {}\n'.format(self.mtls[cur_mat]))
                pstr = 'f '
                for v in self.polygons[pid]:
                    # UGLY!
                    vstr = '{}/{}/{} '.format(v[0]+1,v[1]+1 if v[1] >= 0 else 'X', v[2]+1 if v[2] >= 0 else 'X' )
                    vstr = vstr.replace('/X/','//').replace('/X ', ' ')
                    pstr += vstr
                ofile.write( pstr+'\n')