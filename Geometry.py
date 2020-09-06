#############################################################################
#
#               Operations with lines and polygons
#
#############################################################################


import numpy as np
import itertools
import Algebra,Utils


#===========================================================================#
#
#   tests if a point is Left|On|Right of an infinite line.
#
#---------------------------------------------------------------------------#
"""
   Input: three points P0, P1, and P2
   Return: >0 for P2 left of the line through P0 and P1
           =0 for P2 on the line
           <0 for P2 right of the line
   See: the January 2001 Algorithm "Area of 2D and 3D Triangles and Polygons"
"""

def is_left(P0, P1, P2):
    return (P1[0] - P0[0]) * (P2[1] - P0[1]) - (P2[0] - P0[0]) * (P1[1] - P0[1])





#===========================================================================#
#
# Routines for performing the "point in polygon" inclusion test
# Copyright 2001, softSurfer (www.softsurfer.com)
# translated to Python by Maciej Kalisiak <mac@dgp.toronto.edu>
#   a Point is represented as a tuple: (x,y)
#
#---------------------------------------------------------------------------#

"""
 cn_PnPoly(): crossing number test for a point in a polygon
     Input:  P = a point,
             V[] = vertex points of a polygon
     Return: 0 = outside, 1 = inside
    This code is patterned after [Franklin, 2000]
"""

def PointInPolygon_cn(P, V):


    cn = 0    # the crossing number counter

    # repeat the first vertex at end
    V = tuple(V[:])+(V[0],)

    # loop through all edges of the polygon
    for i in range(len(V)-1):   # edge from V[i] to V[i+1]
        if ((V[i][1] <= P[1] and V[i+1][1] > P[1])   # an upward crossing
            or (V[i][1] > P[1] and V[i+1][1] <= P[1])):  # a downward crossing
            # compute the actual edge-ray intersect x-coordinate
            vt = (P[1] - V[i][1]) / float(V[i+1][1] - V[i][1])
            if P[0] < V[i][0] + vt * (V[i+1][0] - V[i][0]): # P[0] < intersect
                cn += 1  # a valid crossing of y=P[1] right of P[0]

    return cn % 2 == 1  ###  0 if even (out), and 1 if odd (in)

"""
 wn_PnPoly(): winding number test for a point in a polygon
     Input:  P = a point,
             V[] = vertex points of a polygon
     Return: wn = the winding number (=0 only if P is outside V[])
""" 

def PointInPolygon_wn(P, V):

    wn = 0   # the winding number counter

    # repeat the first vertex at end
    V = tuple(V[:]) + (V[0],)

    # loop through all edges of the polygon
    for i in range(len(V)-1):     # edge from V[i] to V[i+1]
        if V[i][1] <= P[1]:        # start y <= P[1]
            if V[i+1][1] > P[1]:     # an upward crossing
                if is_left(V[i], V[i+1], P) > 0: # P left of edge
                    wn += 1           # have a valid up intersect
        else:                      # start y > P[1] (no test needed)
            if V[i+1][1] <= P[1]:    # a downward crossing
                if is_left(V[i], V[i+1], P) < 0: # P right of edge
                    wn -= 1           # have a valid down intersect
#    return wn
    return wn != 0



#===========================================================================#
# 
# Make a polygon convex by removing the vertices inside
#
#---------------------------------------------------------------------------#

def MakePolygonConvex(p):

  p = Order_PolygonVertices(p)

  while len(p)>=3:

    alright = True
   
    for (i,P) in enumerate(p):
  
      newp = p[np.arange(len(p))!=i]
  
      if PointInPolygon_wn(P, newp):
 
        alright = False
  
        p = newp 
  
        break
     
  
    if alright:
      return p

  raise Exception("Could not make the polygon convex")


#===========================================================================#
# 
# Distance between a line and a point (or many vs. many)  
#
#---------------------------------------------------------------------------#

# dist(\vek{R} = \vek{a} + t \vek{n},\vek{p}) = || (a-p) - ((a-p).n)n||
# dist(line(point A, direction u),P) = || AP x u || / ||u||

def Distance_PointLine(a,n,p):

  a,n,p = np.array(a),np.array(n)/la.norm(n),np.array(p)

  if np.ndim(n)!=1:
    raise Exeception("Not yet implemented")


  if a.ndim == p.ndim == 1:

    D = a-p

    parralel = np.dot(D,n)

    return la.norm(D - parallel*n)

  
  if a.ndim == 1: a = a.reshape(1,-1)
  if p.ndim == 1: p = p.reshape(1,-1)


  D = np.zeros((a.shape[0],p.shape[0],a.shape[1]))
 
  for (i,Di) in enumerate(Algebra.OuterDiff(a,p)):
    D[:,:,i] = Di

  parallel = D.dot(n)

  for k in range(D.shape[-1]):

    D[:,:,k] -= parallel*n[k]


  return la.norm(D,axis=-1)





#===========================================================================#
# 
# Maximize the intersection between a set of lines and a polygon
#
#---------------------------------------------------------------------------#

def Maximize_ContactSurface(starting_points,direction,vertices=None,ordered_vertices=None):

    from shapely.geometry import LinearRing,MultiPoint,LineString
    from shapely.ops import nearest_points
  
    if ordered_vertices is not None:
        V = ordered_vertices
    else:
        V = Order_PolygonVertices(vertices)
   
    poly_sides,poly_verts = LinearRing(V),MultiPoint(V)
  

    shifts = np.zeros((2,2))

    for start in starting_points:

        line = LineString([start,start+direction])
       
        if poly_sides.intersection(line).is_empty:
        
            Ps = [p.coords for p in nearest_points(line,poly_verts)]

            s = np.diff(Ps,axis=0).reshape(-1)
            
            for (i,shift) in enumerate(shifts):

                if np.dot(shift,s) >= np.dot(shift,shift):

                    shifts[i] = s

    norms = np.linalg.norm(shifts,axis=1)

    if any(norms<1e-10):
        return shifts[np.argmax(norms),:]

    return shifts[np.argmin(norms),:]





#===========================================================================#
# 
#    Among the intersections between a set of lines and a polygon,
#       choose that one which is closest to a poygon vertex
#
#---------------------------------------------------------------------------#

def ClosestContact_LinesPolygon(starting_points,direction,vertices=None,ordered_vertices=None):

    from shapely.geometry import LinearRing,Point,LineString
    from shapely.geometry import MultiPoint,MultiLineString,GeometryCollection
    from shapely.ops import nearest_points
    
    if ordered_vertices is not None:
        V = ordered_vertices
    else:
        V = Order_PolygonVertices(vertices)
    
    poly_sides,poly_verts = LinearRing(V),MultiPoint(V)
    
    
    def get_coord(A):
    
        if isinstance(A,Point):
        
            return np.array(nearest_points(poly_verts,A)[0].coords[:])
        
        if isinstance(A,LineString):
        
            return np.array(A.coords[:])
        
        if any([isinstance(A,T) for T in [MultiPoint,GeometryCollection,MultiLineString]]):
        
            return np.vstack([get_coord(a) for a in A])
        
        raise Exception("Unexpected type",type(A))
    
    
    out = {}
    
    for (i,start) in enumerate(starting_points):
    
        line = LineString([start,start+direction])
        
        intersection = poly_sides.intersection(line)
        
        
        if intersection.is_empty==False:
        
            Rs = get_coord(intersection)
            
            dists = np.linalg.norm(Rs - start,axis=1)
            
            md = np.min(dists)
            
            if "dist" in out.keys() and out["dist"] < md:
            
                continue
            
            else:
            
                out = {	"start":start,
                        "index":i,
                        "dist":md,
                        "stop":Rs[np.argmin(dists)]
                        }
            
    return out


  
  

  


#===========================================================================#
#
# Order trigonometrically the vertices of a 2D polygon 
#
#---------------------------------------------------------------------------#

def Order_PolygonVertices(V):

  V = np.array(V)

  return V[np.argsort(np.arctan2(*(V-np.mean(V,axis=0)).T[::-1]))]

  



#===========================================================================#
#
# Generate vertices of a d-dimensional body based on d vectors
#
#---------------------------------------------------------------------------#

def BodyVertices_fromVectors(v):

  body = [ np.zeros(len(v),dtype=type(np.reshape(v,-1)[0])) ]

  for d in range(len(v)):

    body += [np.sum(x,axis=0) for x in itertools.combinations(v,d+1)]

  return np.array(body)



#===========================================================================#
#
# Generate polygon based on two vertices
#
#---------------------------------------------------------------------------#

def RegularPolygon_from2Vertices(A,B,inp=None):

  if type(inp)==int:

    n = inp

    d12 = np.array(B)-np.array(A)
  
    Rot = Rotation_Matrix(2*np.pi/n)
  
    verts = [A]
  
    side = d12
  
    for j in range(n-1):
   
      verts.append(verts[-1]+side)
      side = Rot@side 
  

  else:

    O = Assign_Value(inp,np.zeros(len(A)))

    phi = Algebra.Angle_btw_Vectors(A,B,O)

    phis = np.arange(np.ceil(np.abs(2*np.pi/phi)))*phi

    verts = [O + Algebra.Rotation_Matrix(p)@(A-O) for p in phis]

  return np.array(verts)




















#############################################################################
# end
