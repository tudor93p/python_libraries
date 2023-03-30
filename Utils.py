import numpy as np
import sympy as sp
from numpy import linalg as la
import itertools, json 


import Algebra


#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj): 

        if isinstance(obj, np.ndarray):
            return obj.tolist()

        if isinstance(obj, np.int64):
            return int(obj)

        return json.JSONEncoder.default(self, obj) 



#===========================================================================#
#
#   Get a parameter from a set of dicttionaries 
#
#---------------------------------------------------------------------------#


def prioritized_get(*dicts):

    def g(name, last_priority=None):
    
        for d in dicts:
           
            if name in d and d[name] is not None: 

                return d[name] 
    
    
        return last_priority

    return g


#===========================================================================#
#
# Semnificative digits
#
#---------------------------------------------------------------------------#

def NrDigits(x):

    if isinstance(x,int) or abs(x-int(x))<1e-10:
        return 0

    return 1 + NrDigits(10*x)


def OrderMagnitude(x):

  a = np.abs(x)

  if a < 1e-20:

    return 1

  try:
    l = np.abs(np.log10(a))

    return int(-np.ceil(l) if a<1 else np.floor(l)+1)

  except ValueError:
    return int(a<1)*2-1

  except OverflowError:
    return int(a<1)*2-1

def Round_toMaxOrder(x,n=3):

  X = []

  for xi in x:

      if xi in [None,np.inf,np.nan] or np.isnan(xi) or np.isinf(xi):

          X.append(0)

      elif isinstance(xi,int) or isinstance(xi,float):

          X.append(xi)

      else:
          print("##########\t",type(xi),xi)
          X.append(0)
      

  max_ord = OrderMagnitude(np.max(np.abs(X)))
 
  n -= max_ord + (max_ord<0) 


  if n <= 0: 
  
    return np.array(np.round(X,n), dtype=int)

  return np.round(X,n)






#===========================================================================#
#
# 1D x,y -> mgrid X,Y
#
#---------------------------------------------------------------------------#

def mgrid_from_1D(x,y=None,extend=True):

  if y is None:
      return mgrid_from_1D(x,x,extend)

  if extend:
    x,y = [np.append(a,a[-1] + np.mean(np.diff(a))) for a in [x,y]]
                # otherwise pcolormesh ignores last row and column of Z

  X = np.repeat(np.reshape(x,(-1,1)),len(y),axis=1)

  Y = np.repeat(np.reshape(y,(1,-1)),len(x),axis=0)

  return X,Y



#===========================================================================#
#
# rescale an array to the interval [m,M]
#
#---------------------------------------------------------------------------#

def Rescale(A,mM=[0,1],mM_A=None):

  m,M = Algebra.minmax(mM)

  mA,MA = Algebra.minmax(A if mM_A is None else mM_A)

  return (np.array(A)-mA)*(M-m)/(MA-mA) + m







#===========================================================================#
#
# Adaptive subdivisions
#
#---------------------------------------------------------------------------#

#
def Adaptive_Sampling(func, points, tol=0.05, min_points=16, max_level=16,
                    sample_transform=None):#,vectorized=True):
    """
    Sample a 1D function to given tolerance by adaptive subdivision.

    The result of sampling is a set of points that, if plotted,
    produces a smooth curve with also sharp features of the function
    resolved.

    Parameters
    ----------
    func : callable
        Function func(x) of a single argument. It is assumed to be vectorized.
    points : array-like, 1D
        Initial points to sample, sorted in ascending order.
        These will determine also the bounds of sampling.
    tol : float, optional
        Tolerance to sample to. The condition is roughly that the total
        length of the curve on the (x, y) plane is computed up to this
        tolerance.
    min_point : int, optional
        Minimum number of points to sample.
    max_level : int, optional
        Maximum subdivision depth.
    sample_transform : callable, optional
        Function w = g(x, y). The x-samples are generated so that w
        is sampled.

    Returns
    -------
    x : ndarray
        X-coordinates
    y : ndarray
        Corresponding values of func(x)

    Notes
    -----
    This routine is useful in computing functions that are expensive
    to compute, and have sharp features --- it makes more sense to
    adaptively dedicate more sampling points for the sharp features
    than the smooth parts.

    Examples
    --------
    >>> def func(x):
    ...     '''Function with a sharp peak on a smooth background'''
    ...     a = 0.001
    ...     return x + a**2/(a**2 + x**2)
    ...
    >>> x, y = sample_function(func, [-1, 1], tol=1e-3)

    >>> import matplotlib.pyplot as plt
    >>> xx = np.linspace(-1, 1, 12000)
    >>> plt.plot(xx, func(xx), '-', x, y[0], '.')
    >>> plt.show()

    """
#
#    def func2(x):
#
#      if np.ndim(x) == 0:
#        return f(x)
#
#      f = np.zeros_like(x)
#
#      for i in np.ndindex(*x.shape):
#        f[i] = x[i]
#
#      return f
#
#    F = func if vectorized else func2
  
    return _Adaptive_Sampling(func, points, values=None, mask=None, depth=0,
                            tol=tol, min_points=min_points, max_level=max_level,
                            sample_transform=sample_transform)

def _Adaptive_Sampling(func, points, values=None, mask=None, tol=0.05,
                     depth=0, min_points=16, max_level=16,
                     sample_transform=None):
    points = np.asarray(points)

    if values is None:
        values = np.atleast_2d(func(points))

#    if (np.abs(values-np.mean(values)) < 1e-8).all():
#      return points.reshape(-1),values.reshape(-1)

    if mask is None:
        mask = Ellipsis

    if depth > max_level:
        # recursion limit
        return points, values

#    x_a = points[...,:-1][...,mask]
#    x_b = points[...,1:][...,mask]

    x_a = points[...,:-1][mask]
    x_b = points[...,1:][mask]


    x_c = .5*(x_a + x_b)
    y_c = np.atleast_2d(func(x_c))

    x_2 = np.r_[points, x_c]
    y_2 = np.r_['-1', values, y_c]
    j = np.argsort(x_2)

    x_2 = x_2[...,j]
    y_2 = y_2[...,j]


    # -- Determine the intervals at which refinement is necessary

    if len(x_2) < min_points:
        mask = np.ones([len(x_2)-1], dtype=bool)
    else:
        # represent the data as a path in N dimensions (scaled to unit box)
        if sample_transform is not None:
            y_2_val = sample_transform(x_2, y_2)
        else:
            y_2_val = y_2

        p = np.r_['0',
                  x_2[None,:],
                  y_2_val.real.reshape(-1, y_2_val.shape[-1]),
                  y_2_val.imag.reshape(-1, y_2_val.shape[-1])
                  ]

        sz = (p.shape[0]-1)//2

        xscale = x_2.ptp(axis=-1)
        yscale = abs(y_2_val.ptp(axis=-1)).ravel()

        p[0] /= xscale
        p[1:sz+1] /= yscale[:,None]
        p[sz+1:]  /= yscale[:,None]

        # compute the length of each line segment in the path
        dp = np.diff(p, axis=-1)
        s = np.sqrt((dp**2).sum(axis=0))
        s_tot = s.sum()

        # compute the angle between consecutive line segments
        dp /= s
        dcos = np.arccos(np.clip((dp[:,1:] * dp[:,:-1]).sum(axis=0), -1, 1))

        # determine where to subdivide: the condition is roughly that
        # the total length of the path (in the scaled data) is computed
        # to accuracy `tol`
        dp_piece = dcos * .5*(s[1:] + s[:-1])
        mask = (dp_piece > tol * s_tot)

        mask = np.r_[mask, False]
        mask[1:] |= mask[:-1].copy()


    # -- Refine, if necessary

    if mask.any():
        return _Adaptive_Sampling(func, x_2, y_2, mask, tol=tol, depth=depth+1,
                                min_points=min_points, max_level=max_level,
                                sample_transform=sample_transform)
    else:
        return x_2, y_2




#===========================================================================#
#
# numpy amin and argmin
#
#---------------------------------------------------------------------------#

def arg_and_minmax(A,which="min",axis=0):

    if   which == "min": f = np.argmin
    elif which == "max": f = np.argmax
    else: raise ValueError("Please provide either 'min' or 'max'!")

    A = np.array(A)

    axes = np.arange(A.ndim)

#    rest_axes = axes if axis is None else axes[axes!=axis]

    min_idx = f(A,axis=axis)


    rest_idx = [range(A.shape[l]) for l in axes[axes!=axis]]
#rest_axes]

    indices = np.insert(np.meshgrid(*rest_idx,indexing='ij'),axis,min_idx,0)


    return min_idx,A[tuple(np.array(indices,dtype=int))]


#===========================================================================#
#
# Remove duplicates in list
#
#---------------------------------------------------------------------------#

def Remove_Duplicates(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


#===========================================================================#
#
# add the 3rd dimension with a certain value
#
#---------------------------------------------------------------------------#

def Add_dim(a,val=0):


  return np.hstack((a,np.ones((len(a),1),dtype=type(val))*val))

#===========================================================================#
#
# assign value if it's not None
#
#---------------------------------------------------------------------------#

def Assign_Value(input_value,std_value,args_std=None):

  if input_value is not None:
    return input_value

  return std_value if args_std is None else std_value(*args_std)




#===========================================================================#
#
# Unique items of a list, uniqueness being defined via "rule"
#
#---------------------------------------------------------------------------#

def unique(V,rule = lambda v,V: [v==vi for vi in V]):

  if len(V) <=1:
    return V

  inds = np.ones(len(V),dtype=bool)

  i = 0

  while True:

    js = np.nonzero(rule(V[i],V))[0]

    inds[js[js!=i]] = False

    if not any(inds[i+1:]):
      return V[inds]
  
    i += 1 + np.argmax(inds[i+1:])


#===========================================================================#
#
#  Vectors of integers, with elements going between two limits 
#
#---------------------------------------------------------------------------#

def vectors_of_integers(dim,end,start=None,sort=False):


  if type(end)==int:     end = np.ones(dim,dtype=int)*end
  if type(start)==int: start = np.ones(dim,dtype=int)*start

  end = np.array(end)
  start = np.array(Assign_Value(start,-end))

  out = start + np.array(list(np.ndindex(*(end-start+1))),dtype=int)

  if sort == True:

    return out[np.argsort(la.norm(out,axis=1))]
 
  return out


#===========================================================================#
#
# Implements the numerical 'equal', which means is approximately ...
#
#---------------------------------------------------------------------------#


def Same(x,y,n=4):

  return (np.abs(np.reshape(x,(-1))-np.reshape(y,(-1))) < 1/10**n).all()


#  return np.mean(np.abs(np.reshape(np.array(x)-np.array(y),(-1))))<1./10**n



#===========================================================================#
#
#  Returns a path of n points which connects the inputed points
#
#---------------------------------------------------------------------------#

def path_connect(points,n,end_point=True,bounds=[0,1],fdist=lambda x:x):

  points = np.array(points)

  n = n - end_point

  dist = fdist(la.norm(np.diff(points,axis=0),axis=1))
  

  dist = dist/np.sum(dist)

  cumdist = np.cumsum(np.append(0,dist)) 


  ns_ = [max(1,int(ni)) for ni in np.round(dist*n)]

  while sum(ns_) > n: ns_[np.argmax(ns_)] -= 1
  while sum(ns_) < n: ns_[np.argmin(ns_)] += 1
  
  ep = lambda i : (i==len(points)-2)*end_point

#  t = lambda i : np.linspace(0., 1., ns_[i] + ep(i), endpoint = ep(i))
  
  ts = [np.linspace(0., 1., ns_[i] + ep(i), endpoint = ep(i)) for i in range(len(points)-1)]


  segment = lambda i : np.einsum('jk,jl->kl',[1-ts[i],ts[i]],points[i:i+2])



  path = np.vstack(([segment(i) for i in range(len(points)-1)]))

  x = np.concatenate(tuple((1-t)*cumdist[i] + t*cumdist[i+1] for (i,t) in enumerate(ts))) 
 

  xticks = bounds[0] + cumdist * np.diff(bounds)

  return path, xticks, bounds[0] + x * np.diff(bounds)










#############################################################################















