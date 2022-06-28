import numpy as np
import sympy as sp
from sympy.physics.quantum import TensorProduct as spkron
#from numpy import linalg as la
#import itertools
#import time
#import matplotlib.pyplot as plt


#import Utils


##===========================================================================#
##
##
##
##---------------------------------------------------------------------------#
#
#def FindRoot_Bisection(f, u10, u20=None, val10=None, val20=None, 
#                            a=0, b=None, move_right=None, move_left=None,
#                            max_iter=1000, get_fval=False):
#
#    b = Utils.Assign_Value(b, a+1e-6)
#
#
#
#    def bisection(u1, u2, iteration=0):
#
#        um = (u1+u2)/2
#
#        val = f(um)
#
#        if val < a and iteration < max_iter: 
#            return bisection(um, u2, iteration+1)
#
#        elif val > b and iteration < max_iter:
#            return bisection(u1, um, iteration+1)
#
#
#        if get_fval: 
#            return um, val
#
#        return um
#
#
#
#    uvalu = lambda arg: (arg, f(arg))
#
#
#
## --- Find the interval, if not provided (properly) --- #
#
#    def interval(u1, u2, val1, val2):
#
#        for (u,v) in [(u1,val1), (u2,val2)]:
#
#            if a <= v <= b:
#
#                if get_fval:
#
#                    return uvalu(u)
#
#                return u 
#
#
#        if val1<a and val2>b:
#
#            return bisection(u1, u2)
#
#        if val1 > a:
#    
#            if val1 > b:
#                (u2,val2) = (u1,val1)
#    
#            (u1,val1) = uvalu(move_left(u1))
#
#
#
#        if val2 < b:
#    
#            if val2 < a:
#                (u1,val1) = (u2,val2)
#
#            (u2,val2) = uvalu(move_right(u2))
#
#        return interval(u1, u2, val1, val2)
#
#
#
#    if val10 is None:
#
#        val10 = f(u10)
#
#
#
#
#    if u20 is None or abs(u10-u20)<1e-12:
#
#        return interval(u10, u10, val10, val10)
#
#    if val20 is None:
#
#        val20 = f(u20)
#  
#    return interval(u10, u20, val10, val20)
#
#
##===========================================================================#
##
## Both min and max
##
##---------------------------------------------------------------------------#
#
#def minmax(A,**kwargs):
#
#  if A is None:
#    return [None,None]
#
#  Aa = np.array(A)
#
#  if None in Aa:
#      return None 
#
#  return np.array([np.min(Aa,**kwargs),np.max(Aa,**kwargs)])
#
#
#
##===========================================================================#
## angle between vectors
##---------------------------------------------------------------------------#
#
#def Angle_btw_Vectors(A,B,O=None):
#
#  O = Utils.Assign_Value(O,np.zeros(len(A)))
#
#  return np.arctan2(np.append(np.cross(A-O,B-O),[])[-1],np.dot(A-O,B-O))
#
#
##===========================================================================#
## 
##---------------------------------------------------------------------------#
#
#
#def Invert_LattVect(vectors):
#
#
#  inverse = la.inv(np.array(vectors)[:,:len(vectors)]).T*2*np.pi
#
#  for _ in range(np.array(vectors).shape[1]-len(inverse)):
#  
#    inverse = Utils.Add_dim(inverse,0.)
#
#  return inverse
#

#==============================================================================
# Pauli Matrices
#------------------------------------------------------------------------------

def PauliMatrices():

  s0 = np.array([[1.,0],[0,1]],dtype=complex)
  sx = np.array([[0.,1],[1,0]],dtype=complex)
  sy = np.array([[0.,-1j],[1j,0]],dtype=complex)
  sz = np.array([[1.,0],[0,-1]],dtype=complex)

  return [s0,sx,sy,sz]


def spPauliMatrices():

  z,o,i = sp.Integer(0),sp.Integer(1),sp.I

  s0 = sp.Matrix([[o,z],[z,o]])
  sx = sp.Matrix([[z,o],[o,z]])
  sy = sp.Matrix([[z,-i],[i,z]])
  sz = sp.Matrix([[o,z],[z,-o]])

  return [s0,sx,sy,sz]

#===========================================================================#
#
# Dot product for two lists of sympy objects
#
#---------------------------------------------------------------------------#

def spDot(A,B):

  return sp.Add(*[a*b for (a,b) in zip(A,B)])

#===========================================================================#
#
# List of matrices as Kronecker products of Pauli matrices specified in strings
#
#---------------------------------------------------------------------------#

def valid_Pauli_kron(x):

    if isinstance(x,str):

        out = np.array(list(x), dtype=int)  

        if np.all(out<=3) and np.all(out>=0):
            return out 

    raise ValueError("Only strings between \'0\' and \'3\' are supported")
    

def rec_kron(kron,A):

    if len(A) == 1:
      return A[0]

    return kron(rec_kron(kron, A[:-1]), A[-1])



def Kronecker1(PM, kron, x, order=None):

    P = valid_Pauli_kron(x) 

    if len(P)==0:
        raise ValueError("At least one matrix is required for a Kronecker product")


    if order is None:

        return rec_kron(kron, [PM[p] for p in P])

    if callable(order):

        return rec_kron(kron, [PM[P[i]] for i in order(P)])

    return rec_kron(kron, [PM[P[i]] for i in order])



def Kronecker(X, symbolic=False, **kwargs):

    PM = spPauliMatrices() if symbolic else PauliMatrices()

    kron = spkron if symbolic else np.kron 

    if isinstance(X,str):

        if ' ' not in X:

            return Kronecker1(PM, kron, X, **kwargs)

        return Kronecker(X.split(' '), **kwargs, symbolic=symbolic)


    return [Kronecker1(PM, kron, x, **kwargs) for x in X]



  

##===========================================================================#
##
## Discrete Fourier Transform using sympy
##
##---------------------------------------------------------------------------#
#  
#
#def FourierTransform_Function(f,k,R,limits=(-sp.pi,sp.pi),toolong=5,nmax=0):
#
#  iszero = sp.Function('delta',real=True)
#
#  a = sp.Symbol('aux',real=True)
#  seriestype = type(sp.fourier_series(sp.sin(a),(a,-sp.pi,sp.pi)))
#
#
#  def transf_series(series,x,nmax=1):
#
#    if type(series) != seriestype:
#      return series*iszero(x),int(nmax)
#
#    a0, cos, sin  = series.a0, series.an, series.bn
#
#    if int(a0!=0) + len(cos) + len(sin) > toolong:
#      print("The Fourier series has too many terms!")
#
#    s  = a0*iszero(x)
#
#    s += sum([a*(iszero(x+n) + iszero(x-n)) for (n,a) in cos.items()])/2
#  
#    s += sum([b*(iszero(x+n) - iszero(x-n)) for (n,b) in sin.items()])/(2*sp.I)
#
#    return s, int(max([nmax]+list(cos.keys())+list(sin.keys())))
# 
#
#  for (xj,kj) in zip(np.append(R,[]),np.append(k,[])):
#
#    f,nmax = transf_series(sp.fourier_series(f,(kj,*limits)),xj,nmax)
#
#
#  return f,iszero,nmax
#
#  
#
#def FourierTransform(h,k,R,limits=(-sp.pi,sp.pi),toolong=5):
#
#
#  try:
#    h.shape
#  except:
#    return FourierTransform_Function(h,k,R,limits,toolong)
#
#  newh = sp.zeros(*h.shape)
#  nmax = int(0)
#
#  for index in np.ndindex(*h.shape):
#    newh[index],iszero,nmax = FourierTransform_Function(h[index],k,R,limits,toolong,nmax)
#
#  return newh,iszero,nmax
#
#
##def FourierTransformN(spH,spK,limits=(-sp.pi,sp.pi),toolong=5,tol=1e-5):
##
##  spR = sp.symbols(' '.join(['x'+str(i) for i in range(len(spK))]),real=True)
##
##  spH,iszero,nmax = FourierTransform(spH,spK,spR,limits,toolong)
##
##  iszeroN = lambda a: 1.0*(abs(float(a))<tol)
##
##  def numeric(npR):
##    return np.array(spH.subs({s:n for (s,n) in zip(spR,npR)}).replace(iszero,iszeroN)).astype(complex)
##
##  return nmax,numeric
##
#
#  
#
##==============================================================================
## Fourier Transform of e^{ik.R} x (1, cos(a kj), sin(a kj))
##------------------------------------------------------------------------------
#
##def Fourier_Transform(ftype,R,j=None,unit=1):
##
###  R= -R # needed to be consisent with the supp material, but it inverts space
##
##  if ftype == '1':
##    return Utils.Same(R,0)*1
##
##  else:
##    dim = len(R)
##
##    # in general: compare with dist to nn, nnn etc (instead of 1., -1.)
##
##    delta_1 = np.prod([Utils.Same(R[l],0)*1. for l in range(dim) if l!=j])
##    delta_2 = utils.Same(R[j],  unit)*1. 
##    delta_3 = Utils.Same(R[j], -unit)*1.
##
## 
##    if ftype == 'cos': 
##      return  delta_1 *  0.5*(delta_2 + delta_3)
##
##    elif ftype == 'sin':
##      return  delta_1 *(0.5/1.j)*(delta_2 - delta_3)
##
##  return 0 
##
##
###==============================================================================
## Vectors of integers perpendicular to a certain vector
##------------------------------------------------------------------------------
#
##def perpendicular_integer(vs):
##
##  vs = np.array(vs)
##
##  nr,dim = vs.shape
##
##  if nr == dim:
##
##    if Utils.Same(la.det(vs),0):
##      print('** Error: vectors not linearly independent **')
##      exit()
##    return []
##
##  all_perp = []
##
##  nmax = int(np.sum(np.abs(vs)))
##
##  for ind in np.ndindex(*[2*nmax+1 for _ in range(dim)]):
##    a = nmax - np.array(ind)
##
##    if (np.dot(vs,a)==0).all() and not Utils.Same(a,0):
##      all_perp.append(a)
##
##
##
##  magnitude_perp = 1000
##  n_perp = None
##  vm = np.matrix(vs)
##
##  for x in itertools.product(*[all_perp for _ in range(dim-nr)]):
##
##    magnitude = np.sum(np.abs(x))
##    det = la.det(np.vstack((np.matrix(x),vm)))
##
##    if not Utils.Same(det,0) and magnitude < magnitude_perp:
##      magnitude_perp = magnitude
##      n_perp = x
##
##  return list(n_perp)
#
#
#
#
##==============================================================================
## Computes the overlap matrix of two sets of wavefunctions (on lines)
##------------------------------------------------------------------------------
#
#def Overlap_Matrix(wf1,wf2=None):
#  wf2 = wf1.copy() if wf2 is None else wf2
#
#
## Calculates the matrix product of two sets of input wavefunctions
#  out =  np.matrix(np.conjugate(wf1))*(np.matrix(wf2).T)  # faster way
#
##  x = np.round(np.amax(np.abs(0.0001+out-np.diag(np.diag(out)))),4)
##  y = np.round(np.amax(np.abs(0.0001+np.diag(np.diag(out)))),4)
##  print('diag: ',y,'\toff-diag: ',x)
#
#  return out
#
#
##==============================================================================
## Modified Gram-Schmidt orthogonalization
##------------------------------------------------------------------------------ț
#
## it takes between [1,6]*10**(-4) seconds
#
#def mod_GramSchmidt(V):
#
#  V = np.array(V)
#
#  U = np.zeros_like(V)
#
#  for j in range(V.shape[0]):
#    
#    u = V[j].copy()
#
#    for i in range(j):
#
#      u -= np.vdot(U[i],u)*U[i]
#
#    norm = la.norm(u)
#
#    if norm < 1e-8: 
#      raise ValueError("Invalid input matrix for Gram-Schmidt!")
#    
#    U[j] = u/norm
#
#
#  return U
#
##==============================================================================
## GS orthogonalization
##------------------------------------------------------------------------------
#
#def GS_Orthog(psi): #WFs on lines
#  psi = np.matrix(psi)
#  tolerance = 1e-7
#  ov = np.abs(uij(psi,psi))
#  if np.amax(np.abs(ov - np.diag(np.diag(ov)))) > tolerance:
#    Q,R = la.qr(psi.T )  #WFs on columns
#    new_psi = Q.T
#    ov = np.abs(uij(new_psi,new_psi))
#
#    if np.amax(np.abs(ov - np.diag(np.diag(ov)))) > tolerance:
#      print('WFs not orthogonal after GS: ')
#      return 0
#
#     
#    return new_psi
# 
#  return psi
#
#
#
##===========================================================================#
## 
## Efficient outer sum of two lists of vectors U = 
##	Returns a list [X,Y,...], such that U[i]+V[j] = X[i,j] + Y[i,j] + ...
##
##---------------------------------------------------------------------------#
#
#def OuterSum(*args):
#
#  U,V = [np.array(a) for a in args]
# 
#  if np.ndim(U) not in [1,2] or np.ndim(V) not in [0,1,2]:
#    raise ValueError("The arrays must be 1D or 2D!")
#
#  if np.ndim(U) == 1: U = U.reshape(-1,1)
#  if np.ndim(V) == 0: V = np.repeat(V,U.shape[1]).reshape(1,-1)
#  if np.ndim(V) == 1: V = V.reshape(-1,1)
#
##  if U.shape[1] != V.shape[1]:
##    print("\n*** Warning: the vectors do NOT have the same dimension! Extra dimensions ignored.***\n")
#
#
#
#  return [np.add.outer(*a) for a in zip(U.T,V.T)]
#
#  
#def OuterDiff(U,V=0):
#
#  return OuterSum(U,-np.array(V))
#
#
#def OuterDist(*args):
#
##scipy.spatial.distance.cdist
##np.sqrt(np.sum((p[:,np.newaxis,:]-q[np.newaxis,:,:])**2, axis=2)) 
#
#  return la.norm(OuterDiff(*args),axis=0)
#
##===========================================================================#
## 
## Flattened outer sum, each line S[k] is a sum of U[i_k] and V[j_k]
##	S[k] = U[i] + V[j], with (i,j) = unravel_index(k,(len(U),len(V)))
##
##---------------------------------------------------------------------------#
#
#def FlatOuterSum(*args):
#
#  return np.array([s.reshape(-1) for s in OuterSum(*args)]).T
# 
#
#
#def FlatOuterDiff(U,V=0):
#
#  return FlatOuterSum(U,-np.array(V))
#
#
#
#
#def FlatOuterDist(*args):
#
#  return la.norm(FlatOuterDiff(*args),axis=1)
#
#
#
#def FlatOuter_IndexConvert(U,V):
#
#  get_k = lambda ij: np.ravel_multi_index(ij,(len(U),len(V)))
#
#  get_ij = lambda k: np.unravel_index(k,(len(U),len(V)))
#
#  return get_k,get_ij
#
#
#
#
##def FlatDist(Rs,R0=0):
##
##  Rs = np.array(Rs)
## 
##  if len(Rs.shape)!=2 or np.size(R0) > Rs.shape[1]:
##    raise ValueError("The array must be 2D!")
##
##  return la.norm(Rs-R0,axis=1)
#
#
#
##===========================================================================#
##
## Uses sympy to reformulate a matrix problem of the form sum_i (X_i A_i) = B
## 	as linear system Ax=b, with x containing the elements of all X_i
##
##---------------------------------------------------------------------------#
#
#
#def Reformulate_as_LinearSyst(As,B):
#
#  Xs = []
#  As_symb = []
#  dt = type(As[0][0,0])
#
#  for i,A in enumerate(np.array(As)):
#
#    X_shape = (B.shape[0],A.shape[0])
#
#    s = sp.symbols(["X"+str(i)+"_"+str(j) for j in np.arange(np.prod(X_shape))])
#
#    Xs.append(sp.Matrix(s).reshape(*X_shape))
#   
#    As_symb.append(sp.Matrix(A))
#
#
#  Z = sp.Add(*[X*A for (X,A) in zip(Xs,As_symb)]) - sp.Matrix(B)
#
#  eqs = [Z[i] for i in np.ndindex(*Z.shape)]
#
#  symb = [X[i] for X in Xs for i in np.ndindex(*X.shape)]
#
#
#  matr,vect = [np.array(a,dtype=dt) for a in sp.linear_eq_to_matrix(eqs,symb)]
# 
#  def get_Xs(sol,symb_=symb,Xs_=Xs): 
#
#    repl  = list(zip(symb_,sol))
#
#    Xs_ = [X.subs(repl) for X in Xs_]
# 
#    return [np.array(X,dtype=type(s)) for (X,s) in zip(Xs_,sol)]
#
#  def get_sol(Ys,symb_=symb,Xs_=Xs):
#
#    repl = [(X[i],Y[i]) for X,Y in zip(Xs_,Ys) for i in np.ndindex(*X.shape)]
#
#    return np.array([s.subs(repl) for s in symb_],dtype=type(Ys[0][0,0]))
#
#    
#
#
#  return matr,vect,get_Xs,get_sol
#
#
##===========================================================================#
##
## Input:  two lists (A1,A2) of d d-dim linearly independent vectors
##
## Output: -> d pairs of d-dim integer vectors (n1,n2) which satisfy 
##		-> "sum_i n1_i A1_i == sum_j n2_j A2_j" =: L(n1,n2) 
##		-> (n1,n2) are as small as possible: similarly to a LCM probl
##	  -> the corresponding d linearly independent L-s 
##
## In other words: finds the superlattice vectors, given two sets of lattice 
##	vectors (A1,A2) and raises an error if the struct. is incommensurate
##		
##---------------------------------------------------------------------------#
#
#
#
#	# find the integers (n1,n2) with components in [-n,n]
#def Supervectors_by_Trials(n,As,tol=7):
#
#      # candidate integers
#  intvects = -Utils.vectors_of_integers(len(As[0]),n)
#
#      # compute the norm of each n@A1, n@A2
#  n1,n2 = [np.round(la.norm(intvects@A,axis=1),tol) for A in As]
#
#      # filter the (n1,n2) pairs which render |n1@A1|==|n2@A2|
#  
#
#  inds = np.vstack(([Utils.Add_dim(np.isin(n1,n2i).nonzero()[0].reshape(-1,1),i) for (i,n2i) in enumerate(n2)])).T
#
#
#  Vs = [intvects[iis] for iis in inds]
#
#      # take pair (n1,n2) only if n1@A1 == n2@A2 (not minus)
#  diff = la.norm(np.subtract(*[V@A for V,A in zip(Vs,As)]),axis=1)
#
#  return np.hstack((Vs))[np.nonzero(diff < 10**(-tol))]
#
#
##===========================================================================#
##
## Find supervectors by inverting the difference of reciprocal vectors
##
##---------------------------------------------------------------------------#
#
#
#def Supervectors_by_RecipLatt(n,As,tol=7):
#
#
#  Gs1,Gs2 = [Invert_LattVect(A) for A in As]
#
#  Ls = Invert_LattVect(Gs1-Gs2)
#
#
#  dim = len(As[0])
#
#  intvects = Utils.vectors_of_integers(dim,n)#sort=True)
#
#  Vs = np.zeros((0,2*dim),dtype=int)
#
#  for L in Ls:
#    inds = lambda A: np.nonzero(la.norm(intvects@A-L,axis=1) < 10**(-tol))
#		# intvects[inds(A)] gives the list of integer vectors v_j 
#		# 	which satisfy v_j A = L; probably 0 <= j <= 1
#
#    v1v2s = list(itertools.product(*[intvects[inds(A)] for A in As]))
#
#    Vs = np.vstack((Vs,np.array(v1v2s).reshape(-1,2*dim)))
#
#  return Vs
#
#
#
##===========================================================================#
##
##		
##---------------------------------------------------------------------------#
#
#
#
#def Find_Supervectors(A1,A2,method_vect="Trials",max_nr_uc=100,min_nr_uc=1,tol=7,**kwargs):
#
#  dim = len(A1)
#
#	# Separating the square part of the lattice vectors
#  [A1xy,A1z],[A2xy,A2z] = [np.split(A,[dim],axis=1) for A in [A1,A2]]
#
#  As = [A1xy,A2xy]
#
#  if np.array([np.size(A)!=dim**2 for A in As]).any():
#    raise ValueError("There is a problem with the matrix sizes!")
#
#  if np.array([Rank(A)!=dim for A in As]).any():
#    raise ValueError("The vectors are not linearly independent!")
#
#  As = np.array(As)
#
#
#
#  if method_vect == "Trials":
#    find_sol_n = Supervectors_by_Trials
#
#  elif method_vect == "Recip":
#    find_sol_n = Supervectors_by_RecipLatt
#
#  else:
#    raise ValueError("Define the new method '"+str(method_vect)+"' or change it to either 'Trials' or 'Recip',")
#
#
#
#
#  nr_uc = min_nr_uc
#	# start with a small interval of integers [-nr_uc,nr_uc]
#
#  while nr_uc <= max_nr_uc:
# 
#    vs = find_sol_n(nr_uc,As,tol)
# 
#    if len(vs):	# if the collected (n1,n2) contain the desired maximal set
#      if Rank(vs) == dim:
#
#        if len(vs) != dim:
#  		# extract the maximal set of linearly independent vectors 
#          vs = Lin_indep_arrays(vs[np.argsort(la.norm(vs,axis=1))])
#
#		# split back the pair (n1,n2) 
#        vs = np.split(np.array(vs),2,axis=1)
#
#        Ls12 = [v@A for (v,A) in zip(vs,As)]
#
#		# perform an additional sanity check and return n1,n2,L-s
#        if la.norm(np.subtract(*Ls12)) < 10**(-tol):
#
#          Ls = Ls12[0]
# 
#          if np.size(A1z)!=0 and np.size(A2z)!=0:
#            Ls = Utils.Add_dim(Ls,0.) 	#np.mean([A1z,A2z]))
#
#          return vs, Ls
#
#
#	# if a good solution is not found, will try with a larger interval
#    nr_uc += 2
#  
#
#
#
#  
#  raise Exception("Could not find the supercell after looking at +/- [",min_nr_uc,max_nr_uc,"] cells. Either the structure is incommensurate or 'max_nr_uc' is not large enough.")
#
#
#
#
#
##===========================================================================#
##
## Computes the maximal set of linearly independent arrays
## 
##---------------------------------------------------------------------------#
#
#def Lin_indep_arrays(array_list,get_indices=False):
#
#  flatarrays = np.reshape(array_list,(len(array_list),-1))
#
# 
#  Rk = Rank(flatarrays)
#
#  lin_indep = []
#  indices = []
#
#
#  for i,V in enumerate(flatarrays):
#
#    if Rank(lin_indep+[V]) > len(lin_indep):
#
#      lin_indep.append(V)
#      indices.append(i)
#
#      if len(lin_indep) == Rk:
#
#        out = np.array([array_list[i] for i in indices])
#         
#        if get_indices:
#          return out,indices
#
#        return out
#  
#
#  raise Exception("\n*** Could not find the minimal set of linearly independent arrays! ***\n")
#
#
##===========================================================================#
##
## Rank of a matrix
## 
##---------------------------------------------------------------------------#
#
#
#def Rank(M,tol=1e-5):
#
#  if sum(np.array(M).shape) == 0:
#    return 0
#
#
#  s = np.linalg.svd(np.matrix(M),compute_uv=False)
#
#  rank = sum( s > max(s)*tol ) 
#
#  return rank
#
##===========================================================================#
##
## Solves a problem similar to root finding (i.e. similar to bisection method)
##	The "root", i.e. the desired solution, is the interval (xL,xR) where 
##		"val == 0" and which satisfies "goal(xL,xR,val,...) == True"
##	The function "update_boundaries" shrinks or shifts the interval,
##		guided by the function "compute_newval", which computes "val"
##		for the new interval (it might as well adjust the interval)
## 
##---------------------------------------------------------------------------#
#
#
#def Find_interval(goal,hownewval_args0,max_iter=20):
#
#  def update_boundaries(args):
#  
#    xL,xR,val,dx = args
#  
#    if val == 0:
#      return (xL + dx/2 , xR - dx/2)
#  
#    if val > 0:
#      return (xL-dx, xL)
#  
#    if val < 0:
#      return (xR,xR+dx)
#
#  compute_newval, args = hownewval_args0
#
#  nr_iter = 0
#
#  while not goal(args):
#  
#    if nr_iter == max_iter:
#      raise Exception("Did not converge in",max_iter,"iterations.")
#
#    nr_iter += 1 
#
#    args = compute_newval(update_boundaries(args))
#
#  return args
#
#
#
##===========================================================================#
##
## Rotation Matrix
## 
##---------------------------------------------------------------------------#
#
#def Rotation_Matrix(theta,dim=2,axis=2):
#
#  if dim not in [2,3]:
#    raise ValueError('Error: the rotation matrix can only be given for dim in [2,3].')
#
#  R = SO_Matrix(2,[theta])
#
#  if dim == 3: 
#
#    R = np.insert(R,axis,np.zeros(2),axis=0)
#    R = np.insert(R,axis,np.zeros(3),axis=1)
#   
#    R[axis,axis] = 1 
#
#  return R
#
#
#def spRotation_Matrix(theta,dim=2,axis=2):
#
#  if dim not in [2,3]:
#    raise ValueError('Error: the rotation matrix can only be given for dim in [2,3].')
#
#  c,s = sp.cos(theta),sp.sin(theta)
#  
#  R = sp.Matrix([[c,s],[-s,c]])
#
#  onehot = sp.Matrix([0,0]).row_insert(axis,sp.ones(1))
#
#  R = R.row_insert(axis,sp.zeros(1,2)).col_insert(axis,onehot)
#
#  return R[:dim,:dim]
#
#
#
#
#
#
#
#
##==============================================================================
## Generate matrices
##------------------------------------------------------------------------------
#
#def SO_Matrix(dim,param):
#
#  n = dim*(dim-1)//2
#
#  if len(param) != n:
#    raise ValueError('Wrong number of parameters for SO matrix!')
#
#  R = np.zeros((dim,dim))
#
#  for i in range(dim):
#    for j in range(i):
#
#      R[i,j] = param[i*(i-1)//2+j] 
#      R[j,i] = -R[i,j]
#
#
##  return expm(R) # very slow: 0.7-0.8ms for 2x2; diagonalization is 0.1-0.2ms
#
#  E, U = la.eig(R)
#
#  return np.matmul(np.matmul(U,np.diag(np.exp(E))),np.conj(U.T)).real
#
#
#def SU_Matrix(dim,param):
#
#  n = dim**2 - 1
#
#  if len(param) != n:
#    raise ValueError('Wrong number of parameters for SU matrix!')
#
#  R = np.append(param,-np.sum(param[::(dim+1)])).reshape(dim,dim)
#
#  return expm(1.j*R)
#
#
#
#  
#def U_Matrix(dim,param):
#
#  n = 2*dim**2 
#
#  if len(param) != n:
#    raise ValueError('Wrong number of parameters for U matrix!')
#
#  r,i = param.reshape((2,dim,dim))
#
#  return mod_GramSchmidt(r+i*1j)
#
#
#
#  
#def U_Matrix_2x2(param):
#  phi, psi, theta, delta = param
#
#  u_phi = np.exp(0.5j*phi)
#
#  u_psi = np.diag([np.exp(1j*psi),np.exp(-1j*psi)])
#
#  u_theta = [[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]]
#
#  u_delta = np.diag([np.exp(1j*delta),np.exp(-1j*delta)])
#  
#  return u_phi*np.matmul(u_psi,np.matmul(u_theta,u_delta))
#
#def U_Matrix_2x2_sympy(param):
#  import sympy as sp
#
#  n = 3; f = 10**n
#
#  spR = lambda p: sp.Rational(p*f,f) if int(p*f)/f == p else np.round(p,n)
#
#  param = [spR(p)*sp.pi for p in np.array(param)/np.pi]
#
#
#
#  phi,psi,theta,delta = sp.symbols('phi psi theta delta') 
#  
#
#  u_phi = sp.exp(sp.I*phi*sp.Rational(1,2))
#
#  u_psi = sp.diag(*[sp.exp(sp.I*psi),sp.exp(-sp.I*psi)])
#
#  u_theta = sp.Matrix([[sp.cos(theta),sp.sin(theta)],[-sp.sin(theta),sp.cos(theta)]])
#
#  u_delta = sp.diag(*[sp.exp(sp.I*delta),sp.exp(-sp.I*delta)])
#
#
#
#  out = []
#
#  symb = [phi,psi,theta,delta]
#  matr = [u_phi,u_psi,u_theta,u_delta]
# 
#  for s,m,p in zip(symb,matr,param):
#    if p != 0:
#      out.append(sp.simplify(m.subs(s,p)))
#
##  A = sp.I*sp.Matrix([[1,0],[0,-1]])*sp.pi/4
##  out.append(sp.simplify(sp.exp(A)))
#
##  return out
#
#  u_phi,u_psi,u_theta = out
#
#  return out+ ['=',u_phi,1/sp.sqrt(2),sp.Mul(u_psi,u_theta*sp.sqrt(2))]
#
##  return u_phi*np.matmul(u_psi,np.matmul(u_theta,u_delta))
#
#
#
#
#def H_Matrix(dim,param):
#
#  n = dim**2
#
#  if len(param) != n:
#    raise ValueError('Wrong number of parameters for H matrix!')
#
#
#  diag, real,imag = np.split(param, [dim,dim*(dim+1)//2])
#  offdiag = real + 1.j*imag
#
#  R = np.zeros((dim,dim),dtype=complex)
#
#  for i in range(dim):
#
#    R[i,i] = diag[i]
#
#    for j in range(i):
#
#      R[i,j] = offdiag[i*(i-1)//2+j] 
#
#      R[j,i] = np.conj(R[i,j])
#
#
#  return R
#
#
#
#


###############################################################################















