###############################################################################
#
# 
#
###############################################################################

import numpy as np
import os


#============================================================================#
# number to string 
#----------------------------------------------------------------------------#

def nr2str(x,L,R):

    if R==0:
        if L==0: return "" 

        if isinstance(x,str): return x[-L:].zfill(L)

        return ("m" if x<0 else "") + nr2str(str(abs(int(x))),L,R)

    xL,*xR = str(abs(np.round(x,R))).split('.')


    xR_ = nr2str(xR[0][::-1] if len(xR)==1 else '0',R,0)[::-1] 

    return ("m" if x<0 else "")+"p".join([nr2str(xL,L,0),xR_])



def construct_fname(P, keys, digits):
    
    return "_".join([nr2str(P[k],*digits[k]) for k in keys if k in P])


#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#

def get_paramcombs(K,P):

    if isinstance(P,dict):
        return yield_paramcombs(K, [P[k] for k in K])

    return yield_paramcombs(K, P) 



def get_paramcombs_inds(P):

    return np.ndindex(*map(len, P))





def yield_paramcombs(K,P):

    for I in get_paramcombs_inds(P):

        yield {k:p[i] for (i,k,p) in zip(I,K,P)}




#============================================================================#
# Class 
#----------------------------------------------------------------------------#

class ParamFlow():

    def __init__(self, 
            root="./Data",
            name="", allparams={}, digits=[], usedkeys=None, **kwargs):


        self.root=root

        self.name = name 

        if usedkeys is None:
            self.allparams = allparams

        else:
            self.allparams = {k:allparams[k] for k in usedkeys}

        self.not_saved_params = sorted(list(set(self.allparams.keys()).difference([k for (k,aux) in digits])))

        self.digits = {}

        self.saved_params = []
       
        for (k,d) in digits:

            if k in self.allparams:

                self.saved_params.append(k)

                self.digits[k] = d


    def get_params(self):

        return self.saved_params + self.not_saved_params


    def get_paramcombs(self):

        return get_paramcombs(self.get_params(), self.allparams)


#        for I in np.ndindex(*[len(self.allparams[k]) for k in K]):
#
#            yield {k:self.allparams[k][i] for (i,k) in zip(I,K)}


    def get_fname(self, P, *fname):

        return os.path.join(
                    self.root,
                    self.name,
                    construct_fname(P, self.saved_params, self.digits),
                    *fname,
                    )






#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#




#def restrict(PF, usedkeys):
#
#    return ParamFlow(   
#                PF.name,
#                {k:self.allparams[k] for k in usedkeys},
#                [(k,digits[k]) for k in PF.saved_params if k in usedkeys],
#            )
#






#===========================================================================#
#
#
#
#---------------------------------------------------------------------------#




#############################################################################
