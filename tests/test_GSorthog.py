from  AlgebraNew import  overlap_matrix
import numpy as np 

ov = np.vdot 



nr_wf = 6

nr_comp = 7 



psi1 = np.random.rand(nr_wf, nr_comp)# + 1j*np.random.rand(nr_wf, nr_comp) 
psi2 = np.random.rand(nr_wf, nr_comp) + 1j*np.random.rand(nr_wf, nr_comp) 


for i in range(nr_wf):
    for j in range(i,nr_wf):

        if abs(ov(psi1[i],psi1[j]) - overlap_matrix(psi1)[i,j])>1e-10:

            print(i,j,"p1 p1")


        if abs(ov(psi1[i],psi2[j]) - overlap_matrix(psi1,psi2)[i,j])>1e-10:

            print(i,j,"p1 p2")

        if abs(ov(psi1[i],psi2[j]) - overlap_matrix(psi2,psi1).conjugate()[j,i])>1e-10:

            print(i,j,"p2 p1")


print("GS")

from AlgebraNew import GramSchmidt#, GS_Orthog

psi3 = GramSchmidt(psi1) 

#psi4 = GS_Orthog(psi1) 


print(np.round(np.abs(overlap_matrix(psi1)),3),"\n")
print(np.round(np.abs(overlap_matrix(psi3)),2),"\n")
#print(np.round(np.abs(overlap_matrix(psi4)),3),"\n")



