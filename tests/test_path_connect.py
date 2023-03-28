import Utils 
import numpy as np 
import matplotlib.pyplot as plt 


fig,(ax1,ax2) = plt.subplots(1,2,figsize=(9,4))


points = np.random.rand(5,2)

order = points[:,1]##rt(np.linalg.norm(points,axis=1))

points = points[np.argsort(order)]

for (i,p) in enumerate(points):

    ax1.scatter(*p,label=i,zorder=10)

ax1.legend()


path, xticks, x = Utils.path_connect(points, 30, bounds=[0.3,1.1])

#print(xticks) 



ax1.plot(*path.T,zorder=0,c="gray")


ax2.plot(x,path[:,0],label="x")
ax2.plot(x,path[:,1],label="y")

ax2.set_xlim(x.min(), x.max())

ax2.set_xticks(xticks)  



#ax2.set_xticklabels(range(len(xticks)))

ax2.legend() 

plt.show() 


