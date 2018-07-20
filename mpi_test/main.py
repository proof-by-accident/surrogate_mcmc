import pickle
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
nproc = comm.Get_size()
iproc = comm.Get_rank()

#parms_array_send = np.empty( [nproc**2,2], dtype=np.float64 )
#parms_array = np.empty([nproc,2], dtype=np.float64)

x = y = np.empty(nproc, dtype=np.float64)
X = Y = np.empty([nproc, nproc], dtype=np.float64) 

if iproc == 0:
        x = y = np.linspace(0,1,nproc)
        X,Y = np.meshgrid(x,y)
        #parms_array_send = np.array([ [a,b] for a,b in zip([x for x_list in X for x in x_list], [y for y_list in Y for y in y_list]) ])
else:
        pass

#comm.Scatter( parms_array_send, parms_array, root=0)
comm.Scatter(X,x,root=0)
comm.Scatter(Y,y,root=0)

if iproc !=0 :
        print iproc, x==y, x

else:
        pass
        #print iproc, X,Y

#comm.Barrier()
#for i in range(nproc):
#        if i == iproc:
#                print (iproc, senddata )
#
MPI.Finalize()
#
#        while any([ t == None for t in foo ]):
#                i,v = comm.recv()
#                foo[i] = v
#
#        print(foo)
#        with open('foo1.pickle','wb') as handle:
#                pickle.dump(foo, handle)
#        MPI.COMM_WORLD.Barrier()
#                v = np.exp( i )
#                comm.send( (i,v) , dest= 0)
