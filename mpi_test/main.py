import pickle
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
nproc = comm.Get_size()
iproc = comm.Get_rank()

senddata = np.ones(10, dtype=np.float64)
recvdata = np.empty([nproc,10], dtype=np.float64)

comm.Gather( senddata, recvdata, root=0 )

if iproc == 0:
        print recvdata

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
