import pickle
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
nproc = comm.Get_size()
iproc = comm.Get_rank()

senddata = np.ones(100, dtype=np.float64)

if iproc == 0:
        senddata = np.array( range(100), dtype = np.float64 )

else:
        pass

comm.Bcast( [ senddata, MPI.DOUBLE ], root=0 )

comm.Barrier()
for i in range(nproc):
        if i == iproc:
                print (iproc, senddata )

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
