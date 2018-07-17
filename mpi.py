import pickle
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
nproc = comm.Get_size()
iproc = comm.Get_rank()

senddata = None
recvdata = np.empty(nproc, dtype=np.float64)
if iproc == 0:
        gath_req = comm.Igather(senddata,recvdata, root=0)
        print('waiting for other processes to finish')
        gath_req.Waitall()
        print recvdata

else:
        senddata = np.array([ iproc ], dtype=np.float64)
        

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
