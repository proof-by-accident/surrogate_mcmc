import pickle
import os
import numpy as np

saves_dir_name = raw_input( 'where are the savefiles to collect?\n' ) 
saves_dir = os.sep.join( [os.curdir, saves_dir_name])
saves_files = [ os.sep.join([ saves_dir, f ]) for f in os.listdir( saves_dir ) ]

saved_blocks = np.empty([1,3])

for f in saves_files:
    if 'exlk' in f:
        saved_block = pickle.load(open(f,'rb'))[1]
        saved_blocks = np.vstack([ saved_blocks, saved_block ])
    

pickle.dump( saved_blocks[1:,:], open( './saved_exp_like.pl', 'wb' ) )

#f = saves_files.pop()

#parms, l = pickle.load( open( f, 'rb' ) )
#b = parms[:,0]
#g = parms[:,1]
#
#ret_stack = np.vstack([b,g,l]).T
#
#
#for f in saves_files:
#    parms, l = pickle.load( open( f, 'rb' ) )
#    b = parms[:,0]
#    g = parms[:,1]
#
#    ret = np.vstack([b,g,l]).T
#
#    ret_stack = np.vstack([ret_stack, ret])
#
#pickle.dump( ret_stack, open( './saved_exp_like.pl', 'wb' ) )
