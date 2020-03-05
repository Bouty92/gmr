#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from gmr import GMM


######################
# REFERENCE FONCTION #
######################

def F( x, y ) :
    # Rosenbrock function:
	z = ( 0 - x )**2 + 100*( y - x**2 + 1 )**2
	return z/1000

x_limit = ( -2, 2 )
y_limit = ( -2, 2 )


display_resolution = 100

x_scale = np.linspace( *x_limit, display_resolution )
y_scale = np.linspace( *y_limit, display_resolution )

X, Y = np.meshgrid( x_scale, y_scale )

Zr = np.zeros( ( display_resolution, display_resolution ) )
for i, x in enumerate( x_scale ) :
	for j, y in enumerate( y_scale ) :
		Zr[j][i] = F( x, y )



################
# TRAINING SET #
################

sample_resolution = 50

data = []
for x in np.linspace( *x_limit, sample_resolution ) :
	for y in np.linspace( *y_limit, sample_resolution ) :
		data.append( [ x, y, F( x, y ) ] )
data = np.array( data )
X_data = data[:,:2]
y_data = data[:,2]



#######
# GMR #
#######

gmm = GMM( n_components=4 )
gmm.from_samples( data )

#x0 = 0
x0 = 1.5
#y0 = 1.5
y0 = -1.5
z0 = np.squeeze( gmm.predict(np.array([ 0, 1 ]), np.array([ x0, y0 ])[np.newaxis,:] ) )
dxdy = np.squeeze( gmm.condition_derivative( np.array([ 0, 1 ]), np.array([ x0, y0 ]) ) )
print( 'z( %g, %g ):' % ( x0, y0 ), z0 )
print( 'dydx( %g, %g ):' % ( x0, y0 ), dxdy )

Zp = np.zeros_like( Zr )
for i, x in enumerate( x_scale ) :
	for j, y in enumerate( y_scale ) :
		Zp[j][i] = np.squeeze( gmm.predict( np.array([ 0, 1 ]), np.array([ x, y ])[np.newaxis,:] ) )



#########
# PLOTS #
#########

#azim = -120 ; elev = 40
azim = 120 ; elev = 40
#azim = 75 ; elev = 30

fig = plt.figure( 'Reference', figsize=(8,8) )
ax = plt.axes( projection='3d' )
ax.plot_surface( X, Y, Zr )
ax.set_xlabel( 'x' )
ax.set_ylabel( 'y' )
ax.set_zlabel( 'z' )
ax.view_init( elev, azim )


fig = plt.figure( 'Prediction', figsize=(8,8) )
ax = plt.axes( projection='3d' )
ax.plot_surface( X, Y, Zp )
ax.set_xlabel( 'x' )
ax.set_ylabel( 'y' )
ax.set_zlabel( 'z' )
ax.view_init( elev, azim )

d = 0.5
ax.plot( [ x0 - d*dxdy[0], x0 + d*dxdy[0] ], [ y0 - d*dxdy[1], y0 + d*dxdy[1] ],
         [ z0 - d*( dxdy[0]**2 + dxdy[1]**2 ), z0 + d*( dxdy[0]**2 + dxdy[1]**2 ) ], c='r', lw=5 )
#ax.plot( [ x0 - d, x0 + d ], [ y0, y0 ],
         #[ z0 - d*dxdy[0], z0 + d*dxdy[0] ], c='g', lw=5 )
#ax.plot( [ x0, x0 ], [ y0 - d, y0 + d ],
         #[ z0 - d*dxdy[1], z0 + d*dxdy[1] ], c='b', lw=5 )
ax.scatter( x0, y0, z0, c='r', s=100, marker='o' )


plt.show()
