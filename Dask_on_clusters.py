%%time

from PIL import Image
from dask import delayed
import numpy as np
import math
import cv2
import glob
from scipy import ndimage
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage as ndi
import scipy.misc
from skimage import exposure
import os 
import multiprocessing
from dask.distributed import Client, progress
import dask.array as da
import dask_ndmeasure
import dask
import dask.threaded
from scipy.ndimage.measurements import center_of_mass
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
import numpy as np
import scipy.linalg as sla
from dask.multiprocessing import get as mp_get
from dask_ml.cluster import SpectralClustering

import gcsfs

def read_image_file(path, img_type= cv2.IMREAD_GRAYSCALE):
    with fs.open(path,'rb') as f:
        image = np.asarray(bytearray(f.read()), dtype="uint8")
        
        image = cv2.imdecode(image, img_type)
        return image

def read_images(folders_address, file_extension, sample_address, t_slice, z_slice, x_dim, y_dim):

	'''
	This Function reads all the images in a parallel fashion using dask arrays and put them into a 4D numpy array.
	First dimension indicates the frame numbers, the second one shows the slices(z axis) and the 3rd and 4th are the x and y axises. 
	To use this function, you need to put the images of each frame inside a seperate folder. 
	
	Parameters
    ----------
    folders_address: string
		root address of image folders    
	file_extension: string
        it shows the type of the images that should be imported ( e.g.: '*.tiff')
	sample_file_name: string
		a file name of a sample slice of the images
	t_slice: integer
		total number of frames of the video
	z_slice: integer
		the depth of each frame
	x_dim : integer
		length of the image (pixel)
	y_dim : 
		width of the image (pixel)
    
    Returns
    -------
    all_images_nparray : array, shape (frames, z, x, y)
        all the images are arranged in a 4D numpy array

	'''
	
	sample_file_name = sample_address
	addr=os.path.join(folders_address, sample_file_name) 
	sample = read_image_file(addr, cv2.IMREAD_GRAYSCALE)
	
	
	directory_list = list()
	directory_list = list(fs.ls(folders_address, detail=False))

	sorted_path = sorted(directory_list)
	#print(sorted_path)

	all_images = list ()
	extension = file_extension

	images = [delayed(read_image_file)(file,cv2.IMREAD_GRAYSCALE) for i in range(len(sorted_path)) for file in sorted(fs.glob(os.path.join(sorted_path[i],extension))) ] 
	arrays = [da.from_delayed(im, shape = sample.shape, dtype = sample.dtype) for im in images]
	#print(arrays)
	x = da.stack(arrays, axis =0)
	x = x.reshape(t_slice, z_slice, x_dim, y_dim)

	all_images_nparray = x.rechunk((1, 1, x_dim, y_dim))
	all_images_nparray = all_images_nparray.persist()
	progress(all_images_nparray)

	return all_images_nparray


def read_images_old(folders_address, file_extension, sample_address, t_slice, z_slice, x_dim, y_dim):

	'''
	This Function reads all the images in a parallel fashion using dask arrays and put them into a 4D numpy array.
	First dimension indicates the frame numbers, the second one shows the slices(z axis) and the 3rd and 4th are the x and y axises. 
	To use this function, you need to put the images of each frame inside a seperate folder. 
	
	Parameters
    ----------
    folders_address: string
		root address of image folders    
	file_extension: string
        it shows the type of the images that should be imported ( e.g.: '*.tiff')
	sample_file_name: string
		a file name of a sample slice of the images
	t_slice: integer
		total number of frames of the video
	z_slice: integer
		the depth of each frame
	x_dim : integer
		length of the image (pixel)
	y_dim : 
		width of the image (pixel)
    
    Returns
    -------
    all_images_nparray : array, shape (frames, z, x, y)
        all the images are arranged in a 4D numpy array

	'''
	
	sample_file_name = sample_address
	sample = cv2.imread(os.path.join(folders_address, sample_file_name), cv2.IMREAD_GRAYSCALE)
	
	directory_list = list()
	for root, dirs, files in os.walk(folders_address , topdown=False):
	    for name in dirs:
	        if not (name.startswith('.')):
	            directory_list.append(os.path.join(root, name))

	sorted_path = sorted(directory_list)

	all_images = list ()
	extension = file_extension
	images = [delayed(cv2.imread)(file,cv2.IMREAD_GRAYSCALE) for i in range(len(sorted_path)) for file in sorted(glob.glob(os.path.join(sorted_path[i],extension))) ] 
	arrays = [da.from_delayed(im, shape = sample.shape, dtype = sample.dtype) for im in images]

	x = da.stack(arrays, axis =0)
	x = x.reshape(t_slice, z_slice, x_dim, y_dim)

	all_images_nparray = x.rechunk((1, 1, x_dim, y_dim))
	all_images_nparray = all_images_nparray.persist()
	progress(all_images_nparray)

	return all_images_nparray

def thresholding(i):
    ret = []
    for x in range(total.shape[1]):
        ret.append(cv2.LUT(total[i][x], BW))
    return ret


def index_thr(frame,unique_array, denoising_thresh):
    thr_idxs=[]
    n_labels=len(frame)
    for i in range(1, n_labels):
        if frame[i] > denoising_thresh : 
            thr_idxs.append(unique_array[i])
    return thr_idxs 

def tracker(all_centers):

	all_cen = all_centers #[ [x[0] for x in frame ] for frame in all_centers]
	new_objects = [ [(0,x)] for x in all_centers[0] ]

	t_limit = 20

	for i in range (1, len(all_cen)-1):
	    
	    '''in every step we need to check the points in current frame with 
	    last selected points in our object list
	    '''
	    current_frame = all_cen[i]
	    last_known_centers = [ obj[-1][1] for obj in new_objects if len(obj)>0 ] 
	    
	    # We are going to use Hungarian algorithm which is built in scipy
	    # As linear_sum_assignment. we need to pass a cost to that function
	    # the function will assign the points based on minimum cost. Here we 
	    # define the distance between the above mentioned points as our cost 
	    # function 
	    cost = distance.cdist(last_known_centers, current_frame,'euclidean')
	    # in this function row_ind will act as object_ids and the col_ind
	    # will play the role of new_centers_ind for us so we have : 
	    obj_ids, new_centers_ind = linear_sum_assignment(cost)
	    
	    all_center_inds = set(range(len(current_frame)))
	    # now we should iterate on obj_id and new_center_ind 
	    # checking the min acceptable distance , appending the points to 
	    # our frames and finally removing those points from our set.
	    
	    for  obj_id, new_center_ind  in zip(obj_ids,new_centers_ind):
	        if( distance.euclidean(np.array(current_frame[new_center_ind]),np.array(new_objects[obj_id][-1][1]) ) <= t_limit):
	            all_center_inds.remove(new_center_ind)
	            new_objects[obj_id].append((i,current_frame[new_center_ind]))
	    # at the end if the points are not matched with the previous objects 
	    # we will consider them as new objects and appending them to the end 
	    # of our object list.

	    for new_center_ind in all_center_inds:
	        new_objects.append([ (i,current_frame[new_center_ind])])
	xx = [[]]
	yy = [[]]
	zz = [[]]
	for i in range (len(new_objects)):
	    for j in range (len(new_objects[i])):
	        zz[i].append(new_objects[i][j][1][0])
	        xx[i].append(new_objects[i][j][1][1])
	        yy[i].append(new_objects[i][j][1][2])
	    xx.append([])
	    zz.append([])
	    yy.append([])

	zz = [pick for pick in zz if len(pick) > 0 ]
	xx = [pick for pick in xx if len(pick) > 0 ]
	yy = [pick for pick in yy if len(pick) > 0 ]

	xx = np.asarray(xx)
	yy = np.asarray(yy)  
	zz = np.asarray(zz)


	return (xx, yy, zz)


def visualization_3d_detection(all_cnf, image_width, image_height):

	znf3d = [[]]
	xnf3d = [[]]
	ynf3d = [[]]

	for frames in range(all_cnf.shape[0]):
	    for i in range (np.shape(all_cnf[frames])[0]):
	        znf3d[frames].append(all_cnf[frames][i][0])
	        xnf3d[frames].append(all_cnf[frames][i][1])
	        ynf3d[frames].append(all_cnf[frames][i][2])
	    znf3d.append([])
	    xnf3d.append([])
	    ynf3d.append([])

	# ok! Now let us visualize the final detection results:
	fig = plt.figure(figsize=(image_width, image_height))
	ax = fig.add_subplot(111, projection='3d')
	for i in range(all_cnf.shape[0]):
	    ax.scatter(ynf3d[i], xnf3d[i], znf3d[i], 
	               zdir='znf3d[i]', marker = "o",  c= znf3d[i], cmap='gist_heat')

	ax.w_xaxis.set_pane_color((0.2, 0.3, 0.8, 1.0))
	ax.w_yaxis.set_pane_color((0.2, 0.3, 0.8, 1.0))
	ax.w_zaxis.set_pane_color((0.2, 0.3, 0.8, 1.0))
	ax.view_init(35, 45)
	plt.grid(False)
	plt.savefig("demo2center-allnf.png", dpi=200)
	plt.show()

def Trajectory_3D_TimeVarying(frame_num, single_flag, point_num, s, x, y, z, number_of_points, video_file):
    '''*********************************************************** 
    This Function  will  plot a 3D  representation of the motility
    x, y and z are the  axis  values  which are defined inside the 
    main function.  single_flag is a  flag which indicates we want
    1 trajectory plotting or all of  them ?  if True means we need
    just one trajectory. The point_num indicates the number of the
    point which we are going to plot. written by MS.Fazli
    **************************************************************
    '''
    n = frame_num
    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(35, 45)



    if single_flag :
        traj_points = 1
        T = np.linspace(0,1,np.size(x[point_num]))

    else : 
        traj_points = number_of_points
        T = np.linspace(0,1,np.size(x[0]))

    for i in range(traj_points):
        for j in range(0, n-s, s):
            if single_flag :             
                ax.plot(yy[point_num][j:j+s+1],  xx[point_num][j:j+s+1] ,zz[point_num][j:j+s+1], zdir='zz[i]', linewidth =5, color = ( 0.0, 0.9*T[j], 0.0))
#                 plt.pause(0.06)
            else : 
                ax.plot(yy[i][j:j+s+1],  xx[i][j:j+s+1] ,zz[i][j:j+s+1], zdir='zz[i]', linewidth =3, color = (T[j], 0.0, 0.0))
#                 plt.pause(0.06)

    #for angle in range(0, 360):
        # ax.view_init(5, i)
        # plt.pause(0.01)
        # plt.draw()
    ax.w_xaxis.set_pane_color((0.2, 0.3, 0.8, 1.0))
    ax.w_yaxis.set_pane_color((0.2, 0.3, 0.8, 1.0))
    ax.w_zaxis.set_pane_color((0.2, 0.3, 0.8, 1.0))
    ax.view_init(35, 45)

    plt.grid(False)
    
    ax.set_xlabel('Y')
    ax.set_ylabel('X')
    ax.set_zlabel('Z')
    plt.savefig(str(video_file) + '4d_timeVariying_.jpg', dpi=600)

    plt.show()

def simple_visualization_tracked_points(xx, yy, zz, traj_length, image_width, image_height, savefig_quality, savefig_name):


	fig = plt.figure(figsize=(image_width,image_height))
	ax = fig.add_subplot(111, projection='3d')
	for i in range(traj_length):
	    ax.plot(yy[i], xx[i], zz[i], 
	               zdir='zz[i]', linewidth = 3)

	ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
	ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
	ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
	ax.view_init(35, 45)
	plt.grid(False)
	plt.savefig(savefig_name, dpi = savefig_quality)
	plt.show()

def preprocessing_for_clustering(x, y, z, frame_number, object_numbers):
    newxx =[]
    newyy =[]
    newzz =[]
    for i in range(object_numbers):
        if len(x[i]) == frame_number :
            newxx.append(x[i])
            newyy.append(y[i])
            newzz.append(z[i])
    #print ('len obj(' +str(i) +')='+str(len(xx[i])) )
    allx = np.asarray(newxx)
    ally = np.asarray(newyy)
    allz = np.asarray(newzz)
    
    return(allx, ally, allz)

def laplacian(A):
    """Computes the symetric normalized laplacian.
    L = D^{-1/2} A D{-1/2}
    """
    D = np.zeros(A.shape)
    w = np.sum(A, axis=0)
    D.flat[::len(w) + 1] = w ** (-0.5)  # set the diag of D to w
    return D.dot(A).dot(D)
#def apply_martin()
def state_space(raw_data, q):
    
    import numpy as np
    import numpy.linalg as linalg
    """
    Performs the state-space projection of the original data using principal
    component analysis (eigen-decomposition).
    Parameters
    ----------
    raw_data : array, shape (N, M)
        Row-vector data points with M features.
    q : integer
        Number of principal components to keep.
    Returns
    -------
    X : array, shape (q, M)
        State-space projection of the original data.
    C : array, shape (N, q) the PCA matrix (useful for returning to the data space)
        Projection matrix.
    """
    if q <= 0:
        raise Exception('Parameter "q" restricted to positive integer values.')

    # Perform the SVD on the data.
    # For full documentation on this aspect, see page 15 of Midori Hyndman's
    # master's thesis on Autoregressive modeling.
    #
    # Y = U * S * Vt,
    #
    # Y = C * X,
    #
    # So:
    # C = first q columns of U
    # S_hat = first q singular values of S
    # Vt_hat = first q rows of Vt
    #
    # X = S_hat * Vt_hat
    #
    # For the full documentation of SVD, see:
    # http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.svd.html#numpy.linalg.svd
    U, S, Vt = linalg.svd(raw_data, full_matrices = False)
    C = U[:, :q]
    Sh = np.diag(S)[:q, :q]
    Vth = Vt[:q, :]
    X = np.dot(Sh, Vth)
    return [X, C]


def train(X, order = 2):
    import numpy as np
    import numpy.linalg as linalg
    """
    Estimates the transition matrices A (and eventually the error parameters as
    well) for this AR model, given the order of the markov process.
    (in this notation, the parameter to this method "order" has the same value as "q")
    Parameters
    ----------
    X : array, shape (q, M) or (M,)
        Matrix of column vectors of the data (either original or state-space).
    order : integer
        Positive, non-zero integer order value for the order of the Markov process.
    Returns
    -------
    A : array, shape (q, q)
        Transition coefficients for the system
    """
    if order <= 0:
        raise Exception('Parameter "order" restricted to positive integer values')
    W = None

    # A particular special case first.
    if len(X.shape) == 1:
        Xtemp = np.zeros(shape = (1, np.size(X)))
        Xtemp[0, :] = X
        X = Xtemp

    # What happens in this loop is so obscenely complicated that I'm pretty
    # sure I couldn't replicate if I had to, much less explain it. Nevertheless,
    # this loop allows for the calculation of n-th order transition matrices
    # of a high-dimensional system.
    #
    # I know this could be done much more simply with some np.reshape() voodoo
    # magic, but for the time being I'm entirely too lazy to do so. Plus, this
    # works. Which is good.
    for i in range(1, order + 1):
        Xt = X[:, order - i: -i]
        if W is None:
            W = np.zeros((np.size(Xt, axis = 0) * order, np.size(Xt, axis = 1)))
        W[(i - 1) * np.size(Xt, axis = 0):((i - 1) * np.size(Xt, axis = 0)) + np.size(Xt, axis = 0), ...] = Xt
    Xt = X[:, order:]
    A = np.dot(Xt, linalg.pinv(W))
    
    # The data structure "A" is actually all the transition matrices appended
    # horizontally into a single NumPy array. We need to extract them.
    matrices = []
    for i in range(0, order):
        matrices.append(A[..., i * np.size(A, axis = 0):(i * np.size(A, axis = 0)) + np.size(A, axis = 0)])
    return matrices

def martin(A1, C1, A2, C2):
    """
    Computes the pairwise Martin distance between two d-order AR systems.
    Parameters
    ----------
    A1, A2 : lists
        Lists of AR parameters for two systems. They MUST be identical
        in dimensionality (q) and order (d).
    C1, C2 : array, shape (N, q)
        Subspaces for the two systems.
    Returns
    -------
    m : float
        Martin distance between the two systems.
    """
    N, q = C1.shape
    d = len(A1)
    # print(N, q, d)
    A = np.zeros(shape = (2 * q * d, 2 * q * d))
    # print(A.shape, C1t.shape, C2t.shape)

    C1t, C2t = np.zeros(shape = (N * d, q * d)), np.zeros(shape = (N * d, q * d))

    # First, append all the parameters into our gargantuan matrix.
    for i, (a1, a2) in enumerate(zip(A1, A2)):
        A[:q, (i * q):(i * q) + q] = a1
        A[(d * q):(d * q) + q, (d * q) + (i * q):(d * q) + (i * q) + q] = a2

        # Do we have a higher order system?
        if d > 1:
            C1t[(i * N):(i * N) + N, (i * q):(i * q) + q] = C1
            C2t[(i * N):(i * N) + N, (i * q):(i * q) + q] = C2

    # Some clean-up.
    if d == 1:
        C1t = C1
        C2t = C2
    else:
        A[q:(d * q), :(d - 1) * q] = A[((d + 1) * q):, (d * q):-q] = np.identity(q * (d - 1))

    # Create the Q matrix and solve the Lyapunov system.
    Q = np.hstack([C1t, C2t]).T.dot(np.hstack([C1t, C2t]))
    X = sla.solve_discrete_lyapunov(A, Q)

    # Because roundoff errors.
    X = (X + X.T) * 0.5

    # Now continue as usual.
    P11, P12, P22 = X[:(q * d), :(q * d)], X[:(q * d), (q * d):], X[(q * d):, (q * d):]
    PPP = sla.inv(P11).dot(P12).dot(sla.inv(P22)).dot(P12.T)
    
    w = sla.eigvalsh(PPP)
    maxpp = w.flatten().max()
    w = np.true_divide(w, maxpp)
    if w.prod() <= 0.0:
        # Swigert: Hey we've got a problem here.
        w = np.delete(w, np.where(w <= 0.0))
    #print(w)
    #print('Done')
    return -np.log(w.prod())
def computing_affinity_old(traj_pool, frame_numbers, flatten_AR_mat, number_of_points):    
    #first We create a trajectory pool with the dimensions of 3x(Num_of_trajectoris)x(Num_of_frames)
    all_traj_mat = traj_pool.copy()
    all_traj_mat = all_traj_mat.reshape(all_traj_mat.shape[0], all_traj_mat.shape[1]* all_traj_mat.shape[2])
    print(all_traj_mat.shape)
    
    #Parameterization: ( This part is not computationally expensive)
    #Then we set the AR dimensions to be 2 ( A dimensionality reduction ), so that the matrix C will be a 3x2 Matrix 
    #and the matrix X will be a 2x(Num_of_trajectoris)x(Num_of_frames)
    X, C = state_space(all_traj_mat,2)
    print('AR_Matrix, Projection_Matrix dims=', str(X.shape), str(C.shape))
    
    for index in range (number_of_points): 
        traj2 = X[:, frame_numbers*index: (index+1)*frame_numbers]
        A1, A2, A3, A4, A5 = (train(traj2, 5))
        flatten_AR_mat[index] = np.concatenate((A1.flatten(), A2.flatten(), A3.flatten(), A4.flatten(), A5.flatten()))
    print(flatten_AR_mat.shape)
    
    # Now let us create a pairwise Martin Distance:

   #===================================================================================
   # DASK version of a bottleneck: just for faster processing
   #===================================================================================
    import dask.bag as db

    Mrt_dist = [ dask.delayed(martin)(flatten_AR_mat[i], C, flatten_AR_mat[j], C) for i in range(flatten_AR_mat.shape[0]) 
                                                for j in range(flatten_AR_mat.shape[0]) ]

    Mrt_dist_mat = dask.compute(*Mrt_dist)
    Mrt_dist_mat = np.asarray(Mrt_dist_mat)
    #Mrt_dist_mat = Mrt_dist_mat.reshape(flatten_AR_mat.shape[0],flatten_AR_mat.shape[0])
    print(Mrt_dist_mat.shape)


    #OK! Now, It's the to polish the results of computing the distance:
    Mrt_dist_mat = (Mrt_dist_mat.T + Mrt_dist_mat) * .5
    for i in range (Mrt_dist_mat.shape[0]):
        for j in range (Mrt_dist_mat.shape[1]):
            if i == j :
                Mrt_dist_mat[i, j] = 0.0
    print('Check if any value in distance matrix in "Nan": ', str(np.isnan(Mrt_dist_mat).any()))
    print('Check if any value in distance matrix in "inf": ', str(np.isinf(Mrt_dist_mat).any()))
    #Converting the distance to similarity:
    similarity1 = np.exp(-.5 * Mrt_dist_mat / Mrt_dist_mat.std())
    
    #Now, Let us visualize the distance matrix:
    fig = plt.figure(figsize=(15,10))
    plt.imshow(similarity1, cmap = 'Blues')
    plt.colorbar()
    plt.show()
    
    # Now, let us also compute the the Normal distance matrix too.
    lap = laplacian(Mrt_dist_mat)
    for i in range (Mrt_dist_mat.shape[0]):
        for j in range(Mrt_dist_mat.shape[1]):
            if i == j :
                lap[i, j] = 0.0
    lap = (lap + lap.T) * .5
    
    #Converting the distance to similarity:
    similarity2 = np.exp(-.5 * lap / lap.std())

    print(np.isnan(lap).any())
    print(np.isinf(lap).any())
    plt.imshow(similarity2, cmap = 'hot')
    plt.colorbar()
    return similarity1, similarity2

def clustering_dask_ml(traj_pool, frame_numbers, flatten_AR_mat, number_of_points, gamma = -.5):    
    #first We create a trajectory pool with the dimensions of 3x(Num_of_trajectoris)x(Num_of_frames)
    all_traj_mat = traj_pool.copy()
    all_traj_mat = all_traj_mat.reshape(all_traj_mat.shape[0], all_traj_mat.shape[1]* all_traj_mat.shape[2])
    print(all_traj_mat.shape)
    
    #Parameterization:
    #Then we set the AR dimensions to be 2 ( A dimensionality reduction ), so that the matrix C will be a 3x2 Matrix 
    #and the matrix X will be a 2x(Num_of_trajectoris)x(Num_of_frames)
    X, C = state_space(all_traj_mat,2)
    print('AR_Matrix, Projection_Matrix dims=', str(X.shape), str(C.shape))
    
    for index in range (number_of_points): 
        traj2 = X[:, frame_numbers*index: (index+1)*frame_numbers]
        A1, A2, A3, A4, A5 = (train(traj2, 5))
        flatten_AR_mat[index] = np.concatenate((A1.flatten(), A2.flatten(), A3.flatten(), A4.flatten(), A5.flatten()))
    print(flatten_AR_mat.shape)
    
    
    pairwise_martin = lambda x1,x2: np.exp(-gamma * martin(x1,C,x2,C)**2)
    sc = SpectralClustering(affinity = pairwise_martin)
    return sc, sc.fit_predict(flatten_AR_mat)
    
    

def computing_affinity(traj_pool, frame_numbers, flatten_AR_mat, number_of_points):    
    #first We create a trajectory pool with the dimensions of 3x(Num_of_trajectoris)x(Num_of_frames)
    all_traj_mat = traj_pool.copy()
    all_traj_mat = all_traj_mat.reshape(all_traj_mat.shape[0], all_traj_mat.shape[1]* all_traj_mat.shape[2])
    print(all_traj_mat.shape)
    
    #Parameterization:
    #Then we set the AR dimensions to be 2 ( A dimensionality reduction ), so that the matrix C will be a 3x2 Matrix 
    #and the matrix X will be a 2x(Num_of_trajectoris)x(Num_of_frames)
    X, C = state_space(all_traj_mat,2)
    print('AR_Matrix, Projection_Matrix dims=', str(X.shape), str(C.shape))
    
    for index in range (number_of_points): 
        traj2 = X[:, frame_numbers*index: (index+1)*frame_numbers]
        A1, A2, A3, A4, A5 = (train(traj2, 5))
        flatten_AR_mat[index] = np.concatenate((A1.flatten(), A2.flatten(), A3.flatten(), A4.flatten(), A5.flatten()))
    print(flatten_AR_mat.shape)
    
    # Now let us create a pairwise Martin Distance:

   #===================================================================================
   # Parallel version : just for faster data acquisition
   #===================================================================================
    import multiprocessing
    from joblib import Parallel, delayed
    from multiprocessing import Pool
    num_cores = multiprocessing.cpu_count()

    Mrt_dist_mat = Parallel(n_jobs = num_cores)(delayed(martin)(flatten_AR_mat[i], C, flatten_AR_mat[j], C) for i in range(flatten_AR_mat.shape[0]) 
                                                 for j in range(flatten_AR_mat.shape[0]))
    Mrt_dist_mat = np.asarray(Mrt_dist_mat)
    Mrt_dist_mat = Mrt_dist_mat.reshape(flatten_AR_mat.shape[0],flatten_AR_mat.shape[0])
    print(Mrt_dist_mat.shape)
#===========================================================================================================
    
    #Mrt_dist_mat = np.zeros(shape=(flatten_AR_mat.shape[0], flatten_AR_mat.shape[0]))
    #for i in range(flatten_AR_mat.shape[0]):
    #    for j in range(flatten_AR_mat.shape[0]):
    #        Mrt_dist_mat[i, j] = martin(flatten_AR_mat[i], C, flatten_AR_mat[j], C)
    #        if i == 1 and j == 1 :
    #            print(Mrt_dist_mat[i, j])
    #    print(i, j)
#===========================================================================================================    



    #OK! Now, It's the to polish the results of computing the distance:
    Mrt_dist_mat = (Mrt_dist_mat.T + Mrt_dist_mat) * .5
    for i in range (Mrt_dist_mat.shape[0]):
        for j in range (Mrt_dist_mat.shape[1]):
            if i == j :
                Mrt_dist_mat[i, j] = 0.0
    print('Check if any value in distance matrix in "Nan": ', str(np.isnan(Mrt_dist_mat).any()))
    print('Check if any value in distance matrix in "inf": ', str(np.isinf(Mrt_dist_mat).any()))
    #Converting the distance to similarity:
    similarity1 = np.exp(-.5 * Mrt_dist_mat / Mrt_dist_mat.std())
    
    #Now, Let us visualize the distance matrix:
    fig = plt.figure(figsize=(15,10))
    plt.imshow(similarity1, cmap = 'Blues')
    plt.colorbar()
    plt.show()
    
    # Now, let us also compute the the Normal distance matrix too.
    lap = laplacian(Mrt_dist_mat)
    for i in range (Mrt_dist_mat.shape[0]):
        for j in range(Mrt_dist_mat.shape[1]):
            if i == j :
                lap[i, j] = 0.0
    lap = (lap + lap.T) * .5
    
    #Converting the distance to similarity:
    similarity2 = np.exp(-.5 * lap / lap.std())

    print(np.isnan(lap).any())
    print(np.isinf(lap).any())
    plt.imshow(similarity2, cmap = 'hot')
    plt.colorbar()
    return similarity1, similarity2

def clustering(affinity, num_of_clusters, labels_file_name, affinity_file_name):
    
    from sklearn.cluster import spectral_clustering
    import sklearn.cluster as cluster

    nclusters= num_of_clusters
    scB = cluster.SpectralClustering(n_clusters = nclusters, affinity = 'precomputed',assign_labels='discretize')
    #Mrt_dist_mat2 = (Mrt_dist_mat2 - np.mean(Mrt_dist_mat2, axis=0)) / np.std(Mrt_dist_mat2, axis=0)
    scB.fit(affinity)

    yB = scB.labels_

    np.save(labels_file_name, yB)
    np.save(affinity_file_name ,affinity)
    return(yB)

def clustering_dask_ml(traj_pool, frame_numbers, flatten_AR_mat, number_of_points, gamma = -.5):    
    #first We create a trajectory pool with the dimensions of 3x(Num_of_trajectoris)x(Num_of_frames)
    all_traj_mat = traj_pool.copy()
    all_traj_mat = all_traj_mat.reshape(all_traj_mat.shape[0], all_traj_mat.shape[1]* all_traj_mat.shape[2])
    print(all_traj_mat.shape)
    
    #Parameterization:
    #Then we set the AR dimensions to be 2 ( A dimensionality reduction ), so that the matrix C will be a 3x2 Matrix 
    #and the matrix X will be a 2x(Num_of_trajectoris)x(Num_of_frames)
    X, C = state_space(all_traj_mat,2)
    print('AR_Matrix, Projection_Matrix dims=', str(X.shape), str(C.shape))
    
    for index in range (number_of_points): 
        traj2 = X[:, frame_numbers*index: (index+1)*frame_numbers]
        A1, A2, A3, A4, A5 = (train(traj2, 5))
        flatten_AR_mat[index] = np.concatenate((A1.flatten(), A2.flatten(), A3.flatten(), A4.flatten(), A5.flatten()))
    print(flatten_AR_mat.shape)
    

    def pairwise_martin(X, Y = None, **kwargs):
        from sklearn.metrics.pairwise import pairwise_kernels
        #print(X.shape,C.shape, Y.shape if Y is not None else None)
        def _martin(x1,x2):
            m1 = martin(x1,C,x2,C)
            m2 = martin(x2,C,x1,C)
            m = (m1+m2) /2.0
            if np.isnan(m) or np.isinf(m):
                raise Exception("NAN")
            r = np.exp(-kwargs['gamma'] * m)
            if np.isnan(r) or np.isinf(r):
                raise Exception("r is Nan")
            return r
        rbf_martin = _martin
        return pairwise_kernels(X,Y, metric=rbf_martin)
    sc = SpectralClustering(n_clusters= 3, affinity = pairwise_martin, n_components= 40, gamma=gamma)
    return sc, None,flatten_AR_mat

def visualize_clusters(color_list, label, xx, yy, zz, output_png_file):
    
    # Plotting the trajectories and showing the clustering using a specific color for each label
    col = color_list
    # for i in range(Number_of_points):
    #     plt.plot(matrix_nexb[i],matrix_neyb[i] , color= col[yB[i]])

    # plt.gca().invert_yaxis()
    # plt.show()

    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(xx.shape[0]):
        ax.plot(yy[i], xx[i], zz[i], 
                zdir='zz[i]', linewidth = 3,  color= col[label[i]])

    ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
    ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
    ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
    ax.view_init(35, 45)
    plt.grid(False)
    plt.savefig(output_png_file)
    plt.show()


def main():
	global total 
	global fs
	global BW
	global sc

	folders = '/The/Address/To/The_dataset/'
	sample_address = 'The/Adress/To/Sample_TIFF/image/sample_tiff_image.tif'
	folders = 'gcs://uga-toxo/3d_video1/'
	sample_address = 't01/010t01z01.tif'
	client = Client('dask-scheduler:8786')
	fs = gcsfs.GCSFileSystem(project='toxoplasma-motility')
	extension =  '*.tif'
	denoising_thresh = 1
	color_list = ['red', 'y', 'blue', 'green', 'cyan', 'pink','lime','brown']

	num_cores = 20#multiprocessing.cpu_count()
	#client = Client(n_workers=num_cores)

	all_image_arr = read_images(folders, extension, sample_address, 63, 41, 500, 502)
	print(all_image_arr.shape)



	# #================ Thresholding: =============
	BW = np.array([i if i > 42 else 0 for i in range(0,256)]).astype(np.uint8)
	thresh = lambda  x : cv2.LUT (x, BW)
	thresholded_array = all_image_arr.map_blocks(thresh).compute()
	thresholded_dask_arr = da.from_array(thresholded_array, chunks=(1, 1, 500, 502), lock=True) 

	print(thresholded_dask_arr.shape)

	# #================ 3d CCL and labeling: =============

	structure = np.ones((3,3,3), dtype = np.int)
	labl2 = lambda  x : np.asarray(dask_ndmeasure.label(x, structure)[1])


	ccl_res = [dask.delayed(dask_ndmeasure.label)(thresholded_dask_arr[frames], structure)[0] for frames in range(np.shape(thresholded_dask_arr)[0])] 
	ccls = [da.from_delayed( d_r, shape=(41, 500, 502), dtype=thresholded_dask_arr[0].dtype) for d_r in ccl_res]
	ccl_cmpt = [dask.delayed(labl2)(thresholded_dask_arr[frames]) for frames in range(np.shape(thresholded_dask_arr)[0])] 
	ccls_cmpt = [da.from_delayed( d_r, shape=(), dtype='int64') for d_r in ccl_cmpt]
	ccl_arr0 = da.stack(ccls, axis=0)
	ccl_arr1 = da.stack(ccls_cmpt)

	all_cmpts = ccl_arr1.compute()
	all_labels = ccl_arr0.compute()
	print('========================================')
	print(all_labels.shape)
	print('========================================')
	print(all_cmpts.shape)



	#================ Computing the volume of each compnent: ===============

	count_unqe = [ dask.delayed(np.unique)(all_labels[frames], return_counts = True) for frames in range(thresholded_dask_arr.shape[0]) ]
	count_u1 = dask.compute(*count_unqe)


	count_u1 = np.array(count_u1)
	unique = count_u1[:, 0]
	counts = count_u1[:, 1]


	#================ Denoising: ===============

	thr_idxs4 = [ dask.delayed(index_thr)(counts[frame],unique[frame], denoising_thresh) for frame in range(thresholded_dask_arr.shape[0]) ]
	thr_idxs_computed = dask.compute(*thr_idxs4)

	print(len(thr_idxs_computed[0]))

	#================ Computing the centers: ===============

	all_centers_noisefree = [ dask.delayed(center_of_mass)(thresholded_dask_arr[frames], labels=all_labels[frames], index=thr_idxs_computed[frames]) for frames in range(thresholded_dask_arr.shape[0])]
	#a_c_n = dask.compute(*all_centers_noisefree)

	a_c_n_fast = dask.compute(*all_centers_noisefree, scheduler='threads')

	#================= Tracking Part : ======================

	xx, yy, zz = tracker(a_c_n_fast)

	#================= Clustering Part : ======================
	#dask.config.set(scheduler='threads')

	# xx = np.load('/home/vel/Downloads/toxo-clustering/ToxoPlasma-master/Trajectory Clustering/Martin_test/XX.npy')
	# yy = np.load('/home/vel/Downloads/toxo-clustering/ToxoPlasma-master/Trajectory Clustering/Martin_test/YY.npy')
	# zz = np.load('/home/vel/Downloads/toxo-clustering/ToxoPlasma-master/Trajectory Clustering/Martin_test/ZZ.npy')

	#================= Pre Processing for clustering ==========
	#print(all_centers_noisefree.shape)
	tracked_frames = all_cmpts.shape[0]- 1
	object_numbers = xx.shape[0]
	xx, yy, zz = preprocessing_for_clustering(xx, yy, zz, tracked_frames, object_numbers)

	#================= Setting Auto regressive parameters and other initializations =========

	number_of_points = np.shape(xx)[0]
	AR_order = 5
	columns = AR_order * 2 * 2
	flatten_AR_mat = np.zeros(shape = (number_of_points, columns))
	cluster_num = 3
	print((number_of_points, columns), flatten_AR_mat.shape)

	#================= Creating a pool of preprocessed trajectories =========

	traj_pool = np.stack([xx, yy, zz])
	print(traj_pool.shape, xx[0].shape)

	simple_visualization_tracked_points(xx, yy, zz, xx.shape[0], 15, 10, 150, 'test_track_parallel')


	#sim1, sim2 = computing_affinity(traj_pool, tracked_frames, flatten_AR_mat, number_of_points)
	print("here")
	#yB1 = clustering(sim1, cluster_num, 'data_set1_labels.npy', 'data_set1_affinity.npy')
	#sc, yB1 = clustering_dask_ml(traj_pool, tracked_frames, flatten_AR_mat, number_of_points)

	#print(yB1)
	#visualize_clusters(color_list, yB1, xx, yy, zz, 'Dataset-1_clusters_visualization.png')
	#return sc, sc.fit_predict(flatten_AR_mat)
	sc.fit(As)

	eigs = sc.eigenvalues_.compute()
	yB1= sc.labels_.compute()
	visualize_clusters(color_list, yB1, xx, yy, zz, 'Dataset-1_clusters_visualization.png')


	sc, yB1, As = clustering_dask_ml(traj_pool, tracked_frames, flatten_AR_mat, number_of_points, gamma = -0.5)
if __name__ == "__main__":
    main()
