""" PCA exercise
"""

#: import common modules
import numpy as np  # the Python array package
import matplotlib.pyplot as plt  # the Python plotting package
# Display array values to 6 digits of precision
np.set_printoptions(precision=4, suppress=True)
#: import numpy.linalg with a shorter name
import numpy.linalg as npl
import nibabel as nib
import os

#- Load the image 'ds114_sub009_t2r1.nii' with nibabel
#- Get the data array from the image
img = nib.load('ds114_sub009_t2r1.nii')
shape = img.shape
print(shape)
image_data_array = img.get_fdata()
#- Make variables:
#- 'vol_shape' for shape of volumes
#- 'n_vols' for number of volumes
vol_shape = shape[0:3]
n_vols = shape[-1]
#- Slice the image data array to give array with only first two
#- volumes
image_data_array = image_data_array[:, :, :,  0:2]
#- Set N to be the number of voxels in a volume
N = vol_shape[0]*vol_shape[1]*vol_shape[2]
#- Reshape to 2D array with first dimension length N
image_data_array = np.reshape(image_data_array, (N, 2))
#- Transpose to 2 by N array
image_data_array = image_data_array.transpose()
#- Calculate the mean across columns (122880,)
colum_mean = image_data_array.mean(axis=0)
#- Row means copied N times to become a 2 by N array
row_means = image_data_array.mean(axis=1) #(2,)
row_means = np.repeat(row_means, N, axis=0) #(2, 122880)
row_means = row_means.reshape((2, N))
#- Subtract the means for each row, put the result into X
#- Show the means over the columns, after the subtraction
#X = image_data_array - colum_mean
X = image_data_array - row_means
print('new column means; ', X.mean(axis=0))
#- Plot the signal in the first row against the signal in the second
plt.plot(X[:, 0], X[:, 1])
plt.show()
#- Calculate unscaled covariance matrix for X
cov = np.cov(X)
#- Use SVD to return U, S, VT matrices from unscaled covariance
U,S,VT = np.linalg.svd(cov)
#- Show that the columns in U each have vector length 1
print('length of column vectors: ', np.linalg.norm(U, axis=0))

#- Confirm orthogonality of columns in U
print('vector product of columns of U = ', np.dot(U[0, :], U[1, :]))
#- Confirm tranpose of U is inverse of U
print('transpose of U = ', np.transpose(U))
print('inverse of U = ', np.linalg.inv(U))
#- Show the total sum of squares in X
#- Is this (nearly) the same as the sum of the values in S?
print('sum of squares in X = ', np.sum(X**2))
print('sum of values in S = ', np.sum(S))

#- Plot the signal in the first row against the signal in the second
#- Plot line corresponding to a scaled version of the first principal component
#- (Scaling may need to be negative)
plt.plot(X[:, 0], X[:, 1])
plt.plot(U[:, 0])
plt.show()
#- Calculate the scalar projections for projecting X onto the
#- vectors in U.
#- Put the result into a new array C.
C = U.T.dot(X)
#- Transpose C
#- Reshape the first dimension of C to have the 3D shape of the
#- original data volumes.
Ct = np.transpose(C)
Cr = np.reshape(C, (vol_shape[0], -1, -1))
#- Break 4D array into two 3D volumes

#- Show middle slice (over third dimension) from scalar projections
#- for first component

#- Show middle slice (over third dimension) from scalar projections
#- for second component

#- Reshape first dimension of whole image data array to N, and take
#- transpose

#- Calculate mean across columns
#- Expand to (173, N) shape using np.outer
#- Subtract from data array to remove mean over columns (row means)
#- Put result into array X

#- Calculate unscaled covariance matrix of X


#- Use subplots to make axes to plot first 10 principal component
#- vectors
#- Plot one component vector per sub-plot.

#- Calculate scalar projections for projecting X onto U
#- Put results into array C.

#- Transpose C
#- Reshape the first dimension of C to have the 3D shape of the
#- original data volumes.

#- Show middle slice (over third dimension) of first principal
#- component volume

#- Make the mean volume (mean over the last axis)
#- Show the middle plane (slicing over the third axis)

#- Show middle plane (slice over third dimension) of second principal
#- component volume

#- Show middle plane (slice over third dimension) of third principal
#- component volume
