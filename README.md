# Hyperspectral and Multispectral Image Fusion Under Spectrally Varying Spatial Blurs

Hyperspectral imaging has become a significant source of valuable data for astronomers over the past decades. Current instrumental and observing time constraints allow direct acquisition of multispectral images, with high spatial but low spectral resolution, and hyperspectral images, with low spatial but high spectral resolution. 
To enhance scientific interpretation of the data, we propose a data fusion method which combines the benefits of each image to recover a high spatio-spectral resolution datacube. 
The proposed inverse problem accounts for the specificities of astronomical instruments, such as spectrally variant blurs. 
We provide a fast implementation by solving the problem in the frequency domain and in a low-dimensional subspace to efficiently handle the convolution operators as well as the high dimensionality of the data. 

## Files 

- CONSTANTS.py : This module is used to store constants and paths for data fusion.
- main.py : Run this module to test fusion
- fusion.py : This module implents the optimization process for fusion of infrared astronomical hyperspectral and multispectral images as described in [1,2].
- sparse.py : This module computes the sparse matrix A and the vector b used in the fast fusion precedure described in [2].
- sparse_preprocess.py : This module implents the preprocessing of the data as described in [1,2].
- errors.py : This module computes and saves fusion performance measures.
- acp_v2.py : This module implents PCA performed on the HS image for spectral dimension reduction.
- tools : Tools for data fusion.

- produce_HS_MS.py : This code implents forward models of the NIRCam imager and the NIRSpec IFU embedded in the JWST as described in [1].

## Guidelines

Please refer to [2] for the following notations.

Name spatial and spectral degradation operators fits files :
- Store the 2D-spatial fft of M : 'M_fft.fits'
- Store the 2D-spatial fft of H : 'H_fft.fits'
- Lm : 'Lm.fits'
- Lh : 'Lh.fits'

Modify paths and constants in CONSTANTS.py, if necessary. 

Compute HS and MS images, if necessary, or store them in the folder of your choice.

Run 'main.py' to test fusion.

## References

Ref 1 : C. Guilloteau, T. Oberlin, O. Berné, É. Habart, and N. Dobigeon
“Simulated JWST datasets for multispectral and hyperspectral image fusion”
The Astronomical Journal, vol. 160, no. 1, p. 28, Jun. 2020.

Ref 2 : C. Guilloteau, T. Oberlin, O. Berné, É. Habart, and N. Dobigeon
"Hyperspectral and Multispectral Image Fusion Under Spectrally Varying Spatial Blurs – Application to High Dimensional Infrared Astronomical Imaging"
IEEE Transactions on Computatonal Imaging, vol.6, Sept. 2020.
