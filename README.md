# Visual-Information-Fidelity (VIF) - Python

This repository contains python implementation of steerale pyramid version of Visual Information Fidelity (VIF) proposed in [1]
This is a replication of MATLAB version released by the authors of [1] which is available [HERE](http://live.ece.utexas.edu/research/Quality/ifcvec_release.zip).

## Dependencies
1) Python (>=3.5)
2) Numpy (>=1.16)
3) Python Imaging Library (PIL) (>=6.0)
4) Steerable Pyramid Toolbox (PyPyrTools) [link](https://github.com/LabForComputationalVision/pyPyrTools)

## Usage
Let imref and imdist denote reference and distorted images respectively. Then the VIF value is calculated as
VIF = vifvec(imref, imdist)

A demo code is provided in test.py for testing purposes

[1]H.R. Sheikh, A.C. Bovik and G. de Veciana, "An information fidelity criterion for image quality assessment using natural scene statistics," IEEE Transactions on Image Processing , vol.14, no.12pp. 2117- 2128, Dec. 2005.
