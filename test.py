from PIL import Image
import numpy as np
from vifvec import vifvec

#Load reference image
imref = np.array(Image.open('bikes.bmp').convert('L')).astype(float)

# Load distorted image
imdist = np.array(Image.open('img29.bmp').convert('L')).astype(float)

#Calculate VIF score
vif_score = vifvec(imref,imdist)