

## 1. Get the constants

### Pytorch DNN constants

Stored in [model_info.txt](https://github.com/tonyfu97/LearnCompNeuro/blob/main/artiphysiology/model_info.txt). 
Load it with pandas.

### Pasupathy shape constants
```python
#
#  In this cell are the control coordinates for all 51 shapes, in addition
#  to some extra information about the number of control coordinates and
#  the number of unique rotations of the shapes.  The extra information is
#  not used in the demo below.

# Number of coordinate values for each shape (2 x Number of x,y pairs). Probably don't need this one.
pasu_shape_n = [
18,18,26,26,26,30,26,18,30,24,24,18,34,26,26,30,38,34,34,42,42,38,46,26,34,34,34,42,42,42,50,34,42,50,50,58,66,26,26,26,30,30,34,50,34,26,38,42,42,30,34]

# Number of rotations for each shape, within standard set of 370
#   Thus, the sum of this array = 370
pasu_shape_nrot = [1,1,8,8,4,8,4,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,2,8,4,8,8,2,8,8,8,8,8,8,8,8,8,8,8,8,8,8]

# True unique rotations
#    *** Under the assumption that the shape is centered, but shape 4 is
#        not centered.
#    Note, two '8' values have been replaced by '4' relative to the above.
pasu_shape_nrotu = [1,1,8,4,4,8,4,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,2,8,4,8,8,2,8,8,8,8,8,8,8,4,8,8,8,8,8,8]

# Control coordinates for each of the 51 shapes listed as 1D arrays,
#   in the format:  x0,y0,x1,y1,x2,y2,...
# *** WYETH NOTE:  Apr 2014 - slight change in coords to Pasu_shape_29
pasu_shape = [
[-0.4,0.0,-0.283,0.283,0.0,0.4,0.283,0.283,0.4,0.0,0.283,-0.283,0.0,-0.4,-0.283,-0.283,-0.4,0.0],
[-1.6,0.0,-1.131,1.131,0.0,1.6,1.131,1.131,1.6,0.0,1.131,-1.131,0.0,-1.6,-1.131,-1.131,-1.6,0.0],
[-0.4,0.0,-0.375,0.416,-0.3,0.825,-0.174,1.221,0.0,1.6,0.174,1.221,0.3,0.825,0.375,0.416,0.4,0.0,0.283,-0.283,0.0,-0.4,-0.283,-0.283,-0.4,0.0],
[-0.481,0.215,-0.518,0.6,-0.481,0.983,-0.369,1.354,0.0,1.6,0.369,1.354,0.481,0.983,0.518,0.6,0.481,0.215,0.369,-0.154,0.0,-0.4,-0.369,-0.154,-0.481,0.215],
[-0.373,0.0,-0.266,0.828,-0.069,1.37,0.0,1.6,0.069,1.37,0.266,0.828,0.373,0.0,0.266,-0.828,0.069,-1.37,0.0,-1.6,-0.069,-1.37,-0.266,-0.828,-0.373,0.0],
[-0.438,0.195,-0.277,0.916,-0.182,1.19,-0.102,1.387,0.0,1.6,0.102,1.387,0.182,1.19,0.277,0.916,0.438,0.195,0.477,-0.543,0.393,-1.278,0.0,-1.6,-0.393,-1.278,-0.477,-0.543,-0.438,0.195],
[-0.64,0.0,-0.571,0.689,-0.369,1.354,0.0,1.6,0.369,1.354,0.571,0.689,0.64,0.0,0.571,-0.689,0.369,-1.354,0.0,-1.6,-0.369,-1.354,-0.571,-0.689,-0.64,0.0],
[-0.266,0.82,0.0,1.6,0.122,0.988,0.468,0.468,0.988,0.122,1.6,0.0,0.82,-0.266,0.066,0.066,-0.266,0.82],
[-0.386,0.589,-0.386,1.303,0.0,1.6,0.283,1.483,0.4,1.2,0.461,0.894,0.635,0.635,0.894,0.461,1.2,0.4,1.483,0.283,1.6,0.0,1.303,-0.386,0.589,-0.386,-0.029,-0.029,-0.386,0.589],
[-0.082,0.186,-0.289,0.884,0.0,1.6,0.1,1.14,0.351,0.751,0.74,0.5,1.2,0.4,1.483,0.283,1.6,0.0,1.278,-0.393,0.467,-0.294,-0.082,0.186],
[-1.6,0.0,-1.483,0.283,-1.2,0.4,-0.74,0.5,-0.351,0.751,-0.1,1.14,0.0,1.6,0.289,0.884,0.082,0.186,-0.467,-0.294,-1.278,-0.393,-1.6,0.0],
[-0.245,0.781,0.0,1.6,0.075,0.846,0.294,0.122,0.651,-0.546,1.131,-1.131,0.385,-0.733,-0.108,-0.051,-0.245,0.781],
[-0.427,0.573,-0.393,1.278,-0.283,1.483,0.0,1.6,0.283,1.483,0.393,1.122,0.373,0.652,0.487,0.198,0.727,-0.203,1.071,-0.516,1.248,-0.848,1.131,-1.131,0.85,-1.25,0.626,-1.181,0.106,-0.713,-0.256,-0.11,-0.427,0.573],
[-0.123,0.149,-0.167,0.883,0.0,1.6,0.054,0.983,0.257,0.401,0.605,-0.111,1.071,-0.516,1.216,-0.848,1.131,-1.131,0.85,-1.25,0.57,-1.131,0.13,-0.542,-0.123,0.149],
[-0.605,-0.111,-0.257,0.401,-0.054,0.983,0.0,1.6,0.167,0.883,0.123,0.149,-0.13,-0.542,-0.57,-1.131,-0.85,-1.25,-1.131,-1.131,-1.216,-0.848,-1.071,-0.516,-0.605,-0.111],
[-0.533,0.0,-0.397,0.843,-0.176,1.333,0.0,1.6,0.122,0.988,0.468,0.468,0.988,0.122,1.6,0.0,0.988,-0.122,0.468,-0.468,0.122,-0.988,0.0,-1.6,-0.176,-1.333,-0.397,-0.843,-0.533,0.0],
[-0.533,0.0,-0.397,0.843,-0.176,1.333,0.0,1.6,0.1,1.14,0.351,0.751,0.74,0.5,1.2,0.4,1.483,0.283,1.6,0.0,1.483,-0.283,1.2,-0.4,0.74,-0.5,0.351,-0.751,0.1,-1.14,0.0,-1.6,-0.176,-1.333,-0.397,-0.843,-0.533,0.0],
[-0.575,0.172,-0.381,0.923,-0.212,1.273,0.0,1.6,0.122,0.988,0.468,0.468,0.988,0.122,1.6,0.0,1.14,-0.092,0.752,-0.352,0.492,-0.74,0.4,-1.2,0.283,-1.483,0.0,-1.6,-0.369,-1.354,-0.571,-0.605,-0.575,0.172],
[-0.571,0.605,-0.369,1.354,0.0,1.6,0.283,1.483,0.4,1.2,0.492,0.74,0.752,0.352,1.14,0.092,1.6,0.0,0.988,-0.122,0.468,-0.468,0.122,-0.988,0.0,-1.6,-0.212,-1.273,-0.381,-0.923,-0.575,-0.172,-0.571,0.605],
[-0.575,0.172,-0.381,0.923,-0.212,1.273,0.0,1.6,0.1,1.14,0.351,0.751,0.74,0.5,1.2,0.4,1.483,0.283,1.6,0.0,1.483,-0.283,1.2,-0.4,0.894,-0.461,0.635,-0.635,0.461,-0.894,0.4,-1.2,0.283,-1.483,0.0,-1.6,-0.369,-1.354,-0.571,-0.605,-0.575,0.172],
[-0.571,0.605,-0.369,1.354,0.0,1.6,0.283,1.483,0.4,1.2,0.461,0.894,0.635,0.635,0.894,0.461,1.2,0.4,1.483,0.283,1.6,0.0,1.483,-0.283,1.2,-0.4,0.74,-0.5,0.351,-0.751,0.1,-1.14,0.0,-1.6,-0.212,-1.273,-0.381,-0.923,-0.575,-0.172,-0.571,0.605],
[-0.64,0.0,-0.571,0.689,-0.369,1.354,0.0,1.6,0.283,1.483,0.4,1.2,0.492,0.74,0.752,0.352,1.14,0.092,1.6,0.0,1.14,-0.092,0.752,-0.352,0.492,-0.74,0.4,-1.2,0.283,-1.483,0.0,-1.6,-0.369,-1.354,-0.571,-0.689,-0.64,0.0],
[-0.64,0.0,-0.571,0.689,-0.369,1.354,0.0,1.6,0.283,1.483,0.4,1.2,0.461,0.894,0.635,0.635,0.894,0.461,1.2,0.4,1.483,0.283,1.6,0.0,1.483,-0.283,1.2,-0.4,0.894,-0.461,0.635,-0.635,0.461,-0.894,0.4,-1.2,0.283,-1.483,0.0,-1.6,-0.369,-1.354,-0.571,-0.689,-0.64,0.0],
[-0.294,0.122,-0.075,0.846,0.0,1.6,0.075,0.846,0.294,0.122,0.651,-0.546,1.131,-1.131,0.612,-0.785,0.0,-0.663,-0.612,-0.785,-1.131,-1.131,-0.651,-0.546,-0.294,0.122],
[-0.467,0.102,-0.35,0.505,-0.393,1.122,-0.283,1.483,0.0,1.6,0.283,1.483,0.393,1.122,0.35,0.505,0.467,-0.102,0.751,-0.688,1.131,-1.131,0.612,-0.785,0.0,-0.663,-0.612,-0.785,-1.131,-1.131,-0.751,-0.688,-0.467,0.102],
[-0.294,0.122,-0.075,0.846,0.0,1.6,0.054,0.983,0.257,0.401,0.605,-0.111,1.071,-0.516,1.248,-0.848,1.131,-1.131,0.85,-1.25,0.57,-1.131,0.179,-0.871,-0.282,-0.78,-0.742,-0.871,-1.131,-1.131,-0.651,-0.546,-0.294,0.122],
[-0.257,0.401,-0.054,0.983,0.0,1.6,0.075,0.846,0.294,0.122,0.651,-0.546,1.131,-1.131,0.742,-0.871,0.282,-0.78,-0.179,-0.871,-0.571,-1.131,-0.85,-1.25,-1.131,-1.131,-1.248,-0.848,-1.071,-0.516,-0.605,-0.111,-0.257,0.401],
[-0.257,0.401,-0.054,0.983,0.0,1.6,0.054,0.983,0.257,0.401,0.605,-0.111,1.071,-0.516,1.248,-0.848,1.131,-1.131,0.85,-1.25,0.57,-1.131,0.308,-0.957,0.0,-0.896,-0.308,-0.957,-0.57,-1.131,-0.85,-1.25,-1.131,-1.131,-1.248,-0.848,-1.071,-0.516,-0.605,-0.111,-0.257,0.401],
[-0.487,0.198,-0.373,0.652,-0.393,1.122,-0.283,1.483,0.0,1.6,0.283,1.483,0.393,1.122,0.35,0.505,0.467,-0.102,0.751,-0.688,1.131,-1.131,0.742,-0.871,0.282,-0.78,-0.179,-0.871,-0.571,-1.131,-0.85,-1.25,-1.131,-1.131,-1.248,-0.848,-1.071,-0.516,-0.727,-0.203,-0.487,0.198],
[-0.467,-0.102,-0.35,0.505,-0.393,1.122,-0.283,1.483,0.0,1.6,0.283,1.483,0.393,1.122,0.373,0.652,0.487,0.198,0.727,-0.203,1.071,-0.516,1.248,-0.848,1.131,-1.131,0.85,-1.25,0.571,-1.131,0.179,-0.871,-0.282,-0.78,-0.742,-0.871,-1.131,-1.131,-0.751,-0.688,-0.467,-0.102],
[-0.487,0.198,-0.373,0.652,-0.393,1.122,-0.283,1.483,0.0,1.6,0.283,1.483,0.393,1.122,0.373,0.652,0.487,0.198,0.727,-0.203,1.071,-0.516,1.248,-0.848,1.131,-1.131,0.85,-1.25,0.57,-1.131,0.308,-0.957,0.0,-0.896,-0.308,-0.957,-0.57,-1.131,-0.85,-1.25,-1.131,-1.131,-1.248,-0.848,-1.071,-0.516,-0.727,-0.203,-0.487,0.198],
[-1.6,0.0,-0.988,0.122,-0.468,0.468,-0.122,0.988,0.0,1.6,0.122,0.988,0.468,0.468,0.988,0.122,1.6,0.0,0.988,-0.122,0.468,-0.468,0.122,-0.988,0.0,-1.6,-0.122,-0.988,-0.468,-0.468,-0.988,-0.122,-1.6,0.0],
[-1.6,0.0,-0.988,0.122,-0.468,0.468,-0.122,0.988,0.0,1.6,0.122,0.988,0.468,0.468,0.988,0.122,1.6,0.0,1.14,-0.1,0.751,-0.351,0.5,-0.74,0.4,-1.2,0.283,-1.483,0.0,-1.6,-0.283,-1.483,-0.4,-1.2,-0.5,-0.74,-0.751,-0.351,-1.14,-0.1,-1.6,0.0],
[-1.6,0.0,-1.483,0.283,-1.2,0.4,-0.74,0.5,-0.351,0.751,-0.1,1.14,0.0,1.6,0.1,1.14,0.351,0.751,0.74,0.5,1.2,0.4,1.483,0.283,1.6,0.0,1.483,-0.283,1.2,-0.4,0.74,-0.5,0.351,-0.751,0.1,-1.14,0.0,-1.6,-0.1,-1.14,-0.351,-0.751,-0.74,-0.5,-1.2,-0.4,-1.483,-0.283,-1.6,0.0],
[-1.6,0.0,-0.988,0.122,-0.468,0.468,-0.122,0.988,0.0,1.6,0.1,1.14,0.351,0.751,0.74,0.5,1.2,0.4,1.483,0.283,1.6,0.0,1.483,-0.283,1.2,-0.4,0.894,-0.461,0.635,-0.635,0.461,-0.894,0.4,-1.2,0.283,-1.483,0.0,-1.6,-0.283,-1.483,-0.4,-1.2,-0.5,-0.74,-0.751,-0.351,-1.14,-0.1,-1.6,0.0],
[-1.6,0.0,-1.483,0.283,-1.2,0.4,-0.894,0.461,-0.635,0.635,-0.461,0.894,-0.4,1.2,-0.283,1.483,0.0,1.6,0.283,1.483,0.4,1.2,0.461,0.894,0.635,0.635,0.894,0.461,1.2,0.4,1.483,0.283,1.6,0.0,1.483,-0.283,1.2,-0.4,0.74,-0.5,0.351,-0.751,0.1,-1.14,0.0,-1.6,-0.1,-1.14,-0.351,-0.751,-0.74,-0.5,-1.2,-0.4,-1.483,-0.283,-1.6,0.0],
[-1.6,0.0,-1.483,0.283,-1.2,0.4,-0.894,0.461,-0.635,0.635,-0.461,0.894,-0.4,1.2,-0.283,1.483,0.0,1.6,0.283,1.483,0.4,1.2,0.461,0.894,0.635,0.635,0.894,0.461,1.2,0.4,1.483,0.283,1.6,0.0,1.483,-0.283,1.2,-0.4,0.894,-0.461,0.635,-0.635,0.461,-0.894,0.4,-1.2,0.283,-1.483,0.0,-1.6,-0.283,-1.483,-0.4,-1.2,-0.461,-0.894,-0.635,-0.635,-0.894,-0.461,-1.2,-0.4,-1.483,-0.283,-1.6,0.0],
[-0.571,0.605,0.0,1.6,1.131,1.131,1.6,0.0,1.2,-0.4,0.74,-0.5,0.351,-0.751,0.1,-1.14,0.0,-1.6,-0.212,-1.273,-0.381,-0.923,-0.575,-0.172,-0.571,0.605],
[-0.575,0.172,-0.381,0.923,-0.212,1.273,0.0,1.6,0.1,1.14,0.351,0.751,0.74,0.5,1.2,0.4,1.6,0.0,1.131,-1.131,0.0,-1.6,-0.571,-0.605,-0.575,0.172],
[-0.257,0.401,-0.054,0.983,0.0,1.6,0.054,0.983,0.257,0.401,0.605,-0.111,1.071,-0.516,1.131,-1.131,0.0,-1.6,-1.131,-1.131,-1.071,-0.516,-0.605,-0.111,-0.257,0.401],
[-0.64,0.0,-0.571,0.689,0.0,1.6,1.131,1.131,1.6,0.0,1.2,-0.4,0.894,-0.461,0.635,-0.635,0.461,-0.894,0.4,-1.2,0.283,-1.483,0.0,-1.6,-0.369,-1.354,-0.571,-0.689,-0.64,0.0],
[-0.64,0.0,-0.571,0.689,-0.369,1.354,0.0,1.6,0.283,1.483,0.4,1.2,0.461,0.894,0.635,0.635,0.894,0.461,1.2,0.4,1.6,0.0,1.131,-1.131,0.0,-1.6,-0.571,-0.689,-0.64,0.0],
[-0.487,0.198,-0.373,0.652,-0.393,1.122,-0.283,1.483,0.0,1.6,0.283,1.483,0.393,1.122,0.373,0.652,0.487,0.198,0.727,-0.203,1.071,-0.516,1.131,-1.131,0.0,-1.6,-1.131,-1.131,-1.071,-0.516,-0.727,-0.203,-0.487,0.198],
[-1.6,0.0,-1.2,0.4,-0.894,0.461,-0.635,0.635,-0.461,0.894,-0.4,1.2,-0.283,1.483,0.0,1.6,0.283,1.483,0.4,1.2,0.461,0.894,0.635,0.635,0.894,0.461,1.2,0.4,1.483,0.283,1.6,0.0,1.483,-0.283,1.2,-0.4,0.894,-0.461,0.635,-0.635,0.461,-0.894,0.4,-1.2,0.0,-1.6,-1.131,-1.131,-1.6,0.0],
[-1.6,0.0,-1.2,0.4,-0.894,0.461,-0.635,0.635,-0.461,0.894,-0.4,1.2,0.0,1.6,1.131,1.131,1.6,0.0,1.2,-0.4,0.894,-0.461,0.635,-0.635,0.461,-0.894,0.4,-1.2,0.0,-1.6,-1.131,-1.131,-1.6,0.0],
[-1.6,0.0,-1.2,0.4,-0.894,0.461,-0.635,0.635,-0.461,0.894,-0.4,1.2,0.0,1.6,1.131,1.131,1.6,0.0,1.131,-1.131,0.0,-1.6,-1.131,-1.131,-1.6,0.0],
[-1.6,0.0,-1.2,0.4,-0.894,0.461,-0.635,0.635,-0.461,0.894,-0.4,1.2,-0.283,1.483,0.0,1.6,0.283,1.483,0.4,1.2,0.461,0.894,0.635,0.635,0.894,0.461,1.2,0.4,1.6,0.0,1.131,-1.131,0.0,-1.6,-1.131,-1.131,-1.6,0.0],
[-1.6,0.0,-1.131,1.131,0.0,1.6,0.4,1.2,0.461,0.894,0.635,0.635,0.894,0.461,1.2,0.4,1.483,0.283,1.6,0.0,1.483,-0.283,1.2,-0.4,0.74,-0.5,0.351,-0.751,0.1,-1.14,0.0,-1.6,-0.1,-1.14,-0.351,-0.751,-0.74,-0.5,-1.2,-0.4,-1.6,0.0],
[-1.6,0.0,-1.483,0.283,-1.2,0.4,-0.894,0.461,-0.635,0.635,-0.461,0.894,-0.4,1.2,0.0,1.6,1.131,1.131,1.6,0.0,1.2,-0.4,0.74,-0.5,0.351,-0.751,0.1,-1.14,0.0,-1.6,-0.1,-1.14,-0.351,-0.751,-0.74,-0.5,-1.2,-0.4,-1.483,-0.283,-1.6,0.0],
[-1.6,0.0,-1.131,1.131,0.0,1.6,1.131,1.131,1.6,0.0,1.2,-0.4,0.74,-0.5,0.351,-0.751,0.1,-1.14,0.0,-1.6,-0.1,-1.14,-0.351,-0.751,-0.74,-0.5,-1.2,-0.4,-1.6,0.0],
[-1.6,0.0,-0.988,0.122,-0.468,0.468,-0.122,0.988,0.0,1.6,0.1,1.14,0.351,0.751,0.74,0.5,1.2,0.4,1.6,0.0,1.131,-1.131,0.0,-1.6,-0.4,-1.2,-0.5,-0.74,-0.751,-0.351,-1.14,-0.1,-1.6,0.0]]
```


## 2. Generate stimulus set


### 2.1 Generate HD version of the stimulus set

I would recommend generating high-definition version of the stimulus set first, and then resize them later for your specific layers.

Look at this [notebook](https://colab.research.google.com/drive/1Pfu1hcJq6U2f3XTKA_fuPtRV6xH0f72c#scrollTo=G7OCGOQkSFH9)

Here is my version of it:

```python
import numpy as np
import matplotlib.pyplot as plt

ROTATION_INCREMENT = 45  # 45 degrees increment for rotations

class Shape:
    def __init__(self, coords):
        self.coords = np.array(coords).reshape(len(coords)//2, 2)
        self.sampled_coords = self._sample_shape(self.coords)

    @staticmethod
    def _sample_shape(invec):
        sample = 50.0
        num = np.shape(invec)[0]
        inshft = np.vstack((invec[num-2,:], invec, invec[1,:]))
        ip = np.arange(0, 50, 1) / sample

        vtx, vty = [], []
        for i in range(0, num-1):
            bufvrt = inshft[i:i+4,:]
            incr = [
                -ip**3 + 3*ip**2 - 3*ip + 1,
                3*ip**3 - 6*ip**2 + 4,
                -3*ip**3 + 3*ip**2 + 3*ip + 1,
                ip**3
            ]
            vtx.extend(np.sum(np.tile(bufvrt[:,0].reshape(4,1), (1, len(ip))) * incr, axis=0) / 6.0)
            vty.extend(np.sum(np.tile(bufvrt[:,1].reshape(4,1), (1, len(ip))) * incr, axis=0) / 6.0)

        return np.array(list(zip(vtx, vty)))

    def rotate(self, rotation_index):
        rot_angle = rotation_index * ROTATION_INCREMENT * np.pi / 180.0
        rotX = np.cos(rot_angle) * self.sampled_coords[:,0] + np.sin(rot_angle) * self.sampled_coords[:,1]
        rotY = -np.sin(rot_angle) * self.sampled_coords[:,0] + np.cos(rot_angle) * self.sampled_coords[:,1]
        return rotX, rotY

    def render(self, rotation_index=0, img_size=(16, 16), background_color='k', shape_color='w', fill_shape=True):
        plt.figure(figsize=img_size)
        plt.axis('off')
        plt.fill([-4.0, 4.0, 4.0, -4.0], [-4.0, -4.0, 4.0, 4.0], background_color)

        rotX, rotY = self.rotate(rotation_index)
        if fill_shape:
            plt.fill(rotX, rotY, shape_color)
        else:
            plt.plot(rotX, rotY, color=shape_color)
        plt.show()  # You probably want to return the numpy array instead of showing it

# Example Usage:
shape_coords = [-0.294,0.122,-0.075,0.846,0.0,1.6,0.054,0.983,0.257,0.401,0.605,-0.111,1.071,-0.516,1.248,-0.848,1.131,-1.131,0.85,-1.25,0.57,-1.131,0.179,-0.871,-0.282,-0.78,-0.742,-0.871,-1.131,-1.131,-0.651,-0.546,-0.294,0.122]
shape = Shape(shape_coords)
shape.render(1, img_size=(6, 6), background_color='g', shape_color='y', fill_shape=False)
```

* Surface-Property Units: care if the shape is filled or not. Hypothesized to have lower dynamic range. More sensitive to color.
* Boundary-Property Units: don't care if the shape is filled or not. Hypothesized to have greater dynamic range. Invariant to color.


### 2.2 Rescale the stimulus set according to the RF and XN size of the layer.

Check out `model_info.txt` for the RF and XN size of the layer.

## 3. Get the model response

1. Get the pretrained model
2. Get the response up to the layer of interest
3. Get only the center responses of the layer of interest
4. Store the responses in a numpy array? or in a text file? 


## 4. Analyze the model response

Plot some plots, etc.
