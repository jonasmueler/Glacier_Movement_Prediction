from geoarray import GeoArray
from arosics import COREG, COREG_LOCAL
import numpy as np
import functions
import matplotlib.pyplot as plt

arr = functions.openData("/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/datasets/parbati/monthlyAveragedScenesUnalligned/parbatiScenes/0")
arr = GeoArray(arr)
ndarray = arr[:]            # numpy.ndarray with shape (10980, 10980)
gt = arr.geotransform  # GDAL geotransform: (300000.0, 10.0, 0.0, 5900040.0, 0.0, -10.0)
prj = arr.projection

ref_arr = functions.openData("/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/datasets/parbati/monthlyAveragedScenesUnalligned/parbatiScenes/1")
refArr = GeoArray(ref_arr)
refNdarray = refArr[:]            # numpy.ndarray with shape (10980, 10980)
refGt = refArr.geotransform  # GDAL geotransform: (300000.0, 10.0, 0.0, 5900040.0, 0.0, -10.0)
refPrj = refArr.projection

geoArr_reference = GeoArray(refNdarray) #, refGt, refPrj)
geoArr_target = GeoArray(ndarray) #, gt, prj)

kwargs = {
    'grid_res'     : 20,
    'window_size'  : (20,20),
    'path_out'     : '/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/datasets',
    'projectDir'   : 'my_project',
    'q'            : False,
}

CR = COREG_LOCAL(geoArr_reference, geoArr_target, **kwargs)#, wp=(354223, 5805559), ws=(256,256))
res = CR.correct_shifts()
print(res.keys())
print(np.sum(res["arr_shifted"] - arr))

def imageAlignArcosics(img, reference):
    """
    aligns the array to the reference array using fourier space transformations

    img: np.array
    reference: np.array
    return: np.array
        aligned array to the input
    """

    # convert to geoARray
    img = GeoArray(img)
    reference = GeoArray(reference)

    # parameters
    kwargs = {
        'grid_res': 20,
        'window_size': (20, 20),
        'path_out': '/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/datasets',
        'projectDir': 'my_project',
        'q': False,
    }

    # apply
    CR = COREG_LOCAL(img, reference, **kwargs)
    res = CR.correct_shifts()

    return res["arr_shifted"]



