import cv2
import numpy as np

# Load image stack
image_stack = np.random.rand(20, 2, 700, 700).astype('float32')

def enhancedCorAlign(imgStack):
    """
    aligsn images to mean temporal image

    imgStack: np.array
        (N, C, X, Y)
    returns: list of np.array
        [(C, X, Y)...]
    """

    # switch channel dimensions
    imgStack = np.transpose(imgStack, (0, 2, 3, 1))

    # create new array to fill
    imageStack = np.zeros((imgStack.shape[0], imgStack.shape[1], imgStack.shape[2], 3))

    #add 3rd channel dimension
    for i in range(len(imgStack)):
        imageStack[i] = np.stack((imgStack[i, :, :, 0], imgStack[i, :, :, 1], imgStack[i, :, :, 1]), axis = 2)


    # Compute average temporal image
    imgStack = imageStack.astype('float32')
    avgTemporalImage = np.mean(imgStack, axis=0).astype('float32')

    # Compute gradient of averaged RGB channels
    grayAvgTemporalImage = cv2.cvtColor(avgTemporalImage, cv2.COLOR_BGR2GRAY)
    gradientAvgTemporalImage = cv2.Sobel(grayAvgTemporalImage, cv2.CV_64F, 1, 1, ksize=3).astype('float32')

    # Define motion model
    motionModel = cv2.MOTION_TRANSLATION

    # Define ECC algorithm parameters
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10000, 1e-9)
    warpMatrix = np.eye(2, 3, dtype=np.float32)

    # Define ECC mask
    mask = None # all pixels used as missings are already cleared

    # Apply ECC registration
    registeredStack = []
    for frame in imageStack:
        grayFrame = cv2.cvtColor(frame.astype('float32'), cv2.COLOR_BGR2GRAY)
        gradientFrame = cv2.Sobel(grayFrame, cv2.CV_64F, 1, 1, ksize=3).astype('float32')
        (cc, warpMatrix) = cv2.findTransformECC(gradientAvgTemporalImage, gradientFrame, warpMatrix, motionModel, criteria, mask, 1)
        registeredFrame = cv2.warpAffine(frame, warpMatrix, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        registeredFrame = np.transpose(registeredFrame, (2,0,1))[[0,1], :, :]
        registeredStack.append(registeredFrame)

    return registeredStack


#print(enhancedCorAlign(image_stack)[0].shape)


