import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from tqdm import tqdm
from skimage.filters import rank, threshold_otsu
from scipy import ndimage as ndi
from skimage.morphology import disk, ball, remove_small_objects, binary_dilation, binary_erosion, binary_closing, binary_opening
from skimage.segmentation import watershed, find_boundaries, relabel_sequential
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian
import pydensecrf.densecrf as dcrf
from skimage.color import gray2rgb
from skimage.color import rgb2gray

def crf(original_image, predictions, use_2d = True):
    predictions = predictions.squeeze()*255
    original_image = original_image.squeeze()*255
    
    # Converting annotated image to RGB if it is Gray scale
    if(len(predictions.shape)<3):
        predictions = gray2rgb(predictions)
    
    cv2.imwrite("/home/benito/script/NephroBIT/KPIs24/post_processing/testing2.png",predictions)
    predictions = predictions.astype(np.uint32)
    original_image = np.moveaxis(original_image, 0, -1).astype(np.uint32)
    
    # #Converting the annotations RGB color to single 32 bit integer
    annotated_label = predictions[:,:,0].astype(np.uint32) + (predictions[:,:,1]<<8).astype(np.uint32) + (predictions[:,:,2]<<16).astype(np.uint32)
    # annotated_label = predictions
    # Convert the 32bit integer color to 0,1, 2, ... labels.
    colors, labels = np.unique(annotated_label, return_inverse=True)
    
    #Creating a mapping back to 32 bit colors
    colorize = np.empty((len(colors), 3), np.uint8)
    colorize[:,0] = (colors & 0x0000FF)
    colorize[:,1] = (colors & 0x00FF00) >> 8
    colorize[:,2] = (colors & 0xFF0000) >> 16
    
    #Gives no of class labels in the annotated image
    n_labels = len(set(labels.flat)) 
    
    # print("No of labels in the Image are ")
    # print(n_labels)
    
    #Setting up the CRF model
    if use_2d:
        d = dcrf.DenseCRF2D(original_image.shape[1], original_image.shape[0], n_labels)

        # get unary potentials (neg log probability)
        U = unary_from_labels(labels, n_labels, gt_prob=0.90, zero_unsure=False)
        d.setUnaryEnergy(U.astype(np.float32))

        # This adds the color-independent term, features are the locations only.
        d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                          normalization=dcrf.NORMALIZE_SYMMETRIC)

        # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
        d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=np.uint8(original_image.copy(order='C')),
                           compat=10,
                           kernel=dcrf.DIAG_KERNEL,
                           normalization=dcrf.NORMALIZE_SYMMETRIC)
        
    #Run Inference for 5 steps 
    Q = d.inference(5)

    # Find out the most probable class for each pixel.
    MAP = np.argmax(Q, axis=0)

    # Convert the MAP (labels) back to the corresponding colors and save the image.
    # Note that there is no "unknown" here anymore, no matter what we had at first.
    MAP = colorize[MAP,:]
    # cv2.imwrite(output_image,MAP.reshape(original_image.shape))
    output_image = MAP.reshape(original_image.shape)
    #rgb to gray
    output_image = rgb2gray(output_image)
    
    return output_image.astype(np.uint8)

def watershed_refinement(image, mask, save_dir=None):
    """Apply watershed to the given predictions with the goal of refine the boundaries of the artifacts. """

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    watershed_predictions = np.zeros(mask.shape, dtype=np.uint8)
    for ch in range(image.shape[-1]):
        markers = rank.gradient(mask[:,:,ch], disk(12)) < 10
        markers = ndi.label(markers)[0]
        gradient = rank.gradient(image[:,:,ch], disk(2))
        labels = watershed(gradient, markers)
        watershed_predictions[:,:,ch] = labels
        
        if save_dir is not None:
            f = os.path.join(save_dir, "mark_" + str(ch) + ".png")
            cv2.imwrite(f, markers)
    watershed_predictions = np.mean(watershed_predictions, axis=-1)
    watershed_predictions[watershed_predictions==1] = 0
    watershed_predictions[watershed_predictions>1] = 1

    return np.expand_dims(watershed_predictions, -1)


def apply_morphological_operations(mask, operations=['erosion'], radius=[3]):
    """Apply morphological operations to the given mask."""
    morpholofical_func =[]
    for operation in operations:
        if operation == 'erosion':
            morpholofical_func.append(binary_erosion)
        elif operation == 'dilation':
            morpholofical_func.append(binary_dilation)
        elif operation == 'closing':
            morpholofical_func.append(binary_closing)
        elif operation == 'opening':
            morpholofical_func.append(binary_opening)
        else:
            raise ValueError("Invalid operation. Choose either 'erosion', 'dilation', or 'closing' or 'opening'.")
    
    for i, operation in enumerate(operations):
        mask = morpholofical_func[i](mask, disk(radius[0]))
    
    return mask

