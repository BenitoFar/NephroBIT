import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from tqdm import tqdm
from skimage.filters import rank, threshold_otsu
from scipy import ndimage as ndi
from skimage.morphology import disk, ball, remove_small_objects, binary_dilation, binary_erosion, binary_closing, binary_opening
from skimage.segmentation import watershed, find_boundaries, relabel_sequential

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

