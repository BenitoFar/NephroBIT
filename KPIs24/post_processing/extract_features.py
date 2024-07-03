import sys, os
sys.path.append('/home/benito/script/NephroBIT/KPIs24/utilites')
import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor
from radiomics.imageoperations import getWaveletImage, getLoGImage
import pandas as pd
from utilites import prepare_data
from radiomics import logging
import cv2
logger = logging.getLogger("radiomics")
logger.setLevel(logging.ERROR)

def extract_radiomics_features(image_path, segmentation_path, saveWavletImages = False):
    """
    Extract radiomics features from an image and its segmentation.

    Parameters
    ----------
    image_path : str
        The path to the image file.
    segmentation_path : str
        The path to the segmentation file.

    Returns
    -------
    dict
        A dictionary containing the extracted features.
    """
    
    # Load the image and segmentation using SimpleITK
    image = sitk.ReadImage(image_path, sitk.sitkInt8)
    segmentation = sitk.ReadImage(segmentation_path)

    #set the label value for the segmentation to 1 (the label value of the ROI) instead of 255
    segmentation = sitk.BinaryThreshold(segmentation, 0, 0, 0, 1)
    
    # Define the settings for the feature extractor
    settings = {}
    settings['binWidth'] = 25
    settings['resampledPixelSpacing'] = None  # [3,3,3] is an example for defining resampling (voxels with size 3x3x3mm)
    settings['interpolator'] = sitk.sitkBSpline
    settings['Label'] = 1
    # settings['imageType'] = {'Original': {}, 'LoG': {'sigma': [1.0, 3.0]}, 'Wavelet': {'binWidth': 25}}
    # settings['featureClass'] = {'firstorder': {}, 'shape2D': {}, 'glcm': {}, 'glrlm': {}, 'glszm': {}, 'ngtdm': {}, 'gldm': {}}
    settings['additionalInfo'] = False
    
    # Initialize the feature extractor
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)

    # Enable the desired features
    # extractor.enableAllFeatures()
    extractor.enableFeatureClassByName('shape2D')
    extractor.enableImageTypes(Original={}, Wavelet={'binWidth': 10})
    
    #extract from each segmentaion connected components and extract features from each connected component
    # Find connected components
    num_labels, labels = cv2.connectedComponents(sitk.GetArrayFromImage(segmentation).astype(np.uint8))
    # Initialize lists to store shape information
    features_df = pd.DataFrame()
    # Iterate over each connected component
    for label in range(1, num_labels):
        try:
            # Create a mask for the current component
            label_array = np.uint8(labels == label)
            mask = sitk.GetImageFromArray(label_array)
            # sitk.WriteImage(mask*255, os.path.join('/home/benito/script/NephroBIT/KPIs24/post_processing/masks', f'{segmentation_path.split('/')[-1].split('.jpg')[0]}_mask_label{label}.png'))
            #extract features
            features = pd.Series(extractor.execute(image, mask)).to_frame().T
            features_names = features.columns.tolist()
            features_values = features.values.tolist()[0]
            
            features_names = ["id", "class", "id_patch", "n_foreground_pixels", *features_names]
            features_values = [segmentation_path.split('/')[-3], segmentation_path.split('/')[-4], segmentation_path.split('/')[-1].split(".jpg")[0] + f"_{label}", label_array.sum() ,*features_values]
            
            features = pd.DataFrame([features_values], columns=features_names)
            
            #join the features of the connected components to the features of the entire image
            features_df = pd.concat([features_df, features], axis = 0)
        except:
            print(f"Error in extracting features from connected component {label}")
            continue
        
    if saveWavletImages:
        #to check what look like the wavalet and LoG images get them and save them
        #create an image with same dimension of image but with all 1s to extract wavelet and log images for the entire image
        image_mask = sitk.Image(image.GetSize(), sitk.sitkInt8)
        image_mask.CopyInformation(image)
        image_mask = sitk.BinaryThreshold(image_mask, 0, 0, 1, 0)
        
        wavelet_image = getWaveletImage(image, image_mask, binWidth=10)
        for wavimg in wavelet_image:
            sitk.WriteImage(wavimg[0], f"/home/benito/script/KPIs24/results/glomeroulous_features/images/wavelet_{wavimg[1]}_{segmentation_path.split('/')[-1].split('.jpg')[0]}.nrrd", useCompression=True)
            
    return features_df

def main():
    datadir = "/mnt/atlas/data_KPIs/data/KPIs24_Training_Data/Task1_patch_level/train/"
    data_list = prepare_data(datadir)
    
    #for img and maks in data_list extract radiomics features and store them in a dataframe
    features = pd.DataFrame()
    for i in range(len(data_list)):
        #print count of left images to process
        print(f"{i}/{len(data_list)}")
        features = pd.concat([features, extract_radiomics_features(data_list[i]['img'], data_list[i]['label'], saveWavletImages = False)], axis = 0)

    print('Features extracted from N = :', features.shape[0], 'glomeruli')
    # Save the DataFrame to a CSV file
    features.to_csv('/home/benito/script/KPIs24/results/glomeroulous_features/radiomics_features_entire_train_patches_by_glomeruli_updated.csv', index=False)
    
if __name__ == "__main__":
    main()