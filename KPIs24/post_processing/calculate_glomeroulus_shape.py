import cv2
import numpy as np
from utilites import prepare_data
from monai.apps.nuclick.transforms import SplitLabelMined
from monai.transforms import (
    Activations,
    EnsureChannelFirstd,
    AsDiscrete,
    Compose,
    LoadImaged,
    RandCropByPosNegLabeld,
    RandRotate90d,
    ScaleIntensityRangeD,
    RandFlipd,
    SelectItemsd,
    LabelToMaskd,
    Resized,
    OneOf, 
    RandGaussianSmoothd, 
    MedianSmoothd, 
    RandGaussianNoised
)
import SimpleITK as sitk
import radiomics
import pandas as pd
import matplotlib.pyplot as plt

def remove_border_components(mask):
    #Define a mask with the same dimension of mask, but with 1s in the borders and 0s in the center
    border_mask = np.zeros_like(mask)
    border_mask[0, :] = 1
    border_mask[-1, :] = 1
    border_mask[:, 0] = 1
    border_mask[:, -1] = 1
    #Remove the components that are in the border
    masksUnion = mask * border_mask
    #get connected components
    num_labels, labels = cv2.connectedComponents(masksUnion)
    #remove the components that are in the border
    mask = np.uint8(labels == 0)
    
    return mask

def binarize_mask(mask):
    # Binarize mask
    mask = np.uint8(mask > 128)

    # Fill holes
    # mask = cv2.fillPoly(mask, mask, 1)

    return mask

def calculate_shape_features(mask):
    #convert to simpleItk image
    maskImg = sitk.GetImageFromArray(mask) 
    #extract radiomics shape features from mask
    feature_extractor = radiomics.featureextractor.RadiomicsFeatureExtractor()
    feature_extractor.enableImageTypes(Original={})
    feature_extractor.disableAllFeatures()
    feature_extractor.enableFeatureClassByName('shape2D')
    #remove additional info
    feature_extractor.settings['additionalInfo'] = False
    
    feature_vector = feature_extractor.execute(maskImg, maskImg)
    print("Radiomics features:", feature_vector)
    
     # Calculate area
    area = np.sum(mask)
    print('Percentage of mask area over total image area:', area / np.prod(mask.shape) * 100, '%')
    
    #add area to feature vector
    feature_vector['original_shape2D_Area%'] = area / np.prod(mask.shape) * 100
    
    # # Calculate diameter using equivalent diameter
    # _, _, _, (diameter_x, diameter_y) = cv2.minMaxLoc(cv2.distanceTransform(mask, cv2.DIST_L2, 3))

    # # Calculate perimeter using contour
    # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # perimeter = cv2.arcLength(contours[0], True)        
    
    return feature_vector

def get_components_features(mask):
    # Find connected components
    num_labels, labels = cv2.connectedComponents(mask)

    # Initialize lists to store shape information
    feature_df = pd.DataFrame()
    
    # Iterate over each connected component
    for label in range(1, num_labels):
        
        # Create a mask for the current component
        mask = np.uint8(labels == label)
        #extract features
        try:
            feature_vector = calculate_shape_features(mask)
        except:
            #feature_vector set all columns to nan
            print("Error calculating features for component", label)
            feature_vector = np.nan
        
        feature_df = pd.concat([feature_df, pd.DataFrame([feature_vector])], ignore_index=True)

    return feature_df
        

def main():
    datadir =  "/data/KPIs24/KPIs24 Training Data/Task1_patch_level/data/"
    data_list = prepare_data(datadir)
    
    label_list = [d['label'] for d in data_list]

    df_shape_features = pd.DataFrame(columns=['label_path', 'components', 'original_shape2D_Elongation', 'original_shape2D_MajorAxisLength', 'original_shape2D_MaximumDiameter', 'original_shape2D_MeshSurface', 'original_shape2D_MinorAxisLength', 'original_shape2D_Perimeter', 'original_shape2D_PerimeterSurfaceRatio', 'original_shape2D_PixelSurface', 'original_shape2D_Sphericity', 'original_shape2D_Area%'])
    
    # for label in label_list:
    #     print(label)
    #     #read mask label to pass to calculate_shape
    #     label_image = cv2.imread(label, cv2.IMREAD_GRAYSCALE)
    #     #process the image to binarize it
    #     label_image = binarize_mask(label_image)
    #     #extract features
    #     feature_df = get_components_features(label_image)
    #     feature_df['label_path'] = label
    #     feature_df.loc[:, 'components'] = range(len(feature_df))
    #     #merge features
    #     df_shape_features = pd.concat([df_shape_features, feature_df], ignore_index=True)
    
    # #save features to csv
    # df_shape_features.to_csv('glomeroulous_shape_features_patches.csv', index=False)
    df_shape_features = pd.read_csv('glomeroulous_shape_features_patches.csv')
    
    #for each feature plot the histogram and boxplot and save it
    
    for column in df_shape_features.columns[2:-1]:
        df_shape_features_column = df_shape_features[column].astype(float)
        figure = plt.figure()
        ax = df_shape_features_column.plot.hist(bins=100)
        ax.axvline(df_shape_features_column.mean(), color='r', linestyle='--', label='Mean')
        ax.axvline(df_shape_features_column.median(), color='g', linestyle='--', label='Median')
        ax.axvline(df_shape_features_column.quantile(0.25), color='b', linestyle='--', label='25th Percentile')
        ax.axvline(df_shape_features_column.quantile(0.75), color='m', linestyle='--', label='75th Percentile')

        # Add text for the number of each metric
        ax.text(df_shape_features_column.mean(), ax.get_ylim()[1]-100, f"Mean: {df_shape_features_column.mean():.2f}", color='r', ha='center', va='bottom')
        ax.text(df_shape_features_column.median(), ax.get_ylim()[1]-200, f"Median: {df_shape_features_column.median():.2f}", color='g', ha='center', va='bottom')
        ax.text(df_shape_features_column.quantile(0.25), ax.get_ylim()[1]-300, f"25th Percentile: {df_shape_features_column.quantile(0.25):.2f}", color='b', ha='center', va='bottom')
        ax.text(df_shape_features_column.quantile(0.75), ax.get_ylim()[1]-400, f"75th Percentile: {df_shape_features_column.quantile(0.75):.2f}", color='m', ha='center', va='bottom')
        plt.legend()
        plt.savefig(column + '_hist.png')
        
        # figure = plt.figure()
        # ax = df_shape_features.boxplot(column)
        # ax.axhline(df_shape_features_column.mean(), color='r', linestyle='--', label='Mean')
        # ax.axhline(df_shape_features_column.median(), color='g', linestyle='--', label='Median')
        # ax.axhline(df_shape_features_column.quantile(0.25), color='b', linestyle='--', label='25th Percentile')
        # ax.axhline(df_shape_features_column.quantile(0.75), color='m', linestyle='--', label='75th Percentile')
        # plt.legend()
        # plt.savefig(column + '_boxplot.png')
        
        
        
if __name__ == "__main__":
    main()
    