import sys, os
sys.path.append('/home/benito/script/NephroBIT/KPIs24/utilites')
import glob as glob
import pandas as pd
import torch
import torch.nn as nn
import functools
import cv2
from PIL import Image
from torchvision import transforms
from utilites import seed_everything, prepare_data
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)
    
def define_D(input_nc, ndf = 64, netD = 'basic', n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=nn.BatchNorm2d)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return net.to(gpu_ids[0])

    
def save_predictions(data_dict, model, output_file = None):
    # Rest of the code remains unchanged
    predictions = pd.DataFrame(columns = ['id', 'class', 'real_or_fake', 'probability'])
    for image_path, mask_path in zip(data_dict['image'], data_dict['mask']):
        image_tensor = torch.unsqueeze(transforms.ToTensor()(Image.open(image_path).convert('RGB')), 0)
        mask_tensor = transforms.ToTensor()(Image.open(mask_path))[0][None, None, ...]
        input_tensor = torch.cat([mask_tensor, image_tensor], axis = 1)

        prediction = model(input_tensor.to('cuda')).mean().item() #torch.nn.Softmax(0)()
        predictions = pd.concat([predictions, pd.DataFrame({'id': image_path.split('/')[-1].split('_mask')[0], 'class': image_path.split('/')[-1].split('_')[0], 
                                          'real_or_fake': ('fake' if 'fake' in image_path else 'real'), 
                                          'probability': prediction}, index=[0])])

    # Save the dataframe to a file
    if output_file is not None:
        predictions.to_csv(output_file, index=False)
    
    return predictions 

def main():
    
    #define images path
    fakedatapath = '/mnt/atlas/data_KPIs/data/KPIs24_Training_Data/Task1_patch_level/pix2pix_data_augmentation/' #to_56Nx/parches_generados/
    realdatapath = '/mnt/atlas/data_KPIs/data/KPIs24_Training_Data/Task1_patch_level/split_small_patches_updated/'
    
    #read paths of images from the folder (they end with _img.jpg)
    realdata_image = [os.path.join(root, file) for root, dirs, files in os.walk(realdatapath) for file in files if file.endswith(".jpg") and 'img' in file]
    #replace img with mask
    realdata_mask = [path.replace('img','mask') for path in realdata_image]
    
    fakedata_image = [os.path.join(root, file) for root, dirs, files in os.walk(fakedatapath) for file in files if file.endswith(".png") and 'fake' in file and 'parches_generados' in root]
    fakedata_mask = [path.replace('fake_B','real_A') for path in fakedata_image]
    
    #join dictories
    data_image = fakedata_image + realdata_image 
    data_mask = fakedata_mask + realdata_mask
    
    data_dict = {'image': data_image, 'mask': data_mask}
    
    #load model and weights
    model = define_D(input_nc=4, gpu_ids=[0])
    model.load_state_dict(torch.load('/home/benito/script/NephroBIT/KPIs24/evaluate/discriminator_pix2pix.pth'))
    
    # image_tensor = torch.unsqueeze(transforms.ToTensor()(Image.open(data_image[0]).convert('RGB')), 0)
    # mask_tensor = transforms.ToTensor()(Image.open(data_mask[0]))[0][None, None, ...]
    # input_tensor = torch.cat([mask_tensor, image_tensor], axis = 1)
    # prediction = model(input_tensor.to('cuda')).mean().item()
    predictions = save_predictions(data_dict, model, 'predictions_discriminator.csv')
    #get the number of real and fake images
    n_real = len(predictions[predictions['real_or_fake'] == 'real'])
    n_fake = len(predictions[predictions['real_or_fake'] == 'fake'])
    print('Number of real images: ', n_real)
    print('Number of fake images: ', n_fake)
    
    #get the number of real and fake images that were correctly classified
    n_real_correct = len(predictions[(predictions['real_or_fake'] == 'real') & (predictions['probability'] > 0.5)])
    n_fake_correct = len(predictions[(predictions['real_or_fake'] == 'fake') & (predictions['probability'] < 0.5)])
    print('Number of real images correctly classified: ', n_real_correct)
    print('Number of fake images correctly classified: ', n_fake_correct)
    
    #get the number of real and fake images that were incorrectly classified
    n_real_incorrect = len(predictions[(predictions['real_or_fake'] == 'real') & (predictions['probability'] < 0.5)])
    n_fake_incorrect = len(predictions[(predictions['real_or_fake'] == 'fake') & (predictions['probability'] > 0.5)])
    print('Number of real images incorrectly classified: ', n_real_incorrect)
    print('Number of fake images incorrectly classified: ', n_fake_incorrect)
    
    
if __name__ == "__main__":
    main()