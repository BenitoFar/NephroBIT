import copy

import torch

from monai.networks.nets.flexible_unet import FlexibleUNet
from monai.networks.nets import SwinUNETR

class HoVerFlexibleNet(FlexibleUNet):
    """
    EfficientUNet from monai code adapted to add the HoVerNet decoder
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            backbone: str,
            pretrained: bool = False,
            hovermaps: bool = True,
            freeze_encoder: bool = False,
            freeze_decoder_bin: bool = False
    ) -> None:
        self.hovermaps = hovermaps
        self.freeze_encoder = freeze_encoder
        self.freeze_decoder_bin = freeze_decoder_bin
        super().__init__(in_channels, out_channels, backbone, pretrained)
        self.decoder_bin = self.decoder
        self.segmentation_head_bin = self.segmentation_head
        self.decoder_hover = copy.deepcopy(self.decoder)
        self.segmentation_head_hover = copy.deepcopy(self.segmentation_head)
        if not self.hovermaps:
            for name, para in self.decoder_hover.named_parameters():
                para.requires_grad = False
            for name, para in self.segmentation_head_hover.named_parameters():
                para.requires_grad = False

        # to freeze parts of the network
        if self.freeze_encoder:
            for name, para in self.encoder.named_parameters():
                para.requires_grad = False
        if self.freeze_decoder_bin:
            for name, para in self.decoder_bin.named_parameters():
                para.requires_grad = False
            for name, para in self.segmentation_head_bin.named_parameters():
                para.requires_grad = False

    def forward(self, inputs: torch.Tensor):
        """
        Do a typical encoder-decoder-header inference.

        Args:
            inputs: input should have spatially N dimensions ``(Batch, in_channels, dim_0[, dim_1, ..., dim_N])``,
                N is defined by `dimensions`.

        Returns:
            A torch Tensor of "raw" predictions in shape ``(Batch, out_channels, dim_0[, dim_1, ..., dim_N])``.

        """
        x = inputs
        enc_out = self.encoder(x)
        decoder_bin_out = self.decoder_bin(enc_out, self.skip_connect)
        x_bin_seg = self.segmentation_head_bin(decoder_bin_out)

        if self.hovermaps:
            decoder_hover_out = self.decoder_hover(enc_out, self.skip_connect)
            x_hover_seg = self.segmentation_head_hover(decoder_hover_out)

            outputs = {
                "pred_bin": x_bin_seg,
                "pred_hover": x_hover_seg
            }
        else:
            outputs = {
                "pred_bin": x_bin_seg,
            }
        return outputs


class HoVerSwinUNETR(SwinUNETR):
    """
    SwinUNETR from monai code adapted to add the HoVerNet decoder
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            img_size: int = 512,
            depths: list = [2, 2, 2, 2],
            num_heads: list = [3, 6, 12, 24],
            feature_size=24,
            spatial_dims = 2,
            hovermaps: bool = True,
            freeze_encoder: bool = False,
            freeze_decoder_bin: bool = False
    ) -> None:
        self.hovermaps = hovermaps
        self.freeze_encoder = freeze_encoder
        self.freeze_decoder_bin = freeze_decoder_bin
        super().__init__(
            in_channels, out_channels, img_size, depths, num_heads, feature_size, spatial_dims
        )
        self.decoder_bin = self.decoder
        self.segmentation_head_bin = self.segmentation_head
        self.decoder_hover = copy.deepcopy(self.decoder)
        self.segmentation_head_hover = copy.deepcopy(self.segmentation_head)
        if not self.hovermaps:
            for name, para in self.decoder_hover.named_parameters():
                para.requires_grad = False
            for name, para in self.segmentation_head_hover.named_parameters():
                para.requires_grad = False

        # to freeze parts of the network
        if self.freeze_encoder:
            for name, para in self.encoder.named_parameters():
                para.requires_grad = False
        if self.freeze_decoder_bin:
            for name, para in self.decoder_bin.named_parameters():
                para.requires_grad = False
            for name, para in self.segmentation_head_bin.named_parameters():
                para.requires_grad = False

    def forward(self, inputs: torch.Tensor):
        """
        Do a typical encoder-decoder-header inference.

        Args:
            inputs: input   
        
        Returns:
            A torch Tensor of "raw" predictions in shape ``(Batch, out_channels, dim_0[, dim_1, ..., dim_N])``.

        """
        
        x = inputs
        enc_out = self.encoder(x)
        decoder_bin_out = self.decoder_bin(enc_out)
        x_bin_seg = self.segmentation_head_bin(decoder_bin_out)

        if self.hovermaps:
            decoder_hover_out = self.decoder_hover(enc_out)
            x_hover_seg = self.segmentation_head_hover(decoder_hover_out)

            outputs = {
                "pred_bin": x_bin_seg,
                "pred_hover": x_hover_seg
            }
        else:
            outputs = {
                "pred_bin": x_bin_seg,
            }
        return outputs
    
    