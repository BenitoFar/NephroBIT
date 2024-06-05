from typing import Dict

import torch
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss

from monai.losses import DiceCELoss
from monai.transforms import SobelGradients

class ClassificationLoss(_Loss):
    """
    Copied from monai hovernet code
    Loss function for HoVerNet pipeline, which is combination of losses across the three branches.
    The NP (nucleus prediction) branch uses Dice + CrossEntropy.
    The HV (Horizontal and Vertical) distance from centroid branch uses MSE + MSE of the gradient.
    The NC (Nuclear Class prediction) branch uses Dice + CrossEntropy
    The result is a weighted sum of these losses.

    Args:
        lambda_hv_mse: Weight factor to apply to the HV regression MSE part of the overall loss
        lambda_hv_mse_grad: Weight factor to apply to the MSE of the HV gradient part of the overall loss
        lambda_ce: Weight factor to apply to the nuclei prediction CrossEntropyLoss part
            of the overall loss
        lambda_dice: Weight factor to apply to the nuclei prediction DiceLoss part of overall loss
        lambda_nc_ce: Weight factor to apply to the nuclei class prediction CrossEntropyLoss part
            of the overall loss
        lambda_nc_dice: Weight factor to apply to the nuclei class prediction DiceLoss part of the
            overall loss

    """

    def __init__(
        self,
        lambda_ce_bin: float = 1.0,
        lambda_dice_bin: float = 1.0,
        lambda_ce_class: float = 1.0,
        lambda_dice_class: float = 1.0,
        classification: bool = True
    ) -> None:
        self.lambda_ce_bin = lambda_ce_bin
        self.lambda_dice_bin = lambda_dice_bin
        self.lambda_ce_class = lambda_ce_class
        self.lambda_dice_class = lambda_dice_class
        self.classification = classification
        super().__init__()

        self.DiceCELoss_bin = DiceCELoss(softmax=True, lambda_dice=self.lambda_dice_bin, lambda_ce=self.lambda_ce_bin, to_onehot_y=True)
        self.DiceCELoss_class = DiceCELoss(softmax=True, lambda_dice=self.lambda_dice_class, lambda_ce=self.lambda_ce_class, to_onehot_y=True)

    def forward(self, prediction: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            prediction: dictionary of predicted outputs for three branches,
                each of which should have the shape of BNHW.
            target: dictionary of ground truths for three branches,
                each of which should have the shape of BNHW.
        """
        loss_bin = self.DiceCELoss(prediction["pred_bin"], target["label_bin"])

        if self.classification:
            loss_class = self.DiceCELoss(prediction["pred_class"], target["label_class"])
            loss = loss_bin + loss_class
        else:
            loss = loss_bin

        return loss

class HoVerNetLoss(_Loss):
    """
    Copied from monai hovernet code
    Loss function for HoVerNet pipeline, which is combination of losses across the three branches.
    The NP (nucleus prediction) branch uses Dice + CrossEntropy.
    The HV (Horizontal and Vertical) distance from centroid branch uses MSE + MSE of the gradient.
    The NC (Nuclear Class prediction) branch uses Dice + CrossEntropy
    The result is a weighted sum of these losses.

    Args:
        lambda_hv_mse: Weight factor to apply to the HV regression MSE part of the overall loss
        lambda_hv_mse_grad: Weight factor to apply to the MSE of the HV gradient part of the overall loss
        lambda_ce: Weight factor to apply to the nuclei prediction CrossEntropyLoss part
            of the overall loss
        lambda_dice: Weight factor to apply to the nuclei prediction DiceLoss part of overall loss
        lambda_nc_ce: Weight factor to apply to the nuclei class prediction CrossEntropyLoss part
            of the overall loss
        lambda_nc_dice: Weight factor to apply to the nuclei class prediction DiceLoss part of the
            overall loss

    """

    def __init__(
        self,
        lambda_ce: float = 1.0,
        lambda_dice: float = 1.0,
        lambda_hv_mse: float = 2.0,
        lambda_hv_mse_grad: float = 1.0,
        hovermaps: bool = True
    ) -> None:
        self.lambda_ce = lambda_ce
        self.lambda_dice = lambda_dice
        self.lambda_hv_mse = lambda_hv_mse
        self.lambda_hv_mse_grad = lambda_hv_mse_grad
        self.hovermaps = hovermaps
        super().__init__()

        self.DiceCELoss = DiceCELoss(softmax=True, lambda_dice=self.lambda_dice, lambda_ce=self.lambda_ce, to_onehot_y=True)
        self.sobel_v = SobelGradients(kernel_size=5, spatial_axes=0)
        self.sobel_h = SobelGradients(kernel_size=5, spatial_axes=1)

    def _compute_sobel(self, image: torch.Tensor) -> torch.Tensor:
        """Compute the Sobel gradients of the horizontal vertical map (HoVerMap).
        More specifically, it will compute horizontal gradient of the input horizontal gradient map (channel=0) and
        vertical gradient of the input vertical gradient map (channel=1).

        Args:
            image: a tensor with the shape of BxCxHxW representing HoVerMap

        """
        result_h = self.sobel_h(image[:, 0])
        result_v = self.sobel_v(image[:, 1])
        return torch.stack([result_h, result_v], dim=1)

    def _mse_gradient_loss(self, prediction: torch.Tensor, target: torch.Tensor, focus: torch.Tensor) -> torch.Tensor:
        """Compute the MSE loss of the gradients of the horizontal and vertical centroid distance maps"""

        pred_grad = self._compute_sobel(prediction)
        true_grad = self._compute_sobel(target)

        loss = pred_grad - true_grad

        # The focus constrains the loss computation to the detected nuclear regions
        # (i.e. background is excluded)
        focus = focus[:, None, ...]
        focus = torch.cat((focus, focus), 1)

        loss = focus * (loss * loss)
        loss = loss.sum() / (focus.sum() + 1.0e-8)

        return loss

    def forward(self, prediction: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            prediction: dictionary of predicted outputs for three branches,
                each of which should have the shape of BNHW.
            target: dictionary of ground truths for three branches,
                each of which should have the shape of BNHW.
        """
        loss_bin = self.DiceCELoss(prediction["pred_bin"], target["label_bin"])

        if self.hovermaps:
            # Compute the HV branch loss
            loss_hv_mse = (
                F.mse_loss(prediction["pred_hover"], target["label_hover"]) * self.lambda_hv_mse
            )

            # Use the nuclei class, one hot encoded, as the mask
            loss_hv_mse_grad = (
                self._mse_gradient_loss(
                    prediction["pred_hover"],
                    target["label_hover"],
                    target["label_bin"][:, 0],
                )
                * self.lambda_hv_mse_grad
            )
            loss_hover = loss_hv_mse_grad + loss_hv_mse
            loss = loss_bin + loss_hover
        else:
            loss = loss_bin

        return loss
