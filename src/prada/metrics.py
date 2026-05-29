from typing import List, Literal, Optional, Union

import torch
from torch import Tensor, nn
from torchmetrics.functional.image.lpips import (
    _LPIPS,
    _lpips_compute,
    _lpips_update,
    _normalize_tensor,
    _resize_tensor,
    _spatial_average,
    _upsample,
)

from prada.misc import apply_to_dict


class _PatchedLPIPS(_LPIPS):
    """Patched version of LPIPS that only returns second layer, as proposed in AEROBLADE"""

    def forward(
        self, in0: Tensor, in1: Tensor, retperlayer: bool = False, normalize: bool = False
    ) -> Union[Tensor, tuple[Tensor, List[Tensor]]]:
        if normalize:  # turn on this flag if input is [0,1] so it can be adjusted to [-1, +1]
            in0 = 2 * in0 - 1
            in1 = 2 * in1 - 1

        # normalize input
        in0_input, in1_input = self.scaling_layer(in0), self.scaling_layer(in1)

        # resize input if needed
        if self.resize is not None:
            in0_input = _resize_tensor(in0_input, size=self.resize)
            in1_input = _resize_tensor(in1_input, size=self.resize)

        outs0, outs1 = self.net.forward(in0_input), self.net.forward(in1_input)
        feats0, feats1, diffs = {}, {}, {}

        for kk in range(self.L):
            feats0[kk], feats1[kk] = _normalize_tensor(outs0[kk]), _normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        res = []
        for kk in range(self.L):
            if self.spatial:
                res.append(_upsample(self.lins[kk](diffs[kk]), out_hw=tuple(in0.shape[2:])))
            else:
                res.append(_spatial_average(self.lins[kk](diffs[kk]), keep_dim=True))

        val: Tensor = sum(res)  # type: ignore[assignment]
        if retperlayer:
            return (val, res)
        # return val
        return res[1]  # return second layer


class _PatchedNoTrainLpips(_PatchedLPIPS):
    """Wrapper to make sure LPIPS never leaves evaluation mode."""

    def train(self, mode: bool) -> "_NoTrainLpips":  # type: ignore[override]
        """Force network to always be in evaluation mode."""
        return super().train(False)


def aeroblade(
    img1: Tensor,
    img2: Tensor,
    net_type: Literal["alex", "vgg", "squeeze"] = "alex",
    reduction: Optional[Literal["sum", "mean", "none"]] = "mean",
    normalize: bool = False,
) -> Tensor:
    net = _PatchedNoTrainLpips(net=net_type).to(device=img1.device, dtype=img1.dtype)
    loss = _lpips_update(img1, img2, net, normalize)
    return _lpips_compute(loss, reduction)


def icas(
    diff_gt_llh_BL: Tensor,
    a: float = 1.75,
    b: float = 1.3,
) -> Tensor:
    """Yu et al. 2025, ICAS: Detecting Training Data from Autoregressive Image Generative Models
    Reference: https://github.com/Chrisqcwx/ImageAR-MIA/blob/2444168aabb87ef97a3541e202295c8c70cdf93f/icas.py#L156
    """
    omega_BL = 1 / (a + torch.exp(b * diff_gt_llh_BL))
    icas_BL = omega_BL * diff_gt_llh_BL

    return icas_BL