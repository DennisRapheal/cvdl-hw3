import torch
from torch import nn
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn,
    maskrcnn_resnet50_fpn_v2,
    MaskRCNN_ResNet50_FPN_Weights,          # v1 權重列舉
    MaskRCNN_ResNet50_FPN_V2_Weights        # v2 權重列舉
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn   import MaskRCNNPredictor

# model_with_maps.py
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models.detection.mask_rcnn import MaskRCNN
from torchvision.models.detection.image_list import ImageList

# -- your ExtraHead stays exactly the same -------------------------------
class ExtraHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, name: str):
        super().__init__()
        self.name = name
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, out_channels, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


# -----------------------------------------------------------------------
class MaskRCNNWithMaps(MaskRCNN):
    """
    Wrapper that adds two auxiliary heads producing
    * center heat‑map   (B,1,H,W)
    * boundary heat‑map (B,1,H,W)

    Expect each sample’s `targets` dict (training‑time only) to contain
        'center_map':   Tensor[1,H,W]  – binary mask (0/1)
        'boundary_map': Tensor[1,H,W]  – binary mask (0/1)
    """
    def __init__(self, base_model: MaskRCNN,
                 loss_weight: float = 1.0,
                 bce_pos_weight: float | None = None):
        # steal everything from the already‑built base_model
        super().__init__(base_model.backbone,
                         base_model.rpn,
                         base_model.roi_heads,
                         base_model.transform)
        # ----------------------------------------------------------------
        in_ch = base_model.backbone.out_channels  # FPN: 256
        self.center_head   = ExtraHead(in_ch, 1, "center")
        self.boundary_head = ExtraHead(in_ch, 1, "boundary")

        self._aux_w  = loss_weight
        self._pos_w  = None
        if bce_pos_weight is not None:
            self._pos_w = torch.tensor([bce_pos_weight])

    # --------------------------------------------------------------------
    def forward(self, images, targets=None):
        """
        * training  →  losses dict  (main + auxiliary)
        * eval      →  predictions list, each with extra
                       'center_logits' and 'boundary_logits'
        """
        if self.training:
            assert targets is not None, "targets required in train mode"

        # 1) standard pre‑processing (normalisation + batching)
        images, targets = self.transform(images, targets)
        # images -> ImageList  (tensors, image_sizes)

        # 2) backbone
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = {"0": features}

        # ----------------------------------------------------------------
        # -------  AUX HEADS (use highest‑res FPN map ‘0’) ----------------
        fmap = features["0"]
        center_logits   = self.center_head(fmap)      # (B,1,h,w)
        boundary_logits = self.boundary_head(fmap)    # (B,1,h,w)

        # 3) main MaskRCNN forward ----------------------------------------
        detections, detector_losses = \
            self.roi_heads(features, images.image_sizes, targets)
        detections = self.transform.postprocess(detections,
                                                images.image_sizes,
                                                [t["orig_size"] for t in targets]
                                                if self.training else images.image_sizes)

        # 4) attach logits to predictions (in eval mode only)
        if not self.training:
            for d, c, b in zip(detections,
                               center_logits.sigmoid(),   # (1,h,w)
                               boundary_logits.sigmoid()):
                d["center_logits"]   = c.squeeze(0)
                d["boundary_logits"] = b.squeeze(0)
            return detections

        # 5) -------  AUXILIARY LOSSES -----------------------------------
        aux_losses = {}
        pos_w = self._pos_w.to(center_logits.device) if self._pos_w is not None else None
        for name, logits in [("center",   center_logits),
                             ("boundary", boundary_logits)]:
            # up‑/down‑sample gt to match logits size
            gts = torch.stack([t[f"{name}_map"] for t in targets])  # (B,1,H,W)
            gts = F.interpolate(gts.float(), size=logits.shape[-2:],
                                mode="nearest")                     # keep binary
            loss = F.binary_cross_entropy_with_logits(
                logits, gts, pos_weight=pos_w)
            aux_losses[f"loss_{name}"] = self._aux_w * loss

        total_losses = {}
        total_losses.update(detector_losses)
        total_losses.update(aux_losses)
        return total_losses


def get_model(num_classes: int,
              model_type: str = "resnet50",
              with_train_map: bool = False,
              aux_loss_weight: float = 1.0):

    if model_type == "resnet50":
        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        base = maskrcnn_resnet50_fpn(weights=weights)
    elif model_type == "resnet50_v2":
        weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        base = maskrcnn_resnet50_fpn_v2(weights=weights)
    else:
        raise ValueError("…")

    # replace heads to fit your class count
    in_feats = base.roi_heads.box_predictor.cls_score.in_features
    base.roi_heads.box_predictor = FastRCNNPredictor(in_feats, num_classes)

    in_ch = base.roi_heads.mask_predictor.conv5_mask.in_channels
    base.roi_heads.mask_predictor = MaskRCNNPredictor(in_ch, 256, num_classes)

    if not with_train_map:
        return base                          # vanilla MaskRCNN

    # ⇣ wrap with the auxiliary‑head class
    model = MaskRCNNWithMaps(base_model=base,
                             loss_weight=aux_loss_weight)
    return model
