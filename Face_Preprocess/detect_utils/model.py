# code borrowed from https://github.com/jiuxianghedonglu/AnimeHeadDetection/blob/master/detect_image.py
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FasterRCNN


def fasterrcnn_resnet_fpn(resnet_name='resnet50', pretrained_backbone=True, progress=True, num_classes=2, **kwargs):
    """
    Constructs a Faster R-CNN model with a ResNet-FPN backbone.
    The input to the model is expected to be a list of tensors, each of shape ``[C, H, W]``, one for each
    image, and should be in ``0-1`` range. Different images can have different sizes.
    The behavior of the model changes depending if it is in training or evaluation mode.
    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with values
          between ``0`` and ``H`` and ``0`` and ``W``
        - labels (``Int64Tensor[N]``): the class label for each ground-truth box
    The model returns a ``Dict[Tensor]`` during training, containing the classification and regression
    losses for both the RPN and the R-CNN.
    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a ``List[Dict[Tensor]]``, one for each input image. The fields of the ``Dict`` are as
    follows:
        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with values between
          ``0`` and ``H`` and ``0`` and ``W``
        - labels (``Int64Tensor[N]``): the predicted labels for each image
        - scores (``Tensor[N]``): the scores or each prediction
    """
    backbone = resnet_fpn_backbone(resnet_name, pretrained_backbone)
    model = FasterRCNN(backbone, num_classes,
                       box_detections_per_img=100, **kwargs)
    return model


if __name__ == '__main__':
    import torch
    model = fasterrcnn_resnet_fpn()
    x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    model.eval()
    predictions = model(x)
    print(predictions)
    model.train()
    print(model(x, predictions))
