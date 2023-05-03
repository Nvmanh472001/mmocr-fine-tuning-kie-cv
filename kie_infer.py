from mmocr.models.common.backbones import MobileViTUnet


backbone_args = dict(name="mobilevit_xs", in_channels=16)
mobilevit_unet = MobileViTUnet(backbone_args, base_channels=16)
print(mobilevit_unet.eval())
