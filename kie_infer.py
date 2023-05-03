from mmengine.runner import Runner
from mmengine.config import Config

cfg = Config.fromfile("./configs/kie/sdmgr/sdmgr_unet16_60e_wildreceipt.py")
cfg.work_dir = "./work_dir"
runner = Runner.from_cfg(cfg)
