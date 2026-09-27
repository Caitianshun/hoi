"""Add only an explicit foreground-mask property to V3 training cameras."""
from common import *
import cv2,numpy as np,torch
from adapter_4dgs import ProtocolScene as BaseScene,CalibratedCamera,load_manifest
class ProtocolScene(BaseScene):
    def __init__(self,args,gaussians,**kw):
        super().__init__(args,gaussians,**kw)
        for camera,frame in zip(self.train_camera,self.manifest['frames']):
            assert camera.image_name==str(frame['frame_id']) and camera.mask is None
            assert sha(frame['mask_path'])==frame['mask_sha256']
            mask=cv2.imread(frame['mask_path'],cv2.IMREAD_GRAYSCALE)
            assert mask is not None and mask.shape==(camera.image_height,camera.image_width)
            camera.foreground_mask=torch.from_numpy(mask>=128)
            camera.mask_source=identity(frame['mask_path'])
        self.mask_policy='published training mask >=128, additional loss use; upstream mask remains None'
