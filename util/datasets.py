# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/mae
# UM-MAE: https://github.com/implus/UM-MAE
# --------------------------------------------------------

import os
import PIL
import torchvision.transforms
from PIL import Image
import json
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class ImageListFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 ann_file=None, loader=default_loader):
        self.root = root
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.nb_classes = 1000

        assert ann_file is not None
        print('load info from', ann_file)

        self.samples = []
        ann = open(ann_file)
        for elem in ann.readlines():
            cut = elem.split(' ')
            path_current = os.path.join(root, cut[0])
            target_current = int(cut[1])
            self.samples.append((path_current, target_current))
        ann.close()

        print('load finish')

class Reflacx(Dataset):
    def __init__(self, data_root: str, transforms=None) -> None:
        
        self.data_root=data_root
        self.info=json.load(open(os.path.join(self.data_root,"reflacx.json")))
        self.gaze_dir=os.path.join(self.data_root,"attention")
        self.transforms=transforms
        # self.vis_trans=PairedTransform(transforms[0])
        # self.val_trans=transforms[1]

    def getImgPath(self,index):
        image_path=self.info[index]['image_path']
        image_path=os.path.join(self.data_root,image_path)
        return image_path

    def __getitem__(self, index):
        image_path=self.info[index]['image_path']
        study_id=self.info[index]['study_id']
        reflacx_id=self.info[index]['reflacx_id']
        image_path=os.path.join(self.data_root,image_path)
        # gaze_path=os.path.join(self.gaze_dir,study_id,f"{reflacx_id}.png")

        image = Image.open(image_path).convert('RGB')
        # gaze=Image.open(gaze_path)
        
        if self.transforms !=None:
            # image,gaze=self.transforms(image,gaze)
            image=self.transforms(image)
        return image
        # return {"image":image}
        # return {"image":image,"gaze":gaze}
    

    def __len__(self):
        return len(self.info)
        

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    # TODO modify your own dataset here
    # folder = os.path.join(args.data_path, 'train' if is_train else 'val')
    # ann_file = os.path.join(args.data_path, 'train.txt' if is_train else 'val.txt')
    dataset=Reflacx("../data/reflacx-1.0.0/",transform)
    # dataset = ImageListFolder(folder, transform=transform, ann_file=ann_file)

    print(dataset)

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
