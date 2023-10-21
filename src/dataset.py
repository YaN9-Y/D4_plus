import os
import glob
import torch
import random
import numpy as np
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from PIL import Image, ImageFile
import torchvision.transforms as transforms


ImageFile.LOAD_TRUNCATED_IMAGES = True

class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, crop_size,  hazy_flist, clean_flist=None, augment=True, split='unpair'):
        super(Dataset, self).__init__()
        self.augment = augment
        self.config = config
        assert split in ['unpair','pair_train', 'pair_test', 'hazy', 'clean', 'depth', 'hazy_various']

        self.split = split


        self.clean_data = self.load_flist(clean_flist)
        self.noisy_data = self.load_flist(hazy_flist)


        self.input_size = crop_size
        self.jitter = transforms.ColorJitter(brightness=(0.8,1),contrast=(1.4,1.8),saturation=(0.4,1))



        self.transforms = transforms.Compose(([
                                                  transforms.RandomCrop((self.input_size, self.input_size)),
                                              ] if crop_size else [])
                                             + ([transforms.RandomHorizontalFlip()]
                                                if self.augment else [])
                                             + [transforms.ToTensor()]
                                             )
        print(self.clean_data)


    def __len__(self):
        if self.split in ['pair_test', 'hazy','hazy_various']:
            return len(self.noisy_data)
        else:
            return len(self.clean_data)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error: ' + self.data[index])
            item = self.load_item(0)

        return item

    def load_name(self, index):
        if self.split in ['clean','depth']:
            name = self.clean_data[index]
        else:
            name = self.noisy_data[index]
        return os.path.basename(name)

    def load_item(self, index):
        # load image
            if self.split in ['hazy','hazy_various']:
                img_noisy = Image.open(self.noisy_data[index])
                img_noisy = self.convert_to_rgb(img_noisy)
                img_noisy = TF.to_tensor(img_noisy)
                return img_noisy

            elif self.split in ['clean','depth']:
                img_clean = Image.open(self.clean_data[index])
                img_clean = self.convert_to_rgb(img_clean)
                img_clean = TF.to_tensor(img_clean)
                if self.config.INDOOR_CROP:
                    img_clean = img_clean[:,10:-10, 10:-10]
                return img_clean

            elif self.split in ['unpair']:
                while (True):
                    clean_index = int(np.random.random() * len(self.clean_data))
                    img_clean = Image.open((self.clean_data[clean_index]))

                    if self.config.IS_REAL_MODEL == 1:
                        mask_clean = Image.open(self.get_mask_path(orig_path=self.clean_data[clean_index],
                                                                   tar_path=self.config.PATH_CLEAN_MASK))

                    if np.array(img_clean).shape is None:
                        print(self.clean_data[clean_index])
                    if min(np.array(img_clean).shape[0:2]) > self.config.CROP_SIZE:
                        break

                while(True):
                    noisy_index = int(np.random.random() * len(self.noisy_data))
                    img_noisy = Image.open((self.noisy_data[noisy_index]))
                    if self.config.IS_REAL_MODEL == 1:
                        mask_hazy = Image.open(self.get_mask_path(orig_path=self.noisy_data[noisy_index],tar_path=self.config.PATH_HAZY_MASK))

                    if np.array(img_noisy).shape is None:
                        print(self.noisy_data[noisy_index])
                    if min(np.array(img_noisy).shape[0:2]) > self.config.CROP_SIZE:
                        break

                while (True):
                    clean_index = int(np.random.random() * len(self.clean_data))
                    img_clean_2 = Image.open((self.clean_data[clean_index]))
                    if np.array(img_clean_2).shape is None:
                        print(self.clean_data[clean_index])
                    if min(np.array(img_clean_2).shape[0:2]) > self.config.CROP_SIZE:
                        break

                while (True):
                    noisy_index = int(np.random.random() * len(self.noisy_data))
                    img_noisy_2 = Image.open((self.noisy_data[noisy_index]))
                    if np.array(img_noisy_2).shape is None:
                        print(self.noisy_data[noisy_index])
                    if min(np.array(img_noisy_2).shape[0:2]) > self.config.CROP_SIZE:
                        break


                if self.config.IS_REAL_MODEL == 0:

                    img_noisy = self.convert_to_rgb(img_noisy)
                    img_clean = self.convert_to_rgb(img_clean)

                    img_clean = self.get_square_img(img_clean)
                    img_noisy = self.get_square_img(img_noisy)

                    img_clean = TF.resize(img_clean, size=self.input_size, interpolation=Image.BICUBIC)
                    img_noisy = TF.resize(img_noisy, size=self.input_size, interpolation=Image.BICUBIC)

                    img_clean = self.transforms(img_clean)
                    img_noisy = self.transforms(img_noisy)


                    """
                    data for contrastive loss
                    """
                    img_noisy_2 = self.convert_to_rgb(img_noisy_2)
                    img_clean_2 = self.convert_to_rgb(img_clean_2)

                    img_clean_2 = self.get_square_img(img_clean_2)
                    img_noisy_2 = self.get_square_img(img_noisy_2)

                    img_clean_2 = TF.resize(img_clean_2, size=self.input_size, interpolation=Image.BICUBIC)
                    img_noisy_2 = TF.resize(img_noisy_2, size=self.input_size, interpolation=Image.BICUBIC)

                    img_clean_2 = self.transforms(img_clean_2)
                    img_noisy_2 = self.transforms(img_noisy_2)

                    return img_clean, img_noisy, img_clean_2, img_noisy_2

                elif self.config.IS_REAL_MODEL == 1:
                    img_noisy = self.convert_to_rgb(img_noisy)
                    img_clean = self.convert_to_rgb(img_clean)

                    img_noisy, mask_hazy = self.get_square_imgs(img_noisy, mask_hazy)
                    img_clean, mask_clean = self.get_square_imgs(img_clean, mask_clean)

                    img_clean = TF.resize(img_clean, size=self.input_size, interpolation=Image.BICUBIC)
                    img_noisy = TF.resize(img_noisy, size=self.input_size, interpolation=Image.BICUBIC)
                    mask_clean = TF.resize(mask_clean, size=self.input_size, interpolation=Image.BICUBIC)
                    mask_hazy = TF.resize(mask_hazy, size=self.input_size, interpolation=Image.BICUBIC)

                    img_clean = self.jitter(img_clean)
                    img_clean, mask_clean = self.apply_transforms(img_clean, mask_clean)
                    img_noisy, mask_hazy = self.apply_transforms(img_noisy, mask_hazy)

                    """
                    Extra data for contrastive loss
                    """
                    img_noisy_2 = self.convert_to_rgb(img_noisy_2)
                    img_clean_2 = self.convert_to_rgb(img_clean_2)

                    img_clean_2 = self.get_square_img(img_clean_2)
                    img_noisy_2 = self.get_square_img(img_noisy_2)

                    img_clean_2 = TF.resize(img_clean_2, size=self.input_size, interpolation=Image.BICUBIC)
                    img_noisy_2 = TF.resize(img_noisy_2, size=self.input_size, interpolation=Image.BICUBIC)

                    img_clean_2 = self.transforms(img_clean_2)
                    img_noisy_2 = self.transforms(img_noisy_2)

                    img_clean = img_clean.clamp(0.04, 1)
                    img_noisy = img_noisy.clamp(0.04, 1)

                    return img_clean, img_noisy, img_clean_2, img_noisy_2, mask_clean, mask_hazy


            elif self.split in ['pair_train', 'pair_test']:

                img_noisy = Image.open(self.noisy_data[index])
                img_clean = self.get_gt_path(self.noisy_data[index])

                img_clean = Image.open(img_clean)

                img_noisy = self.convert_to_rgb(img_noisy)
                img_clean = self.convert_to_rgb(img_clean)

                if img_clean.size != img_noisy.size:
                    img_clean = TF.center_crop(img_clean, img_noisy.size[::-1])

                img_clean = TF.to_tensor(img_clean)
                img_noisy = TF.to_tensor(img_noisy)


                return img_clean, img_noisy


    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist

        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))+list(glob.glob(flist + '/*.jpeg'))
                flist.sort()
                return flist

            if os.path.isfile(flist):
                return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')

        return []


    def load_image_to_memory(self, flist):
        filelist = np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
        images_list = []
        for i in range(len(filelist)):
            images_list.append(np.array(Image.open(filelist[i])))
            if i%100 == 0:
                print('loading data: %d / %d', i+1, len(filelist))
        return images_list

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item

    def RandomRot(self, img, angle=90, p=0.5):
        if random.random() > p:
            return transforms.functional.rotate(img, angle)
        return img


    def get_gt_path(self, path):
        filename = os.path.basename(path)

        if self.split == 'pair_train':
            prefix = str.split(filename, '_')[0]
            gt_path = os.path.join(self.config.TRAIN_CLEAN_PATH, prefix+filename[-4:])

        elif self.split == 'pair_test':
            prefix = str.split(filename,'_')[0]
            gt_path = os.path.join(self.config.VAL_CLEAN_PATH, prefix+'.png')


        return gt_path


    def convert_to_rgb(self, img):
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        return img

    def apply_transforms(self, *imgs):
        imgs = list(imgs)

        if self.augment:
            if random.random()>0.5:
                for i in range(len(imgs)):
                    imgs[i] = TF.hflip(imgs[i])


        for i in range(len(imgs)):
            imgs[i] = TF.to_tensor(imgs[i])

        return imgs



    def get_square_img(self, img):
        h,w= img.size
        if h < w:
            return TF.crop(img, random.randint(0,w-h), 0,  h, h)
        elif h >= w:
            return TF.crop(img, 0, random.randint(0,h-w), w, w)

    def get_square_imgs(self, *imgs):
        h,w= imgs[0].size
        imgs = list(imgs)
        if h < w:
            border = random.randint(0,w-h)
            for i in range(len(imgs)):
                imgs[i] = TF.crop(imgs[i], border, 0,  h, h)
        elif h >= w:
            border = random.randint(0, h - w)
            for i in range(len(imgs)):
                imgs[i] = TF.crop(imgs[i], 0, border, w, w)

        return imgs

    def get_mask_path(self, orig_path, tar_path ):
        basename = os.path.basename(orig_path)
        new_path = os.path.join(tar_path,basename)

        return new_path
