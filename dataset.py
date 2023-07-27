import os
import re
from torchvision import transforms as T
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from PIL import ImageFile, Image
# from transform.randaugment import RandomAugment
ImageFile.LOAD_TRUNCATED_IMAGES = True


IMAGE_NET_MEAN = [0.48145466, 0.4578275, 0.40821073]
IMAGE_NET_STD = [0.26862954, 0.26130258, 0.27577711]
normalize = T.Normalize(
    mean=IMAGE_NET_MEAN,
    std=IMAGE_NET_STD)

# def _convert_image_to_rgb(image):
#     return image.convert("RGB")
#
#
# def _transform(n_px):
#     return T.Compose([
#         Resize(n_px, interpolation=BICUBIC),
#         T.CenterCrop(n_px),
#         _convert_image_to_rgb,
#         ToTensor(),
#         Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
#     ])
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


class AVA_Comment_Dataset(Dataset):
    def __init__(self, path_to_csv, images_path, if_train):
        self.df = pd.read_csv(path_to_csv)
        self.images_path = images_path
    #     normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    #     if if_train:
    #          self.transform = transforms.Compose([
    #             transforms.RandomResizedCrop((224, 224), scale=(0.5, 1.0),
    #                                          interpolation=InterpolationMode.BICUBIC),
    #             transforms.RandomHorizontalFlip(),
    #             RandomAugment(2, 5, isPIL=True, augs=['Identity', 'AutoContrast', 'Brightness', 'Sharpness', 'Equalize',
    #                                                   'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
    #             transforms.ToTensor(),
    #             normalize,
    #         ])
    #     else:
    #         self.transform = transforms.Compose([
    #             transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
    #             transforms.ToTensor(),
    #             normalize,
    #         ])
        if if_train:
            self.transform = T.Compose([
                T.Resize((256, 256), interpolation=BICUBIC),
                T.RandomHorizontalFlip(),
                T.RandomCrop((224, 224)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        else:
            self.transform = T.Compose([
                T.Resize((224, 224), interpolation=BICUBIC),
                # T.Resize((224, 224), interpolation=BICUBIC),
                # T.CenterCrop((224, 224)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                # T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        row = self.df.iloc[item]
        score = row['label'].split()
        y = np.array([int(k) for k in score]).astype('float32')
        p = y / y.sum()

        image_id = row['index']
        image_path = os.path.join(self.images_path, f'{image_id}.jpg')
        image = default_loader(image_path)
        image = self.transform(image)

        caption = row['comment']
        caption = self.pre_caption(caption)

        # return image, p.astype('float32')
        return image, caption, p.astype('float16')
        # return image, caption, p.astype('float32')

    def pre_caption(self, caption, max_words=100):
        caption = re.sub(
            r"[\[(,.!?\'\"()*#:;~)\]]",
            ' ',
            caption.lower(),
        )
        caption = caption.replace('\\n', ' ')

        caption = re.sub(
            r"\s{2,}",
            ' ',
            caption,
        )
        # caption = caption.strip('\\n')
        caption = caption.strip(' ')

        #truncate caption
        caption_words = caption.split(' ')
        if len(caption_words) > max_words:
            caption = ' '.join(caption_words[:max_words])

        return caption



class AVA_Comment_Dataset_bert(Dataset):
    def __init__(self, path_to_csv, images_path, if_train):
        self.df = pd.read_csv(path_to_csv)
        self.images_path = images_path
        if if_train:
            self.transform = T.Compose([
                T.Resize((256, 256), interpolation=BICUBIC),
                T.RandomHorizontalFlip(),
                T.RandomCrop((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
                # normalize])
        else:
            self.transform = T.Compose([
                T.Resize((224, 224), interpolation=BICUBIC),
                # T.CenterCrop((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
                # normalize])

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        row = self.df.iloc[item]
        score = row['label'].split()
        y = np.array([int(k) for k in score]).astype('float32')
        p = y / y.sum()

        image_id = row['index']
        image_path = os.path.join(self.images_path, f'{image_id}.jpg')
        image = default_loader(image_path)
        image = self.transform(image)

        caption = row['comment']
        caption = self.pre_caption(caption)

        return image, caption, p##.astype('float32')

    def pre_caption(self, caption, max_words=200):
        caption = re.sub(
            r"[\[(\'\"()*#:~)\]]",
            ' ',
            caption,
        )
        caption = caption.replace('\\n', ' ')

        caption = re.sub(
            r"\s{2,}",
            ' ',
            caption,
        )
        # caption = caption.strip('\\n')
        caption = caption.strip(' ')

        #truncate caption
        caption_words = caption.split(' ')
        if len(caption_words) > max_words:
            caption = ' '.join(caption_words[:max_words])

        return caption


class AVA_Comment_Dataset_vit_bert(Dataset):
    def __init__(self, path_to_csv, images_path, if_train):
        self.df = pd.read_csv(path_to_csv)
        self.images_path = images_path
        if if_train:
            self.transform = T.Compose([
                T.Resize((256, 256), interpolation=BICUBIC),
                T.RandomHorizontalFlip(),
                T.RandomCrop((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
                # normalize])
        else:
            self.transform = T.Compose([
                T.Resize((224, 224), interpolation=BICUBIC),
                # T.CenterCrop((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
                # normalize])

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        row = self.df.iloc[item]
        score = row['label'].split()
        y = np.array([int(k) for k in score]).astype('float32')
        p = y / y.sum()

        image_id = row['index']
        image_path = os.path.join(self.images_path, f'{image_id}.jpg')
        image = default_loader(image_path)
        image = self.transform(image)

        caption = row['comment']
        caption = self.pre_caption(caption)

        return image, caption, p##.astype('float32')

    def pre_caption(self, caption, max_words=200):
        caption = re.sub(
            r"[\[(\'\"()*#:~)\]]",
            ' ',
            caption,
        )
        caption = caption.replace('\\n', ' ')

        caption = re.sub(
            r"\s{2,}",
            ' ',
            caption,
        )
        # caption = caption.strip('\\n')
        caption = caption.strip(' ')

        #truncate caption
        caption_words = caption.split(' ')
        if len(caption_words) > max_words:
            caption = ' '.join(caption_words[:max_words])

        return caption

class AVA_Comment_Dataset_bert_semantic(Dataset):
    def __init__(self, path_to_csv, images_path, if_train):
        self.df = pd.read_csv(path_to_csv)
        self.images_path = images_path
        if if_train:
            self.transform = T.Compose([
                T.Resize((256, 256), interpolation=BICUBIC),
                T.RandomHorizontalFlip(),
                T.RandomCrop((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        else:
            self.transform = T.Compose([
                T.Resize((224, 224), interpolation=BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        row = self.df.iloc[item]
        score = row['label'].split()
        y = np.array([int(k) for k in score]).astype('float32')
        p = y / y.sum()

        image_id = row['index']
        image_path = os.path.join(self.images_path, f'{image_id}.jpg')
        image = default_loader(image_path)
        image = self.transform(image)

        caption = row['comment']
        caption = self.pre_caption(caption)

        semantic_id_1 = row['semetic_id_1']
        semantic_id_2 = row['semetic_id_2']
        challenge_id = row['challege_id']

        return image, caption, p, semantic_id_1, semantic_id_2, challenge_id##.astype('float32')

    def pre_caption(self, caption, max_words=200):
        caption = re.sub(
            r"[\[(\'\"()*#:~)\]]",
            ' ',
            caption,
        )
        caption = caption.replace('\\n', ' ')

        caption = re.sub(
            r"\s{2,}",
            ' ',
            caption,
        )
        # caption = caption.strip('\\n')
        caption = caption.strip(' ')

        #truncate caption
        caption_words = caption.split(' ')
        if len(caption_words) > max_words:
            caption = ' '.join(caption_words[:max_words])

        return caption


class AVA_Comment_Dataset_bert_ce(Dataset):
    def __init__(self, path_to_csv, images_path, if_train):
        self.df = pd.read_csv(path_to_csv)
        self.images_path = images_path

        if if_train:
            self.transform = T.Compose([
                T.Resize((256, 256), interpolation=BICUBIC),
                T.RandomHorizontalFlip(),
                T.RandomCrop((224, 224)),
                T.ToTensor(),
                normalize])
        else:
            self.transform = T.Compose([
                T.Resize((224, 224), interpolation=BICUBIC),
                # T.CenterCrop((224, 224)),
                T.ToTensor(),
                normalize])

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        row = self.df.iloc[item]
        score = row['label'].split()
        y = np.array([int(k) for k in score]).astype('float32')
        p = y / y.sum()

        image_id = row['index']
        image_path = os.path.join(self.images_path, f'{image_id}.jpg')
        image = default_loader(image_path)
        image = self.transform(image)

        caption = row['comment']
        caption = self.pre_caption(caption)
        cls = row['class']
        return image, caption, p, cls##.astype('float32')

    def pre_caption(self, caption, max_words=200):
        caption = re.sub(
            r"[\[(\'\"()*#:~)\]]",
            ' ',
            caption,
        )
        caption = caption.replace('\\n', ' ')

        caption = re.sub(
            r"\s{2,}",
            ' ',
            caption,
        )
        # caption = caption.strip('\\n')
        caption = caption.strip(' ')

        #truncate caption
        caption_words = caption.split(' ')
        if len(caption_words) > max_words:
            caption = ' '.join(caption_words[:max_words])

        return caption


class AVA_Comment_Dataset_vit(Dataset):
    def __init__(self, path_to_csv, images_path, if_train):
        self.df = pd.read_csv(path_to_csv)
        self.images_path = images_path
        if if_train:
            self.transform = T.Compose([
                T.Resize((256, 256), interpolation=BICUBIC),
                T.RandomHorizontalFlip(),
                T.RandomCrop((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        else:
            self.transform = T.Compose([
                T.Resize((224, 224), interpolation=BICUBIC),
                # T.CenterCrop((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        row = self.df.iloc[item]
        score = row['label'].split()
        y = np.array([int(k) for k in score]).astype('float32')
        p = y / y.sum()

        image_id = row['index']
        image_path = os.path.join(self.images_path, f'{image_id}.jpg')
        image = default_loader(image_path)
        image = self.transform(image)

        caption = row['comment']
        caption = self.pre_caption(caption)

        return image, caption, p##.astype('float32')

    def pre_caption(self, caption, max_words=200):
        caption = re.sub(
            r"[\[(\'\"()*#:~)\]]",
            ' ',
            caption,
        )
        caption = caption.replace('\\n', ' ')

        caption = re.sub(
            r"\s{2,}",
            ' ',
            caption,
        )
        # caption = caption.strip('\\n')
        caption = caption.strip(' ')

        #truncate caption
        caption_words = caption.split(' ')
        if len(caption_words) > max_words:
            caption = ' '.join(caption_words[:max_words])

        return caption


class AVA_Comment_Dataset_bert_resample(Dataset):
    def __init__(self, path_to_csv, images_path, if_train):
        self.df = pd.read_csv(path_to_csv)
        self.images_path = images_path

        if if_train:
            self.transform = T.Compose([
                T.Resize((256, 256), interpolation=BICUBIC),
                T.RandomHorizontalFlip(),
                T.RandomCrop((224, 224)),
                T.ToTensor(),
                normalize])
        else:
            self.transform = T.Compose([
                T.Resize((224, 224), interpolation=BICUBIC),
                # T.CenterCrop((224, 224)),
                T.ToTensor(),
                normalize])

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        row = self.df.iloc[item]
        score = row['label'].split()
        y = np.array([int(k) for k in score]).astype('float32')
        p = y / y.sum()

        image_id = row['index']
        image_path = os.path.join(self.images_path, f'{image_id}.jpg')
        image = default_loader(image_path)
        image = self.transform(image)

        caption = row['comment']
        caption = self.pre_caption(caption)
        cls = row['class']
        return image, caption, p, cls##.astype('float32')

    def pre_caption(self, caption, max_words=200):
        caption = re.sub(
            r"[\[(\'\"()*#:~)\]]",
            ' ',
            caption,
        )
        caption = caption.replace('\\n', ' ')

        caption = re.sub(
            r"\s{2,}",
            ' ',
            caption,
        )
        # caption = caption.strip('\\n')
        caption = caption.strip(' ')

        #truncate caption
        caption_words = caption.split(' ')
        if len(caption_words) > max_words:
            caption = ' '.join(caption_words[:max_words])

        return caption


class ava_comment_InstanceSample(Dataset):
    """
    CIFAR100Instance+Sample Dataset
    """
    def __init__(self, path_to_csv, images_path, k=4096, mode='exact', is_sample=True, percent=1.0):
        super().__init__()
        self.df = pd.read_csv(path_to_csv)
        self.images_path = images_path
        self.k = k
        self.mode = mode
        self.is_sample = is_sample
        self.transform = T.Compose([
            T.Resize((256, 256), interpolation=BICUBIC),
            T.RandomHorizontalFlip(),
            T.RandomCrop((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            # normalize])

        num_classes = 10

        num_samples = self.df.shape[0]
        # label = self.train_labels

        self.cls_positive = [[] for i in range(num_classes)]
        for i in range(num_samples):
            row = self.df.iloc[i]
            score = row['label'].split()
            y = np.array([int(k) for k in score]).astype('float32')
            label = np.argmax(y)
            self.cls_positive[label].append(i)

        self.cls_negative = [[] for i in range(num_classes)]
        for i in range(num_classes):
            for j in range(num_classes):
                if j == i:
                    continue
                self.cls_negative[i].extend(self.cls_positive[j])

        self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
        self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]

        if 0 < percent < 1:
            n = int(len(self.cls_negative[0]) * percent)
            self.cls_negative = [np.random.permutation(self.cls_negative[i])[0:n]
                                 for i in range(num_classes)]

        self.cls_positive = np.asarray(self.cls_positive)
        self.cls_negative = np.asarray(self.cls_negative)

    def __getitem__(self, index):
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        row = self.df.iloc[index]
        score = row['label'].split()
        y = np.array([int(k) for k in score]).astype('float32')
        target = np.argmax(y)
        p = y / y.sum()

        img_id = row['index']
        img_path = os.path.join(self.images_path, f'{img_id}.jpg')
        img = default_loader(img_path)
        if self.transform is not None:
            img = self.transform(img)

        caption = row['comment']
        caption = self.pre_caption(caption)

        # sample contrastive examples
        if self.mode == 'exact':
            pos_idx = index
        else:
            raise NotImplementedError(self.mode)

        replace = True if self.k > len(self.cls_negative[target]) else False
        neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=replace)
        sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
        return img, caption, p, index, sample_idx

    # def pre_caption(self, caption, max_words=200):
    #     caption = re.sub(
    #         r"[\[(\'\"()*#:~)\]]",
    #         ' ',
    #         caption,
    #     )
    #     caption = caption.replace('\\n', ' ')
    #
    #     caption = re.sub(
    #         r"\s{2,}",
    #         ' ',
    #         caption,
    #     )
    #     # caption = caption.strip('\\n')
    #     caption = caption.strip(' ')
    #
    #     #truncate caption
    #     caption_words = caption.split(' ')
    #     if len(caption_words) > max_words:
    #         caption = ' '.join(caption_words[:max_words])
    #
    #     return caption

    def pre_caption(self, caption, max_words=100):
        caption = re.sub(
            r"[\[(,.!?\'\"()*#:;~)\]]",
            ' ',
            caption.lower(),
        )
        caption = caption.replace('\\n', ' ')

        caption = re.sub(
            r"\s{2,}",
            ' ',
            caption,
        )
        # caption = caption.strip('\\n')
        caption = caption.strip(' ')

        #truncate caption
        caption_words = caption.split(' ')
        if len(caption_words) > max_words:
            caption = ' '.join(caption_words[:max_words])

        return caption

    def __len__(self):
        return self.df.shape[0]