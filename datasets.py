import os
import numpy as np
from PIL import Image

from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
import torch


class SiameseMNIST(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset

        self.train = self.mnist_dataset.train
        self.transform = self.mnist_dataset.transform

        if self.train:
            self.train_labels = self.mnist_dataset.train_labels
            self.train_data = self.mnist_dataset.train_data
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                     for label in self.labels_set}
        else:
            # generate fixed pairs for testing
            self.test_labels = self.mnist_dataset.test_labels
            self.test_data = self.mnist_dataset.test_data
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            positive_pairs = [[i,
                               random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                               1]
                              for i in range(0, len(self.test_data), 2)]

            negative_pairs = [[i,
                               random_state.choice(self.label_to_indices[
                                                       np.random.choice(
                                                           list(self.labels_set - set([self.test_labels[i].item()]))
                                                       )
                                                   ]),
                               0]
                              for i in range(1, len(self.test_data), 2)]
            self.test_pairs = positive_pairs + negative_pairs

    def create_images(self, img1, img2, idx1, idx2):
        img1 = Image.fromarray(img1.numpy(), mode='L')
        img2 = Image.fromarray(img2.numpy(), mode='L')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2

    def __getitem__(self, index):
        index1 = None
        index2 = None
        if self.train:
            target = np.random.randint(0, 2)
            index1 = index
            img1, label1 = self.train_data[index1], self.train_labels[index1].item()
            if target == 1:
                siamese_index = index1
                while siamese_index == index1:
                    siamese_index = np.random.choice(self.label_to_indices[label1])
            else:
                siamese_label = np.random.choice(list(self.labels_set - set([label1])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            index2 = siamese_index
            img2 = self.train_data[index2]
        else:
            index1 = self.test_pairs[index][0]
            index2 = self.test_pairs[index][1]
            img1 = self.test_data[index1]
            img2 = self.test_data[index2]
            target = self.test_pairs[index][2]

        return self.create_images(img1, img2, index1, index2), target

    def __len__(self):
        return len(self.mnist_dataset)


class SiameseMMFashion(SiameseMNIST):

    def __init__(self, mmfashion_dataset, train, classes=True):
        self.mmfashion_dataset = mmfashion_dataset
        self.train = train
        self.transform = self.mmfashion_dataset.transform

        if self.train:
            if classes:
                self.train_labels = torch.tensor(self.mmfashion_dataset.class_ids)
            else:
                self.train_labels = torch.tensor(list(self.mmfashion_dataset.idx2id.values()))
            self.train_data = self.mmfashion_dataset.img_list
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                     for label in self.labels_set}
        else:
            # generate fixed pairs for testing
            if classes:
                self.test_labels = torch.tensor(self.mmfashion_dataset.class_ids)
            else:
                self.test_labels = torch.tensor(list(self.mmfashion_dataset.idx2id.values()))
            self.test_data = self.mmfashion_dataset.img_list
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            positive_pairs = [[i,
                               random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                               1]
                              for i in range(0, len(self.test_data), 2)]

            negative_pairs = [[i,
                               random_state.choice(self.label_to_indices[
                                                       np.random.choice(
                                                           list(self.labels_set - set([self.test_labels[i].item()]))
                                                       )
                                                   ]),
                               0]
                              for i in range(1, len(self.test_data), 2)]
            self.test_pairs = positive_pairs + negative_pairs

    def create_images(self, img1, img2, idx1, idx2):
        image1 = self.get_mmfashion_img(img1, idx1)
        image2 = self.get_mmfashion_img(img2, idx2)
        return image1, image2

    def get_mmfashion_img(self, img, idx):
        image = Image.open(os.path.join(self.mmfashion_dataset.img_path, img))

        if self.mmfashion_dataset.with_bbox:
            bbox_cor = self.mmfashion_dataset.bboxes[idx]
            x1 = max(0, int(bbox_cor[0]) - 20)
            y1 = max(0, int(bbox_cor[1]) - 20)
            x2 = int(bbox_cor[2]) + 20
            y2 = int(bbox_cor[3]) + 20
            image = image.crop(box=(x1, y1, x2, y2))

        image.thumbnail(self.mmfashion_dataset.img_size, Image.ANTIALIAS)
        image = image.convert('RGB')
        image = self.mmfashion_dataset.transform(image)
        return image

    def __len__(self):
        return len(self.mmfashion_dataset)


class TripletMNIST(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset
        self.train = self.mnist_dataset.train
        self.transform = self.mnist_dataset.transform

        if self.train:
            self.train_labels = self.mnist_dataset.train_labels
            self.train_data = self.mnist_dataset.train_data
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                     for label in self.labels_set}

        else:
            self.test_labels = self.mnist_dataset.test_labels
            self.test_data = self.mnist_dataset.test_data
            # generate fixed triplets for testing
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                         random_state.choice(self.label_to_indices[
                                                 np.random.choice(
                                                     list(self.labels_set - set([self.test_labels[i].item()]))
                                                 )
                                             ])
                         ]
                        for i in range(len(self.test_data))]
            self.test_triplets = triplets

    def create_images(self, img1, img2, img3, idx1, idx2, idx3):
        img1 = Image.fromarray(img1.numpy(), mode='L')
        img2 = Image.fromarray(img2.numpy(), mode='L')
        img3 = Image.fromarray(img3.numpy(), mode='L')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        return img1, img2, img3

    def __getitem__(self, index):
        index1 = None
        index2 = None
        index3 = None
        if self.train:
            index1 = index
            img1, label1 = self.train_data[index], self.train_labels[index].item()
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            index2 = positive_index
            index3 = negative_index
            img2 = self.train_data[positive_index]
            img3 = self.train_data[negative_index]
        else:
            index1 = self.test_triplets[index][0]
            index2 = self.test_triplets[index][1]
            index3 = self.test_triplets[index][2]
            img1 = self.test_data[index1]
            img2 = self.test_data[index2]
            img3 = self.test_data[index3]

        return self.create_images(img1, img2, img3, index1, index2, index3), []

    def __len__(self):
        return len(self.mnist_dataset)


class TripletMMFashion(TripletMNIST):

    def __init__(self, mmfashion_dataset, train, classes=True):
        self.mmfashion_dataset = mmfashion_dataset
        self.train = train
        self.transform = self.mmfashion_dataset.transform

        if self.train:
            if classes:
                self.train_labels = torch.tensor(self.mmfashion_dataset.class_ids)
            else:
                self.train_labels = torch.tensor(list(self.mmfashion_dataset.idx2id.values()))
            self.train_data = self.mmfashion_dataset.img_list
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                     for label in self.labels_set}

        else:
            if classes:
                self.test_labels = torch.tensor(self.mmfashion_dataset.class_ids)
            else:
                self.test_labels = torch.tensor(list(self.mmfashion_dataset.idx2id.values()))
            self.test_data = self.mmfashion_dataset.img_list
            # generate fixed triplets for testing
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                         random_state.choice(self.label_to_indices[
                                                 np.random.choice(
                                                     list(self.labels_set - set([self.test_labels[i].item()]))
                                                 )
                                             ])
                         ]
                        for i in range(len(self.test_data))]
            self.test_triplets = triplets

    def create_images(self, img1, img2, img3, idx1, idx2, idx3):
        image1 = self.get_mmfashion_img(img1, idx1)
        image2 = self.get_mmfashion_img(img2, idx2)
        image3 = self.get_mmfashion_img(img3, idx3)
        return image1, image2, image3

    def get_mmfashion_img(self, img, idx):
        image = Image.open(os.path.join(self.mmfashion_dataset.img_path, img))

        if self.mmfashion_dataset.with_bbox:
            bbox_cor = self.mmfashion_dataset.bboxes[idx]
            x1 = max(0, int(bbox_cor[0]) - 20)
            y1 = max(0, int(bbox_cor[1]) - 20)
            x2 = int(bbox_cor[2]) + 20
            y2 = int(bbox_cor[3]) + 20
            image = image.crop(box=(x1, y1, x2, y2))

        image.thumbnail(self.mmfashion_dataset.img_size, Image.ANTIALIAS)
        image = image.convert('RGB')
        image = self.mmfashion_dataset.transform(image)
        return image

    def __len__(self):
        return len(self.mmfashion_dataset)


class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size
