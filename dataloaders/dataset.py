import os
import sys
import shutil
import csv
from sklearn.model_selection import train_test_split

import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
# check current python path
# ROOT_DIR = os.path.abspath("./")
# sys.path.insert(0, ROOT_DIR)
# print(sys.path)
from mypath import Path
from tqdm import tqdm
import multiprocessing
from itertools import product

class VideoDataset(Dataset):
    r"""A Dataset for a folder of videos. Expects the directory structure to be
    directory->[train/val/test]->[class labels]->[videos]. Initializes with a list
    of all file names, along with an array of labels, with label being automatically
    inferred from the respective folder names.

        Args:
            dataset (str): Name of dataset. Defaults to 'ucf101'.
            split (str): Determines which folder of the directory the dataset will read from. Defaults to 'train'.
            clip_len (int): Determines how many frames are there in each clip. Defaults to 16.
            preprocess (bool): Determines whether to preprocess dataset. Default is False.
    """

    def __init__(self, dataset='ucf101', split='train', clip_len=16, preprocess=False, subset=[]):
        self.root_dir, self.output_dir = Path.db_dir(dataset)
        folder = os.path.join(self.output_dir, split)
        self.clip_len = clip_len
        self.split = split

        # The following three parameters are chosen as described in the paper section 4.1
        self.resize_height = 100
        self.resize_width = 160
        self.crop_size = 368

        if not self.check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You need to download it from official website.')

        if preprocess or (not self.check_preprocess()):
            # print(preprocess, not self.check_preprocess())
            print('Preprocessing of {} dataset, this will take long, but it will be done only once.'.format(dataset))
            if dataset == "20bn-jester":
                self.preprocess_jester()
            else:
                self.preprocess()

        # check argument 'subset'  if use part of the classes
        if len(subset) > 0:
            print("Using a subset [{}] of dataset: {}".format(len(subset), subset))
        # Obtain all the filenames of files inside all the class folders
        # Going through each class folder one at a time
        self.fnames, labels = [], []
        for label in sorted(os.listdir(folder)):
            if label == ".DS_Store":
                continue
            if label in subset:
                for fname in os.listdir(os.path.join(folder, label)):
                    self.fnames.append(os.path.join(folder, label, fname))
                    labels.append(label)

        assert len(labels) == len(self.fnames)
        print('Number of {} videos: {:d}'.format(split, len(self.fnames)))
        self.class_names = sorted(set(labels))
        print('Double checked using {} classes: {}'.format(len(self.class_names), self.class_names))
        # Prepare a mapping between the label names (strings) and indices (ints)
        self.label2index = {label: index for index, label in enumerate(self.class_names)}
        # Convert the list of label names into an array of label indices
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)

        if dataset == "ucf101":
            if not os.path.exists('dataloaders/ucf_labels.txt'):
                with open('dataloaders/ucf_labels.txt', 'w') as f:
                    for id, label in enumerate(sorted(self.label2index)):
                        f.writelines(str(id+1) + ' ' + label + '\n')

        elif dataset == 'hmdb51':
            if not os.path.exists('dataloaders/hmdb_labels.txt'):
                with open('dataloaders/hmdb_labels.txt', 'w') as f:
                    for id, label in enumerate(sorted(self.label2index)):
                        f.writelines(str(id+1) + ' ' + label + '\n')
        elif dataset == '20bn-jester':
            # re log the label mapping
            if len(subset) > 0 and os.path.exists('dataloaders/20bn-jester_labels.txt'):
                os.remove('dataloaders/20bn-jester_labels.txt')
            if not os.path.exists('dataloaders/20bn-jester_labels.txt'):
                with open('dataloaders/20bn-jester_labels.txt', 'w') as f:
                    for id, label in enumerate(sorted(self.label2index)):
                        f.writelines(str(id+1) + ' ' + label + '\n')


    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        # index = 112257
        try:
            # Loading and preprocessing.
            buffer = self.load_frames(self.fnames[index])
            # resize to 368 by 368
            buffer = self.resize(buffer)
            buffer = self.crop(buffer, self.clip_len, self.crop_size)
            labels = np.array(self.label_array[index])

            if self.split == 'test':
                # Perform data augmentation
                buffer = self.randomflip(buffer)
            buffer = self.normalize(buffer)
            buffer = self.to_tensor(buffer)
            return torch.from_numpy(buffer), torch.from_numpy(labels)
        except:
            print("error when loading [{}]{}.".format(index, self.fnames[index]))
            return None, None

    def check_integrity(self):
        if not os.path.exists(self.root_dir):
            return False
        else:
            return True

    def check_preprocess(self):
        # TODO: Check image size in output_dir
        if not os.path.exists(self.output_dir):
            return False
        elif not os.path.exists(os.path.join(self.output_dir, 'train')):
            return False

        for ii, video_class in enumerate(os.listdir(os.path.join(self.output_dir, 'train'))):
            if video_class == ".DS_Store":
                continue
            for video in os.listdir(os.path.join(self.output_dir, 'train', video_class)):
                video_name = os.path.join(os.path.join(self.output_dir, 'train', video_class, video),
                                    sorted(os.listdir(os.path.join(self.output_dir, 'train', video_class, video)))[0])
                image = cv2.imread(video_name)
                if image.shape[0] != self.resize_height or image.shape[1] != self.resize_width:
                    return False
                else:
                    break

            if ii == 10:
                break

        return True

    def preprocess(self):
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
            os.mkdir(os.path.join(self.output_dir, 'train'))
            os.mkdir(os.path.join(self.output_dir, 'val'))
            os.mkdir(os.path.join(self.output_dir, 'test'))

        # Split train/val/test sets
        for file in os.listdir(self.root_dir):
            if file == ".DS_Store":
                continue
            file_path = os.path.join(self.root_dir, file)
            video_files = [name for name in os.listdir(file_path) if ".DS_Store" not in name]

            train_and_valid, test = train_test_split(video_files, test_size=0.2, random_state=42)
            train, val = train_test_split(train_and_valid, test_size=0.2, random_state=42)

            train_dir = os.path.join(self.output_dir, 'train', file)
            val_dir = os.path.join(self.output_dir, 'val', file)
            test_dir = os.path.join(self.output_dir, 'test', file)

            if not os.path.exists(train_dir):
                os.mkdir(train_dir)
            if not os.path.exists(val_dir):
                os.mkdir(val_dir)
            if not os.path.exists(test_dir):
                os.mkdir(test_dir)

            for video in train:
                self.process_video(video, file, train_dir)

            for video in val:
                self.process_video(video, file, val_dir)

            for video in test:
                self.process_video(video, file, test_dir)

        print('Preprocessing finished.')

    def process_video(self, video, action_name, save_dir):
        # Initialize a VideoCapture object to read video data into a numpy array
        video_filename = video.split('.')[0]
        if not os.path.exists(os.path.join(save_dir, video_filename)):
            os.mkdir(os.path.join(save_dir, video_filename))

        capture = cv2.VideoCapture(os.path.join(self.root_dir, action_name, video))

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Make sure splited video has at least 16 frames
        EXTRACT_FREQUENCY = 4
        if frame_count // EXTRACT_FREQUENCY <= 16:
            EXTRACT_FREQUENCY -= 1
            if frame_count // EXTRACT_FREQUENCY <= 16:
                EXTRACT_FREQUENCY -= 1
                if frame_count // EXTRACT_FREQUENCY <= 16:
                    EXTRACT_FREQUENCY -= 1

        count = 0
        i = 0
        retaining = True

        while (count < frame_count and retaining):
            retaining, frame = capture.read()
            if frame is None:
                continue

            if count % EXTRACT_FREQUENCY == 0:
                if (frame_height != self.resize_height) or (frame_width != self.resize_width):
                    frame = cv2.resize(frame, (self.resize_width, self.resize_height))
                cv2.imwrite(filename=os.path.join(save_dir, video_filename, '0000{}.jpg'.format(str(i))), img=frame)
                i += 1
            count += 1

        # Release the VideoCapture once it is no longer needed
        capture.release()

    def copy_train(self, train_label):
        src_dir = os.path.join(self.root_dir, "20bn-jester-v1", train_label[0])
        dst_dir = os.path.join(self.output_dir, "train", train_label[1], train_label[0])
        shutil.copytree(src_dir, dst_dir)

    def copy_val(self, val_label):
        src_dir = os.path.join(self.root_dir, "20bn-jester-v1", val_label[0])
        dst_dir = os.path.join(self.output_dir, "val", val_label[1], val_label[0])
        shutil.copytree(src_dir, dst_dir)

    def resize_train_dataset(self, train_label):
        dst_dir = os.path.join(self.output_dir, "train", train_label[1], train_label[0])
        for frame_name in os.listdir(dst_dir):  # frame - 0001.jpg
            frame_full_path = os.path.join(dst_dir, frame_name)
            frame = cv2.imread(frame_full_path)
            if frame.shape != (self.resize_width, self.resize_height, 3):
                frame = cv2.resize(frame, (self.resize_width, self.resize_height))
                cv2.imwrite(filename=frame_full_path, img=frame)

    def resize_val_dataset(self, val_label):
        dst_dir = os.path.join(self.output_dir, "val", val_label[1], val_label[0])
        for frame_name in os.listdir(dst_dir):  # frame - 0001.jpg
            frame_full_path = os.path.join(dst_dir, frame_name)
            frame = cv2.imread(frame_full_path)
            if frame.shape != (self.resize_width, self.resize_height, 3):
                frame = cv2.resize(frame, (self.resize_width, self.resize_height))
                cv2.imwrite(filename=frame_full_path, img=frame)

    def preprocess_jester(self):
        """
        The origin jester dataset structure: 

        [dataset dir] dataset/20bn-jester/20bn-jester-v1/
        |---- [folder] 1/
                |---- [frame] 00001.jpg
                |---- ...
        |---- [folder] 2/
        |---- ...

        """
        # if os.path.exists(self.output_dir):
        #     shutil.rmtree(self.output_dir)


        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
            train_base_dir = os.path.join(self.output_dir, 'train')
            val_base_dir = os.path.join(self.output_dir, 'val')
            test_base_dir = os.path.join(self.output_dir, 'test')
            os.mkdir(train_base_dir)
            os.mkdir(val_base_dir)
            os.mkdir(test_base_dir)
            # get class labels from csv file
            labels_dir = os.path.join(self.root_dir, 'jester-v1-labels.csv')
            labels = self.read_csv_input(labels_dir, [0])
            labels = [label[0] for label in labels]
            # make directory for each class in each folder
            for label in labels:
                for base_dir in [train_base_dir, val_base_dir, test_base_dir]:
                    label_dir = os.path.join(base_dir, label)
                    os.mkdir(label_dir)

        # read train val sets, notice that test set has no label
        train_csv_dir = os.path.join(self.root_dir, 'jester-v1-train.csv')
        val_csv_dir = os.path.join(self.root_dir, 'jester-v1-validation.csv')
        train_labels = self.read_csv_input(train_csv_dir, [0, 1])
        val_labels = self.read_csv_input(val_csv_dir, [0, 1])
        # print(val_labels[:10])
        # copy the whole folder to target dir
        # Create thread pool
        pool = multiprocessing.Pool(4)
        copy = True
        resize = True

        if copy:
            print("copy files from {} to {}...".format("20bn-jester-v1", "train"))
            # for train_label in tqdm(train_labels):

            for _ in tqdm(pool.imap_unordered(self.copy_train, train_labels), total=len(train_labels)):
                pass
            # pool.map(copy_train, train_labels)
            print("copy files from {} to {}...".format("20bn-jester-v1", "val"))
            # for val_label in tqdm(val_labels):

            # pool.map(copy_val, val_labels)
            for _ in tqdm(pool.imap_unordered(self.copy_val, val_labels), total=len(val_labels)):
                pass


        # resize image
        if resize == True:
            print("resize images to ({},{})...".format(self.resize_width, self.resize_height))


            for _ in tqdm(pool.imap_unordered(self.resize_train_dataset, train_labels), total=len(train_labels)):
                pass


            for _ in tqdm(pool.imap_unordered(self.resize_val_dataset, val_labels), total=len(val_labels)):
                pass

        pool.close()
        print('Preprocessing finished.')

    def read_csv_input(self, csv_path, selected_entries):
        csv_data = []
        with open(csv_path) as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=';')
            for row in csv_reader:
                item = []
                for entry in selected_entries:
                    item += [row[entry]]
                # row[0] video id; row[1] class label
                # item = ListDataJpeg(row[0],[1])
                csv_data.append(item)
        return csv_data

    def randomflip(self, buffer):
        """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                frame = cv2.flip(buffer[i], flipCode=1)
                buffer[i] = cv2.flip(frame, flipCode=1)

        return buffer


    def normalize(self, buffer):
        for i, frame in enumerate(buffer):
            frame -= np.array([[[90.0, 98.0, 102.0]]])
            buffer[i] = frame

        return buffer

    def to_tensor(self, buffer):
        return buffer.transpose((3, 0, 1, 2))

    def load_frames(self, file_dir):
        frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
        frame_count = len(frames) if len(frames) >= self.clip_len else self.clip_len
        # change empty to zeros, since some video has less frames; also change frame_count to clip_len
        buffer = np.zeros((frame_count, self.resize_height, self.resize_width, 3), np.dtype('float32'))
        for i, frame_name in enumerate(frames):
            frame = np.array(cv2.imread(frame_name)).astype(np.float64)
            buffer[i] = frame

        return buffer

    def crop(self, buffer, clip_len, crop_size):
        # randomly select time index for temporal jittering
        time_padding = buffer.shape[0] - clip_len
        time_index = np.random.randint(time_padding) if time_padding > 0 else 0
        # Randomly select start indices in order to crop the video
        height_index = np.random.randint(buffer.shape[1] - crop_size)
        width_index = np.random.randint(buffer.shape[2] - crop_size)

        # Crop and jitter the video using indexing. The spatial crop is performed on
        # the entire array, so each frame is cropped in the same location. The temporal
        # jitter takes place via the selection of consecutive frames
        buffer = buffer[time_index:time_index + clip_len,
                 height_index:height_index + crop_size,
                 width_index:width_index + crop_size, :]

        return buffer

    def resize(self, buffer, size=(400, 640)):
        B = buffer.shape[0]
        new_buffer = np.empty((B,) + size + (3,), np.dtype('float32'))
        for b in range(B):
            new_buffer[b] = cv2.resize(buffer[b], dsize=(size[1], size[0]))
        return new_buffer 





if __name__ == "__main__":
    from torch.utils.data import DataLoader
    subset = ['No gesture', 'Thumb Down', 'Thumb Up', 'Swiping Left', 'Swiping Right']
    train_data = VideoDataset(dataset='20bn-jester', split='train', clip_len=2, preprocess=True, subset=subset)
    train_loader = DataLoader(train_data, batch_size=2, shuffle=True, num_workers=1)

    for i, sample in enumerate(train_loader):
        inputs = sample[0]
        labels = sample[1]
        print(inputs.size())
        print(labels)

        if i == 1:
            break