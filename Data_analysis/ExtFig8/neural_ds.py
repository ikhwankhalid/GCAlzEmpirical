import torch
import numpy as np


class NeuralDatasetSimple(torch.utils.data.Dataset):
    """
    Represent our pose and neural data.

    """

    def __init__(
        self,
        ifr,
        pose,
        time,
        seq_length,
        ifr_normalization_means=None,
        ifr_normalization_stds=None,
    ):
        """
        ifr: instantaneous firing rate
        pose: position of the animal
        seq_length: length of the data passed to the network
        """
        super().__init__()
        self.ifr = ifr.astype(np.float32)
        self.oriIfr = ifr.astype(np.float32)
        self.pose = pose.astype(np.float32)
        self.time = time.astype(np.float32)
        self.seq_length = seq_length

        self.ifr_normalization_means = ifr_normalization_means
        self.ifr_normalization_stds = ifr_normalization_stds

        self.normalize_ifr()

        self.validIndices = np.argwhere(~np.isnan(self.pose[:, 0])).squeeze()
        # print(self.validIndices)
        self.validIndices = self.validIndices[
            self.validIndices > seq_length
        ]  # make sure we have enough neural dat leading to the pose

    def normalize_ifr(self):
        """
        Set the mean of each neuron to 0 and std to 1
        Neural networks work best with inputs in this range
        Set maximal values at -5.0 and 5 to avoid extreme data points

        ###########
        # warning #
        ###########

        In some situation, you should use the normalization of the training set to normalize your test set.
        For instance, if the test set is very short, you might have a very poor estimate of the mean and std, or the std might be undefined if a neuron is silent.
        """
        if self.ifr_normalization_means is None:
            self.ifr_normalization_means = self.ifr.mean(axis=0)
            self.ifr_normalization_stds = self.ifr.std(axis=0)

        self.ifr = (
            self.ifr - np.expand_dims(self.ifr_normalization_means, 0)
        ) / np.expand_dims(self.ifr_normalization_stds, axis=0)
        self.ifr[self.ifr > 5.0] = 5.0
        self.ifr[self.ifr < -5.0] = -5.0

    def __len__(self):
        return len(self.validIndices)

    def __getitem__(self, index):
        """
        Function to get an item from the dataset

        Returns pose, neural data

        """
        neuralData = self.ifr[
            self.validIndices[index] - self.seq_length : self.validIndices[index], :
        ]
        pose = self.pose[self.validIndices[index] : self.validIndices[index] + 1, :]  #
        time = self.time[self.validIndices[index] : self.validIndices[index] + 1]

        return (
            torch.from_numpy(neuralData),
            torch.from_numpy(pose).squeeze(),
            torch.from_numpy(time),
        )
