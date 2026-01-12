import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn.utils as utils

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


class LSTMSimple(torch.nn.Module):
    def __init__(
        self, input_size, hidden_size, num_layers, num_outputs, sequence_length, device
    ):
        super().__init__()
        """
        For more information about nn.LSTM -> https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        """

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # input : batch_size x sequence x features
        self.device = device
        self.fc = torch.nn.Linear(
            hidden_size * sequence_length, num_outputs
        )  # if you onely want to use the last hidden state (hidden_state,num_classes)

    def forward(self, x):

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out.reshape(out.shape[0], -1)
        out = self.fc(
            out
        )  # if you want to use only the last hidden state, remove previous line, # out = self.fc(out[:,-1,:])

        return out


def poseToGridSpace(
    pose,
    period=np.array([40, 40, 40]),
    orientation=np.array([0, np.pi / 3, np.pi / 3 * 2]),
):
    """
    Function to transfrom the x,y position of the mouse to
    a position within the internal representation of grid cells.

    The internal representation is 3 angles (x,y,z) which represents the distance along 3 axes
    The 3 axes are at 60 degrees of each other.
    To get from distance to angle, we get the modulo of the distance and the underlying spacing.
    Then set the range to -np.pi to pi.
    Each angle is represented by a cos and sin component to avoid discontinuity (0-360).

    Arguments:
    pose: 2D numpy array with x and y position, 2 columns
    period: spacing of the underlying band pattern
    orientation: angle of the 3 main axes of the grid pattern
    """

    Rx0 = np.array(
        [[np.cos(-orientation[0])], [-np.sin(-orientation[0])]]
    )  # minus sign because we want to rotate the inverse of the angle to bring it back to 1,0
    Rx1 = np.array([[np.cos(-orientation[1])], [-np.sin(-orientation[1])]])
    Rx2 = np.array([[np.cos(-orientation[2])], [-np.sin(-orientation[2])]])

    d0 = pose @ Rx0
    d1 = pose @ Rx1
    d2 = pose @ Rx2

    c0 = (d0 % period[0]) / period[0] * np.pi * 2 - np.pi
    c1 = (d1 % period[1]) / period[1] * np.pi * 2 - np.pi
    c2 = (d2 % period[2]) / period[2] * np.pi * 2 - np.pi

    c0c = np.cos(c0)
    c0s = np.sin(c0)
    c1c = np.cos(c1)
    c1s = np.sin(c1)
    c2c = np.cos(c2)
    c2s = np.sin(c2)

    return np.stack(
        [
            c0c.flatten(),
            c0s.flatten(),
            c1c.flatten(),
            c1s.flatten(),
            c2c.flatten(),
            c2s.flatten(),
        ]
    ).T


def gridSpaceToMovementPath(grid_coord, grid_period=40, orientation=0):
    """
    Function to go from grid cell coordinate (2 angles) to movement path

    gridSpace is a representation of the internal activity of the grid manifold. It has 3 dimensions that are circular. But we are only using 2 dimensions here
    When the active representation in grid space changes, we can transform this into movement in the real world.
    We don't know the absolute position of the animal, but we can recreate the movement path.

    We use 2 of the 3 components of the grid space to reconstruct the movement path.
    For each time sample, we know the movement in the grid cells space along these 2 directions.
    If we know that the mouse moved 2 cm along the first grid vector, the mouse can be at any position on a line that passes by 2*unitvector0 and is perpendicular to unitvector0
    If we know that the mouse moved 3 cm along the second grid vector, the mouse can be at any position on a line that passes by 3*unitvector1 and is perpendicular to unitvector1
    We just find the intersection of the two lines to know the movement of the mouse in x,y space.


    Arguments:
    grid_coord: is a 2D numpy array with the cos and sin component of the first 2 axes of the grid (4 columns)
    """

    # get angle from the cos and sin components
    ga0 = np.arctan2(grid_coord[:, 1], grid_coord[:, 0])
    ga1 = np.arctan2(grid_coord[:, 3], grid_coord[:, 2])

    # get how many cm per radian

    cm_per_radian = grid_period / (2 * np.pi)

    # get the movement along the 3 vector of the grid
    dga0 = mvtFromAngle(ga0, cm_per_radian[0])
    dga1 = mvtFromAngle(ga1, cm_per_radian[1])

    # unit vector and unit vector perpendicular to the grid module orientation vectors
    uv0 = np.array([[np.cos(orientation[0]), np.sin(orientation[0])]])  # unit vector v0
    puv0 = np.array(
        [[np.cos(orientation[0] + np.pi / 2), np.sin(orientation[0] + np.pi / 2)]]
    )  # unit vector perpendicular to uv0
    uv1 = np.array([[np.cos(orientation[1]), np.sin(orientation[1])]])  # unit vector v1
    puv1 = np.array(
        [[np.cos(orientation[1] + np.pi / 2), np.sin(orientation[1] + np.pi / 2)]]
    )  # unit vector perpendicular to uv1

    # two points in the x,y coordinate system that are on a line perpendicular to v0
    p1 = np.expand_dims(dga0, 1) * uv0  # x,y coordinate of movement along v0
    p2 = (
        p1 + puv0
    )  # a second x,y coordinate that is p1 plus a vector perpendicular to uv0

    # two points in the x,y coordinate system that are on a line perpendicular to v1
    p3 = np.expand_dims(dga1, 1) * uv1  # coordinate of the point 1 on line 1
    p4 = p3 + puv1  # coordinate of point 2 on line 1

    # find the intersection between 2 lines, using 2 points that are part of line 1 and 2 points that are part of line 2
    # https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
    px_num = (p1[:, 0] * p2[:, 1] - p1[:, 1] * p2[:, 0]) * (p3[:, 0] - p4[:, 0]) - (
        p1[:, 0] - p2[:, 0]
    ) * (p3[:, 0] * p4[:, 1] - p3[:, 1] * p4[:, 0])
    px_den = (p1[:, 0] - p2[:, 0]) * (p3[:, 1] - p4[:, 1]) - (p1[:, 1] - p2[:, 1]) * (
        p3[:, 0] - p4[:, 0]
    )
    reconstructedX = px_num / px_den
    py_num = (p1[:, 0] * p2[:, 1] - p1[:, 1] * p2[:, 0]) * (p3[:, 1] - p4[:, 1]) - (
        p1[:, 1] - p2[:, 1]
    ) * (p3[:, 0] * p4[:, 1] - p3[:, 1] * p4[:, 0])
    py_den = (p1[:, 0] - p2[:, 0]) * (p3[:, 1] - p4[:, 1]) - (p1[:, 1] - p2[:, 1]) * (
        p3[:, 0] - p4[:, 0]
    )
    reconstructedY = py_num / py_den

    return np.stack([reconstructedX, reconstructedY]).T


def mvtFromAngle(ga, cm_per_radian):
    """
    Go from an angle in the one grid coordinate (one of the 3 axes) to a change in position along this axis
    """
    dga = np.diff(ga, prepend=np.nan)  # this is the change in the angle
    dga = np.where(
        dga > np.pi, dga - 2 * np.pi, dga
    )  # correct for positive jumps because of circular data
    dga = np.where(dga < -np.pi, dga + 2 * np.pi, dga)  # correct for negative jumps
    dga = dga * cm_per_radian  # transform from change in angle to change in cm
    return dga


# Utility functions


def loss_on_dataset(model, data_loader, device, loss_fn):
    model.eval()
    loss_sum = 0
    total_attention_weights_n = 0
    total_attention_weights_t = 0

    with torch.no_grad():
        for imgs, labels, _, _, _ in data_loader:
            imgs = imgs.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.float32)
            outputs, attn_weights_t = model(imgs)

            # total_attention_weights_n += attn_weights_n.sum(dim=0)
            total_attention_weights_t += attn_weights_t.sum(dim=0)
            loss = loss_fn(outputs, labels)
            loss_sum += loss.item()

        average_attention_weights_n = total_attention_weights_n / len(
            data_loader.dataset
        )
        average_attention_weights_t = total_attention_weights_t / len(
            data_loader.dataset
        )

    return (
        loss_sum / len(data_loader),
        average_attention_weights_n,
        average_attention_weights_t,
    )


def train_epoch(model, data_loader, optimizer, loss_fn, device):
    model.train()
    loss_sum = 0

    total_attention_weights_n = 0
    total_attention_weights_t = 0
    max_grad_norm = 2

    for imgs, labels, _, _, _ in data_loader:
        imgs = imgs.to(device)
        labels = labels.to(device, dtype=torch.float32)

        outputs, attn_weights_t = model(
            imgs
        )  # outputs, attn_weights_t, attn_weights_n = model(imgs, time_diff)

        # total_attention_weights_n += attn_weights_n.sum(dim=0)
        total_attention_weights_t += attn_weights_t.sum(dim=0)

        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Plot the graph to show how gradient flows
        # plot_grad_flow(model.cpu().named_parameters())
        # model.to(device)

        utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        loss_sum += loss.item()

    average_attention_weights_n = total_attention_weights_n / len(data_loader.dataset)
    average_attention_weights_t = total_attention_weights_t / len(data_loader.dataset)

    return (
        loss_sum / len(data_loader),
        average_attention_weights_n,
        average_attention_weights_t,
    )


def evaluate_model(model, data_loader, device, loss_fn):
    return loss_on_dataset(model, data_loader, device, loss_fn)


def save_training_history_to_csv(history, file_path):
    print("Saving training history:", file_path)
    history.to_csv(file_path, index=False)


# Check of flow of gradients in the network
def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)


def vl_mvt_direction_error(mvtDirError):
    """
    Calculate the mean direction of the mvt direction error
    """
    xMean = np.mean(np.cos(mvtDirError))
    yMean = np.mean(np.sin(mvtDirError)) 
    return np.sqrt(xMean*xMean+yMean*yMean)