import random
import torch

def decide_slice_range(max_length=500, frames=50):
    left = random.randint(0, max_length-frames)
    right = left + frames
    return (left, right)


def slice_features(z, slice_range):
    left, right = slice_range[0], slice_range[1]
    return z[:, :, left:right]


def slice_waveform(wf, slice_range, frame_size):
    left, right = slice_range[0], slice_range[1]
    return wf[:, left*frame_size:right*frame_size]