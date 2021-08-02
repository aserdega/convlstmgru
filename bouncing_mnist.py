from __future__ import division

import sys
import h5py
import numpy as np


class BouncingMnist():
  """Data Handler that creates Bouncing MNIST dataset on the fly."""
  def __init__(self, mnistDataset='mnist.h5', mode='standard', transform=None, background='zeros', num_frames=20, batch_size=1, image_size=64, num_digits=2, step_length=0.1):
    self.mode_ = mode
    self.background_ = background
    self.seq_length_ = num_frames
    self.batch_size_ = batch_size
    self.image_size_ = image_size
    self.num_digits_ = num_digits
    self.step_length_ = step_length
    self.dataset_size_ = 20000  # The dataset is really infinite. This is just for validation.
    self.digit_size_ = 28
    self.frame_size_ = self.image_size_ ** 2
    self.num_channels_ = 1
    self.transform_ = transform

    try:
      f = h5py.File(mnistDataset)
    except:
      print('Please set the correct path to MNIST dataset')
      sys.exit()

    self.data_ = f['train'][()].reshape(-1, 28, 28)
    #self.test = f['test'][()].reshape(-1, 28, 28)

    f.close()
    self.indices_ = np.arange(self.data_.shape[0])
    self.row_ = 0
    np.random.shuffle(self.indices_)

  def __len__(self):
    return self.dataset_size_

  def __getitem__(self, idx):
    item = self.get_batch()[0,:,:,:,:]
    item_t = self.transform_(item)
    return item_t

  def GetRandomTrajectory(self, batch_size):
    length = self.seq_length_
    canvas_size = self.image_size_ - self.digit_size_

    # Initial position uniform random inside the box.
    y = np.random.rand(batch_size)
    x = np.random.rand(batch_size)

    # Choose a random velocity.
    theta = np.random.rand(batch_size) * 2 * np.pi
    v_y = np.sin(theta)
    v_x = np.cos(theta)

    start_y = np.zeros((length, batch_size))
    start_x = np.zeros((length, batch_size))
    for i in range(length):
      # Take a step along velocity.
      y += v_y * self.step_length_
      x += v_x * self.step_length_

      # Bounce off edges.
      for j in range(batch_size):
        if x[j] <= 0:
          x[j] = 0
          v_x[j] = -v_x[j]
        if x[j] >= 1.0:
          x[j] = 1.0
          v_x[j] = -v_x[j]
        if y[j] <= 0:
          y[j] = 0
          v_y[j] = -v_y[j]
        if y[j] >= 1.0:
          y[j] = 1.0
          v_y[j] = -v_y[j]
      start_y[i, :] = y
      start_x[i, :] = x

    # Scale to the size of the canvas.
    start_y = (canvas_size * start_y).astype(np.int32)
    start_x = (canvas_size * start_x).astype(np.int32)
    return start_y, start_x

  def Overlap(self, a, b):
    return np.maximum(a, b)

  def get_batch(self, verbose=False):
    start_y, start_x = self.GetRandomTrajectory(self.batch_size_ * self.num_digits_)

    # minibatch data
    if self.background_ == 'zeros':
        data = np.zeros((self.batch_size_, self.num_channels_, self.image_size_, self.image_size_, self.seq_length_), dtype=np.float32)
    elif self.background_ == 'rand':
        data = np.random.rand(self.batch_size_, self.num_channels_, self.image_size_, self.image_size_, self.seq_length_)

    for j in range(self.batch_size_):
      for n in range(self.num_digits_):

        # get random digit from dataset
        ind = self.indices_[self.row_]
        self.row_ += 1
        if self.row_ == self.data_.shape[0]:
          self.row_ = 0
          np.random.shuffle(self.indices_)
        digit_image = self.data_[ind, :, :]
        digit_size = self.digit_size_

        if self.mode_ == 'squares':
            digit_size = np.random.randint(5,20)
            digit_image = np.ones((digit_size, digit_size), dtype=np.float32)

        # generate video
        for i in range(self.seq_length_):
          top    = start_y[i, j * self.num_digits_ + n]
          left   = start_x[i, j * self.num_digits_ + n]
          bottom = top  + digit_size
          right  = left + digit_size
          data[j, :, top:bottom, left:right, i] = self.Overlap(data[j, :, top:bottom, left:right, i], digit_image)

    dum = np.moveaxis(data, -1, 1)
    return dum

