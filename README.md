# ConvLSTM and ConvGRU | Pytorch
**Implementation of ConvolutionalLSTM and ConvolutonalGRU in PyTorch**

Inspired by [this](https://github.com/ndrplz/ConvLSTM_pytorch) repository but has been refactored and got new features such as peephole option and usage examples in implementations of video predicton seq-to-seq models on moving MNIST dataset.

### How to Use
The `ConvLSTM` and `ConvGRU` module are inherited from `nn.Module`.

The ConvLSTM and ConvGRU allow using any number of layers. In this case, it can be specified the hidden dimension (that is, the number of channels) and the kernel size of each layer. In the case more layers are present but a single value is provided, this is replicated for all the layers. For example, in the following snippet each of the three layers has a different hidden dimension but the same kernel size.

Short code snippet of usage
```
conv_lstm_encoder = ConvLSTM(
                   input_size=(hidden_spt,hidden_spt),
                   input_dim=hidden_dim,
                   hidden_dim=lstm_dims,
                   kernel_size=(3,3),
                   num_layers=3,
                   peephole=True,
                   batchnorm=False,
                   batch_first=True,
                   activation=F.tanh
                  )
                  
hidden = conv_lstm_encoder.get_init_states(batch_size)
output, encoder_state = conv_lstm_encoder(input, hidden)
```

### Project Structure
## Main Files
convlstm.py - contains main classes for ConvLSTMCell(represents one layer) and ConvLSTM modules
convgru.py  - same as for convlstm
## Other
train_gru_predictor.py and train_lstm_predictor.py - train video prediction models based on ConvGru and ConvLSTM respectively
cnn.py - file that contains simple convolutional networks for encoding and decoding frames representations
bouncing_mnist.py - contains dataloader that generates moving MNIST dataset from plain MNIST on a fly (requires file)
generate_test_set.py - used to generate testing data for trained models
test.py - contains tester for trained models
