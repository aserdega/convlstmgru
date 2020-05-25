import torch
import torch.nn as nn

import torch.optim as optim
import torch.nn.functional as F

import torchvision.utils
import torchvision.datasets as dset
import torchvision.transforms as transforms

import numpy as np

from convlstm import ConvLSTM
from cnn import ConvEncoder, ConvDecoder

from bouncing_mnist import BouncingMnist

DATA_PATH = "./mnist_test/mnist_test_500.npy"
b_size = 32

CUDA1 = "cuda:1"

class TesterMnist:
    def __init__(self, data_path="./mnist_test/mnist_test_500.npy", out_dir="./test_out"):
        self.dataset = np.load(data_path)
        self.size = self.dataset.shape[0]
        self.iter = 0
        self.out_dir = out_dir

    def run_test(self, cnn_encoder, cnn_decoder, lstm_encoder, lstm_decoder):
        """
        cnn_encoder = cnn_encoder.eval()
        cnn_decoder = cnn_decoder.eval()
        lstm_encoder = lstm_encoder.eval()
        lstm_decoder = lstm_decoder.eval()
        """

        hidden_dim = lstm_encoder.input_dim
        hidden_spt = lstm_encoder.height

        mse_hist = []
        bce_hist = []
        pnsr_hist = torch.zeros(10).cuda()#, device=CUDA1)#divide by number of batches!!!
        fr_mse_hist = torch.zeros(10).cuda()#, device=CUDA1)#divide by number of batches!!!

        i = 0
        batch_n = 1
        for sl in range(0,self.size - b_size,b_size):
            seqs = torch.from_numpy(self.dataset[sl:sl+b_size,:,:,:,:])
            nextf_raw = seqs[:,10:,:,:,:].cuda()
            #----cnn encoder----

            prevf_raw = seqs[:,:10,:,:,:].contiguous().view(-1,1,64,64).cuda()
            prevf_enc = cnn_encoder(prevf_raw).view(b_size,10,hidden_dim,hidden_spt,hidden_spt)

            #----lstm encoder---

            hidden           = lstm_encoder.get_init_states(b_size)
            _, encoder_state = lstm_encoder(prevf_enc, hidden)

            #----lstm decoder---

            decoder_output_list = []

            for s in range(10):
                if s == 0:
                    decoder_out, decoder_state = lstm_decoder(prevf_enc[:,-1:,:,:,:], encoder_state)
                else:
                    decoder_out, decoder_state = lstm_decoder(decoder_out, decoder_state)
                decoder_output_list.append(decoder_out)

            final_decoder_out = torch.cat(decoder_output_list, 1)

            #----cnn decoder----

            decoder_out_rs  = final_decoder_out.view(-1,hidden_dim,hidden_spt,hidden_spt)

            cnn_decoder_out_raw = F.sigmoid(cnn_decoder(decoder_out_rs).detach())
            cnn_decoder_out     = cnn_decoder_out_raw.view(b_size,10,1,64,64)

            #-----calculate mse and bce----------
            slice_mse = F.mse_loss(cnn_decoder_out, nextf_raw).item()
            slice_bce = F.binary_cross_entropy(cnn_decoder_out, nextf_raw).item()

            mse_hist.append(slice_mse)
            bce_hist.append(slice_bce)

            cur_pnsr, cur_fr_mse = self.calculate_pnsr(cnn_decoder_out, nextf_raw)
            pnsr_hist += cur_pnsr
            fr_mse_hist += cur_fr_mse

            batch_n += 1

        pnsr_hist = pnsr_hist / batch_n
        fr_mse_hist =  fr_mse_hist / batch_n

        mse_total = np.mean(mse_hist)
        bce_total = np.mean(bce_hist)

        l = []
        for j in range(3):
            l.append(seqs[j,:10,:,:,:])
            l.append(cnn_decoder_out[j,:,:,:,:].data.cpu())
            l.append(seqs[j,10:,:,:,:])
        samples = torch.cat(l)
        torchvision.utils.save_image(samples,
                                     self.out_dir + "/{0:0>5}iter.png".format(self.iter), nrow=10)
        self.iter += 1

        """
        cnn_encoder = cnn_encoder.train()
        cnn_decoder = cnn_decoder.train()
        lstm_encoder = lstm_encoder.train()
        lstm_decoder = lstm_decoder.train()
        """

        return mse_total, bce_total, pnsr_hist, fr_mse_hist

    #sum, not mean for mse, wrong
    def calculate_pnsr(self, pred, target):
        elw_mse = F.mse_loss(pred,target,size_average=False,reduce=False).squeeze()
        fr_mse  = elw_mse.sum(-1).sum(-1).mean(0)
        pnsr = 10 * torch.log10(1 / elw_mse.mean(-1).mean(-1).mean(0))
        return pnsr, fr_mse

