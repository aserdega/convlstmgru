import os
import numpy as np

import torch
import torch.nn as nn

import torch.optim as optim
import torch.nn.functional as F

import torchvision.utils
import torchvision.datasets as dset
import torchvision.transforms as transforms

from convlstm import ConvLSTM
from cnn import ConvEncoder, ConvDecoder

import test
from bouncing_mnist import BouncingMnist


#-------parameters----

b_size     = 32

hidden_dim = 64
hidden_spt = 16

lstm_dims = [64,64,64]

teacher_forcing = True

torch.manual_seed(2)
np.random.seed(2)
torch.backends.cudnn.deterministic = True

#----------create some dirs---------

root_log_dir = './results_verylong_' + ('' if teacher_forcing else 'ntf_') + \
                                                    str(lstm_dims)[1:-1].replace(', ','x')

exists = True
while exists:
    exists = os.path.exists(root_log_dir)
    if not exists:
        os.makedirs(root_log_dir)
    else:
        root_log_dir += '_'

train_out_dir = root_log_dir + '/train'
test_out_dir = root_log_dir + '/test'

chkpnt_dir = root_log_dir + '/checkpoint'

os.makedirs(train_out_dir)
os.makedirs(test_out_dir)
os.makedirs(chkpnt_dir)

log_file = root_log_dir + '/out.log'
f = open(log_file,"w+")
f.close()

#-------some methods---------

def save_model(step, train_bce, test_bce):
    new_dir =  chkpnt_dir + "/{0:0>5}_iter|test_bce:{1:.4f}|train_bce:{2:.4f}".format(step, test_bce, train_bce)
    os.makedirs(new_dir)

    torch.save(lstm_encoder.state_dict(), new_dir + '/lstm_encoder.msd')
    torch.save(lstm_decoder.state_dict(), new_dir + '/lstm_decoder.msd')
    torch.save(cnn_encoder.state_dict(), new_dir + '/cnn_encoder.msd')
    torch.save(cnn_decoder.state_dict(), new_dir + '/cnn_decoder.msd')

def get_sample_prob(step):
    alpha = 2450#1150
    beta  = 8000
    return alpha / (alpha + np.exp((step + beta) / alpha))

#----------data---------

transform = transforms.Compose([
    transforms.Lambda(lambda x: torch.from_numpy(x))
    ])

raw_data  = BouncingMnist(transform=transform)
dloader   = torch.utils.data.DataLoader(raw_data, batch_size=b_size,
                                      shuffle=True, drop_last=True, num_workers=4)

tester = test.TesterMnist(out_dir=test_out_dir)

#--------model---------

cnn_encoder = ConvEncoder(hidden_dim)
cnn_decoder = ConvDecoder(b_size=b_size,inp_dim=hidden_dim)

cnn_encoder.cuda()
cnn_decoder.cuda()

lstm_encoder = ConvLSTM(
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

lstm_decoder = ConvLSTM(
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

lstm_encoder.cuda()
lstm_decoder.cuda()


sigmoid = nn.Sigmoid()
crit = nn.BCELoss()
crit.cuda()


params = list(cnn_encoder.parameters()) + list(cnn_decoder.parameters()) + \
         list(lstm_encoder.parameters()) + list(lstm_decoder.parameters())

p_optimizer = optim.Adam(params)

#--------train---------

i = 0

for e in range(100):
    for _, batch in enumerate(dloader):
        p_optimizer.zero_grad()

        seqs = batch
        nextf_raw = seqs[:,10:,:,:,:].cuda()

        #----cnn encoder----

        prevf_raw = seqs[:,:10,:,:,:].contiguous().view(-1,1,64,64).cuda()
        prevf_enc = cnn_encoder(prevf_raw).view(b_size,10,hidden_dim,hidden_spt,hidden_spt)

        if teacher_forcing:
            cnn_encoder_out = cnn_encoder(seqs[:,10:,:,:,:].contiguous().view(-1,1,64,64).cuda())
            nextf_enc       = cnn_encoder_out.view(b_size,10,hidden_dim,hidden_spt,hidden_spt)

        #----lstm encoder---

        hidden           = lstm_encoder.get_init_states(b_size)
        _, encoder_state = lstm_encoder(prevf_enc, hidden)

        #----lstm decoder---

        sample_prob =  get_sample_prob(i) if teacher_forcing else 0
        decoder_output_list = []
        r_hist = []

        for s in range(10):
            if s == 0:
                decoder_out, decoder_state = lstm_decoder(prevf_enc[:,-1:,:,:,:], encoder_state)
            else:
                r = np.random.rand()
                r_hist.append(int(r > sample_prob)) #debug

                if r > sample_prob:
                    decoder_out, decoder_state = lstm_decoder(decoder_out, decoder_state)
                else:
                    decoder_out, decoder_state = lstm_decoder(nextf_enc[:,s-1:s,:,:,:], decoder_state)

            decoder_output_list.append(decoder_out)

        final_decoder_out = torch.cat(decoder_output_list, 1)

        #----cnn decoder----

        decoder_out_rs  = final_decoder_out.view(-1,hidden_dim,hidden_spt,hidden_spt)

        cnn_decoder_out_raw = F.sigmoid(cnn_decoder(decoder_out_rs))
        cnn_decoder_out     = cnn_decoder_out_raw.view(b_size,10,1,64,64)

        #-----predictor loss----------

        pred_loss = crit(cnn_decoder_out, nextf_raw)

        total_loss = pred_loss
        total_loss.backward()
        p_optimizer.step()

        #----ouputs---------

        if i % 100 == 0:
            output_str = " | Epoch: {0} | Iter: {1} |\n".format(e, i)
            output_str += "   TotalLoss:     {0:.6f}\n".format(total_loss.item())
            output_str += "-------------------------------------------\n"
            output_str += "   Sampling prob: {0:.3f}\n".format(sample_prob)
            output_str += "   Sampling:\n"
            output_str += "    " + str(r_hist) + "\n"

            l = []
            for j in range(3):
                l.append(seqs[j,:10,:,:,:])
                l.append(cnn_decoder_out[j,:,:,:,:].data.cpu())
                l.append(seqs[j,10:,:,:,:])
            samples = torch.cat(l)
            torchvision.utils.save_image(samples,
                                         train_out_dir + "/{0:0>5}iter.png".format(i), nrow=10)

        if i % 200 == 0:
            #evaluate on the test set
            test_mse, test_bce, pnsr, fr_mse = tester.run_test(cnn_encoder,cnn_decoder, lstm_encoder, lstm_decoder)
            if i == 0:
                min_test_bce = test_bce

            output_str += "-------------------------------------------\n"
            output_str += "   Test BCE:      {0:.6f}\n".format(test_bce)
            output_str += "   Test MSE:      {0:.6f}\n".format(test_mse)
            output_str += "-------------------------------------------\n"
            output_str += "   Test PNSR:\n"
            output_str += "    " + str(pnsr.data.cpu()) + "\n"
            output_str += "   Frame mse:\n"
            output_str += "    " + str(fr_mse.data.cpu()) + "\n"

        if i % 100 == 0:
            output_str += "\n=================================================\n\n"
            print(output_str)
            with open(log_file, "a") as lf:
                lf.write(output_str)

        if i > 35000 and (i % 400 == 0 or test_bce < min_test_bce):
            min_test_bce = min(min_test_bce, test_bce)
            save_model(i, total_loss.item(), test_bce)

        #--------------------

        i += 1
    if i >= 60000:
        exit()

