import torch.nn.functional as F
from torch.nn import Parameter
import torch

from numpy import arange
from numpy.random import mtrand
import numpy as np


class DEC(torch.nn.Module):
    def __init__(self, args):
        super(DEC, self).__init__()
        self.args = args

        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")

        if args.dec_rnn == 'gru':
            RNN_MODEL = torch.nn.GRU
        elif args.dec_rnn == 'lstm':
            RNN_MODEL = torch.nn.LSTM
        else:
            RNN_MODEL = torch.nn.RNN

        self.dropout = torch.nn.Dropout(args.dropout)

        self.dec1_rnns = RNN_MODEL(int(args.code_rate_n / args.code_rate_k), args.dec_num_unit,
                                   num_layers=2, bias=True, batch_first=True,
                                   dropout=args.dropout)

        self.dec2_rnns = RNN_MODEL(int(args.code_rate_n / args.code_rate_k), args.dec_num_unit,
                                   num_layers=2, bias=True, batch_first=True,
                                   dropout=args.dropout)

        self.dec_outputs = torch.nn.Linear(2*(args.dec_num_unit+args.enc_num_unit), 1)

        self.attn = torch.nn.Linear(args.dec_num_unit+args.enc_num_unit, args.dec_num_unit, bias=False)

        self.v = torch.nn.Linear(args.dec_num_unit, 1, bias=False)

    def dec_act(self, inputs):
        if self.args.dec_act == 'tanh':
            return F.tanh(inputs)
        elif self.args.dec_act == 'elu':
            return F.elu(inputs)
        elif self.args.dec_act == 'relu':
            return F.relu(inputs)
        elif self.args.dec_act == 'selu':
            return F.selu(inputs)
        elif self.args.dec_act == 'sigmoid':
            return F.sigmoid(inputs)
        elif self.args.dec_act == 'linear':
            return inputs
        else:
            return inputs

    def attention(self, s, encoder_hidden):
        s = s[1:2, :, :]
        s = s.transpose(0, 1)
        src_len = encoder_hidden.shape[1]

        # repeat decoder hidden state src_len times
        # s = [batch_size, src_len, dec_hid_dim]
        # enc_output = [batch_size, src_len, enc_hid_dim * 2]
        s = s.repeat(1, src_len, 1)
        # energy = [batch_size, src_len, dec_hid_dim]
        energy = torch.tanh(self.attn(torch.cat((s, encoder_hidden), dim=2)))
        # attention = [batch_size, src_len]
        attention = self.v(energy).squeeze(2)
        return F.softmax(attention, dim=1)

    def forward(self, received, encoder_hidden, h0):
        received = received.type(torch.FloatTensor).to(self.this_device)
        for i in range(self.args.block_len):
            if i==0:
                out1, decoder_hidden1 = self.dec1_rnns(received[:, i:i+1, :], h0)
                out2, decoder_hidden2 = self.dec2_rnns(received[:, i:i+1, :], h0)
                a = self.attention(h0, encoder_hidden).unsqueeze(1)
                c = torch.bmm(a, encoder_hidden)
                rnn_out1 = torch.cat((out1,c), dim=2)
                rnn_out2 = torch.cat((out2,c), dim=2)
            else:
                a1 = self.attention(decoder_hidden1, encoder_hidden).unsqueeze(1)
                a2 = self.attention(decoder_hidden2, encoder_hidden).unsqueeze(1)
                c1 = torch.bmm(a1, encoder_hidden)
                c2 = torch.bmm(a2, encoder_hidden)
                out1, decoder_hidden1 = self.dec1_rnns(received[:, i:i + 1, :], decoder_hidden1)
                out2, decoder_hidden2 = self.dec2_rnns(received[:, i:i + 1, :], decoder_hidden2)
                out1 = torch.cat((out1, c1), dim=2)
                out2 = torch.cat((out2, c2), dim=2)
                rnn_out1 = torch.cat((rnn_out1, out1), dim=1)
                rnn_out2 = torch.cat((rnn_out2, out2), dim=1)

        for i in range(self.args.block_len):
            if (i >= self.args.block_len - self.args.D - 1):
                rt_d = rnn_out2[:, self.args.block_len - 1:self.args.block_len, :]
            else:
                rt_d = rnn_out2[:, i + self.args.D:i + self.args.D + 1, :]
            rt = rnn_out1[:, i:i + 1, :]
            rnn_out = torch.cat((rt, rt_d), dim=2)
            dec_out = self.dec_outputs(rnn_out)
            if i == 0:
                final = dec_out
            else:
                final = torch.cat((final, dec_out), dim=1)
        final = torch.sigmoid(final)
        return final