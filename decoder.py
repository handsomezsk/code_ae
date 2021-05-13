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

        self.dec_outputs = torch.nn.Linear(2*args.dec_num_unit, 1)
        self.fc = torch.nn.Linear(2*args.dec_num_unit, args.dec_num_unit)

        self.attn = torch.nn.Linear(2*args.dec_num_unit, args.dec_num_unit, bias=False)
        self.v = torch.nn.Linear(args.dec_num_unit, 2, bias=False)

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

    def attention(self, s, hidden):

        src_len = int(hidden.shape[1]/2)

        # repeat decoder hidden state src_len times
        # s = [batch_size, src_len, dec_hid_dim]
        # enc_output = [batch_size, src_len, enc_hid_dim * 2]
        s = s.repeat(1, src_len, 1)
        # energy = [batch_size, src_len, dec_hid_dim]
        energy = torch.tanh(self.attn(torch.cat((s, hidden), dim=2)))
        # attention = [batch_size, src_len]
        attention = self.v(energy)
        attention = attention.transpose(1, 2)
        return F.softmax(attention, dim=2)

    def forward(self, received):
        received = received.type(torch.FloatTensor).to(self.this_device)
        for i in range(self.args.block_len):
            if i == 0:
                out1, hidden1 = self.dec1_rnns(received[:, i:i+1, :])
                out2, hidden2 = self.dec2_rnns(received[:, i:i+1, :])
                hidden1_tra = hidden1.transpose(0, 1)
                hidden2_tra = hidden2.transpose(0, 1)
                hiddens1 = hidden1_tra
                hiddens2 = hidden2_tra
                a1 = self.attention(hidden1_tra, hiddens1)
                a2 = self.attention(hidden2_tra, hiddens2)
                
                c1 = torch.bmm(a1, hiddens1)
                c1 = c1.transpose(0, 1)
                hidden1 = torch.cat((c1, hidden1), dim=2)
                hidden1 = self.fc(hidden1)
                c2 = torch.bmm(a2, hiddens2)
                c2 = c2.transpose(0, 1)
                hidden2 = torch.cat((c2, hidden2), dim=2)
                hidden2 = self.fc(hidden2)

                rnn_out1 = out1
                rnn_out2 = out2
            else:
                out1, hidden1 = self.dec1_rnns(received[:, i:i+1, :], hidden1)
                out2, hidden2 = self.dec2_rnns(received[:, i:i+1, :], hidden2)
                hidden1_tra = hidden1.transpose(0, 1)
                hidden2_tra = hidden2.transpose(0, 1)
                hiddens1 = torch.cat((hiddens1, hidden1_tra), dim=1)
                hiddens2 = torch.cat((hiddens2, hidden2_tra), dim=1)
                a1 = self.attention(hidden1_tra, hiddens1)
                a2 = self.attention(hidden2_tra, hiddens2)

                c1 = torch.bmm(a1, hiddens1)
                c1 = c1.transpose(0, 1)
                hidden1 = torch.cat((c1, hidden1), dim=2)
                hidden1 = self.fc(hidden1)
                c2 = torch.bmm(a2, hiddens2)
                c2 = c2.transpose(0, 1)
                hidden2 = torch.cat((c2, hidden2), dim=2)
                hidden2 = self.fc(hidden2)

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