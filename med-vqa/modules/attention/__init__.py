from .ban import *
from .dan import *
from .mfb import *
from .san import *



def get_attention(name,args,dataset):

    if name == 'BAN':
        return BAN(args,dataset)
    elif name == 'MFB':
        args.HIGH_ORDER = False
        args.HIDDEN_SIZE = 512
        args.MFB_K = 5
        args.MFB_O = 1024
        args.LSTM_OUT_SIZE = 1024
        args.DROPOUT_R = 0.1
        args.I_GLIMPSES = 2
        args.Q_GLIMPSES = 2
        return MFB(args,dataset.v_dim,args.hid_dim,True)
    elif name == 'MFBCo':
        args.HIGH_ORDER = False
        args.HIDDEN_SIZE = 512
        args.MFB_K = 5
        args.MFB_O = 1024
        args.LSTM_OUT_SIZE = 1024
        args.DROPOUT_R = 0.1
        args.I_GLIMPSES = 2
        args.Q_GLIMPSES = 2
        return CoAtt(args)

    elif name == 'MFH':
        args.HIGH_ORDER = True
        args.HIDDEN_SIZE = 512
        args.MFB_K = 5
        args.MFB_O = 1024
        args.LSTM_OUT_SIZE = 1024
        args.DROPOUT_R = 0.1
        args.I_GLIMPSES = 2
        args.Q_GLIMPSES = 2
        return CoAtt(args)

    elif name == 'SAN':
        return StackedAttention(args.num_stacks, dataset.v_dim, args.hid_dim, args.hid_dim,args.dropout)
      
    elif name == 'DAN':
        return rDAN(args)

    elif name == 'VO':
        return nn.Linear(128,1024)

    elif name == 'QO':
        return nn.Linear(1024,1024)    


