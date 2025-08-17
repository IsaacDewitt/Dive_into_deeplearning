from torch import nn

class Encoder(nn.Module):
    def __init(self,**kwargs):
        super(Encoder,self).__init__(**kwargs)
    def forward(self,x,**kwargs):
        raise NotImplementedError

class Decoder(nn.Module):
    def __intit(self,**kwargs):
        super(Decoder,self).__init__(**kwargs)
    def init_state(self,enc_outputs, *args):
        raise NotImplementedError
    def forward(self,x,state):
        raise NotImplementedError

class EncoderDecoder(nn.Module):
    def __init__(self,encoder,decoder, **kwargs):
        super(EncoderDecoder,self).__init__(**kwargs)
        self.encoder = encoder
        self.deocoder = decoder
    def foward(self,enc_x,dec_x,*args):
        # *args接受任意数量的参数，打包成一个元组
        # **kwargs接受任意数量的参数，打包成一个字典
        enc_outputs = self.encoder(enc_x,*args)
        dec_state = self.decoder.init_state(enc_outputs,*args)
        return self.decoder(dec_x,dec_state)
