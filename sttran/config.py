from argparse import ArgumentParser

BATCHNORM_MOMENTUM = 0.01

class Config():
    """Wrapper class for model hyperparameters."""

    def __init__(self, mode='predcls', save_path='data/', model_path='', datasize='large',
                 data_path='/data/scene_understanding/action_genome',
                 ckpt='checkpoint', optimizer='adamw', lr=1e-5, enc_layer=1, dec_layer=3,
                 bce_loss=False, nepoch=10, interactive=False):
        """
        Defaults
        """
        self.mode = mode
        self.save_path = save_path
        self.model_path = model_path
        self.data_path = data_path
        self.datasize = datasize
        self.ckpt = ckpt
        self.optimizer = optimizer
        self.bce_loss = bce_loss
        self.lr = lr
        self.enc_layer = enc_layer
        self.dec_layer = dec_layer
        self.nepoch = nepoch

        self.interactive = interactive
        if not self.interactive:
            self.parser = self.setup_parser()
            self.args = vars(self.parser.parse_args())
            self.__dict__.update(self.args)

    def setup_parser(self):
        """
        Sets up an argument parser
        :return:
        """
        parser = ArgumentParser(description='training code')
        parser.add_argument('-mode', dest='mode', help='predcls/sgcls/sgdet', default='predcls', type=str)
        parser.add_argument('-save_path', default='data/', type=str)
        parser.add_argument('-model_path', default=None, type=str)
        parser.add_argument('-data_path', default='/data/scene_understanding/action_genome/', type=str)
        parser.add_argument('-datasize', dest='datasize', help='mini dataset or whole', default='large', type=str)
        parser.add_argument('-ckpt', dest='ckpt', help='checkpoint', default=None, type=str)
        parser.add_argument('-optimizer', help='adamw/adam/sgd', default='adamw', type=str)
        parser.add_argument('-lr', dest='lr', help='learning rate', default=1e-5, type=float)
        parser.add_argument('-nepoch', help='epoch number', default=10, type=float)
        parser.add_argument('-enc_layer', dest='enc_layer', help='spatial encoder layer', default=1, type=int)
        parser.add_argument('-dec_layer', dest='dec_layer', help='temporal decoder layer', default=3, type=int)
        parser.add_argument('-bce_loss', action='store_true')
        return parser
