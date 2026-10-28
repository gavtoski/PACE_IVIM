
import torch
import os


class net_pars:
    def __init__(self):
        self.use_three_compartment = True
        self.fitS0 = True
        self.norm_input = True
        self.IR = False

        class RelTimes:
            def __init__(self):
                self.echotime = 75.0
                self.repetitiontime = 5000.0
                self.inversiontime = 500.0
                self.tissueT1 = 1200.0
                self.tissueT2 = 80.0
                self.isfT1 = 3000.0
                self.isfT2 = 300.0
                self.bloodT1 = 1650.0
                self.bloodT2 = 275.0

        self.rel_times = RelTimes()
        self.profile = "brain3" if self.use_three_compartment else "brain2"

        self.depth = 2
        self.width = 46
        self.dropout = 0.1
        self.batch_norm = True
        self.cube_size = 3
        self.patch_size = 3
        self.latent_output = 64
        self.cnn_channels = 2
        self.use_struct = True
        self.con = "sigmoid"

        self.train_batch_size = 128
        self.lr = 1e-3
        self.maxit = 10000
        self.patience = 5
        self.scheduler = True
        self.optim = "adam"


        if self.use_three_compartment:
            
            dpar_bounds = [0.0001, 0.0015]

            fint_bounds = [0.000, 0.400]

            dint_bounds = [0.0015, 0.0040]

            fmv_bounds = [0.000, 0.200]

            dmv_bounds = [0.0040, 0.2000]

            s0_bounds  = [0.7, 1.3]

            self.cons_min = [
                dpar_bounds[0], fint_bounds[0], dint_bounds[0], 
                fmv_bounds[0],  dmv_bounds[0],  s0_bounds[0]
            ]

            self.cons_max = [
                dpar_bounds[1], fint_bounds[1], dint_bounds[1], 
                fmv_bounds[1],  dmv_bounds[1],  s0_bounds[1]
            ]


        else:
            self.cons_min = [0.0003, 0.0,   0.005, 0.5]
            self.cons_max = [0.005,  0.7,   0.30,  2.5]


        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.result_dir = "./results"
        os.makedirs(self.result_dir, exist_ok=True)
