# ======================================================
# Bin Hoang, University of Rochester
# pace.hyperparams.py
# Fully self-contained version with hardcoded defaults
# ======================================================

import torch
import os


class net_pars:
    def __init__(self):
        # ----------------------
        # Model configuration
        # ----------------------
        self.use_three_compartment = True   # default 3-compartment IVIM
        self.fitS0 = True                   # fit S0 explicitly
        self.norm_input = True
        self.IR = False                     # disable inversion recovery

        # IR timing defaults (only used if IR=True)
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

        # ----------------------
        # Network architecture
        # ----------------------
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

        # ----------------------
        # Training hyperparameters
        # ----------------------
        self.train_batch_size = 128
        self.lr = 1e-3
        self.maxit = 10000
        self.patience = 5
        self.scheduler = True
        self.optim = "adam"

        # -------------------------------------------------------------
        # Constraint bounds — ORIGINAL PAPER LIMITS (Voorter et al.)
        # -------------------------------------------------------------
        # No tissue-specific merging. No padding. No range expansion.
        #
        # Network order: [Dpar, Fint, Dint, Fmv, Dmv, S0]
        # -------------------------------------------------------------
        # In pace.hyperparams.py

        if self.use_three_compartment:
            # ---------------------------------------------------------
            # VOORTER TABLE S1 (Supporting Information)
            # ---------------------------------------------------------
            
            # 1. Dpar (Parenchymal Diffusion)
            dpar_bounds = [0.0001, 0.0015]

            # 2. Fint (Intermediate Fraction)
            fint_bounds = [0.000, 0.400]

            # 3. Dint (Intermediate Diffusion)
            dint_bounds = [0.0015, 0.0040]

            # 4. Fmv (Microvascular Fraction)
            fmv_bounds = [0.000, 0.200]

            # 5. Dmv (Pseudo-diffusion)
            dmv_bounds = [0.0040, 0.2000]

            # 6. S0 (Signal Intercept)
            s0_bounds  = [0.7, 1.3]

            # ---------------------------------------------------------
            # FINAL ARRAY 
            # ---------------------------------------------------------
            self.cons_min = [
                dpar_bounds[0], fint_bounds[0], dint_bounds[0], 
                fmv_bounds[0],  dmv_bounds[0],  s0_bounds[0]
            ]

            self.cons_max = [
                dpar_bounds[1], fint_bounds[1], dint_bounds[1], 
                fmv_bounds[1],  dmv_bounds[1],  s0_bounds[1]
            ]


        else:
            # Original 2C unchanged
            self.cons_min = [0.0003, 0.0,   0.005, 0.5]
            self.cons_max = [0.005,  0.7,   0.30,  2.5]

        # NOTE: We intentionally remove padding. 
        # The OG paper uses exact constraint limits.

        # ----------------------
        # Device and result directory
        # ----------------------
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.result_dir = "./results"
        os.makedirs(self.result_dir, exist_ok=True)
