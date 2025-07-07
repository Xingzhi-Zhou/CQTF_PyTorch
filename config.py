import os
import torch
import triton.language as tl

### GPU ################################################################
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### Quantizaiton config ################################################

# Total bits
TB = 10

# Fractional bits
FB = 7

# CQTF config

QUANT_MODE = 'RN'
# QUANT_MODE = 'RC'
# QUANT_MODE = 'RF'

# Need to manually replace the quantization function in CQTF_matmul_kernel
# Line 42 of .\models\CQTF_module.py
if QUANT_MODE == 'RN':
    round_func = lambda x: tl.floor(x + 0.5)
elif QUANT_MODE == 'RC':
    round_func = tl.ceil
elif QUANT_MODE == 'RF':
    round_func = tl.floor   

#########################################################################


### training config ####################################################
save_model = True

load_model = False
start_epoch = 0

model_name = 'CQTF_VGG_SMALL'

batch_size = 128
epochs = 2000

learning_rate = 0.007
momentum = 0.9
weight_decay = 1e-4

#########################################################################

quant_step = 1 / (2 ** FB)
max_val = (2 ** (TB - 1) - 1) * quant_step
min_val = -max_val