
import torch
from easydict import EasyDict as edict
import torch.backends.cudnn as cudnn
import os


C = edict()
config = C
cfg = C
##Select the Nvidia card
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'  #'1'
#from torch.autograd import Variable
##----------------------------------Common Settings----------------------------
##Fix seed to reproduce result
#random.seed(1234)
#torch.manual_seed(1234)

C.seed = 12345

##Network setting
C.pre_trained=True
##Optimization
C.num_epoch = 20000
C.lr_S = 2e-4
C.lr_D = 2e-5
C.momentum_S=0.9
C.momentum_D=0.9
C.step_size_S = 5000
C.step_size_D = 5000
C.beta1=0.9
C.beta2=0.999
C.batch_train = 4
##CUDNN
cudnn.enabled = True
cudnn.benchmark=True

##Data setting
#xdim = 164
#ydim = 144
#zdim = 192
C.traindata_path = '/your/path/to/your/train_h5file'
C.valdata_path = '/your/path/to/your/val_h5file'
C.input_dim = 2
C.ignore_label = 9
C.num_classes= 4
C.crop_size = (64, 64, 64)
## Note
C.checkpoint_name= '3dbrainseg'
C.note_S='3dbrainseg(Adam lr_S: ' + str(C.lr_S) + ',w_decay:1e-4' + 'beta:' +str(C.beta1)+ ',' + str(C.beta2) + ',' + 'step:' + str(C.step_size_S) + ' , lr_step)'
C.note_D='3dbrainseg(Adam lr_S: ' + str(C.lr_S) + ',w_decay:1e-4' + 'beta:' +str(C.beta1)+ ',' + str(C.beta2) + ',' + 'step:' + str(C.step_size_S) + ' , lr_step)'

C.num_checkpoint='20000'
C.note= str(C.num_checkpoint) +'_' + C.checkpoint_name
#Testing
C.checkpoint='./checkpoints/'+str(C.num_checkpoint) +'_' + C.checkpoint_name + '.pth'

#---------------------------------------------------------------------------------
print('@%s:  ' % os.path.basename(__file__))

# if 1:
#     torch.backends.cudnn.benchmark = True  ##uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms. -
#     torch.backends.cudnn.enabled   = True
#     print ('\tset cuda environment')
#     print ('\t\ttorch.__version__              =', torch.__version__)
#     print ('\t\ttorch.version.cuda             =', torch.version.cuda)
#     print ('\t\ttorch.backends.cudnn.version() =', torch.backends.cudnn.version())
#     try:
#         print ('\t\tos[\'CUDA_VISIBLE_DEVICES\']     =',os.environ['CUDA_VISIBLE_DEVICES'])
#         NUM_CUDA_DEVICES = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
#     except Exception:
#         print ('\t\tos[\'CUDA_VISIBLE_DEVICES\']     =','None')
#         NUM_CUDA_DEVICES = 1
#
#     print ('\t\ttorch.cuda.device_count()      =', torch.cuda.device_count())
#     print ('\t\ttorch.cuda.current_device()    =', torch.cuda.current_device())
#
#
# print('')

#---------------------------------------------------------------------------------

##----------------------------------Common Functions----------------------------
