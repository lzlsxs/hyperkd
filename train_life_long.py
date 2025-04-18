import os
import random
import torch
#from networks.vit_pytorch import ViT
from networks.LSGA_ViT import LSGAVIT
from networks.network import LLL_Net
import torch.nn as nn
import argparse
import numpy as np
#from networks import allmodels
import approach
import importlib
from approach.incremental_learning import Inc_Learning_Appr
from utils.utils import print_summary
from datasets.my_dataset import get_loader


parser = argparse.ArgumentParser("HSI")
#dataset args
parser.add_argument('--batch_size', type=int, default=64, help='number of batch size')
#parser.add_argument('--nc_first_task', default=None, type=int, required=False,
#                        help='Number of classes of the first task (default=%(default)s)')
parser.add_argument('--stop_at_task', default=0, type=int, required=False,
                        help='Stop training after specified task (default=%(default)s)')

#model args
parser.add_argument('--mode', choices=['ViT', 'CAF'], default='ViT', help='mode choice')
parser.add_argument('--patches', type=int, default=7, help='number of patches')
parser.add_argument('--band_patches', type=int, default=3, help='number of related band')
#parser.add_argument('--max_band', type=int, default=224, help='number of band')

# training args
parser.add_argument('--approach', default='finetuning', type=str, choices=approach.__all__,
                        help='Learning approach used (default=%(default)s)', metavar="APPROACH")
parser.add_argument('--nepochs', type=int, default=300, help='epoch number')
parser.add_argument('--lr', default=5e-4, type=float, required=False,
                        help='Starting learning rate (default=%(default)s)')
parser.add_argument('--lr_min', default=1e-8, type=float, required=False,
                        help='Minimum learning rate (default=%(default)s)')
parser.add_argument('--lr_factor', default=1/0.9, type=float, required=False,
                        help='Learning rate decreasing factor (default=%(default)s)')
parser.add_argument('--lr_patience', default=5, type=int, required=False,
                        help='Maximum patience to wait before decreasing learning rate (default=%(default)s)')
parser.add_argument('--clipping', default=10000, type=float, required=False,
                        help='Clip gradient norm (default=%(default)s)')
parser.add_argument('--momentum', default=0.0, type=float, required=False,
                        help='Momentum factor (default=%(default)s)')
parser.add_argument('--weight_decay', default=0.0, type=float, required=False,
                        help='Weight decay (L2 penalty) (default=%(default)s)')
parser.add_argument('--warmup_nepochs', default=0, type=int, required=False,
                        help='Number of warm-up epochs (default=%(default)s)')
parser.add_argument('--warmup_lr_factor', default=1.0, type=float, required=False,
                        help='Warm-up learning rate factor (default=%(default)s)')
parser.add_argument('--multi_softmax', action='store_true',
                        help='Apply separate softmax for each task (default=%(default)s)')
parser.add_argument('--fix_bn', action='store_true',
                        help='Fix batch normalization after first task (default=%(default)s)')
parser.add_argument('--eval_on_train', action='store_true',
                        help='Show train loss and accuracy (default=%(default)s)')
#miscellaneous args
parser.add_argument('--seed', type=int, default=0, help='number of seed')
parser.add_argument('--results_path', type=str, default='./results',
                        help='Results path (default=%(default)s)')
parser.add_argument('--exp_name', default=None, type=str,
                        help='Experiment name (default=%(default)s)')
parser.add_argument('--log', default=['disk'], type=str, choices=['disk', 'tensorboard'],
                        help='Loggers used (disk, tensorboard) (default=%(default)s)', nargs='*', metavar="LOGGER")
parser.add_argument('--save_models', action='store_true',
                        help='Save trained models (default=%(default)s)')
parser.add_argument('--last_layer_analysis', action='store_true',
                        help='Plot last layer analysis (default=%(default)s)')

# gridsearch args
parser.add_argument('--gridsearch-tasks', default=-1, type=int,
                        help='Number of tasks to apply GridSearch (-1: all tasks) (default=%(default)s)')
args,extra_args = parser.parse_known_args()
base_kwargs = dict(nepochs=args.nepochs, lr=args.lr, lr_min=args.lr_min, lr_factor=args.lr_factor,
                       lr_patience=args.lr_patience, clipgrad=args.clipping, momentum=args.momentum,
                       wd=args.weight_decay, multi_softmax=args.multi_softmax, wu_nepochs=args.warmup_nepochs,
                       wu_lr_factor=args.warmup_lr_factor, fix_bn=args.fix_bn, eval_on_train=args.eval_on_train)
#设置随机种子
np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print('=' * 108)
print('Arguments =')
for arg in np.sort(list(vars(args).keys())):
    print('\t' + arg + ':', getattr(args, arg))
print('=' * 108)


#Loaders
all_datasets =['Pavia','Indian','Salinas','Houston']
#dataname_to_trainpath = {'Indian':'./data/Indian_pines_train.mat', 'Houston':'./data/Houston_train.mat','Salinas':'./data/Salinas_train.mat','Pavia':'./data/Pavia_train.mat'}
#dataname_to_testpath = {'Indian':'./data/Indian_pines_test.mat', 'Houston':'./data/Houston_test.mat','Salinas':'./data/Salinas_test.mat','Pavia':'./data/Pavia_test.mat'}
dataname_to_path = {'Indian':'./data/IndianPine.mat', 'Houston':'./data/Houston.mat','Salinas':'./data/Salinas.mat','Pavia':'./data/Pavia.mat'}
label_offset = 0
trn_loader=[]
tst_loader=[]
taskcla =[]
bands =[]
print("dataset: "+ str(all_datasets))
for i in range(len(all_datasets)):
    data_path = os.path.expanduser(dataname_to_path[all_datasets[i]])
    #test_path = os.path.expanduser(dataname_to_testpath[all_datasets[i]]) 
    train_loader ,test_loader,num_classes,band = get_loader(data_path,args.batch_size,args.patches,args.band_patches,is_shuffle=True, tsk_offset= label_offset)
    taskcla.append(num_classes)
    bands.append(band)
    trn_loader.append(train_loader)
    tst_loader.append(test_loader)
    label_offset += num_classes
    print(str(all_datasets[i]) + '_train_samples_num : '+str(len(train_loader.dataset)))
    print(str(all_datasets[i]) + '_test_samples_num : '+str(len(test_loader.dataset)))

max_task = len(taskcla) if args.stop_at_task == 0 else args.stop_at_task
max_band = max(bands)
print("taskcla: " + str(taskcla))
print('max_task: ' + str(max_task))
print('band: '+ str(bands))
print("max_band: "+str(max_band))
print('length_per_band:'+str(args.patches*args.patches*args.band_patches))
print('=' * 108)
#Args --model
# init_model = ViT(
#     image_size = args.patches,
#     near_band= args.band_patches,
#     num_patches= max_band,
#     dim = 64,
#     depth= 5,
#     heads= 4,
#     mlp_dim= 8,
#     dropout= 0.1,
#     emb_dropout= 0.1,
#     mode= args.mode   
# )
init_model = LSGAVIT(img_size=args.patches,
                         patch_size=3,
                         in_chans=36,
                         num_classes=num_classes,
                         embed_dim=120,
                         depths=[2],
                         num_heads=[12, 12, 12, 24],
                         )
model = LLL_Net(init_model)

#Args -- continual learning approach
Appr = getattr(importlib.import_module(name='approach.' + args.approach), 'Appr')
assert issubclass(Appr, Inc_Learning_Appr)
appr_args, extra_args = Appr.extra_parser(extra_args)
print('Approach arguments =')
for arg in np.sort(list(vars(appr_args).keys())):
    print('\t' + arg + ':', getattr(appr_args, arg))
print('=' * 108)


# Args -- Exemplars Management
from datasets.exemplars_dataset import ExemplarsDataset
Appr_ExemplarsDataset = Appr.exemplars_dataset_class()
if Appr_ExemplarsDataset:
    assert issubclass(Appr_ExemplarsDataset, ExemplarsDataset)
    appr_exemplars_dataset_args, extra_args = Appr_ExemplarsDataset.extra_parser(extra_args)
    print('Exemplars dataset arguments =')
    for arg in np.sort(list(vars(appr_exemplars_dataset_args).keys())):
        print('\t' + arg + ':', getattr(appr_exemplars_dataset_args, arg))
    print('=' * 108)
else:
    appr_exemplars_dataset_args = argparse.Namespace()

appr_kwargs = {**base_kwargs, **dict(logger=None, **appr_args.__dict__)}
if Appr_ExemplarsDataset:
    appr_kwargs['exemplars_dataset'] = Appr_ExemplarsDataset(None, None,
                                                **appr_exemplars_dataset_args.__dict__)



assert len(extra_args) == 0, "Unused args: {}".format(' '.join(extra_args))

appr = Appr(model, device, **appr_kwargs)
print('Approach_kwargs:')
print(appr_kwargs)
print('=' * 108)



#Loop tasks
print(taskcla)
acc_taw = np.zeros((max_task, max_task))
acc_tag = np.zeros((max_task, max_task))
forg_taw = np.zeros((max_task, max_task))
forg_tag = np.zeros((max_task, max_task))
for t,  ncla in enumerate(taskcla):
    if t >= max_task:
        continue
    print('*' * 108)
    print('Task {:2d}'.format(t))
    print('*' * 108)
    # Add head for current task
    appr.model.add_head(taskcla[t])
    appr.model.to(device)
    
    appr.train(t,trn_loader[t],tst_loader[t])
    for u in range(t + 1):
        test_loss, acc_taw[t, u], acc_tag[t, u],OA_tar, AA_mean_tar, Kappa_tar, AA_tar,OA_tag, AA_mean_tag, Kappa_tag, AA_tag  = appr.eval(u, tst_loader[u])
        if u < t:
            forg_taw[t, u] = acc_taw[:t, u].max(0) - acc_taw[t, u]
            forg_tag[t, u] = acc_tag[:t, u].max(0) - acc_tag[t, u]
        print('>>> Test on task {:2d} : loss={:.3f} | TAw acc={:5.1f}%, forg={:5.1f}%'
            '| TAg acc={:5.1f}%, forg={:5.1f}% <<<'.format(u, test_loss,100 * acc_taw[t, u], 100 * forg_taw[t, u],100 * acc_tag[t, u], 100 * forg_tag[t, u]))
        print("OA_tar: {:.4f} | AA_tar: {:.4f} | Kappa_tar: {:.4f}".format(OA_tar, AA_mean_tar, Kappa_tar))
        print("OA_tag: {:.4f} | AA_tag: {:.4f} | Kappa_tag: {:.4f}".format(OA_tag, AA_mean_tag, Kappa_tag))
        print(AA_tar)
        print(AA_tag)
print_summary(acc_taw, acc_tag, forg_taw, forg_tag)
print('Done!')
