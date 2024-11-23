from robustbench.utils import load_model, clean_accuracy
import torch
import torch.nn as nn
import os
from robustbench.model_zoo.defense import inr
from typing import Callable, Dict, Optional, Sequence, Set, Tuple
import torchvision.datasets as datasets
import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import argparse
import json
import numpy as np
from tqdm import tqdm
from advertorch.attacks import LinfPGDAttack
from advertorch.bpda import BPDAWrapper
import torchattacks
from torchattacks import *
from robustbench.model_zoo.defense import bit_depth_reduction
from robustbench.model_zoo.defense import jpeg_compression 
from robustbench.model_zoo.defense import randomization
from robustbench.model_zoo.defense import inr
from robustbench.model_zoo.defense import autoencoder
from robustbench.model_zoo.defense import stl
import torch.nn as nn
from advertorch.defenses import MedianSmoothing2D
from attacker.iterative_gradient_attack import FGM_L2
import foolbox as fb
from advertorch.attacks import L2BasicIterativeAttack
from my_network import get_model_path, get_network
from my_data import *


parser = argparse.ArgumentParser(description='Parser')
parser.add_argument('--attack', type=str, help='attack')
parser.add_argument('--input_defense', type=str, default="disco", help='defense type')
parser.add_argument('--model_name', type=str, default="Standard", help='model name')
parser.add_argument('--disco_path', nargs='+', help='path to the disco model')
parser.add_argument('--stl_npy_name', type=str, default="64_p8_lm0.1", help='path to the stl numpy file')
parser.add_argument('--debug', action="store_true", default=False, help='debug mode')
parser.add_argument('--trial', type=str, help='trial')
parser.add_argument('--batch_size', type=int, default=100, help='bs')
parser.add_argument('--norm', type=str, default="Linf", help='L norm (Linf or L2)')
parser.add_argument('--dataset', type=str, default="cifar10", help='dataset')
parser.add_argument('--repeat', type=int, default=1, help='repeat')
parser.add_argument('--recursive_num', type=int, default=1, help='recursive_num for disco')
parser.add_argument('--uap_path', type=str, default=1)
parser.add_argument('--uap_target', type=int, default=0)


args = parser.parse_args()
assert args.trial is not None

PREPROCESSINGS = {
    'Res256Crop224':
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ]),
    'Crop288':
    transforms.Compose([transforms.CenterCrop(288),
                        transforms.ToTensor()]),
    None:
    transforms.Compose([transforms.ToTensor()]),
}



def load_cifar10(
    n_examples: Optional[int] = None,
    data_dir: str = './data',
    transforms_test: Callable = PREPROCESSINGS[None],
) -> Tuple[torch.Tensor, torch.Tensor]:
    dataset = datasets.CIFAR10(root=data_dir,
                               train=False,
                               transform=transforms_test,
                               download=True)
    
    test_loader = data.DataLoader(dataset,
                                  batch_size=n_examples,
                                  shuffle=False,
                                  num_workers=4)
    
    return test_loader
 

def metrics_evaluate_test(data_loader, target_model, uap, targeted, target_class, mask=None, defense=None, use_cuda=True):
    # switch to evaluate mode
    target_model.eval()

    clean_acc = AverageMeter()
    perturbed_acc = AverageMeter()
    attack_success_rate = AverageMeter() # Among the correctly classified samples, the ratio of being different from clean prediction (same as gt)
    if targeted:
        all_to_target_success_rate = AverageMeter() # The ratio of samples going to the sink classes
        all_to_target_success_rate_filtered = AverageMeter()

    total_num_samples = 0
    num_same_classified = 0
    num_diff_classified = 0

    for input, gt in data_loader:
        if use_cuda:
            gt = gt.cuda()
            input = input.cuda()

        # compute output
        with torch.no_grad():
            clean_output = target_model(input)
            if mask is None:
                adv_x = (input + uap).float()
            else:
                adv_x = torch.mul((1 - mask), input) + torch.mul(mask, uap).float()

            if args.input_defense != "no_input_defense":
                input = defense.forward(input)
                adv_x = defense.forward(adv_x)

            attack_output = target_model(adv_x)

        correctly_classified_mask = torch.argmax(clean_output, dim=-1).cpu() == gt.cpu()
        cl_acc = accuracy(clean_output.data, gt, topk=(1,))
        clean_acc.update(cl_acc[0].item(), input.size(0))
        pert_acc = accuracy(attack_output.data, gt, topk=(1,))
        perturbed_acc.update(pert_acc[0].item(), input.size(0))

        # Calculating Fooling Ratio params
        clean_out_class = torch.argmax(clean_output, dim=-1)
        uap_out_class = torch.argmax(attack_output, dim=-1)

        total_num_samples += len(clean_out_class)
        num_same_classified += torch.sum(clean_out_class == uap_out_class).cpu().numpy()
        num_diff_classified += torch.sum(~(clean_out_class == uap_out_class)).cpu().numpy()

        if torch.sum(correctly_classified_mask)>0:
            with torch.no_grad():
                pert_output_corr_cl = target_model(adv_x[correctly_classified_mask])
            attack_succ_rate = accuracy(pert_output_corr_cl, gt[correctly_classified_mask], topk=(1,))
            attack_success_rate.update(attack_succ_rate[0].item(), pert_output_corr_cl.size(0))


        # Calculate Absolute Accuracy Drop
        aad_source = clean_acc.avg - perturbed_acc.avg
        # Calculate Relative Accuracy Drop
        if clean_acc.avg != 0:
            rad_source = (clean_acc.avg - perturbed_acc.avg)/clean_acc.avg * 100.
        else:
            rad_source = 0.
        # Calculate fooling ratio
        fooling_ratio = num_diff_classified/total_num_samples * 100.

        if targeted:
            # 2. How many of all samples go the sink class (Only relevant for others loader)
            target_cl = torch.ones_like(gt) * target_class
            all_to_target_succ_rate = accuracy(attack_output, target_cl, topk=(1,))
            all_to_target_success_rate.update(all_to_target_succ_rate[0].item(), attack_output.size(0))

            # 3. How many of all samples go the sink class, except gt sink class (Only relevant for others loader)
            # Filter all idxs which are not belonging to sink class
            non_target_class_idxs = [i != target_class for i in gt]
            non_target_class_mask = torch.Tensor(non_target_class_idxs)==True
            if torch.sum(non_target_class_mask)>0:
                gt_non_target_class = gt[non_target_class_mask]
                pert_output_non_target_class = attack_output[non_target_class_mask]

                target_cl = torch.ones_like(gt_non_target_class) * target_class
                all_to_target_succ_rate_filtered = accuracy(pert_output_non_target_class, target_cl, topk=(1,))
                all_to_target_success_rate_filtered.update(all_to_target_succ_rate_filtered[0].item(), pert_output_non_target_class.size(0))
    print('\n\t#######################')
    print('\tClean model accuracy: {:.3f}'.format(clean_acc.avg))
    print('\tPerturbed model accuracy: {:.3f}'.format(perturbed_acc.avg))
    print('\tAbsolute Accuracy Drop: {:.3f}'.format(aad_source))
    print('\tRelative Accuracy Drop: {:.3f}'.format(rad_source))
    print('\tAttack Success Rate: {:.3f}'.format(100-attack_success_rate.avg))
    print('\tFooling Ratio: {:.3f}'.format(fooling_ratio))
    if targeted:
        print('\tAll --> Target Class {} Prec@1 {:.3f}'.format(target_class, all_to_target_success_rate.avg))
        print('\tAll (w/o sink samples) --> Sink {} Prec@1 {:.3f}'.format(target_class, all_to_target_success_rate_filtered.avg))



def main():

    if args.debug:
        root = "log/defense_transfer/debug"
    else:
        root = "log/defense_transfer" 

    device = torch.device('cuda')
    batch_size=args.batch_size


    num_classes, (mean, std), input_size, num_channels = get_data_specs(args.dataset)

    ####################################
    # Init model, criterion, and optimizer
    test_arch = 'wideresnet'
    model_name = 'wideresnet_cifar10.pth'
    print_log("=> Creating model '{}'".format(test_arch), log)
    # get a path for loading the model to be attacked
    model_path = get_model_path(dataset_name=args.dataset,
                                network_arch=test_arch,
                                random_seed=123)
    model_weights_path = os.path.join(model_path, model_name)

    target_network = get_network(test_arch,
                                 input_size=input_size,
                                 num_classes=num_classes,
                                 finetune=False)

    # Set the target model into evaluation mode
    target_network.eval()

    if args.dataset == "caltech" or args.dataset == 'asl':
        if 'repaired' in model_name:
            target_network = torch.load(model_weights_path, map_location=torch.device('cpu'))
        else:
            #state dict
            orig_state_dict = torch.load(model_weights_path, map_location=torch.device('cpu'))
            new_state_dict = OrderedDict()
            for k, v in target_network.state_dict().items():
                if k in orig_state_dict.keys():
                    new_state_dict[k] = orig_state_dict[k]

            target_network.load_state_dict(new_state_dict)
    elif args.dataset == 'eurosat':
        target_network = torch.load(model_weights_path, map_location=torch.device('cpu'))
        if 'repaired' in model_name:
            adaptive = '_adaptive'
    elif args.dataset == "imagenet" and 'repaired' in model_name:
        target_network = torch.load(model_weights_path, map_location=torch.device('cpu'))
    elif args.dataset == "cifar10":
        if 'repaired' in model_name:
            target_network = torch.load(model_weights_path, map_location=torch.device('cpu'))
        else:
            target_network.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))

    total_params = get_num_parameters(target_network)
    print_log("Target Network Total # parameters: {}".format(total_params), log)

    model = target_network.cuda()
    
    #model = load_model(model_name=args.model_name, dataset='cifar10', threat_model='Linf').to(device)

    uap_path = get_uap_path(uap_data=args.dataset,
                            model_data=args.dataset,
                            network_arch=test_arch,
                            random_seed=123)

    uap_fn = os.path.join(uap_path, 'uap_' + target_name + '.npy')
    uap = np.load(uap_fn) / np.array(std).reshape(1, 3, 1, 1)
    tuap = torch.from_numpy(uap)


    if args.input_defense == "disco":
        defense = inr.INR(device, args.disco_path, height=32, width=32)
    else:
        return

    metrics_evaluate_test(data_loader=data_test_loader,
                          target_model=target_network,
                          uap=tuap,
                          targeted=args.targeted,
                          target_class=args.target_class,
                          mask=mask,
                          defense=defense,
                          use_cuda=args.use_cuda)

if __name__ == "__main__":
    main()

