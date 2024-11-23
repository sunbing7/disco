root="/root/autodl-tmp/sunbing/workspace/uap/my_result/disco/training/save/cifar10/"
DiscoPth=$root"pgd/imnet_mlp/trial_1/epoch-best.pth"

python cifar10_transfer_defense_ours.py --attack=uap  --trial 1 --input_defense="disco" --disco_path $DiscoPth --uap_path=/root/autodl-tmp/sunbing/workspace/uap/my_result/uap_virtual_data.pytorch/uap/cifar10_cifar10_wideresnet_123 --uap_target=0


python cifar10_transfer_defense_ours.py --attack=uap --dataset=cifar10 --trial 1 --input_defense="disco" --disco_path=/root/autodl-tmp/sunbing/workspace/uap/my_result/disco/training/save/cifar10/pgd/imnet_mlp/trial_1/epoch-best.pth --uap_path=/root/autodl-tmp/sunbing/workspace/uap/my_result/uap_virtual_data.pytorch/uap/cifar10_cifar10_wideresnet_123 --uap_target=0