
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
from sparselearning.snip import SNIP, GraSP
import numpy as np
import math
import random
# from sparselearning.funcs import global_magnitude_prune
from sparselearning.funcs import redistribution_funcs
# from sparselearning.flops import print_model_param_nums,count_model_param_flops,print_inf_time
import wandb

def add_sparse_args(parser):
    parser.add_argument('--growth', type=str, default='random', help='Growth mode. Choose from: momentum, random, and momentum_neuron.')
    parser.add_argument('--death', type=str, default='magnitude', help='Death mode / pruning mode. Choose from: magnitude, SET, threshold, CS_death.')
    parser.add_argument('--redistribution', type=str, default='none', help='Redistribution mode. Choose from: momentum, magnitude, nonzeros, or none.')
    parser.add_argument('--death-rate', type=float, default=0.05, help='The pruning rate / death rate for DST.')
    parser.add_argument('--PF-rate', type=float, default=0.8, help='The pruning rate / death rate for Pruning and Finetuning.')
    parser.add_argument('--large-death-rate', type=float, default=0.80, help='The pruning rate / death rate.')
    parser.add_argument('--density', type=float, default=0.05, help='The density of the overall sparse network.')
    parser.add_argument('--sparse', action='store_true', help='Enable sparse mode. Default: True.')
    parser.add_argument('--fix', action='store_true', help='Fix topology during training. Default: True.')
    parser.add_argument('--sparse_init', type=str, default='ER', help='sparse initialization')
    parser.add_argument('--update_frequency', type=int, default=1000, metavar='N', help='how many iterations to train between parameter exploration')


    # for filter grow/prune
    parser.add_argument('--filter_dst', action='store_true', help='filter_dst')
    parser.add_argument('--new_zero', action='store_true', help='Init with zero momentum buffer')
    parser.add_argument('--fix_num_operation', type=int, default=0,help='fix_num_operation in prune and grow')
    parser.add_argument('--bound_ratio', type=float, default=5.0, help='The density of the overall sparse network.')
    parser.add_argument('--minumum_ratio', type=float, default=0.5, help='The density of the overall sparse network.')
    parser.add_argument('--grow_mask_not', action='store_true', help='grow_mask_not')
    parser.add_argument('--no_grow_mask', action='store_true', help='no_grow_mask')
    parser.add_argument('--random_init', action='store_true', help='no_grow_mask')

    parser.add_argument('--connection_wise', action='store_true', help='connection_wise')
    parser.add_argument('--kernal_wise', action='store_true', help='kernal_wise')
    parser.add_argument('--mask_wise', action='store_true', help='mask_wise')
    parser.add_argument('--mag_wise', action='store_true', help='mag_wise')

    parser.add_argument('--grad_flow', action='store_true', help='grad_flow')
    parser.add_argument('--stop_dst_epochs', type=int, default=30,help='stop_dst_epochs in prune and grow')

    parser.add_argument('--stop_gmp_epochs', type=int, default=130,help='stop_dst_epochs in prune and grow')


    parser.add_argument('--mest', action='store_true', help='mest')
    parser.add_argument('--mest_dst', action='store_true', help='mest')

    parser.add_argument('--dst', action='store_true', help='mest')
    parser.add_argument('--gpm_filter_pune', action='store_true', help='gpm_filter_pune')

    parser.add_argument('--he_threshold', type=float, default=2)
    parser.add_argument('--he_selective', action='store_true')
    parser.add_argument('--he_model', type=str, default='half_mhe')
    parser.add_argument('--he_power', type=int, default=-2)
    

class CosineDecay(object):
    def __init__(self, death_rate, T_max, eta_min=0.0, last_epoch=-1):
        self.sgd = optim.SGD(torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1))]), lr=death_rate)
        self.cosine_stepper = torch.optim.lr_scheduler.CosineAnnealingLR(self.sgd, T_max, eta_min, last_epoch)

    def step(self):
        self.cosine_stepper.step()

    def get_dr(self):
        return self.sgd.param_groups[0]['lr']

class LinearDecay(object):
    def __init__(self, death_rate, factor=0.99, frequency=600):
        self.factor = factor
        self.steps = 0
        self.frequency = frequency

    def step(self):
        self.steps += 1

    def get_dr(self, death_rate):
        if self.steps > 0 and self.steps % self.frequency == 0:
            return death_rate*self.factor
        else:
            return death_rate



class Masking(object):
    def __init__(self, optimizer, death_rate=0.3, growth_death_ratio=1.0, death_rate_decay=None, death_mode='magnitude', growth_mode='gradient', redistribution_mode='none', args=None, train_loader=None, device=None):
        growth_modes = ['random', 'momentum', 'momentum_neuron', 'gradient']
        if growth_mode not in growth_modes:
            print('Growth mode: {0} not supported!'.format(growth_mode))
            print('Supported modes are:', str(growth_modes))


        self.args = args
        self.train_loader = train_loader
        self.device = torch.device("cuda")
        self.growth_mode = growth_mode
        self.death_mode = death_mode
        self.growth_death_ratio = growth_death_ratio
        self.redistribution_func = args.redistribution

        self.death_rate_decay = death_rate_decay
        self.PF_rate = args.PF_rate



        self.masks = {}
        self.nonzero_masks = {}
        self.new_masks = {}
        self.pre_tensor = {}
        self.pruning_rate = {}
        self.modules = []
        self.names = []
        self.optimizer = optimizer

        self.adjusted_growth = 0
        self.adjustments = []
        self.baseline_nonzero = None
        self.name2baseline_nonzero = {}
        self.total_params=0
        # stats
        self.name2variance = {}
        self.name2zeros = {}
        self.name2nonzeros = {}
        self.total_variance = 0
        self.total_removed = 0
        self.total_zero = 0
        self.total_nonzero = 0
        self.death_rate = death_rate
        self.name2death_rate = {}
        self.steps = 0




        # channel_wise statisc
        self.channel_variance={}

        # if fix, then we do not explore the sparse connectivity
        if self.args.fix: self.prune_every_k_steps = None
        else: self.prune_every_k_steps = self.args.update_frequency


        # global growth/prune state
        self.prune_threshold = 0.001
        self.growth_threshold = 0.001
        self.growth_increment = 0.2
        self.increment = 0.2
        self.tolerance = 0.02



        #  for filter
        self.baseline_filter_num=0
        # self.layer_rate_decay=CosineDecay(self.args.start_layer_rate, 15)
        self.layer_rate_decay=CosineDecay(self.args.start_layer_rate, math.floor((self.args.stop_gmp_epochs)*len(self.train_loader)))
        # gmp channel prune

        self.initial_prune_time=0.0
        self.final_prune_time=math.floor(self.args.stop_gmp_epochs*len(self.train_loader))



        self.dst_decay=CosineDecay(0.5, math.floor((self.args.epochs)*len(self.train_loader)),0.005)

        self.active_new_mask={}
        self.passtive_new_mask={}


        jump_decay = CosineDecay(1, (args.update_frequency)*80/len(self.train_loader), 0)

        self.temp_mask={}

        self.overlap_history = []



    '''
      CHANEL EXPLORE

    '''
    def get_module(self,key):
        if "MobileNetV1" in  self.module.__class__.__name__ :
            # print ("key",key)
            return getattr(self.module,key[0])[key[1]][key[2]]

        if "VGG" in self.module.__class__.__name__ :
            return self.module.features[key]
        if self.module.__class__.__name__ =="ResNet" or self.module.__class__.__name__ =="ResNetbasic" or self.module.__class__.__name__ =="Model":
            return getattr(getattr(self.module, key[0])[key[1]],key[2])

    def update_filter_mask(self):
        print ("update_filter_mask")
        for module in self.modules:     # this is a list containing the entire model
            for name, tensor in self.module.named_parameters(): # iterate layers
                if name in self.filter_names:   # active layer
                    self.filter_masks[name][~self.filter_names[name].bool()]=0  # copies over masks from filter_names to filter_masks
                    self.filter_masks[name][self.filter_names[name].bool()]=1


                    # print (name,  self.filter_masks[name].sum())
                
            for name, tensor in self.module.named_parameters():
                if name in self.passive_names:
                    if name in self.filter_names:
                        filter_mask=self.filter_masks[name]
                    else:
                        filter_mask=torch.ones_like(tensor, dtype=torch.float32, requires_grad=False).cuda()

                    # transpose because we want to prune output channel (dim 0) of layer A and input channel (dim 1) of layer B
                    filter_mask =filter_mask.transpose(0, 1)
                    filter_mask[~self.passive_names[name].bool()]=0
                    filter_mask.transpose_(0, 1)

                    self.filter_masks[name]=filter_mask
                    # print (name,  self.filter_masks[name].sum())


        
    def get_cfg(self):
        print ("now filter structure")

        total=0
        for name, tensor in self.filter_names.items():
            print (name, tensor.sum().item())
            total+=tensor.sum().item()

        print ("total fitler number total",total)



    def get_filter_name(self):

        filter_names={}
        passive_names={}

        
        for ind in self.module.layer2split:

            dim= self.get_module(ind).weight.shape[0]
            
            mask=torch.ones(dim)
            # mask[int(dim/2):]=0
            filter_names[self.get_mask_name(ind)]=mask

            passive_ind=self.module.next_layers[ind][0]
            passive_names[self.get_mask_name(passive_ind)]=mask
        
            

        self.filter_names=filter_names
        self.passive_names=passive_names

    def get_mask_name(self,key):
        if "VGG" in self.module.__class__.__name__ :
            return "features."+str(key)+".weight"
        if self.module.__class__.__name__ =="ResNet" or self.module.__class__.__name__ =="ResNetbasic":      
            weight_name=[]
            bias_name=[]
            for i in key:

                weight_name.append(str(i))
                bias_name.append(str(i))

            weight_name.append("weight")
            weight_name=".".join(weight_name)


            return weight_name



    def filter_num(self):


        total=0
        for name, tensor in self.filter_names.items():
            # print (name, tensor.sum().item())
            total+=tensor.sum().item()


        return  total



    def merge_filters (self):
    
        # print ("self.filter_names",self.filter_names)
        for active_grow_key in self.module.layer2split:
            passive_grow_key,norm_key=self.module.next_layers[active_grow_key]




                #load original ===============================

            m1=self.get_module(active_grow_key)
            m2=self.get_module(passive_grow_key)
            bnorm=self.get_module(norm_key)

            no_bias= (m1.bias==None)
            # weight, bias

            w1 = m1.weight.data
            w2 = m2.weight.data
            if not no_bias: b1 = m1.bias.data


            #mask    
            m1_mask=self.masks[self.get_mask_name(active_grow_key)]
            m2_mask=self.masks[self.get_mask_name(passive_grow_key)]





            #init new ===============================================================================     
            # weight, bias
            old_width = w1.size(0)
            nw1 = w1.clone()
            nw2 = w2.clone()
            if not no_bias: nb1 = b1.clone()


            nrunning_mean = bnorm.running_mean.clone()
            nrunning_var = bnorm.running_var.clone()       
            nweight = bnorm.weight.data.clone()
            nbias = bnorm.bias.data.clone()

            #new_mask        
            n_m1_mask= m1_mask.clone()
            n_m2_mask= m2_mask.clone()


            #deleting   ==============================================================================
            # transpose from original
            w2 = w2.transpose(0, 1)
            nw2 = nw2.transpose(0, 1)

            m2_mask=m2_mask.transpose(0, 1)
            n_m2_mask=n_m2_mask.transpose(0, 1)   


            # layer mask
            mask=m1_mask

            #grow based on weight norm
            del_mask=self.filter_names[self.get_mask_name(active_grow_key) ].bool()
            # print (self.get_mask_name(active_grow_key),del_mask.sum().item())


            # delet============================   
            # weight, bias   
            nw1 = nw1[del_mask] 
            nw2 = nw2[del_mask]
            if not no_bias: nb1 = nb1[del_mask] 


            # bn weight, bias  
            nrunning_mean=nrunning_mean[del_mask]
            nrunning_var=nrunning_var[del_mask]
            nweight = nweight[del_mask] 
            nbias = nbias[del_mask] 


            # masks
            n_m1_mask=n_m1_mask[del_mask]    
            n_m2_mask=n_m2_mask[del_mask]


            # transpose back
            w2.transpose_(0, 1)
            nw2.transpose_(0, 1)


            m2_mask.transpose_(0, 1)
            n_m2_mask.transpose_(0, 1)


            new_width=del_mask.sum().item()


            m1.out_channels = new_width
            m2.in_channels = new_width
            bnorm.num_features = new_width


            #finalize ===============================  
            # weight bias
            m1.weight = torch.nn.Parameter(nw1)
            m2.weight = torch.nn.Parameter(nw2)
            if not no_bias: m1.bias= torch.nn.Parameter(nb1)


            # norm
            bnorm.running_var = nrunning_var
            bnorm.running_mean = nrunning_mean
            bnorm.weight = torch.nn.Parameter(nweight)
            bnorm.bias = torch.nn.Parameter(nbias)

            # mask
            self.masks[self.get_mask_name(active_grow_key)]=n_m1_mask
            self.masks[self.get_mask_name(passive_grow_key)]=n_m2_mask




    def distribute_del_func(self,new_width,active_grow_key, passive_grow_key):
        
       
        #load original ===============================

        m1=self.get_module(active_grow_key)
        m2=self.get_module(passive_grow_key)
        no_bias= (m1.bias==None)

        w1 = m1.weight.clone()
        w2 = m2.weight.clone()
        if not no_bias: b1 = m1.bias.data
        old_width = int(self.filter_names[self.get_mask_name(active_grow_key)].sum().item())
        # print ("old_width",old_width)




    
        #mask    
        m1_mask=self.masks[self.get_mask_name(active_grow_key)]

        

        
        #del based on weight norm
        mask=m1_mask

        grad=w1
        grad_all=[]
        for filter_grad, filter_mask in zip(grad,mask):
            if self.args.mask_wise:
                filter_single=torch.abs(filter_grad[filter_mask.bool()]).mean().item()
            elif self.args.mag_wise:
                filter_single=torch.abs(filter_grad).mean().item()
        #     print (filter_single)
            if np.isnan(filter_single):
                filter_single=0 
            grad_all.append(filter_single)



        grad_all=torch.FloatTensor(grad_all)

        # operate on select ind
        select_bool=self.filter_names[self.get_mask_name(active_grow_key)].bool()
        select_ind=torch.arange(len(grad_all))[select_bool]


        # sort
        y, idx = torch.sort(torch.abs(grad_all[select_bool]), descending=False)
        del_ind=idx[:old_width-new_width]

        # del

        del_ind=select_ind[del_ind]

        # print ("del_ind",del_ind)


        self.filter_names[self.get_mask_name(active_grow_key)][del_ind]=0
        self.passive_names[self.get_mask_name(passive_grow_key)][del_ind]=0




        assert self.filter_names[self.get_mask_name(active_grow_key)].sum().item() == self.passive_names[self.get_mask_name(passive_grow_key)].sum().item() , "Wrong deletling"
   

        self.update_filter_mask()

        self.apply_mask()



    def del_func(self,del_ind,active_grow_key, passive_grow_key):
        
   
        # print (self.get_mask_name(active_grow_key),"del",len(del_ind))


        self.filter_names[self.get_mask_name(active_grow_key)][del_ind]=0
        self.passive_names[self.get_mask_name(passive_grow_key)][del_ind]=0




        assert self.filter_names[self.get_mask_name(active_grow_key)].sum().item() == self.passive_names[self.get_mask_name(passive_grow_key)].sum().item() , "Wrong deletling"
   

        self.update_filter_mask()

        self.apply_mask()



    def prune_score(self,prune_layer_index,total_to_prune):

        # Gather all scores in a single vector and normalise
        all_scores=[]

        for index in prune_layer_index:


            # single metric
            weight=  torch.abs(self.get_module(index).weight.clone())


            m1_mask=self.masks[self.get_mask_name(index)]

            filter_mask=self.filter_names[self.get_mask_name(index)].bool()
            weight=weight[filter_mask]
            mask=m1_mask[filter_mask]

            for filter_weight, filter_mask in zip(weight,mask):

                if self.args.mask_wise:
                    weight_magnitude = torch.abs(filter_weight)  [filter_mask.bool()]  .mean().item()

                elif self.args.mag_wise:
                    weight_magnitude = torch.abs(filter_weight) .mean().item()

                elif self.args.kernal_wise:
                    vector = filter_weight.view(filter_weight.size(0), -1).sum(dim=1)
                    weight_magnitude=((vector!=0).sum().int().item()/vector.numel())


                elif self.args.connection_wise:
                    vector= filter_weight
                    weight_magnitude=((vector!=0).sum().int().item()/vector.numel())

                all_scores.append(weight_magnitude)

        print ("current score lengh",len(all_scores))
        # print ("all_scores",all_scores)

        acceptable_score = np.sort(np.array(all_scores))[-int(len(all_scores)-total_to_prune)]
        print ("acceptable_score",acceptable_score)
        # if acceptable_score==0:
        #     real_acceptable_score=np.sort(np.array(list(set(all_scores))))[1]

        # else:
        #     real_acceptable_score=acceptable_score

        return acceptable_score

    def gradual_pruning_rate(self,
            step: int,
            initial_threshold: float,
            final_threshold: float,
            initial_time: int,
            final_time: int,
    ):
        if step <= initial_time:
            threshold = initial_threshold
        elif step > final_time:
            threshold = final_threshold
        else:
            mul_coeff = 1 - (step - initial_time) / (final_time - initial_time)
            threshold = final_threshold + (initial_threshold - final_threshold) * (mul_coeff ** 3)

        return threshold

    def hyperspherical_channel_energy(self, index, model='half_mhe', power=-2):
        """Compute the hyperspherical energy of channels."""
        weights = self.get_module(index).weight.clone()
        filter_mask = self.filter_names[self.get_mask_name(index)].bool()
        weights = weights[filter_mask]

        if weights.size(0) <= 1:
            return torch.zeros(filter_mask.size(0), device=weights.device)

        W_flat = weights.view(weights.size(0), -1)

        if model == 'half_mhe':
            W_neg = -W_flat
            W_combined = torch.cat([W_flat, W_neg], dim=0)
        else:
            W_combined = W_flat

        n_original = W_flat.size(0)

        # Normalize vectors
        norms = torch.sqrt(torch.sum(W_combined * W_combined, dim=1, keepdim=True) + 1e-4)
        W_normalized = W_combined / norms

        # Compute pairwise dot products
        similarity_matrix = torch.matmul(W_normalized, W_normalized.t())

        # Clamp similarity to avoid numerical issues
        epsilon = 1e-4
        similarity_matrix = torch.clamp(similarity_matrix, -1 + epsilon, 1 - epsilon)

        # Compute energy matrix based on power
        if power > 0:  # Euclidean distance-based energy
            # ||u-v||² = 2 - 2(u·v) for unit vectors
            distance_squared = 2.0 - 2.0 * similarity_matrix

            eye = torch.eye(distance_squared.size(0), device=distance_squared.device)
            distance_squared = distance_squared + eye * epsilon

            if power == 1:
                energy_matrix = 1.0 / torch.sqrt(distance_squared)
            else:  # power == 2 or other
                energy_matrix = 1.0 / distance_squared  # For power=2, it's inverse squared

        elif power == 0:  # Logarithmic energy
            distance_squared = 2.0 - 2.0 * similarity_matrix
            distance_squared = distance_squared + torch.eye(distance_squared.size(0),
                                                            device=distance_squared.device) * epsilon
            energy_matrix = -torch.log(distance_squared)

        else:  # Angular distance-based energy
            # arccos(u·v)/π gives normalized angular distance
            angular_distances = torch.acos(similarity_matrix) / math.pi
            angular_distances = angular_distances + epsilon  # avoid division by zero

            # For power = -1, use inverse; for power = -2, use inverse squared
            energy_matrix = torch.pow(angular_distances, power)

        # Zero out diagonal elements (self-interactions)
        eye = torch.eye(energy_matrix.size(0), device=energy_matrix.device)
        energy_matrix = energy_matrix * (1.0 - eye)

        # Only consider upper triangular part to avoid counting pairs twice
        energy_matrix = torch.triu(energy_matrix, diagonal=1)

        # Calculate total energy for each channel
        if model == 'half_mhe':
            # For half-space, we need to handle original and virtual channels
            row_sum = energy_matrix.sum(dim=1)
            col_sum = energy_matrix.sum(dim=0)

            # Sum contributions for original channels only
            channel_energies = row_sum[:n_original] + col_sum[:n_original]
        else:
            # For full-space, just sum rows and columns
            channel_energies = energy_matrix.sum(dim=1) + energy_matrix.sum(dim=0)

        # Map back to original size including pruned channels
        full_energies = torch.zeros(filter_mask.size(0), device=weights.device)
        full_energies[filter_mask] = channel_energies

        return full_energies



    def track_hyperspherical_energy(self):
        he_values = {}
        layer_channel_counts = {}
        layer_stds = {}
        total_he = 0
        active_channel_count = 0
        all_layer_means = []

        # For each prunable layer
        for index in self.module.layer2split:
            layer_name = ".".join([str(i) for i in index])
            # Get HE for each channel in this layer
            channel_he = self.hyperspherical_channel_energy(index, model=self.args.he_model, power=self.args.he_power)

            # Filter out pruned channels
            filter_mask = self.filter_names[self.get_mask_name(index)].bool()
            active_channels = filter_mask.sum().item()

            if active_channels > 0:  # Avoid empty layers
                # Get HE only for active channels
                active_he = channel_he[filter_mask]

                # Calculate mean and standard deviation within this layer
                avg_he = active_he.mean().item()
                std_he = active_he.std().item()  # Standard deviation within layer

                # Store metrics
                he_values[layer_name] = avg_he
                layer_channel_counts[layer_name] = active_channels
                layer_stds[layer_name] = std_he

                all_layer_means.append(avg_he)
                total_he += active_he.sum().item()
                active_channel_count += active_channels

        # Prepare metrics dictionary - global metrics
        metrics = {
            "global/total_he": total_he,
            "global/mean_he": np.mean(all_layer_means) if all_layer_means else 0,
            "global/min_he": min(he_values.values()) if he_values else 0,
            "global/max_he": max(he_values.values()) if he_values else 0,
            "global/active_channels": active_channel_count,
            "global/step": self.steps
        }

        # Add per-layer metrics
        for layer_name, avg_he in he_values.items():
            channels = layer_channel_counts[layer_name]
            norm_factor = channels / 64.0  # Normalize relative to a 64-channel layer

            metrics[f"layer/{layer_name}/channel_count"] = channels
            metrics[f"layer/{layer_name}/mean_he"] = avg_he
            metrics[f"layer/{layer_name}/normalized_he"] = avg_he / norm_factor
            metrics[f"layer/{layer_name}/std"] = layer_stds[layer_name]

        wandb.log(metrics)

    def del_layer(self, selective=False):
        print("===========del layer with layer-wise HE===============")
        self.track_hyperspherical_energy()

        print(f"Total filter_names items: {sum(mask.sum().item() for mask in self.filter_names.values())}")
        print(f"Total filter_masks items: {sum(mask.sum().item() for mask in self.filter_masks.values())}")
        filter_number = self.filter_num()  # current num of filters in the network
        print(filter_number, '/', self.baseline_filter_num, 'channels')

        rate = 1 - self.layer_rate  # target density
        total_to_prune = filter_number - self.baseline_filter_num * rate
        print(f"Total channels to prune: {total_to_prune}")
        if total_to_prune <= 0:
            print("No channels to prune")
            return

        pruned_so_far = 0

        threshold = self.args.he_threshold

        for active_prune_key in self.module.layer2split:
            passive_prune_key, norm_key = self.module.next_layers[active_prune_key]

            name_mask = self.get_mask_name(active_prune_key)
            filter_mask = self.filter_names[name_mask]  # active filter mask
            active_channels = filter_mask.sum().item()

            layer_prune_amount = int(total_to_prune * (active_channels / filter_number))

            # Ensure we don't over-prune
            remaining_to_prune = total_to_prune - pruned_so_far
            layer_prune_amount = min(layer_prune_amount, remaining_to_prune)

            min_size = self.minimum_layer.get(active_prune_key, 1)
            max_prune = active_channels - min_size
            layer_prune_amount = min(layer_prune_amount, max_prune)

            if layer_prune_amount <= 0:
                continue

            he_scores = self.hyperspherical_channel_energy(active_prune_key, model=self.args.he_model,
                                                           power=self.args.he_power)

            # Get indices to prune (highest HE = most redundant)
            active_indices = torch.where(filter_mask.bool())[0].to(he_scores.device)
            active_he_scores = he_scores[filter_mask.bool()]

            if selective:

                # norm_he = active_he_scores / (active_channels * (active_channels - 1))
                # cv = norm_he.std() / (norm_he.mean() + 1e-8)

                cv = active_he_scores.std() / (active_he_scores.mean() + 1e-8)
                if cv >= 0.02:
                    print(f"layer {active_prune_key}: HE std = {active_he_scores.std():.4f}, cv = {cv}, using HE")
                    _, sorted_indices = torch.sort(active_he_scores, descending=True)  # higher HE = more prunable
                else:
                    print(f"layer {active_prune_key}: HE std = {active_he_scores.std():.4f}, cv = {cv}, using UMM")
                    weight = self.get_module(active_prune_key).weight.data
                    umm = weight.abs().mean(dim=(1, 2, 3))[filter_mask.bool()].cpu()
                    _, sorted_indices = torch.sort(umm, descending=False)  # lower UMM = more prunable

                # he_std = active_he_scores.std()
                # print(f"Layer {active_prune_key}: HE std = {he_std:.4f}, using {'UMM' if he_std < self.args.he_threshold else 'HE'}")
                #
                # if he_std < threshold:  # fall back to UMM
                #     weight = self.get_module(active_prune_key).weight.data
                #     umm = weight.abs().mean(dim=(1, 2, 3))[filter_mask.bool()].cpu()
                #
                #     _, sorted_indices = torch.sort(umm, descending=False)    # lower UMM = more prunable
                # else:  # use HE
                #     _, sorted_indices = torch.sort(active_he_scores, descending=True)   # higher HE = more prunable

                del_ind = active_indices[sorted_indices[:layer_prune_amount]].tolist()
            else:
                _, sorted_indices = torch.sort(active_he_scores, descending=True)
                del_ind = active_indices[sorted_indices[:layer_prune_amount]].tolist()

            # Set pruning masks
            self.filter_names[self.get_mask_name(active_prune_key)][del_ind] = 0
            self.passive_names[self.get_mask_name(passive_prune_key)][del_ind] = 0

            pruned_so_far += len(del_ind)
            print(f"Layer {active_prune_key}: Pruned {len(del_ind)}/{active_channels}")

        # Apply masks
        self.update_filter_mask()
        print("After update_filter_mask:")
        print(f"Total filter_names items: {sum(mask.sum().item() for mask in self.filter_names.values())}")
        print(f"Total filter_masks items: {sum(mask.sum().item() for mask in self.filter_masks.values())}")

        self.apply_mask()
        self.track_hyperspherical_energy()

        print("After apply_mask:")
        print(f"Total filter_names items: {sum(mask.sum().item() for mask in self.filter_names.values())}")
        print(f"Total filter_masks items: {sum(mask.sum().item() for mask in self.filter_masks.values())}")

        print(f"Total pruned: {pruned_so_far}/{total_to_prune} channels")

    def combined_pruning_score(self, active_prune_key, beta=0.1, prune_amount=None):
        """
        Simple weighted combination of UMM and HE for channel pruning with monitoring metrics.

        Args:
            active_prune_key: The layer key
            beta: Weight for HE contribution (0-1)
            prune_amount: Number of channels to prune (for monitoring purposes)

        Returns:
            Combined pruning score and monitoring metrics
        """
        # Get filter mask and active channels
        name_mask = self.get_mask_name(active_prune_key)
        filter_mask = self.filter_names[name_mask].bool()

        # Get HE scores (higher means more redundant)
        he_scores = self.hyperspherical_channel_energy(
            active_prune_key, model=self.args.he_model, power=self.args.he_power)

        # Get UMM scores (lower means more prunable)
        weight = self.get_module(active_prune_key).weight.data
        umm_scores = weight.abs().mean(dim=(1, 2, 3))

        # Extract scores for active channels
        active_he = he_scores[filter_mask]
        active_umm = umm_scores[filter_mask]

        # Normalize both metrics to [0,1] range
        if len(active_he) > 1:
            he_min, he_max = active_he.min(), active_he.max()
            if he_max > he_min:
                norm_he = (active_he - he_min) / (he_max - he_min)
            else:
                norm_he = torch.zeros_like(active_he)
        else:
            norm_he = torch.zeros_like(active_he)

        if len(active_umm) > 1:
            umm_min, umm_max = active_umm.min(), active_umm.max()
            if umm_max > umm_min:
                # Invert UMM so higher means more prunable
                norm_umm = 1.0 - (active_umm - umm_min) / (umm_max - umm_min)
            else:
                norm_umm = torch.zeros_like(active_umm)
        else:
            norm_umm = torch.zeros_like(active_umm)

        # Simple weighted combination
        combined_active_scores = (1.0 - beta) * norm_umm + beta * norm_he

        # Map back to original size
        combined_scores = torch.zeros_like(he_scores)
        combined_scores[filter_mask] = combined_active_scores

        # Calculate monitoring metrics if prune_amount is provided
        monitoring_metrics = {}
        if prune_amount is not None and prune_amount > 0:
            # Get indices sorted by UMM only (baseline approach)
            _, umm_sorted_indices = torch.sort(norm_umm, descending=True)
            umm_to_prune = umm_sorted_indices[:prune_amount].tolist()

            # Get indices sorted by combined score
            _, combined_sorted_indices = torch.sort(combined_active_scores, descending=True)
            combined_to_prune = combined_sorted_indices[:prune_amount].tolist()

            # 1. Rank correlation (Spearman's rho)
            try:
                import scipy.stats
                rho, _ = scipy.stats.spearmanr(norm_umm.cpu().numpy(), combined_active_scores.cpu().numpy())
                monitoring_metrics['rank_correlation'] = rho
            except:
                # Fallback if scipy not available
                monitoring_metrics['rank_correlation'] = None

            # 2. Calculate overlap between the two pruning decisions
            umm_set = set(umm_to_prune)
            combined_set = set(combined_to_prune)
            overlap = len(umm_set.intersection(combined_set))
            overlap_percentage = 100.0 * overlap / prune_amount if prune_amount > 0 else 100.0

            # 3. Calculate changed decisions
            changed_channels = prune_amount - overlap
            changed_percentage = 100.0 - overlap_percentage

            # Record metrics
            monitoring_metrics['overlap_count'] = overlap
            monitoring_metrics['overlap_percentage'] = overlap_percentage
            monitoring_metrics['changed_count'] = changed_channels
            monitoring_metrics['changed_percentage'] = changed_percentage

            # 4. Calculate average rank change for changed decisions
            rank_changes = []
            for idx in combined_to_prune:
                if idx not in umm_set:
                    # Find where this channel ranked in UMM-only sorting
                    umm_rank = torch.where(umm_sorted_indices == idx)[0].item()
                    rank_change = umm_rank - combined_to_prune.index(idx)
                    rank_changes.append(rank_change)

            if rank_changes:
                monitoring_metrics['avg_rank_improvement'] = sum(rank_changes) / len(rank_changes)
                monitoring_metrics['max_rank_improvement'] = max(rank_changes)
            else:
                monitoring_metrics['avg_rank_improvement'] = 0
                monitoring_metrics['max_rank_improvement'] = 0

        return combined_scores, monitoring_metrics

    # def del_layer(self):
    #     print("===========del layer with HE-influenced pruning===============")
    #
    #     filter_number = self.filter_num()
    #     print(filter_number, '/', self.baseline_filter_num, 'channels')
    #
    #     rate = 1 - self.layer_rate  # target density
    #     total_to_prune = filter_number - self.baseline_filter_num * rate
    #     print(f"Total channels to prune: {total_to_prune}")
    #
    #     if total_to_prune <= 0:
    #         print("No channels to prune")
    #         return
    #
    #     pruned_so_far = 0
    #
    #     # Beta: weight of HE in the pruning decision
    #     beta = self.args.he_beta if hasattr(self.args, 'he_beta') else 0.1
    #
    #     # Global monitoring metrics
    #     global_metrics = {
    #         'total_changed_count': 0,
    #         'total_pruned_count': 0,
    #         'layers_with_changes': 0,
    #         'total_layers_pruned': 0,
    #     }
    #
    #     for active_prune_key in self.module.layer2split:
    #         passive_prune_key, norm_key = self.module.next_layers[active_prune_key]
    #
    #         name_mask = self.get_mask_name(active_prune_key)
    #         filter_mask = self.filter_names[name_mask]
    #         active_channels = filter_mask.sum().item()
    #
    #         layer_prune_amount = int(total_to_prune * (active_channels / filter_number))
    #
    #         # Adjust for remaining pruning budget and minimum layer size
    #         remaining_to_prune = total_to_prune - pruned_so_far
    #         layer_prune_amount = min(layer_prune_amount, remaining_to_prune)
    #         min_size = self.minimum_layer.get(active_prune_key, 1)
    #         max_prune = active_channels - min_size
    #         layer_prune_amount = min(layer_prune_amount, max_prune)
    #
    #         if layer_prune_amount <= 0:
    #             continue
    #
    #         # Get combined scores and monitoring metrics
    #         combined_scores, metrics = self.combined_pruning_score(
    #             active_prune_key, beta, layer_prune_amount)
    #
    #         # Get indices to prune (highest scores = most prunable)
    #         active_indices = torch.where(filter_mask.bool())[0].to(combined_scores.device)
    #         active_combined_scores = combined_scores[filter_mask.bool()]
    #
    #         _, sorted_indices = torch.sort(active_combined_scores, descending=True)
    #         del_ind = active_indices[sorted_indices[:layer_prune_amount]].tolist()
    #
    #         # Set pruning masks
    #         self.filter_names[self.get_mask_name(active_prune_key)][del_ind] = 0
    #         self.passive_names[self.get_mask_name(passive_prune_key)][del_ind] = 0
    #
    #         pruned_so_far += len(del_ind)
    #
    #         # Update global metrics
    #         global_metrics['total_pruned_count'] += layer_prune_amount
    #         global_metrics['total_layers_pruned'] += 1
    #
    #         if metrics.get('changed_count', 0) > 0:
    #             global_metrics['total_changed_count'] += metrics.get('changed_count', 0)
    #             global_metrics['layers_with_changes'] += 1
    #
    #         # Log layer-specific metrics to wandb
    #         layer_name = ".".join([str(i) for i in active_prune_key])
    #         metrics_dict = {
    #             f"layer/{layer_name}/prune_count": layer_prune_amount,
    #             f"layer/{layer_name}/active_channels": active_channels,
    #         }
    #
    #         # Add monitoring metrics
    #         for metric_name, value in metrics.items():
    #             metrics_dict[f"layer/{layer_name}/{metric_name}"] = value
    #
    #         wandb.log(metrics_dict)
    #
    #         # Print summary
    #         change_info = ""
    #         if 'changed_percentage' in metrics:
    #             change_info = f" (Changed: {metrics['changed_percentage']:.1f}% of decisions)"
    #
    #         print(f"Layer {active_prune_key}: Pruned {len(del_ind)}/{active_channels}{change_info}")
    #
    #     # Apply masks
    #     self.update_filter_mask()
    #     self.apply_mask()
    #
    #     # Calculate and log global statistics
    #     if global_metrics['total_pruned_count'] > 0:
    #         global_change_percentage = 100.0 * global_metrics['total_changed_count'] / global_metrics[
    #             'total_pruned_count']
    #     else:
    #         global_change_percentage = 0.0
    #
    #     wandb.log({
    #         "global/pruned_channels": pruned_so_far,
    #         "global/target_to_prune": total_to_prune,
    #         "global/he_beta": beta,
    #         "global/changed_decisions_count": global_metrics['total_changed_count'],
    #         "global/changed_decisions_percentage": global_change_percentage,
    #         "global/layers_with_changes": global_metrics['layers_with_changes'],
    #         "global/total_layers_pruned": global_metrics['total_layers_pruned'],
    #     })
    #
    #     print(f"Total pruned: {pruned_so_far}/{total_to_prune} channels")
    #     print(
    #         f"HE influenced {global_metrics['total_changed_count']} pruning decisions ({global_change_percentage:.1f}%)")
    #     print(
    #         f"HE made changes in {global_metrics['layers_with_changes']}/{global_metrics['total_layers_pruned']} layers")


    '''

    CORE


    '''

    def channel_stastic(self):
        total_variance=0.0
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                if len(weight.size()) == 4:
                    filter_vector = weight.view(weight.size(0)*weight.size(1), -1).sum(dim=1)
                    kernal_sparsity=((filter_vector==0).sum().int().item()/filter_vector.numel())
                    print(f"{name}, kernal sparsity is {(filter_vector==0).sum().int().item()/filter_vector.numel()}")
                    # channel sparsity
                    channel_vector = weight.view(weight.size(0), -1).sum(dim=1)
                    print(f"{name}, filter sparsity is {(channel_vector==0).sum().int().item()/filter_vector.numel()}")
                    
                    
                    print ("-------")
                    # redistribution
                    self.channel_variance[name] = kernal_sparsity

                    if not np.isnan(self.channel_variance[name]):
                        total_variance += self.channel_variance[name]


            for name in self.channel_variance:
                if total_variance != 0.0:
                    self.channel_variance[name] /= total_variance


    def init_growth_prune_and_redist(self):
        if isinstance(self.redistribution_func, str) and self.redistribution_func in redistribution_funcs:
            self.redistribution_func = redistribution_funcs[self.redistribution_func]
        elif isinstance(self.redistribution_func, str):
            print('='*50, 'ERROR', '='*50)
            print('Redistribution mode function not known: {0}.'.format(self.redistribution_func))
            print('Use either a custom redistribution function or one of the pre-defined functions:')
            for key in redistribution_funcs:
                print('\t{0}'.format(key))
            print('='*50, 'ERROR', '='*50)
            raise Exception('Unknown redistribution mode.')




    def gather_statistics(self):
        self.name2nonzeros = {}
        self.name2zeros = {}
        self.name2variance = {}


        self.total_variance = 0.0

        self.total_nonzero = 0
        self.total_zero = 0.0
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                if name in self.filter_masks:
                    mask = self.masks[name][self.filter_masks[name].bool()]
                else:
                    mask = self.masks[name]

                        
                self.name2nonzeros[name] = mask.sum().item()
                self.name2zeros[name] = mask.numel() - self.name2nonzeros[name]

                sparsity = self.name2zeros[name]/float(self.masks[name].numel())

                self.total_nonzero += self.name2nonzeros[name]
                self.total_zero += self.name2zeros[name]






    def step(self):
        self.steps += 1  
        self.optimizer.step()
 
        if self.args.sparse:
            self.apply_mask()

            # mest decay
            self.death_rate_decay.step()
            if self.args.decay_schedule == 'cosine':
                self.death_rate = self.death_rate_decay.get_dr()
            elif self.args.decay_schedule == 'constant':
                self.death_rate = self.args.death_rate


            # filter_dst_decay

            if self.args.gpm_filter_pune:
                self.layer_rate= self.gradual_pruning_rate(self.steps, 0.0, self.args.start_layer_rate, self.initial_prune_time, self.final_prune_time)
            else:
                self.layer_rate_decay.step()
                self.layer_rate= self.layer_rate_decay.get_dr()


            # filter_dst_decay
            self.dst_decay.step()
            self.dst_rate= self.dst_decay.get_dr()




            if self.args.filter_dst:
                if  self.steps >= self.initial_prune_time and self.steps < self.final_prune_time :
                    if (self.steps+ (1000/2)) % self.args.layer_interval== 0 :
                

                        print ("current layer rate",self.layer_rate)
                        
                        print ('===========del layer===============')
                        self.del_layer(selective=self.args.he_selective)
                        # self.del_layer()

                        print ('===========done ===============')

                    
                        if "global" not in self.args.growth:
                            self.update_erk_dic()



            if self.args.mest:
                if self.steps< len(self.train_loader)*(self.args.epochs-self.args.stop_dst_epochs):
                    if self.prune_every_k_steps is not None:
                        if (self.steps % self.prune_every_k_steps == 0):

                            print ("current mest  death_rate", self.death_rate)

                        
                            self.truncate_weights_prune(self.death_rate)
                            self.print_nonzero_counts()

                            self.truncate_weights_grow(self.death_rate)
                            self.print_nonzero_counts()

                        


                            


                elif self.args.mest_dst:
                    if self.prune_every_k_steps is not None:
                        if (self.steps % self.prune_every_k_steps == 0):

                            print ("current dst rate",self.dst_rate)

                            self.truncate_weights(self.dst_rate)
     

            elif self.args.dst:

                if self.prune_every_k_steps is not None:
                    if (self.steps % self.prune_every_k_steps == 0):

                        print ("current dst rate",self.dst_rate)

                        self.truncate_weights(self.dst_rate)
                        self.print_nonzero_counts()
                        

                    

    def update_erk_dic(self):
        erk_power_scale=1.0
        print('retriving sparsity dic by fixed_ERK')

        total_params = 0
        for name, weight in self.masks.items():
            if name in self.filter_masks:
                weight=weight[self.filter_masks[name].bool()]

            total_params  += weight.numel()



        print ("baseline_nonzero",self.baseline_nonzero)
        print ("total_params",total_params)


        density=self.baseline_nonzero/total_params




        ### temp mask
        self.temp_mask=copy.deepcopy(self.masks)

        for name, weight in self.masks.items():

            if name in self.filter_names:
                self.temp_mask[name]=self.temp_mask[name][self.filter_names[name].bool()]


        for name, weight in self.masks.items():
    
            if name in self.passive_names:


                temp=self.temp_mask[name]

                    # transpose
                temp =temp.transpose(0, 1)
                temp=temp[self.passive_names[name].bool()]
                temp.transpose_(0, 1)

                self.temp_mask[name]=temp




        is_epsilon_valid = False
        # # The following loop will terminate worst case when all masks are in the
        # custom_sparsity_map. This should probably never happen though, since once
        # we have a single variable or more with the same constant, we have a valid
        # epsilon. Note that for each iteration we add at least one variable to the
        # custom_sparsity_map and therefore this while loop should terminate.
        dense_layers = set()
        while not is_epsilon_valid:
            # We will start with all layers and try to find right epsilon. However if
            # any probablity exceeds 1, we will make that layer dense and repeat the
            # process (finding epsilon) with the non-dense layers.
            # We want the total number of connections to be the same. Let say we have
            # for layers with N_1, ..., N_4 parameters each. Let say after some
            # iterations probability of some dense layers (3, 4) exceeded 1 and
            # therefore we added them to the dense_layers set. Those layers will not
            # scale with erdos_renyi, however we need to count them so that target
            # paratemeter count is achieved. See below.
            # eps * (p_1 * N_1 + p_2 * N_2) + (N_3 + N_4) =
            #    (1 - default_sparsity) * (N_1 + N_2 + N_3 + N_4)
            # eps * (p_1 * N_1 + p_2 * N_2) =
            #    (1 - default_sparsity) * (N_1 + N_2) - default_sparsity * (N_3 + N_4)
            # eps = rhs / (\sum_i p_i * N_i) = rhs / divisor.

            divisor = 0
            rhs = 0
            raw_probabilities = {}
            for name, mask in self.temp_mask.items():
                n_param = np.prod(mask.shape)

                n_zeros = n_param * (1 - density)
                n_ones = n_param * density
                


                if name in dense_layers:
                    # See `- default_sparsity * (N_3 + N_4)` part of the equation above.
                    rhs -= n_zeros

                else:
                    # Corresponds to `(1 - default_sparsity) * (N_1 + N_2)` part of the
                    # equation above.
                    rhs += n_ones
                    # Erdos-Renyi probability: epsilon * (n_in + n_out / n_in * n_out).
                    raw_probabilities[name] = (
                                                        np.sum(mask.shape) / np.prod(mask.shape)
                                                ) ** erk_power_scale
                    # Note that raw_probabilities[mask] * n_param gives the individual
                    # elements of the divisor.
                    divisor += raw_probabilities[name] * n_param
            # By multipliying individual probabilites with epsilon, we should get the
            # number of parameters per layer correctly.
            epsilon = rhs / divisor
            # If epsilon * raw_probabilities[mask.name] > 1. We set the sparsities of that
            # mask to 0., so they become part of dense_layers sets.
            max_prob = np.max(list(raw_probabilities.values()))
            max_prob_one = max_prob * epsilon
            if max_prob_one > 1:
                is_epsilon_valid = False
                for mask_name, mask_raw_prob in raw_probabilities.items():
                    if mask_raw_prob == max_prob:
                        print(f"Sparsity of var:{mask_name} had to be set to 0.")
                        dense_layers.add(mask_name)
            else:
                is_epsilon_valid = True

       
        self.density_dict = {}
        total_nonzero = 0.0
        # With the valid epsilon, we can set sparsities of the remaning layers.
        for name, mask in self.temp_mask.items():
            n_param = np.prod(mask.shape)
            if name in dense_layers:
                self.density_dict[name] = 1.0
            else:
                probability_one = epsilon * raw_probabilities[name]
                self.density_dict[name] = probability_one
            print(
                f"layer: {name}, shape: {mask.shape}, density: {self.density_dict[name]}"
            )

            total_nonzero += self.density_dict[name] * mask.numel()

        print(f"Overall sparsity {total_nonzero / total_params}",total_nonzero,total_params)


    def init(self, mode='ER', density=0.05,erk_power_scale=1.0):

        ## init for layer
        for index in self.module.layer2split:
            self.baseline_filter_num+=self.get_module(index).weight.shape[0]
            
        print ("baseline fitler num",self.baseline_filter_num)           


        self.bound_layer={}
        for index in self.module.layer2split:
            self.bound_layer[index] = int(self.get_module(index).out_channels)

        self.minimum_layer={}
        for index in self.module.layer2split:
            self.minimum_layer[index] = int(self.get_module(index).out_channels*(1-self.args.start_layer_rate)*self.args.minumum_ratio)

        




        ## init for connection
        self.density = density
        if mode == 'GMP':
            self.baseline_nonzero = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    self.masks[name] = torch.ones_like(weight, dtype=torch.float32, requires_grad=False).cuda()
            self.apply_mask()

    

        elif mode == 'snip':
            print('initialize by snip')
            layer_wise_sparsities = SNIP(self.module, self.density, self.train_loader, self.device)
            # re-sample mask positions
            for sparsity_, name in zip(layer_wise_sparsities, self.masks):
                self.masks[name][:] = (torch.rand(self.masks[name].shape) < (1-sparsity_)).float().data.cuda()

        elif mode == 'GraSP':
            print('initialize by GraSP')
            layer_wise_sparsities = GraSP(self.module, self.density, self.train_loader, self.device)
            # re-sample mask positions
            for sparsity_, name in zip(layer_wise_sparsities, self.masks):
                self.masks[name][:] = (torch.rand(self.masks[name].shape) < (1-sparsity_)).float().data.cuda()



        elif mode == 'pruning':
            print('initialize by pruning')
            weight_abs = []
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    weight_abs.append(torch.abs(weight))

            # Gather all scores in a single vector and normalise
            all_scores = torch.cat([torch.flatten(x) for x in weight_abs])
            num_params_to_keep = int(len(all_scores) * (1 - self.PF_rate))

            threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
            acceptable_score = threshold[-1]

            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    self.masks[name] = ((torch.abs(weight)) >= acceptable_score).float()

        elif mode == 'resume':
            # Initializes the mask according to the weights
            # which are currently zero-valued. This is required
            # if you want to resume a sparse model but did not
            # save the mask.
            print('initialize by resume')
            # self.baseline_nonzero = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    print((weight != 0.0).sum().item()/weight.numel())
                    self.masks[name] = (weight != 0.0).float().data.cuda()
                    # self.baseline_nonzero += weight.numel()*density
            self.apply_mask()

            # for module in self.modules:
            #     for name, weight in module.named_parameters():
            #         if name not in self.masks: continue
            #         print(f"The sparsity of layer {name} is {(self.masks[name]==0).sum()/self.masks[name].numel()}")

        elif mode == 'uniform':
            index = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    name_cur = name + '_' + str(index)
                    index += 1
                    if name_cur not in self.masks: continue
                    self.masks[name_cur][:] = (torch.rand(weight.shape) < density).float().data.cuda() #lsw
                    # self.masks[name_cur][:] = (torch.rand(weight.shape) < density).float().data #lsw
            self.apply_mask()

        elif mode == 'fixed_ERK':
            print('initialize by fixed_ERK')

 

            total_params = 0
            for name, weight in self.masks.items():
                total_params += weight.numel()

            self.total_params=total_params
            is_epsilon_valid = False
            # # The following loop will terminate worst case when all masks are in the
            # custom_sparsity_map. This should probably never happen though, since once
            # we have a single variable or more with the same constant, we have a valid
            # epsilon. Note that for each iteration we add at least one variable to the
            # custom_sparsity_map and therefore this while loop should terminate.
            dense_layers = set()
            while not is_epsilon_valid:
                # We will start with all layers and try to find right epsilon. However if
                # any probablity exceeds 1, we will make that layer dense and repeat the
                # process (finding epsilon) with the non-dense layers.
                # We want the total number of connections to be the same. Let say we have
                # for layers with N_1, ..., N_4 parameters each. Let say after some
                # iterations probability of some dense layers (3, 4) exceeded 1 and
                # therefore we added them to the dense_layers set. Those layers will not
                # scale with erdos_renyi, however we need to count them so that target
                # paratemeter count is achieved. See below.
                # eps * (p_1 * N_1 + p_2 * N_2) + (N_3 + N_4) =
                #    (1 - default_sparsity) * (N_1 + N_2 + N_3 + N_4)
                # eps * (p_1 * N_1 + p_2 * N_2) =
                #    (1 - default_sparsity) * (N_1 + N_2) - default_sparsity * (N_3 + N_4)
                # eps = rhs / (\sum_i p_i * N_i) = rhs / divisor.

                divisor = 0
                rhs = 0
                raw_probabilities = {}
                for name, mask in self.masks.items():
                    n_param = np.prod(mask.shape)
                    n_zeros = n_param * (1 - density)
                    n_ones = n_param * density

                    self.n_ones=n_ones

                    if name in dense_layers:
                        # See `- default_sparsity * (N_3 + N_4)` part of the equation above.
                        rhs -= n_zeros

                    else:
                        # Corresponds to `(1 - default_sparsity) * (N_1 + N_2)` part of the
                        # equation above.
                        rhs += n_ones
                        # Erdos-Renyi probability: epsilon * (n_in + n_out / n_in * n_out).
                        raw_probabilities[name] = (
                                                          np.sum(mask.shape) / np.prod(mask.shape)
                                                  ) ** erk_power_scale
                        # Note that raw_probabilities[mask] * n_param gives the individual
                        # elements of the divisor.
                        divisor += raw_probabilities[name] * n_param
                # By multipliying individual probabilites with epsilon, we should get the
                # number of parameters per layer correctly.
                epsilon = rhs / divisor
                # If epsilon * raw_probabilities[mask.name] > 1. We set the sparsities of that
                # mask to 0., so they become part of dense_layers sets.
                max_prob = np.max(list(raw_probabilities.values()))
                max_prob_one = max_prob * epsilon
                if max_prob_one > 1:
                    is_epsilon_valid = False
                    for mask_name, mask_raw_prob in raw_probabilities.items():
                        if mask_raw_prob == max_prob:
                            print(f"Sparsity of var:{mask_name} had to be set to 0.")
                            dense_layers.add(mask_name)
                else:
                    is_epsilon_valid = True

            self.density_dict = {}
            total_nonzero = 0.0
            # With the valid epsilon, we can set sparsities of the remaning layers.
            for name, mask in self.masks.items():

                n_param = np.prod(mask.shape)
                if name in dense_layers:
                    self.density_dict[name] = 1.0
                else:
                    probability_one = epsilon * raw_probabilities[name]
                    self.density_dict[name] = probability_one
                print(
                    f"layer: {name}, shape: {mask.shape}, density: {self.density_dict[name]}"
                )
                self.masks[name][:] = (torch.rand(mask.shape) < self.density_dict[name]).float().data.cuda()

                total_nonzero += self.density_dict[name] * mask.numel()
            self.baseline_nonzero=total_nonzero
            print(f"Overall sparsity {total_nonzero / total_params}")
            self.temp_mask=copy.deepcopy(self.masks)


        elif mode == 'ER':
            print('initialize by SET')
            # initialization used in sparse evolutionary training
            total_params = 0
            index = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    name_cur = name + '_' + str(index)
                    index += 1
                    if name_cur not in self.masks: continue
                    total_params += weight.numel()

            target_params = total_params *density
            tolerance = 5
            current_params = 0
            new_nonzeros = 0
            epsilon = 10.0
            growth_factor = 0.5
            # searching for the right epsilon for a specific sparsity level
            while not ((current_params+tolerance > target_params) and (current_params-tolerance < target_params)):
                new_nonzeros = 0.0
                index = 0
                for name, weight in module.named_parameters():
                    name_cur = name + '_' + str(index)
                    index += 1
                    if name_cur not in self.masks: continue
                    # original SET formulation for fully connected weights: num_weights = epsilon * (noRows + noCols)
                    # we adapt the same formula for convolutional weights
                    growth =  epsilon*sum(weight.shape)
                    new_nonzeros += growth
                current_params = new_nonzeros
                if current_params > target_params:
                    epsilon *= 1.0 - growth_factor
                else:
                    epsilon *= 1.0 + growth_factor
                growth_factor *= 0.95

            index = 0
            for name, weight in module.named_parameters():
                name_cur = name + '_' + str(index)
                index += 1
                if name_cur not in self.masks: continue
                growth =  epsilon*sum(weight.shape)
                prob = growth/np.prod(weight.shape)
                self.masks[name_cur][:] = (torch.rand(weight.shape) < prob).float().data.cuda() #lsw
                # self.masks[name_cur][:] = (torch.rand(weight.shape) < prob).float().data

        self.apply_mask()

        total_size = 0
        for name, weight in self.masks.items():
            if name in self.filter_masks:
                weight=weight[self.filter_masks[name].bool()]
            print (name,weight.numel())

            
            total_size  += weight.numel()
        

        sparse_size = 0
        for name, weight in self.masks.items():
            if name in self.filter_masks:
                weight=weight[self.filter_masks[name].bool()]
            sparse_size += (weight != 0).sum().int().item()


        print('Total Model parameters after init:', sparse_size, total_size)
        print('Total parameters under sparsity level of {0}: {1}'.format(density, sparse_size / total_size))



    def add_module(self, module, density, sparse_init='ER'):
        self.module = module
        print (module)
        self.sparse_init = sparse_init
        self.modules.append(module)
        print ("add module")

        self.get_filter_name()

        for name, tensor in module.named_parameters():
            self.names.append(name)
            self.masks[name] = torch.zeros_like(tensor, dtype=torch.float32, requires_grad=False).cuda()


        self.filter_masks={}
        for name, tensor in module.named_parameters():
            if name in self.filter_names:
                self.filter_masks[name]=torch.ones_like(tensor, dtype=torch.float32, requires_grad=False).cuda()
                self.filter_masks[name][~self.filter_names[name].bool()]=0

                # print (name,  self.filter_masks[name].sum())


        for name, tensor in module.named_parameters():
            if name in self.passive_names:
                if name in self.filter_names:
                    filter_mask=self.filter_masks[name]
                else:
                    filter_mask=torch.ones_like(tensor, dtype=torch.float32, requires_grad=False).cuda()

                # print (filter_mask.shape)
                filter_mask =filter_mask.transpose(0, 1)
                filter_mask[~self.passive_names[name].bool()]=0
                filter_mask.transpose_(0, 1)
                self.filter_masks[name]=filter_mask

                # print (name,  self.filter_masks[name].sum())
        print('Removing biases...')
        self.remove_weight_partial_name('bias')
        print('Removing 2D batch norms...')
        self.remove_type(nn.BatchNorm2d)
        print('Removing 1D batch norms...')
        self.remove_type(nn.BatchNorm1d)
        self.init(mode=sparse_init, density=density)







    def remove_weight(self, name):
        if name in self.masks:
            print('Removing {0} of size {1} = {2} parameters.'.format(name, self.masks[name].shape, self.masks[name].numel()))
            self.masks.pop(name)
        elif name + '.weight' in self.masks:
            print('Removing {0} of size {1} = {2} parameters.'.format(name, self.masks[name + '.weight'].shape, self.masks[name + '.weight'].numel()))
            self.masks.pop(name + '.weight')
        else:
            print('ERROR', name)

    def remove_weight_partial_name(self, partial_name):
        removed = set()
        for name in list(self.masks.keys()):
            if partial_name in name:

                # print('Removing {0} of size {1} with {2} parameters...'.format(name, self.masks[name].shape,
                #                                                                    np.prod(self.masks[name].shape)))
                removed.add(name)
                self.masks.pop(name)

        print('Removed {0} layers.'.format(len(removed)))

        i = 0
        while i < len(self.names):
            name = self.names[i]
            if name in removed:
                self.names.pop(i)
            else:
                i += 1

    def remove_type(self, nn_type):
        for module in self.modules:
            for name, module in module.named_modules():
                if isinstance(module, nn_type):
                    self.remove_weight(name)

    def apply_mask(self):
        # print ("fusing masks")
        for name, mask in self.masks.items():
            
            if name in self.filter_masks.keys():
                self.masks[name]= torch.logical_and(self.filter_masks[name], self.masks [name]).float()

        for module in self.modules:
            for name, tensor in module.named_parameters():
                if "bias" in name:
                    weight_name=name[:-4]+"weight"
                    if weight_name in self.filter_names:
                        tensor.data = tensor.data*self.filter_names[weight_name].float().cuda()

        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name in self.masks:
                    tensor.data = tensor.data*self.masks[name]
                    if 'momentum_buffer' in self.optimizer.state[tensor]:
                        self.optimizer.state[tensor]['momentum_buffer'] = self.optimizer.state[tensor]['momentum_buffer']*self.masks[name]




    '''
                   DST
    '''






    def truncate_weights_GMP_global(self, epoch):
        '''
        Implementation  of global pruning version of GMP To prune, or not to prune: exploring the efficacy of pruning for model compression https://arxiv.org/abs/1710.01878
        :param epoch: current training epoch
        :return:
        '''
        prune_rate = 1 - self.density
        curr_prune_epoch = epoch
        total_prune_epochs = self.args.multiplier * self.args.final_prune_epoch - self.args.multiplier * self.args.init_prune_epoch + 1
        if epoch >= self.args.multiplier * self.args.init_prune_epoch and epoch <= self.args.multiplier * self.args.final_prune_epoch:
            prune_decay = (1 - ((
                                            curr_prune_epoch - self.args.multiplier * self.args.init_prune_epoch) / total_prune_epochs)) ** 3
            curr_prune_rate = prune_rate - (prune_rate * prune_decay)
            weight_abs = []
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    weight_abs.append(torch.abs(weight))

            # Gather all scores in a single vector and normalise
            all_scores = torch.cat([torch.flatten(x) for x in weight_abs])
            num_params_to_keep = int(len(all_scores) * (1-curr_prune_rate))

            threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
            acceptable_score = threshold[-1]

            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    self.masks[name] = ((torch.abs(weight)) >= acceptable_score).float()
            self.apply_mask()

        total_size = 0
        for name, weight in self.masks.items():
            total_size += weight.numel()
        print('Total Model parameters:', total_size)

        sparse_size = 0
        for name, weight in self.masks.items():
            sparse_size += (weight != 0).sum().int().item()

        print('Total parameters under sparsity level of {0}: {1} after epoch of {2}'.format(self.density,
                                                                                            sparse_size / total_size,
                                                                                            epoch))
    def truncate_weights_GMP(self, epoch):
        '''
        Implementation  of GMP To prune, or not to prune: exploring the efficacy of pruning for model compression https://arxiv.org/abs/1710.01878
        :param epoch: current training epoch
        :return:
        '''
        prune_rate = 1 - self.density
        curr_prune_epoch = epoch
        total_prune_epochs = self.args.multiplier * self.args.final_prune_epoch - self.args.multiplier * self.args.init_prune_epoch + 1
        if epoch >= self.args.multiplier * self.args.init_prune_epoch and epoch <= self.args.multiplier * self.args.final_prune_epoch:
            prune_decay = (1 - ((curr_prune_epoch - self.args.multiplier * self.args.init_prune_epoch) / total_prune_epochs)) ** 3
            curr_prune_rate = prune_rate - (prune_rate * prune_decay)

            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue

                    x, idx = torch.sort(torch.abs(weight.data.view(-1)))
                    p = int(curr_prune_rate * weight.numel())
                    self.masks[name].data.view(-1)[idx[:p]] = 0.0
            self.apply_mask()
        total_size = 0
        for name, weight in self.masks.items():
            total_size += weight.numel()
        print('Total Model parameters:', total_size)

        sparse_size = 0
        for name, weight in self.masks.items():
            sparse_size += (weight != 0).sum().int().item()

        print('Total parameters under sparsity level of {0}: {1} after epoch of {2}'.format(self.density, sparse_size / total_size, epoch))



    def truncate_weights(self, pruning_rate):
        print ("\n")
        print('dynamic sparse change')

        self.gather_statistics()




        #################################prune weights#############################
        if self.death_mode == 'global_magnitude':
            to_kill = math.ceil(pruning_rate*self.total_nonzero)
            self.total_removed=self.global_magnitude_death(to_kill)


        else:
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    mask = self.masks[name]

                    # death
                    if self.death_mode == 'magnitude':
                        new_mask = self.magnitude_death(mask, weight, name, pruning_rate)
                    elif self.death_mode == 'SET':
                        new_mask = self.magnitude_and_negativity_death(mask, weight, name)
                    elif self.death_mode == 'Taylor_FO':
                        new_mask = self.taylor_FO(mask, weight, name)
                    elif self.death_mode == 'threshold':
                        new_mask = self.threshold_death(mask, weight, name)

                    # if self.args.fix_num_operation:
                    #     # print ("fix_num_operation")
                    #     self.pruning_rate[name] = 
                    # else:
                    #     self.pruning_rate[name] = int(self.masks[name].sum().item() - new_mask.sum().item())
                    
                    
                    # self.pruning_rate[name] = int(self.density_dict[name]*self.masks[name].numel()- new_mask.sum().item())
                    # self.pruning_rate[name] = int(self.masks[name].sum().item() - new_mask.sum().item())
                    # print ( name, int(self.density_dict[name]*self.masks[name].numel()), int(self.masks[name].sum().item()))
                    self.masks[name][:] = new_mask

                    togrow= int(self.density_dict[name]*self.temp_mask[name].numel()- new_mask.sum().item())
                    
                    self.pruning_rate[name] =togrow

        self.apply_mask()

        self.print_nonzero_counts()

       #################################grow weights#############################


        self.gather_statistics()
        if self.growth_mode == 'global_gradients':
            print ("self.baseline_nonzero-self.total_nonzero",self.baseline_nonzero-self.total_nonzero)
            total_nonzero_new=self.global_gradient_growth(self.baseline_nonzero-self.total_nonzero)
            print ("total_nonzero_new",total_nonzero_new)
            print ("self.total_removed",self.total_removed)


        

        else:
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue
                    new_mask = self.masks[name].data.byte()
                    
                    # self.pruning_rate[name] = int(self.density_dict[name]*self.temp_mask[name].numel()- new_mask.sum().item())
                    

                    to_grow= int(self.pruning_rate[name])


                    if (to_grow +  int(new_mask.sum().item()))  > int( (self.temp_mask[name].numel())):

                        to_grow= int( (self.temp_mask[name].numel())) -  int(new_mask.sum().item())

                        
                        print ("layer_wise dst no room to grow in each layer in",name )
                        self.pruning_rate[name]=to_grow


                    if to_grow>1:
                        # growth
                        if self.growth_mode == 'random':
                            new_mask = self.random_growth(name, new_mask, self.pruning_rate[name], weight)

                        elif self.growth_mode == 'momentum':
                            new_mask = self.momentum_growth(name, new_mask, self.pruning_rate[name], weight)

                        elif self.growth_mode == 'gradients':
                            new_mask = self.gradient_growth(name, new_mask, self.pruning_rate[name], weight)

                        elif self.growth_mode == 'momentum_neuron':
                            new_mask = self.momentum_neuron_growth(name, new_mask,  self.pruning_rate[name], weight)
                        # exchanging masks
                        self.masks.pop(name)
                        self.masks[name] = new_mask.float()

                    else:
                        print ("layer_wise dst no room to grow in layer")

        self.apply_mask()


        self.print_nonzero_counts()


    def truncate_weights_prune(self, pruning_rate):
        print ("\n")
        print('dynamic sparse change prune')

        self.gather_statistics()




        #################################prune weights#############################
        tokill=self.total_nonzero-self.baseline_nonzero
        print ("to kill", tokill,"expect", self.baseline_nonzero)
        if tokill>0:

            if self.death_mode == 'global_magnitude':
                self.total_removed=self.global_magnitude_death(tokill)

            else:

                pruning_rate=tokill/self.total_nonzero
                print ("calulate prune ratio",pruning_rate)


                for module in self.modules:
                    for name, weight in module.named_parameters():
                        if name not in self.masks: continue
                        mask = self.masks[name]

                        # death
                        if self.death_mode == 'magnitude':
                            new_mask = self.magnitude_death(mask, weight, name, pruning_rate)
                        elif self.death_mode == 'SET':
                            new_mask = self.magnitude_and_negativity_death(mask, weight, name)
                        elif self.death_mode == 'Taylor_FO':
                            new_mask = self.taylor_FO(mask, weight, name)
                        elif self.death_mode == 'threshold':
                            new_mask = self.threshold_death(mask, weight, name)

      
                        self.masks[name][:] = new_mask
                        # self.pruning_rate[name] = int(self.name2nonzeros[name] - new_mask.sum().item())


            self.apply_mask()


    def truncate_weights_grow(self, pruning_rate):
       #################################grow weights#############################


        self.gather_statistics()
        print('dynamic sparse change grow')

        togrow=self.total_params*pruning_rate-self.total_nonzero
        print ("self.total_params*pruning_rate",self.total_params*pruning_rate)
        print ("self.total_nonzero",self.total_nonzero)
        print ("to grow",togrow)




        if togrow>0:
            if self.growth_mode == 'global_gradients':
    
                total_nonzero_new=self.global_gradient_growth(togrow)
                print ("total_nonzero_new",total_nonzero_new)



            else:


                real_d_num=0
                for name, mask in self.masks.items():
                    if self.density_dict[name]==1.0:
                        d_layernum=self.masks[name].sum().item()
                        real_d_num+=d_layernum

                print ("real_d_num",real_d_num)
                d_num=0
                for name, mask in self.temp_mask.items():
                    if self.density_dict[name]==1.0:
                        d_layernum=mask.numel()
                        d_num+=d_layernum

                print ("d_num",d_num)



                expect=0
                for name, mask in self.masks.items():
                    if self.density_dict[name]!=1.0:
                        new_mask = self.masks[name].data.byte()
                        layer_e=int((new_mask.sum().item()))
                        expect+=layer_e

                print ("expect dst",expect)

                print ("total_nonzero",self.total_nonzero)
                print ("baseline_nonzero",self.baseline_nonzero )
                grow_ratio=(togrow-(d_num-real_d_num))/(self.total_nonzero-real_d_num)

                print ("calulate extra grow ratio",grow_ratio)

                if grow_ratio>0:
                    for module in self.modules:
                        for name, weight in module.named_parameters():
                            if name not in self.masks: continue
                            new_mask = self.masks[name].data.byte()

                            to_grow= int((new_mask.sum().item())*grow_ratio)

                            if (to_grow +  int(new_mask.sum().item()))  > int( (self.temp_mask[name].numel())):

                                to_grow= int( (self.temp_mask[name].numel())) -  int(new_mask.sum().item())
                                print ("to_grow",to_grow)
                                print ( "int( (self.masks[name].numel()))",int(self.temp_mask[name].numel()))
                                print ("(to_grow +  int(new_mask.sum().item()))",(to_grow +  int(new_mask.sum().item())) )

                                print ("layer_wise dst no room to grow in each layer in",name , (to_grow +  int(new_mask.sum().item())) - int(self.density_dict[name]* (self.temp_mask[name].numel())) )

                            self.pruning_rate[name] =to_grow

                            if to_grow>1:
                                # growth
                                if self.growth_mode == 'random':
                                    new_mask = self.random_growth(name, new_mask, self.pruning_rate[name], weight)

                                elif self.growth_mode == 'momentum':
                                    new_mask = self.momentum_growth(name, new_mask, self.pruning_rate[name], weight)

                                elif self.growth_mode == 'gradients':
                                    new_mask = self.gradient_growth(name, new_mask, self.pruning_rate[name], weight)

                                elif self.growth_mode == 'momentum_neuron':
                                    new_mask = self.momentum_neuron_growth(name, new_mask,  self.pruning_rate[name], weight)
                                # exchanging masks
                                self.masks.pop(name)
                                self.masks[name] = new_mask.float()
                            print ("to grow smaller than one, so skip")

       


                else:
                    print ("layer_wise dst no room to grow")

        self.apply_mask()









    def pruning(self):
        print('pruning...')
        print('death rate:', self.args.density)
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
                num_remove = math.ceil((1-self.args.density) * weight.numel())
                x, idx = torch.sort(torch.abs(weight.data.view(-1)))
                self.masks[name].data.view(-1)[idx[:num_remove]] = 0.0
        self.apply_mask()
        total_size = 0
        for name, weight in self.masks.items():
            total_size  += weight.numel()
        print('Total Model parameters:', total_size)

        sparse_size = 0
        for name, weight in self.masks.items():
            sparse_size += (weight != 0).sum().int().item()

        print('Total parameters under sparsity level of {0}: {1}'.format(self.args.density, sparse_size / total_size))




    '''
                    DEATH
    '''

    def threshold_death(self, mask, weight, name):
        return (torch.abs(weight.data) > self.threshold)

    def taylor_FO(self, mask, weight, name):

        num_remove = math.ceil(self.name2death_rate[name] * self.name2nonzeros[name])
        num_zeros = self.name2zeros[name]
        k = math.ceil(num_zeros + num_remove)

        x, idx = torch.sort((weight.data * weight.grad).pow(2).flatten())
        mask.data.view(-1)[idx[:k]] = 0.0

        return mask

    def magnitude_death(self, mask, weight, name, pruning_rate):

        num_zeros = (mask == 0).sum().item()
        num_remove = math.ceil(pruning_rate * (mask.sum().item()))
        if num_remove == 0.0: return weight.data != 0.0
        # num_remove = math.ceil(self.name2death_rate[name]*self.name2nonzeros[name])

        x, idx = torch.sort(torch.abs(weight.data.reshape(-1)))

        k = math.ceil(num_zeros + num_remove)
        threshold = x[k - 1].item()

        return (torch.abs(weight.data) > threshold)






    def global_magnitude_death(self,tokill):

        
        ### prune
        tokill=tokill+self.total_zero
        tokill=int(tokill)
        weight_abs = []

        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue

                if name in self.filter_masks:
                    remain = (torch.abs(weight.data))[self.filter_masks[name].bool()] 
                else:
                    remain = torch.abs(weight.data) 

                weight_abs.append(remain)


   
        # Gather all scores in a single vector and normalise
        all_scores = torch.cat([torch.flatten(x) for x in weight_abs])

        threshold, _ = torch.topk(all_scores, tokill,largest=False, sorted=True)


        acceptable_score = threshold[-1]



        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue
  
                new_mask = torch.abs(weight.data) >=acceptable_score
                self.masks[name][:] = new_mask

                if name in self.filter_masks.keys():
                    self.masks[name]= torch.logical_and(self.filter_masks[name], self.masks [name]).float()
                
        return None

    def global_gradient_growth(self, total_regrowth):
        

        togrow = total_regrowth

        togrow=int(togrow)

        ### prune

        weight_abs = []

        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue

                grad = self.get_gradient_for_weights(weight)

                if name in self.filter_masks:
                    remain = (torch.abs(grad * (self.masks[name]==0)))  [self.filter_masks[name].bool()] 
                else:
                    remain = torch.abs(grad *(self.masks[name]==0) )

                weight_abs.append(remain)



        # Gather all scores in a single vector and normalise
        all_scores = torch.cat([torch.flatten(x) for x in weight_abs])

        print ("len(all_scores)",len(all_scores),all_scores.bool().sum().item())
        if togrow>all_scores.bool().sum().item():
            print (togrow, all_scores.bool().sum().item())
            togrow=all_scores.bool().sum().item()
            print ("already full=====================")
        if togrow>0:
            threshold, _ = torch.topk(all_scores, togrow,largest=True, sorted=True)
            acceptable_score = threshold[-1]



            increse=0
            before_mask=0
            after_mask=0

            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue


                    new_mask = self.masks[name]


                    grad = self.get_gradient_for_weights(weight)

                    grad = grad*(new_mask==0).float()
                    if  name in self.filter_masks.keys():                 
                        grad= grad*self.filter_masks[name].float() 

                    increse+=(torch.abs(grad.data) >=acceptable_score).float().sum().item()
                    
                    self.masks[name][:] = (new_mask.byte() | (torch.abs(grad.data) > acceptable_score)).float()

                    before_mask+=self.masks[name].sum().item()

                    if name in self.filter_masks.keys():
                        self.masks[name]= torch.logical_and(self.filter_masks[name], self.masks [name]).float()
                    
                    after_mask+= self.masks[name].sum().item()

            print ("increse", increse,"before_mask",before_mask,"after_mask",after_mask)

        else:
            print ("no room to grow")

        return None

    def global_momentum_growth(self, total_regrowth):
        togrow = total_regrowth
        total_grown = 0
        last_grown = 0
        while total_grown < togrow*(1.0-self.tolerance) or (total_grown > togrow*(1.0+self.tolerance)):
            total_grown = 0
            total_possible = 0
            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks: continue

                    new_mask = self.masks[name]
                    grad = self.get_momentum_for_weight(weight)
                    grad = grad*(new_mask==0).float()
                    possible = (grad !=0.0).sum().item()
                    total_possible += possible
                    grown = (torch.abs(grad.data) > self.growth_threshold).sum().item()
                    total_grown += grown
            print(total_grown, self.growth_threshold, togrow, self.growth_increment, total_possible)
            if total_grown == last_grown: break
            last_grown = total_grown


            if total_grown > togrow*(1.0+self.tolerance):
                self.growth_threshold *= 1.02
                #self.growth_increment *= 0.95
            elif total_grown < togrow*(1.0-self.tolerance):
                self.growth_threshold *= 0.98
                #self.growth_increment *= 0.95

        total_new_nonzeros = 0
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue

                new_mask = self.masks[name]
                grad = self.get_momentum_for_weight(weight)
                grad = grad*(new_mask==0).float()
                self.masks[name][:] = (new_mask.byte() | (torch.abs(grad.data) > self.growth_threshold)).float()
                total_new_nonzeros += new_mask.sum().item()
        return total_new_nonzeros


    def magnitude_and_negativity_death(self, mask, weight, name):
        num_remove = math.ceil(self.name2death_rate[name]*self.name2nonzeros[name])
        num_zeros = self.name2zeros[name]

        # find magnitude threshold
        # remove all weights which absolute value is smaller than threshold
        x, idx = torch.sort(weight[weight > 0.0].data.view(-1))
        k = math.ceil(num_remove/2.0)
        if k >= x.shape[0]:
            k = x.shape[0]

        threshold_magnitude = x[k-1].item()

        # find negativity threshold
        # remove all weights which are smaller than threshold
        x, idx = torch.sort(weight[weight < 0.0].view(-1))
        k = math.ceil(num_remove/2.0)
        if k >= x.shape[0]:
            k = x.shape[0]
        threshold_negativity = x[k-1].item()


        pos_mask = (weight.data > threshold_magnitude) & (weight.data > 0.0)
        neg_mask = (weight.data < threshold_negativity) & (weight.data < 0.0)


        new_mask = pos_mask | neg_mask
        return new_mask

    '''
                    GROWTH
    '''

    def random_growth(self, name, new_mask, total_regrowth, weight):

        if self.density_dict[name]==1.0:
            new_mask = torch.ones_like(new_mask, dtype=torch.float32, requires_grad=False).cuda()


        else:


            if name in self.filter_masks:
                temp_mask = new_mask [self.filter_masks[name].bool()]
            else:
                temp_mask=  new_mask

            n = (temp_mask==0).sum().item()
            if n == 0: return new_mask

            expeced_growth_probability = (total_regrowth/n)
            new_weights = torch.rand(new_mask.shape).cuda() < expeced_growth_probability #lsw
            # new_weights = torch.rand(new_mask.shape) < expeced_growth_probability

            new_mask=new_mask.byte() | new_weights

        if name in self.filter_masks:
            new_mask= torch.logical_and(self.filter_masks[name], new_mask).float()


        return new_mask


    def momentum_growth(self, name, new_mask, total_regrowth, weight):
        grad = self.get_momentum_for_weight(weight)
        grad = grad*(new_mask==0).float()
        y, idx = torch.sort(torch.abs(grad).flatten(), descending=True)
        new_mask.data.view(-1)[idx[:total_regrowth]] = 1.0

        return new_mask

    def gradient_growth(self, name, new_mask, total_regrowth, weight):
        if self.density_dict[name]==1.0:
            new_mask = torch.ones_like(new_mask, dtype=torch.float32, requires_grad=False).cuda()

            if name in self.filter_masks:
               new_mask = torch.logical_and(self.filter_masks[name], new_mask).float()  


        else:
            grad = self.get_gradient_for_weights(weight)
            grad = grad*(new_mask==0).float()

            if name in self.filter_masks:
                remain = (torch.abs(grad * (self.masks[name]==0)))  [self.filter_masks[name].bool()] .float()
            else:
                remain = torch.abs(grad *(self.masks[name]==0) ).float()

            all_scores = torch.cat([torch.flatten(x) for x in remain])
            print ("total_regrowth",total_regrowth)
            threshold, _ = torch.topk(all_scores, total_regrowth,largest=True, sorted=True)
            acceptable_score = threshold[-1]


            new_mask = (new_mask.byte() | (torch.abs(grad.data) > acceptable_score)).float()

        return new_mask

    # def gradient_growth(self, name, new_mask, total_regrowth, weight):

    #     grad = self.get_gradient_for_weights(weight)


    #     # operate on zero mask ind
    #     all_ind=torch.arange(grad.numel()).view(grad.shape)
    #     select_ind=all_ind[new_mask==0]

    #     # sort the gradients

    #     y, idx = torch.sort(torch.abs(grad[new_mask==0]), descending=True)

    #     # grow back in the zeo mask ind
    #     new_mask.data.reshape(-1)[select_ind[idx[:total_regrowth]]]=1.0


    #     # print ("after grow", (new_mask!=0).sum().int().item())



    #     return new_mask

    def momentum_neuron_growth(self, name, new_mask, total_regrowth, weight):
        grad = self.get_momentum_for_weight(weight)

        M = torch.abs(grad)
        if len(M.shape) == 2: sum_dim = [1]
        elif len(M.shape) == 4: sum_dim = [1, 2, 3]

        v = M.mean(sum_dim).data
        v /= v.sum()

        slots_per_neuron = (new_mask==0).sum(sum_dim)

        M = M*(new_mask==0).float()
        for i, fraction  in enumerate(v):
            neuron_regrowth = math.floor(fraction.item()*total_regrowth)
            available = slots_per_neuron[i].item()

            y, idx = torch.sort(M[i].flatten())
            if neuron_regrowth > available:
                neuron_regrowth = available
            threshold = y[-(neuron_regrowth)].item()
            if threshold == 0.0: continue
            if neuron_regrowth < 10: continue
            new_mask[i] = new_mask[i] | (M[i] > threshold)

        return new_mask

    '''
                UTILITY
    '''
    def get_momentum_for_weight(self, weight):
        if 'exp_avg' in self.optimizer.state[weight]:
            adam_m1 = self.optimizer.state[weight]['exp_avg']
            adam_m2 = self.optimizer.state[weight]['exp_avg_sq']
            grad = adam_m1/(torch.sqrt(adam_m2) + 1e-08)
        elif 'momentum_buffer' in self.optimizer.state[weight]:
            grad = self.optimizer.state[weight]['momentum_buffer']
        return grad

    def get_gradient_for_weights(self, weight):
        grad = weight.grad.clone()
        return grad

    def print_grad_flow(self):




        all_act_grad=0
        for name, weight in self.module.named_parameters():
            if name not in self.active_new_mask: continue
            grad = self.get_gradient_for_weights(weight)
            mask=self.active_new_mask[name]

            
            # act_grad = torch.abs(  grad*mask   ) .mean().item()

            act_grad=grad*mask

            # print (name)
            # for i in range(len(mask)):
            #     print ("mask",i,mask[i].sum())

            # for i in range(len(grad)):
            #     print ("grad",i,grad[i].sum())

            # for i in range(len(grad)):
            #     print ("mag",i,weight.clone()[i].sum())

            act_grad=torch.norm(act_grad)
            all_act_grad+=act_grad

        all_pass_grad=0
        for name, weight in self.module.named_parameters():
            if name not in self.passtive_new_mask: continue
            grad = self.get_gradient_for_weights(weight)
            mask=self.passtive_new_mask[name]

            # print (name)
            # for i in range(len(mask)):
            #     print ("mask",i,mask[i].sum())

            # for i in range(len(grad)):
            #     print ("grad",i,grad[i].sum())




            # pas_grad= torch.abs(  grad*mask  ) .mean().item()
            pas_grad=grad*mask
            pas_grad=torch.norm(pas_grad)
            all_pass_grad+=pas_grad


        print ("active grad flow",all_act_grad)
        print ( "all_pass_grad",all_pass_grad)

        total_size = 0
        for name, weight in self.active_new_mask.items():
            total_size  += weight.sum().item() 
        print ("self.active_new_mask",total_size)

        total_size = 0
        for name, weight in self.passtive_new_mask.items():
            total_size  += weight.sum().item() 

        print ("self.passtive_new_mask",total_size) 



    # def print_nonzero_counts(self):
    #     for module in self.modules:
    #         for name, tensor in module.named_parameters():
    #             if name not in self.masks: continue
    #             mask = self.masks[name]
    #             num_nonzeros = (mask != 0).sum().item()
    #             val = '{0}: {1}->{2}, density: {3:.3f}'.format(name, self.name2nonzeros[name], num_nonzeros,
    #                                                            num_nonzeros / float(mask.numel()))
    #             print(val)


    #     total_size = 0
    #     sparse_size = 0
    #     for name, weight in self.masks.items():
    #         total_size += weight.numel()
    #         sparse_size += (weight != 0).sum().int().item()
    #         density =  sparse_size / total_size
    #     print(60 * '=')
    #     print('the current density is {0}: {1} {2}'.format(density,sparse_size,total_size))
    #     print(60 * '=')

    def print_nonzero_counts(self):



        total_size = 0
        for name, weight in self.masks.items():
            if name in self.filter_masks:
                weight=weight[self.filter_masks[name].bool()]

            total_size  += weight.numel()
        

        sparse_size = 0
        for name, weight in self.masks.items():
            if name in self.filter_masks:
                weight=weight[self.filter_masks[name].bool()]
            sparse_size += (weight != 0).sum().int().item()




        total_size = 0
        total_mask_nonzeros=0
        total_weight_nonzeros=0

        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name not in self.masks: continue


                if name in self.filter_masks:

                    print ("before",  "tensor", tensor.numel(),"mask",self.masks[name].sum().item(), self.masks[name].numel())
                    mask=self.masks[name][self.filter_masks[name].bool()]
                    tensor=tensor[self.filter_masks[name].bool()]

                else:
                    mask=self.masks[name]

                total_size += tensor.numel()
                #mask nonzero num
                mask_nonzeros=(mask!= 0.0).sum().item()
                total_mask_nonzeros+=mask_nonzeros

                # weight nonzero num
                weight_nonzeros=(tensor != 0.0).sum().item()
                total_weight_nonzeros+=weight_nonzeros


                print('{0}, mask/weight parameters,{1},{2} density {3},{4}'.format(name,mask_nonzeros,weight_nonzeros,mask_nonzeros /tensor.numel(),weight_nonzeros/tensor.numel()))             

        print('Total Model parameters after dst:', total_mask_nonzeros, total_weight_nonzeros)
        print('Total parameters under density level of {0}: {1} {2} after dst'.format(self.args.density,total_mask_nonzeros /total_size, total_weight_nonzeros /total_size))





        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name not in self.masks: continue
                print('Death rate: {0}\n'.format(self.death_rate))
                break



    def print_layerwise_density(self):

        temp_model=copy.deepcopy(self.module )
                    
        for name, weight in temp_model:
            if name in self.filter_names:
                temp_model[name]=self.temp_mask[name][self.filter_names[name].bool()]

        for name, weight in self.masks.items():
    
            if name in self.passive_names:


                temp=temp_model[name]

                    # transpose
                temp =temp.transpose(0, 1)
                temp=temp[self.passive_names[name].bool()]
                temp.transpose_(0, 1)

                temp_model[name]=temp


  
        for name, weight in temp_model:
            if name not in self.masks: continue

            if len(weight.shape)==4:
        #         print (name)
                
                # channel sparsity
                for channel_vector in weight:
            
                    channel_zero=(channel_vector!=0).sum().int().item()
                    channel_all=channel_vector.numel()

                    print("check in", name, "density is",channel_zero/channel_all,"weight density is",channel_zero/channel_all,"weight magnitue", torch.abs(channel_vector).mean().item()  )
        


    def reset_momentum(self):
        """
        Taken from: https://github.com/AlliedToasters/synapses/blob/master/synapses/SET_layer.py
        Resets buffers from memory according to passed indices.
        When connections are reset, parameters should be treated
        as freshly initialized.
        """
        for module in self.modules:
            for name, tensor in module.named_parameters():
                if name not in self.masks: continue
                mask = self.masks[name]
                weights = list(self.optimizer.state[tensor])
                for w in weights:
                    if w == 'momentum_buffer':
                        # momentum
                        self.optimizer.state[tensor][w][mask==0] = torch.mean(self.optimizer.state[tensor][w][mask.byte()])
                        # self.optimizer.state[tensor][w][mask==0] = 0
                    elif w == 'square_avg' or \
                        w == 'exp_avg' or \
                        w == 'exp_avg_sq' or \
                        w == 'exp_inf':
                        # Adam
                        self.optimizer.state[tensor][w][mask==0] = torch.mean(self.optimizer.state[tensor][w][mask.byte()])

