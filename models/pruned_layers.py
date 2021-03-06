import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from quantize import *

device = "cuda" if torch.cuda.is_available() else "cpu"

class PrunedLinear(nn.Module):
    def __init__(self, linear_module):
        assert isinstance(linear_module, torch.nn.Linear), "Input Module is not a valid linear operator!"
        super(PrunedLinear, self).__init__()
        self.in_features = linear_module.in_features
        self.out_features = linear_module.out_features
        self.linear = linear_module
        #self.linear = nn.Linear(in_features, out_features)
        self.mask = np.ones([self.out_features, self.in_features])
        m = self.in_features
        n = self.out_features
        self.sparsity = 1.0
        self.finetune = False
        # Initailization
        #self.linear.weight.data.normal_(0, math.sqrt(2. / (m+n)))
        
    def forward(self, x):
        if not self.finetune:
            device = torch.device('cuda', torch.distributed.get_rank())
            self.gl_loss = self.compute_group_lasso_v2(device=device)
        out = self.linear(x)
        #out = quant8(out, None) # last layer should NOT be quantized

        return out

    def prune_by_percentage(self, q=5.0):
        """
        Pruning the weight paramters by threshold.
        :param q: pruning percentile. 'q' percent of the least 
        significant weight parameters will be pruned.
        """
        # get bounds
        max = torch.max(torch.abs(self.linear.weight.data))
        min = torch.min(torch.abs(self.linear.weight.data))
        # calculate cutoff
        cutoff = ((max - min) * (q / 100.0)) + min
        # generate mask
        self.mask = torch.abs(self.linear.weight.data) > cutoff
        # prune the weights
        self.linear.weight.data = self.linear.weight.float() * self.mask.float()
        # calculate sparsity
        self.sparsity = self.linear.weight.data.numel() - self.linear.weight.data.nonzero().size(0)


    def prune_by_std(self, s=0.25):
        """
        Pruning by a factor of the standard deviation value.
        :param std: (scalar) factor of the standard deviation value. 
        Weight magnitude below np.std(weight)*std
        will be pruned.
        """

        # generate mask
        self.mask = torch.abs(self.linear.weight.data) >= (torch.std(self.linear.weight.data)*s)
        # prune the weights
        self.linear.weight.data = self.linear.weight.data.float() * self.mask.float()
        # calculate sparsity
        self.sparsity = self.linear.weight.data.numel() - self.linear.weight.data.nonzero().size(0)
        
        #print("WEIGHTS: ",self.linear.weight.data)
        #print("MASK: ",self.mask)

    def prune_towards_dilation(self):
        # do nothing for the linear layers
        mask = torch.tensor([True])
        self.mask = torch.tensor(mask.repeat(self.out_features, self.in_features)).cuda()

    def prune_towards_asym_dilation(self):
        # do nothing for linear layers
        mask = torch.tensor([True])
        self.mask = torch.tensor(mask.repeat(self.out_features, self.in_features)).cuda()

    def prune_structured_interfilter(self, q):
        # get bounds
        max = torch.max(torch.abs(self.linear.weight.data))
        min = torch.min(torch.abs(self.linear.weight.data))
        # calculate cutoff
        cutoff = ((max - min) * (q / 100.0)) + min
        # generate mask
        means = torch.abs(self.linear.weight.data).mean(axis=(0))
        mask = torch.tensor(torch.abs(means) > cutoff)
        self.mask = torch.tensor(mask.repeat(self.out_features, 1)).cuda()
        # prune the weights
        self.linear.weight.data = self.linear.weight.float() * self.mask.float()
        # calculate sparsity
        self.sparsity = self.linear.weight.data.numel() - self.linear.weight.data.nonzero().size(0)

    def prune_chunk(self, chunk_size = 32, q = 0.75):
        last_chunk =  self.out_features % chunk_size
        n_chunks = self.out_features // chunk_size + (last_chunk != 0)

        linear_mat = self.linear.weight.data
        mask = torch.full(linear_mat.shape, True, dtype=bool).cuda()
        cutoff = torch.std(linear_mat)*q

        for chunk_idx in range(n_chunks):
            if chunk_idx == n_chunks - 1 and last_chunk != 0:
                current_chunk = linear_mat[chunk_idx * chunk_size:, :]
                l1_norm = torch.sum(torch.abs(current_chunk), dim=0) / last_chunk
                next_mask = (l1_norm > cutoff).repeat(last_chunk, 1)
                mask[chunk_idx * chunk_size:, :] = torch.logical_and(mask[chunk_idx * chunk_size:, :], next_mask)
            else:
                current_chunk = linear_mat[chunk_idx * chunk_size:(chunk_idx + 1) * chunk_size, :]
                l1_norm = torch.sum(torch.abs(current_chunk), dim=0) / chunk_size
                next_mask = (l1_norm > cutoff).repeat(chunk_size, 1)
                mask[chunk_idx * chunk_size:(chunk_idx + 1) * chunk_size, :] = torch.logical_and(mask[chunk_idx * chunk_size:(chunk_idx + 1) * chunk_size, :], next_mask)
        self.mask = mask
        # prune the weights
        self.linear.weight.data = self.linear.weight.float() * self.mask.float()
        # calculate sparsity
        self.sparsity = self.linear.weight.data.numel() - self.linear.weight.data.nonzero().size(0)
        
    def prune_cascade_l1(self, chunk_size = 32, q = 0.75):
        last_chunk =  self.out_features % chunk_size
        n_chunks = self.out_features // chunk_size + (last_chunk != 0)
        
        linear_mat = self.linear.weight.data
        mask = torch.full(linear_mat.shape, True, dtype=bool).cuda()
        cutoff = torch.std(linear_mat)*q
        cutoff = cutoff	* (1.0 / (n_chunks*(n_chunks+1)/2))
        
        for chunk_idx in range(n_chunks):
            current_cascade = linear_mat[chunk_idx * chunk_size:, :]
            l1_norm = torch.sum(torch.abs(current_cascade), dim=0) / (self.out_features - (chunk_idx * chunk_size))
            # scale norm
            l1_norm = l1_norm * ((n_chunks - chunk_idx) / (n_chunks*(n_chunks+1)/2))
            next_mask = (l1_norm > cutoff).repeat((self.out_features - (chunk_idx * chunk_size)), 1)
            mask[chunk_idx * chunk_size:, :] = torch.logical_and(mask[chunk_idx * chunk_size:, :], next_mask)

            # PRUNE FILTER CHUNK
            #if (chunk_idx + 1) * chunk_size > self.out_features:
            #    end = self.out_features
            #else:
            #    end = (chunk_idx + 1) * chunk_size
            #current_chunk = linear_mat[chunk_idx * chunk_size:end, :]
            #l1_norm = torch.sum(torch.abs(current_chunk)) / ((end - (chunk_idx * chunk_size)) * self.in_features)
            #next_mask = (l1_norm > cutoff).repeat((end - (chunk_idx * chunk_size)), self.in_features)
            #mask[chunk_idx * chunk_size:end, :] = torch.logical_and(mask[chunk_idx * chunk_size:end, :], next_mask)

        self.mask = mask
        # prune the weights
        self.linear.weight.data = self.linear.weight.float() * self.mask.float()
        # calculate sparsity
        self.sparsity = self.linear.weight.data.numel() - self.linear.weight.data.nonzero().size(0)

    def prune_filter_chunk(self, chunk_size = 32, q = 0.75):
        """
        last_chunk =  self.out_features % chunk_size
        n_chunks = self.out_features // chunk_size + (last_chunk != 0)
        
        linear_mat = self.linear.weight.data
        mask = self.mask
        cutoff = torch.std(linear_mat)*q
        
        for chunk_idx in range(n_chunks):
            if (chunk_idx + 1) * chunk_size > self.out_features:
                end = self.out_features
            else:
                end = (chunk_idx + 1) * chunk_size
            current_chunk = linear_mat[chunk_idx * chunk_size:end, :]
            l1_norm = torch.sum(torch.abs(current_chunk)) / ((end - (chunk_idx * chunk_size)) * self.in_features)
            next_mask = (l1_norm > cutoff).repeat((end - (chunk_idx * chunk_size)), self.in_features)
            mask[chunk_idx * chunk_size:end, :] = torch.logical_and(mask[chunk_idx * chunk_size:end, :], next_mask)
            
        self.mask = mask
        # prune the weights
        self.linear.weight.data = self.linear.weight.float() * self.mask.float()
        # calculate sparsity
        self.sparsity = self.linear.weight.data.numel() - self.linear.weight.data.nonzero().size(0)
        """
        pass
        
    def prune_SSL(self, q):
        linear_mat = self.linear.weight.data
        mask = torch.full(linear_mat.shape, True, dtype=bool).cuda()
        cutoff = torch.std(linear_mat)*q
        
        l1_norm = torch.sum(torch.abs(linear_mat), dim=0) / self.out_features
        next_mask = (l1_norm > cutoff).repeat(self.out_features, 1)
        mask = torch.logical_and(mask, next_mask)
        
        self.mask = mask
        # prune the weights
        self.linear.weight.data = self.linear.weight.float() * self.mask.float()
        # calculate sparsity
        self.sparsity = self.linear.weight.data.numel() - self.linear.weight.data.nonzero().size(0)

    # Group Lasso for v1 chunk pruning
    def compute_group_lasso_v1(self, chunk_size = 32):
        layer_loss = torch.zeros(1).cuda()
        
        last_chunk =  self.out_features % chunk_size
        n_chunks = self.out_features // chunk_size + (last_chunk != 0)
        
        linear_mat = self.linear.weight.view((self.out_features, -1))
        
        for chunk_idx in range(n_chunks):
            if chunk_idx == n_chunks - 1 and last_chunk != 0:
                current_chunk = linear_mat[chunk_idx * chunk_size:, :]
                l2_norm = torch.sqrt(torch.sum(current_chunk ** 2, dim=0) / last_chunk)
            else:
                current_chunk = linear_mat[chunk_idx * chunk_size:(chunk_idx + 1) * chunk_size, :]
                l2_norm = torch.sqrt(torch.sum(current_chunk ** 2, dim=0) / chunk_size)
                
            chunk_loss = torch.sum(torch.abs(l2_norm))
            layer_loss += chunk_loss
            
        return layer_loss

    # cascading bounded sparsity - attempt 1
    def compute_group_lasso_v2(self, chunk_size = 32, device=None):        
        last_chunk =  self.out_features % chunk_size
        n_chunks = self.out_features // chunk_size + (last_chunk != 0)
        
        linear_mat = self.linear.weight.view((self.out_features, -1))
        layer_loss = torch.zeros(n_chunks).to(device)

        chunk_ids = torch.arange(n_chunks-1, -1, -1).to(device)
        scaling_factor = ((n_chunks - chunk_ids) / (n_chunks*(n_chunks+1)/2))

        # #print(linear_mat.shape)

        # linear_mat = nn.functional.pad(linear_mat, pad=( 0, 0, 0, chunk_size - last_chunk))

        # #print(linear_mat.shape)
        # #Linear Mat: chunk_size * n_chunks, other_dim
        # linear_mat = linear_mat.reshape(chunk_size, n_chunks, -1).sum(dim=0)

        # #print(linear_mat.shape)
        # #Linear Mat: n_chunks, other_dim
        # linear_mat = linear_mat.fliplr().pow(2).cumsum(dim=0).div((self.out_features - chunk_ids * chunk_size).reshape(-1, 1)).abs().sum(dim=1)

        
        for chunk_idx in range(n_chunks-1, -1, -1):
            current_cascade = linear_mat[chunk_idx * chunk_size:, :]
            
            l2_norm = torch.norm(current_cascade, p=2, dim=0)#torch.sqrt(torch.sum(current_cascade ** 2, dim=0) / (self.out_features - (chunk_idx * chunk_size)))
            # use triangular number to scale norm
            #l2_norm = l2_norm *	((n_chunks - chunk_idx) / (n_chunks*(n_chunks+1)/2))
            layer_loss[chunk_idx] = l2_norm.abs().sum()
    
        #return torch.sum(layer_loss * scaling_factor)
        return torch.sum(layer_loss) ### NOTE: REMOVE SCALING FACTOR FOR NOW
        
    def compute_SSL(self):
        layer_loss = torch.zeros(1).cuda()
        
        conv_mat = self.linear.weight.view((self.out_features, -1))
        l2_norm = torch.sqrt(torch.sum(conv_mat ** 2, dim=0) / self.out_features)
        layer_loss += torch.sum(torch.abs(l2_norm))
        
        return layer_loss
        
class PrunedConv(nn.Module):
    def __init__(self, conv2d_module):
        super(PrunedConv, self).__init__()
        assert isinstance(conv2d_module, torch.nn.Conv2d), "Input Module is not a valid conv operator!"                                                                                      
        self.in_channels = conv2d_module.in_channels
        self.out_channels = conv2d_module.out_channels
        self.kernel_size = conv2d_module.kernel_size
        self.stride = conv2d_module.stride
        self.padding = conv2d_module.padding
        self.dilation = conv2d_module.dilation
        self.bias = conv2d_module.bias
        self.conv = conv2d_module
        self.finetune = False
        #self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias, dilation=dilation)

        # Expand and Transpose to match the dimension
        self.mask = np.ones_like([self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]])

        # Initialization
        n = self.kernel_size[0] * self.kernel_size[1] * self.out_channels
        m = self.kernel_size[0] * self.kernel_size[1] * self.in_channels
        #self.conv.weight.data.normal_(0, math.sqrt(2. / (n+m) ))
        self.sparsity = 1.0

    def forward(self, x):
        if not self.finetune:
        #Compute Gorup Lasso at forward
            device = torch.device('cuda', torch.distributed.get_rank())
            self.gl_loss = self.compute_group_lasso_v2(device=device)

        out = self.conv(x)
        #out = quant8(out, None)
        return out

    def prune_by_percentage(self, q=5.0):
        """
        Pruning the weight paramters by threshold.
        :param q: pruning percentile. 'q' percent of the least 
        significant weight parameters will be pruned.
        """
        
        # get bounds
        max = torch.max(torch.abs(self.conv.weight.data))
        min = torch.min(torch.abs(self.conv.weight.data))
        # calculate cutoff
        cutoff = ((max - min) * (q / 100.0)) + min
        # generate mask
        self.mask = torch.abs(self.conv.weight.data) > cutoff
        # prune the weights
        self.conv.weight.data = self.conv.weight.float() * self.mask.float()
        # calculate sparsity
        self.sparsity = self.conv.weight.data.numel() - self.conv.weight.data.nonzero().size(0)
        

    def prune_by_std(self, q=0.25):
        """
        Pruning by a factor of the standard deviation value.
        :param s: (scalar) factor of the standard deviation value. 
        Weight magnitude below np.std(weight)*std
        will be pruned.
        """
        
        # generate mask
        self.mask = torch.abs(self.conv.weight.data) >= (torch.std(self.conv.weight.data)*q)
        # prune the weights
        self.conv.weight.data = self.conv.weight.data.float() * self.mask.float()
        # calculate sparsity
        self.sparsity = self.conv.weight.data.numel() - self.conv.weight.data.nonzero().size(0)

    def prune_towards_dilation(self):
        # generate mask
        if self.kernel_size[0] == 5:
            mask = torch.tensor([[True, False, True, False, True],
                    [False, False, False, False, False],
                    [True, False, True, False, True],
                    [False, False, False, False, False],
                    [True, False, True, False, True]])
        else:
            mask = torch.tensor([[True, True, True],
                    [True, True, True],
                    [True, True, True]])
        self.mask = torch.tensor(mask.repeat(self.out_channels, self.in_channels, 1, 1)).cuda()
        # prune the weights
        self.conv.weight.data = self.conv.weight.data.float() * self.mask.float()
        # calculate sparsity
        self.sparsity = self.conv.weight.data.numel() - self.conv.weight.data.nonzero().size(0)
        
    def	prune_towards_asym_dilation(self):
        # generate mask
        if self.kernel_size[0] == 5:
            # compute kernel-normalized magnitudes of each element in each kernel across filters
            means = torch.abs(self.conv.weight.data).mean(axis=(2,3)).cpu()

            scaled = np.array([[[[0]*self.kernel_size[0]]*self.kernel_size[1]]*self.in_channels]*self.out_channels)
            weight_data = self.conv.weight.data.cpu()
            for out_channel in range(self.out_channels):
                for in_channel in range(self.in_channels):
                    scaled[out_channel][in_channel] = np.divide(weight_data[out_channel][in_channel], means[out_channel][in_channel])
            #scaled = np.divide(self.conv.weight.data.cpu(), means.cpu())
            magnitudes = np.abs(scaled).sum(axis=0)
            # generate mask based on magnitudes
            mask = torch.tensor([True])
            mask = mask.repeat(self.in_channels, self.kernel_size[0], self.kernel_size[1])
            for in_channel in range(self.in_channels):
                sortIdx = np.argsort(magnitudes[in_channel], axis=None)
                target_kernel_size = 3
                for i in range((self.kernel_size[0]*self.kernel_size[1]) - (target_kernel_size*target_kernel_size)):
                    mask[in_channel][sortIdx[i] // self.kernel_size[0]][sortIdx[i] % self.kernel_size[0]] = False
            self.mask = torch.tensor(mask.repeat(self.out_channels, 1, 1, 1)).cuda()
        else:
            mask = torch.tensor([True])
            self.mask = mask.repeat(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]).cuda()
        # prune the weights
        self.conv.weight.data = self.conv.weight.data.float() * self.mask.float()
        # calculate sparsity
        self.sparsity = self.conv.weight.data.numel() - self.conv.weight.data.nonzero().size(0)

    def prune_structured_interfilter(self, q):
        # get bounds
        max = torch.max(torch.abs(self.conv.weight.data))
        min = torch.min(torch.abs(self.conv.weight.data))
        # calculate cutoff
        cutoff = ((max - min) * (q / 100.0)) + min
        # generate mask
        means = torch.abs(self.conv.weight.data).mean(axis=(0))
        mask = torch.tensor(torch.abs(means) > cutoff)
        self.mask = torch.tensor(mask.repeat(self.out_channels, 1, 1, 1)).cuda()
        # prune the weights
        self.conv.weight.data = self.conv.weight.float() * self.mask.float()
        # calculate sparsity
        self.sparsity = self.conv.weight.data.numel() - self.conv.weight.data.nonzero().size(0)

    def prune_chunk(self, chunk_size = 32, q = 0.75):
        last_chunk =  self.out_channels % chunk_size
        n_chunks = self.out_channels // chunk_size + (last_chunk != 0)
        
        conv_mat = self.conv.weight.data
        mask = torch.full(conv_mat.shape, True, dtype=bool).cuda()
        cutoff = torch.std(conv_mat)*q
        
        for chunk_idx in range(n_chunks):
            if chunk_idx == n_chunks - 1 and last_chunk != 0:
                current_chunk = conv_mat[chunk_idx * chunk_size:, :]
                l1_norm = torch.sum(torch.abs(current_chunk), dim=0) / last_chunk
                next_mask = (l1_norm > cutoff).repeat(last_chunk, 1, 1, 1)
                mask[chunk_idx * chunk_size:, :, :, :] = torch.logical_and(mask[chunk_idx * chunk_size:, :, :,:], next_mask)
            else:
                current_chunk = conv_mat[chunk_idx * chunk_size:(chunk_idx + 1) * chunk_size, :]
                l1_norm = torch.sum(torch.abs(current_chunk), dim=0) / chunk_size
                next_mask = (l1_norm > cutoff).repeat(chunk_size, 1, 1, 1)
                mask[chunk_idx * chunk_size:(chunk_idx + 1) * chunk_size, :, :, :] = torch.logical_and(mask[chunk_idx * chunk_size:(chunk_idx + 1) * chunk_size, :, :, :], next_mask)
            
        self.mask = mask
        # prune the weights
        self.conv.weight.data = self.conv.weight.float() * self.mask.float()
        # calculate sparsity
        self.sparsity = self.conv.weight.data.numel() - self.conv.weight.data.nonzero().size(0)
        
    def prune_cascade_l1(self, chunk_size = 32, q = 0.75):
        last_chunk =  self.out_channels % chunk_size
        n_chunks = self.out_channels // chunk_size + (last_chunk != 0)

        conv_mat = self.conv.weight.data
        mask = torch.full(conv_mat.shape, True, dtype=bool).cuda()
        cutoff = torch.std(conv_mat)*q
        cutoff = cutoff * (1.0 / (n_chunks*(n_chunks+1)/2))
        
        for chunk_idx in range(n_chunks):
            current_cascade = conv_mat[chunk_idx * chunk_size:, :, :, :]
            l1_norm = torch.sum(torch.abs(current_cascade), dim=0) / (self.out_channels - (chunk_idx * chunk_size))
            # scale the norm
            l1_norm = l1_norm * ((n_chunks - chunk_idx) / (n_chunks*(n_chunks+1)/2))
            next_mask = (l1_norm > cutoff).repeat((self.out_channels - (chunk_idx * chunk_size)), 1, 1, 1)
            mask[chunk_idx * chunk_size:, :, :, :] = torch.logical_and(mask[chunk_idx * chunk_size:, :, :, :], next_mask)

            # PRUNE FILTER CHUNKS
            #if (chunk_idx+1) * chunk_size > self.out_channels:
            #    end = self.out_channels
            #else:
            #    end = (chunk_idx+1) * chunk_size
            #current_chunk = conv_mat[chunk_idx * chunk_size:end, :, :, :]
            #l1_norm = torch.sum(torch.abs(current_chunk)) / ((end - (chunk_idx * chunk_size)) * self.in_channels * self.kernel_size[0] * self.kernel_size[1])
            #next_mask = (l1_norm > cutoff).repeat((end - (chunk_idx * chunk_size)), self.in_channels, self.kernel_size[0], self.kernel_size[1])
            #mask[chunk_idx * chunk_size:end, :, :, :] = torch.logical_and(mask[chunk_idx * chunk_size:end, :, :, :], next_mask)
            
        self.mask = mask
        # prune the weights
        self.conv.weight.data = self.conv.weight.float() * self.mask.float()
        # calculate sparsity
        self.sparsity = self.conv.weight.data.numel() - self.conv.weight.data.nonzero().size(0)

    def prune_filter_chunk(self, chunk_size = 32, q = 0.75):
        last_chunk =  self.out_channels % chunk_size
        n_chunks = self.out_channels // chunk_size + (last_chunk != 0)
        
        conv_mat = self.conv.weight.data
        mask = torch.full(conv_mat.shape, True, dtype=bool).cuda()
        cutoff = torch.std(conv_mat)*q
        
        for chunk_idx in range(n_chunks):
            if (chunk_idx+1) * chunk_size > self.out_channels:
                end = self.out_channels
            else:
                end = (chunk_idx+1) * chunk_size
            current_chunk = conv_mat[chunk_idx * chunk_size:end, :, :, :]
            l1_norm = torch.sum(torch.abs(current_chunk)) / ((end - (chunk_idx * chunk_size)) * self.in_channels * self.kernel_size[0] * self.kernel_size[1])
            next_mask = (l1_norm > cutoff).repeat((end - (chunk_idx * chunk_size)), self.in_channels, self.kernel_size[0], self.kernel_size[1])
            mask[chunk_idx * chunk_size:end, :, :, :] = torch.logical_and(mask[chunk_idx * chunk_size:end, :, :, :], next_mask)
            
        self.mask = mask
        # prune the weights
        self.conv.weight.data = self.conv.weight.float() * self.mask.float()
        # calculate the sparsity
        self.sparsity = self.conv.weight.data.numel() - self.conv.weight.data.nonzero().size(0)

    def prune_SSL(self, q = 0.75):
        conv_mat = self.conv.weight.data
        mask = torch.full(conv_mat.shape, True, dtype=bool).cuda()
        cutoff = torch.std(conv_mat)*q

        l1_norm = torch.sum(torch.abs(conv_mat), dim=0) / self.out_channels
        next_mask = (l1_norm > cutoff).repeat(self.out_channels, 1, 1, 1)
        mask = torch.logical_and(mask, next_mask)

        self.mask = mask
        # prune the weights
        self.conv.weight.data = self.conv.weight.float() * self.mask.float()
        # calculate sparsity
        self.sparsity = self.conv.weight.data.numel() - self.conv.weight.data.nonzero().size(0)
        
    # Group Lasso for v1 chunk pruning
    def compute_group_lasso_v1(self, chunk_size = 32):
        layer_loss = torch.zeros(1).cuda()
        
        last_chunk =  self.out_channels % chunk_size
        n_chunks = self.out_channels // chunk_size + (last_chunk != 0)
        
        conv_mat = self.conv.weight.view((self.out_channels, -1))
    
        for chunk_idx in range(n_chunks):
            if chunk_idx == n_chunks - 1 and last_chunk != 0:
                current_chunk = conv_mat[chunk_idx * chunk_size:, :]
                l2_norm = torch.sqrt(torch.sum(current_chunk ** 2, dim=0) / last_chunk)
            else:
                current_chunk = conv_mat[chunk_idx * chunk_size:(chunk_idx + 1) * chunk_size, :]
                l2_norm = torch.sqrt(torch.sum(current_chunk ** 2, dim=0) / chunk_size)
                
            chunk_loss = torch.sum(torch.abs(l2_norm))
            layer_loss += chunk_loss
            
        return layer_loss

    # cascading bounded sparsity - attempt 1
    def compute_group_lasso_v2(self, chunk_size = 32, device = None):
        last_chunk =  self.out_channels % chunk_size
        n_chunks = self.out_channels // chunk_size + (last_chunk != 0)
        
        conv_mat = self.conv.weight.view((self.out_channels, -1))

        layer_loss = torch.zeros(n_chunks).to(device)

        chunk_ids = torch.arange(n_chunks-1, -1, -1).to(device)
        scaling_factor = ((n_chunks - chunk_ids) / (n_chunks*(n_chunks+1)/2))

        # #print(linear_mat.shape)

        # linear_mat = nn.functional.pad(linear_mat, pad=( 0, 0, 0, chunk_size - last_chunk))

        # #print(linear_mat.shape)
        # #Linear Mat: chunk_size * n_chunks, other_dim
        # linear_mat = linear_mat.reshape(chunk_size, n_chunks, -1).sum(dim=0)

        # #print(linear_mat.shape)
        # #Linear Mat: n_chunks, other_dim
        # linear_mat = linear_mat.fliplr().pow(2).cumsum(dim=0).div((self.out_features - chunk_ids * chunk_size).reshape(-1, 1)).abs().sum(dim=1)

        
        for chunk_idx in range(n_chunks-1, -1, -1):
            current_cascade = conv_mat[chunk_idx * chunk_size:, :]
            
            l2_norm = torch.norm(current_cascade, p=2, dim=0)#torch.sqrt(torch.sum(current_cascade ** 2, dim=0) / (self.out_features - (chunk_idx * chunk_size)))
            # use triangular number to scale norm
            #l2_norm = l2_norm *	((n_chunks - chunk_idx) / (n_chunks*(n_chunks+1)/2))
            layer_loss[chunk_idx] = l2_norm.abs().sum()
        # layer_loss = torch.zeros(1).cuda()
        
        
        
        # for chunk_idx in range(n_chunks-1, -1, -1):
        #     current_cascade = conv_mat[chunk_idx * chunk_size:, :]

        #     l2_norm = torch.sqrt(torch.sum(current_cascade ** 2, dim=0) / (self.out_channels - (chunk_idx * chunk_size)))
        #     # use triangular number to scale norm
        #     l2_norm = l2_norm * ((n_chunks - chunk_idx) / (n_chunks*(n_chunks+1)/2))
        #     chunk_loss = torch.sum(torch.abs(l2_norm))
        #     layer_loss += chunk_loss

        #return torch.sum(layer_loss * scaling_factor)
        return torch.sum(layer_loss) ### NOTE: REMOVE SCALING FACTOR FOR NOW
    
    def compute_SSL(self):
        layer_loss = torch.zeros(1).cuda()

        conv_mat = self.conv.weight.view((self.out_channels, -1))
        l2_norm = torch.sqrt(torch.sum(conv_mat ** 2, dim=0) / self.out_channels)
        layer_loss += torch.sum(torch.abs(l2_norm))

        return layer_loss
