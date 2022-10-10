import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

class MLP_dict_softmax(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=(1024, 512), activation='relu', discrim=False, dropout=-1,edge_types=10):
        super(MLP_dict_softmax, self).__init__()
        self.bottleneck_dim = edge_types
        self.MLP_distribution = MLP(input_dim = input_dim, output_dim = self.bottleneck_dim, hidden_size=hidden_size)
        # self.dict_layer = conv1x1(self.bottleneck_dim,output_dim)
        # self.dict_layer = nn.Linear(self.bottleneck_dim,output_dim,bias=False)
        self.MLP_factor = MLP(input_dim = input_dim, output_dim = 1, hidden_size=hidden_size)
        self.init_MLP = MLP(input_dim = input_dim, output_dim = input_dim, hidden_size=hidden_size)

    def forward(self, x):
        x = self.init_MLP(x)
        distribution = gumbel_softmax(self.MLP_distribution(x),tau=1/2, hard=False)
        # embed = self.dict_layer(distribution)
        factor = torch.sigmoid(self.MLP_factor(x))
        # factor = 1
        out = factor * distribution
        return out, distribution

class MS_HGNN_oridinary(nn.Module):
    """Pooling module as proposed in our paper"""
    def __init__(
        self, embedding_dim=64, h_dim=64, mlp_dim=1024, bottleneck_dim=1024,
        activation='relu', batch_norm=True, dropout=0.0, nmp_layers=4, vis=False
    ):
        super(MS_HGNN_oridinary, self).__init__()

        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.bottleneck_dim = bottleneck_dim
        self.embedding_dim = embedding_dim
        self.nmp_layers = nmp_layers
        self.batch_norm = batch_norm
        self.activation = activation
        self.vis = vis

        hdim_extend = 64
        self.hdim_extend = hdim_extend
        self.edge_types = 6
        self.nmp_mlp_start = MLP_dict_softmax(input_dim = hdim_extend, output_dim = h_dim, hidden_size=(128,),edge_types=self.edge_types)
        self.nmp_mlps = self.make_nmp_mlp()
        self.nmp_mlp_end = MLP(input_dim = h_dim*2, output_dim = bottleneck_dim, hidden_size=(128,))
        attention_mlp = []
        for i in range(nmp_layers):
            attention_mlp.append(MLP(input_dim=hdim_extend*2, output_dim=1, hidden_size=(32,)))
        self.attention_mlp = nn.ModuleList(attention_mlp)
        node2edge_start_mlp = []
        for i in range(nmp_layers):
            node2edge_start_mlp.append(MLP(input_dim = h_dim, output_dim = hdim_extend, hidden_size=(256,)))
        self.node2edge_start_mlp = nn.ModuleList(node2edge_start_mlp)
        edge_aggregation_list = []
        for i in range(nmp_layers):
            edge_aggregation_list.append(edge_aggregation(input_dim = h_dim, output_dim = bottleneck_dim, hidden_size=(128,),edge_types=self.edge_types))
        self.edge_aggregation_list = nn.ModuleList(edge_aggregation_list)

    def make_nmp_mlp(self):
        nmp_mlp = []
        for i in range(self.nmp_layers-1):
            mlp1 = MLP(input_dim = self.h_dim*2, output_dim = self.h_dim, hidden_size=(128,))
            mlp2 = MLP_dict_softmax(input_dim = self.hdim_extend, output_dim = self.h_dim, hidden_size=(128,),edge_types=self.edge_types)
            nmp_mlp.append(mlp1)
            nmp_mlp.append(mlp2)
        nmp_mlp = nn.ModuleList(nmp_mlp)
        return nmp_mlp

    def repeat(self, tensor, num_reps):
        """
        Inputs:
        -tensor: 2D tensor of any shape
        -num_reps: Number of times to repeat each row
        Outpus:
        -repeat_tensor: Repeat each row such that: R1, R1, R2, R2
        """
        col_len = tensor.size(1)
        tensor = tensor.unsqueeze(dim=1).repeat(1, num_reps, 1)
        tensor = tensor.view(-1, col_len)
        return tensor

    def edge2node(self, x, rel_rec, rel_send, ori, idx):
        # NOTE: Assumes that we have the same graph across all samples.
        H = rel_rec + rel_send
        incoming = self.edge_aggregation_list[idx](x,H,ori)
        return incoming / incoming.size(1)

    def node2edge(self, x, rel_rec, rel_send, idx):
        # NOTE: Assumes that we have the same graph across all samples.
        H = rel_rec + rel_send
        x = self.node2edge_start_mlp[idx](x)
        # import pdb; pdb.set_trace()
        edge_init = torch.matmul(H,x)
        node_num = x.shape[1]
        edge_num = edge_init.shape[1]
        x_rep = (x[:,:,None,:].transpose(2,1)).repeat(1,edge_num,1,1)
        edge_rep = edge_init[:,:,None,:].repeat(1,1,node_num,1)
        node_edge_cat = torch.cat((x_rep,edge_rep),dim=-1)
        attention_weight = self.attention_mlp[idx](node_edge_cat)[:,:,:,0]
        H_weight = attention_weight * H
        H_weight = F.softmax(H_weight,dim=2)
        H_weight = H_weight * H
        edges = torch.matmul(H_weight,x)
        return edges

    def init_adj(self, num_ped, batch):
        off_diag = np.ones([num_ped, num_ped])

        rel_rec = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float64)
        rel_send = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float64)
        rel_rec = torch.FloatTensor(rel_rec)
        rel_send = torch.FloatTensor(rel_send)

        rel_rec = rel_rec.cuda()
        rel_send = rel_send.cuda()

        rel_rec = rel_rec[None,:,:].repeat(batch,1,1)
        rel_send = rel_send[None,:,:].repeat(batch,1,1)

        return rel_rec, rel_send

    def forward(self, h_states):
        batch = h_states.shape[0]
        actor_num = h_states.shape[1]

        curr_hidden = h_states

        # Neural Message Passing
        rel_rec, rel_send = self.init_adj(actor_num,batch)
        # iter 1
        edge_feat = self.node2edge(curr_hidden, rel_rec, rel_send,0) # [num_edge, h_dim*2]
        # edge_feat = torch.cat([edge_feat, curr_rel_embedding], dim=2)    # [num_edge, h_dim*2+embedding_dim]
        edge_feat, factors = self.nmp_mlp_start(edge_feat)                      # [num_edge, h_dim]
        node_feat = curr_hidden

        nodetoedge_idx = 0
        if self.nmp_layers <= 1:
            pass
        else:
            for nmp_l, nmp_mlp in enumerate(self.nmp_mlps):
                if nmp_l%2==0:
                    node_feat = nmp_mlp(self.edge2node(edge_feat, rel_rec, rel_send,node_feat,nodetoedge_idx)) # [num_ped, h_dim]
                    nodetoedge_idx += 1
                else:    
                    edge_feat, _ = nmp_mlp(self.node2edge(node_feat, rel_rec, rel_send,nodetoedge_idx)) # [num_ped, h_dim] -> [num_edge, 2*h_dim] -> [num_edge, h_dim]
        node_feat = self.nmp_mlp_end(self.edge2node(edge_feat, rel_rec, rel_send, node_feat, nodetoedge_idx))
        return node_feat, factors


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=(1024, 512), activation='relu', discrim=False, dropout=-1):
        super(MLP, self).__init__()
        dims = []
        dims.append(input_dim)
        dims.extend(hidden_size)
        dims.append(output_dim)
        self.layers = nn.ModuleList()
        for i in range(len(dims)-1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()

        self.sigmoid = nn.Sigmoid() if discrim else None
        self.dropout = dropout

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i != len(self.layers)-1:
                x = self.activation(x)
                if self.dropout != -1:
                    x = nn.Dropout(min(0.1, self.dropout/3) if i == 1 else self.dropout)(x)
            elif self.sigmoid:
                x = self.sigmoid(x)
        return x


class MLP_dict(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=(1024, 512), activation='relu', discrim=False, dropout=-1,edge_types=10):
        super(MLP_dict, self).__init__()
        self.bottleneck_dim = edge_types
        self.MLP_distribution = MLP(input_dim = input_dim, output_dim = self.bottleneck_dim, hidden_size=hidden_size)
        # self.dict_layer = conv1x1(self.bottleneck_dim,output_dim)
        # self.dict_layer = nn.Linear(self.bottleneck_dim,output_dim,bias=False)
        self.MLP_factor = MLP(input_dim = input_dim, output_dim = 1, hidden_size=hidden_size)
        self.init_MLP = MLP(input_dim = input_dim, output_dim = input_dim, hidden_size=hidden_size)

    def forward(self, x):
        x = self.init_MLP(x)
        distribution = torch.abs(self.MLP_distribution(x))
        return distribution, distribution

class edge_aggregation(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=(1024, 512), activation='relu', discrim=False, dropout=-1, edge_types=5):
        super(edge_aggregation, self).__init__()
        self.edge_types = edge_types
        self.dict_dim = input_dim
        self.agg_mlp = []
        for i in range(edge_types):
            self.agg_mlp.append(MLP(input_dim=input_dim, output_dim=input_dim, hidden_size=(128,)))
        self.agg_mlp = nn.ModuleList(self.agg_mlp)
        # self.embed_dict = nn.Parameter(torch.Tensor(self.edge_types, self.dict_dim))
        self.mlp = MLP(input_dim=input_dim, output_dim=input_dim, hidden_size=(128,))

    def forward(self,edge_distribution,H,ori):
        batch = edge_distribution.shape[0]
        edges = edge_distribution.shape[1]
        edge_feature = torch.zeros(batch,edges,ori.shape[-1]).type_as(ori)
        edges = torch.matmul(H,ori)
        for i in range(self.edge_types):
            edge_feature += edge_distribution[:,:,i:i+1]*self.agg_mlp[i](edges)

        node_feature = torch.cat((torch.matmul(H.permute(0,2,1), edge_feature),ori),dim=-1)
        return node_feature

class MS_HGNN_hyper(nn.Module):
    """Pooling module as proposed in our paper"""
    def __init__(
        self, embedding_dim=64, h_dim=64, mlp_dim=1024, bottleneck_dim=1024,
        activation='relu', batch_norm=True, dropout=0.0, nmp_layers=4, scale=2, vis=False, actor_number=11
    ):
        super(MS_HGNN_hyper, self).__init__()

        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.bottleneck_dim = bottleneck_dim
        self.embedding_dim = embedding_dim
        self.nmp_layers = nmp_layers
        self.batch_norm = batch_norm
        self.activation = activation
        self.scale = scale
        self.vis = vis

        mlp_pre_dim = embedding_dim + h_dim
        self.vis = vis
        self.spatial_embedding = nn.Linear(2, embedding_dim)
        self.spatial_transform = nn.Linear(h_dim,h_dim)
        hdim_extend = 64
        self.hdim_extend = hdim_extend
        self.edge_types = 10

        self.nmp_mlp_start = MLP_dict_softmax(input_dim=hdim_extend, output_dim=h_dim, hidden_size=(128,),edge_types=self.edge_types)
        self.nmp_mlps = self.make_nmp_mlp()
        self.nmp_mlp_end = MLP(input_dim=h_dim*2, output_dim=bottleneck_dim, hidden_size=(128,))
        attention_mlp = []
        for i in range(nmp_layers):
            attention_mlp.append(MLP(input_dim=hdim_extend*2, output_dim=1, hidden_size=(32,)))
        self.attention_mlp = nn.ModuleList(attention_mlp)

        node2edge_start_mlp = []
        for i in range(nmp_layers):
            node2edge_start_mlp.append(MLP(input_dim = h_dim, output_dim = hdim_extend, hidden_size=(256,)))
        self.node2edge_start_mlp = nn.ModuleList(node2edge_start_mlp)
        edge_aggregation_list = []
        for i in range(nmp_layers):
            edge_aggregation_list.append(edge_aggregation(input_dim = h_dim, output_dim = bottleneck_dim, hidden_size=(128,),edge_types=self.edge_types))
        self.edge_aggregation_list = nn.ModuleList(edge_aggregation_list)
        self.listall = False
        if self.listall:
            if scale < actor_number:
                group_size = scale
                all_combs = []
                for i in range(actor_number):
                    # tensor_a = torch.arange(actor_number)
                    tensor_a = torch.arange(actor_number).cuda()
                    tensor_a = torch.cat((tensor_a[0:i],tensor_a[i+1:]),dim=0)
                    padding = (1,0,0,0)
                    all_comb = F.pad(torch.combinations(tensor_a,r=group_size-1),padding,value=i)
                    all_combs.append(all_comb[None,:,:])
                self.all_combs = torch.cat(all_combs,dim=0)
                self.all_combs = self.all_combs.cuda()

    def make_nmp_mlp(self):
        nmp_mlp = []
        for i in range(self.nmp_layers-1):
            mlp1 = MLP(input_dim=self.h_dim*2, output_dim=self.h_dim, hidden_size=(128,))
            mlp2 = MLP_dict_softmax(input_dim=self.hdim_extend, output_dim=self.h_dim, hidden_size=(128,),edge_types=self.edge_types)
            nmp_mlp.append(mlp1)
            nmp_mlp.append(mlp2)
        nmp_mlp = nn.ModuleList(nmp_mlp)
        return nmp_mlp

    def repeat(self, tensor, num_reps):
        """
        Inputs:
        -tensor: 2D tensor of any shape
        -num_reps: Number of times to repeat each row
        Outpus:
        -repeat_tensor: Repeat each row such that: R1, R1, R2, R2
        """
        col_len = tensor.size(1)
        tensor = tensor.unsqueeze(dim=1).repeat(1, num_reps, 1)
        tensor = tensor.view(-1, col_len)
        return tensor

    def edge2node(self, x, ori, H, idx):
        # NOTE: Assumes that we have the same graph across all samples.
        incoming = self.edge_aggregation_list[idx](x,H,ori)
        return incoming/incoming.size(1)

    def node2edge(self, x, H, idx):
        x = self.node2edge_start_mlp[idx](x)
        edge_init = torch.matmul(H,x)
        node_num = x.shape[1]
        edge_num = edge_init.shape[1]
        x_rep = (x[:,:,None,:].transpose(2,1)).repeat(1,edge_num,1,1)
        edge_rep = edge_init[:,:,None,:].repeat(1,1,node_num,1)
        node_edge_cat = torch.cat((x_rep,edge_rep),dim=-1)
        attention_weight = self.attention_mlp[idx](node_edge_cat)[:,:,:,0]
        H_weight = attention_weight * H
        H_weight = F.softmax(H_weight,dim=2)
        H_weight = H_weight * H
        edges = torch.matmul(H_weight,x)
        return edges
    
    def init_adj_attention(self, feat,feat_corr, scale_factor=2):
        batch = feat.shape[0]
        actor_number = feat.shape[1]
        if scale_factor == actor_number:
            H_matrix = torch.ones(batch,1,actor_number).type_as(feat)
            return H_matrix
        group_size = scale_factor
        if group_size < 1:
            group_size = 1

        _,indice = torch.topk(feat_corr,dim=2,k=group_size,largest=True)
        H_matrix = torch.zeros(batch,actor_number,actor_number).type_as(feat)
        H_matrix = H_matrix.scatter(2,indice,1)

        return H_matrix

    def init_adj_attention_listall(self, feat,feat_corr, scale_factor=2):
        batch = feat.shape[0]
        actor_number = feat.shape[1]
        if scale_factor == actor_number:
            H_matrix = torch.ones(batch,1,actor_number).type_as(feat)
            return H_matrix
        group_size = scale_factor
        if group_size < 1:
            group_size = 1

        all_indice = self.all_combs.clone() #(N,C,m)
        all_indice = all_indice[None,:,:,:].repeat(batch,1,1,1)
        all_matrix = feat_corr[:,None,None,:,:].repeat(1,actor_number,all_indice.shape[2],1,1)
        all_matrix = torch.gather(all_matrix,3,all_indice[:,:,:,:,None].repeat(1,1,1,1,actor_number))
        all_matrix = torch.gather(all_matrix,4,all_indice[:,:,:,None,:].repeat(1,1,1,group_size,1))
        score = torch.sum(all_matrix,dim=(3,4),keepdim=False)
        _,max_idx = torch.max(score,dim=2)
        indice = torch.gather(all_indice,2,max_idx[:,:,None,None].repeat(1,1,1,group_size))[:,:,0,:]

        H_matrix = torch.zeros(batch,actor_number,actor_number).type_as(feat)
        H_matrix = H_matrix.scatter(2,indice,1)

        return H_matrix

    def forward(self, h_states, corr):
        curr_hidden = h_states #(num_pred, h_dim)

        if self.listall:
            H = self.init_adj_attention_listall(curr_hidden,corr,scale_factor=self.scale)
        else:
            H = self.init_adj_attention(curr_hidden,corr,scale_factor=self.scale)

        edge_hidden = self.node2edge(curr_hidden, H, idx=0) 
        edge_feat, factor = self.nmp_mlp_start(edge_hidden)                      
        node_feat = curr_hidden
        node2edge_idx = 0
        if self.nmp_layers <= 1:
            pass
        else:
            for nmp_l, nmp_mlp in enumerate(self.nmp_mlps):
                if nmp_l%2==0:
                    node_feat = nmp_mlp(self.edge2node(edge_feat,node_feat,H,node2edge_idx)) 
                    node2edge_idx += 1
                else:    
                    edge_feat, _ = nmp_mlp(self.node2edge(node_feat, H, idx=node2edge_idx)) 
        node_feat = self.nmp_mlp_end(self.edge2node(edge_feat,node_feat, H,node2edge_idx))
        return node_feat, factor


def sample_gumbel(shape, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3
    Sample from Gumbel(0, 1)
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    U = torch.rand(shape).float()
    return - torch.log(eps - torch.log(U + eps))


def gumbel_softmax_sample(logits, tau=1, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3
    Draw a sample from the Gumbel-Softmax distribution
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
    (MIT license)
    """
    gumbel_noise = sample_gumbel(logits.size(), eps=eps)
    if logits.is_cuda:
        gumbel_noise = gumbel_noise.cuda()
    y = logits + Variable(gumbel_noise)
    return my_softmax(y / tau, axis=-1)

def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3
    Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      tau: non-negative scalar temperature
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probability distribution that sums to 1 across classes
    Constraints:
    - this implementation only works on batch_size x num_features tensor for now
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    y_soft = gumbel_softmax_sample(logits, tau=tau, eps=eps)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        # this bit is based on
        # https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/5
        y_hard = torch.zeros(*shape)
        if y_soft.is_cuda:
            y_hard = y_hard.cuda()
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
        # this cool bit of code achieves two things:
        # - makes the output value exactly one-hot (since we add then
        #   subtract y_soft value)
        # - makes the gradient equal to y_soft gradient (since we strip
        #   all other gradients)
        y = Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y

def my_softmax(input, axis=1):
    trans_input = input.transpose(axis, 0).contiguous()
    soft_max_1d = F.softmax(trans_input)
    return soft_max_1d.transpose(axis, 0)

class MS_HGNN_combined(nn.Module):
    """Pooling module by combining MS_HGNN_ordinary and MS_HGNN_hyper"""
    def __init__(
        self, model_dim=128, hyper_scales=[]
    ):
        super(MS_HGNN_combined, self).__init__()

        self.hyper_scales = hyper_scales
        self.num_hyper_scales = len(hyper_scales)

        # Official hyper-param from paper
        self.interaction = MS_HGNN_oridinary(embedding_dim=16, h_dim=model_dim, mlp_dim=64, bottleneck_dim=model_dim, 
                                            batch_norm=0, nmp_layers=1)
        
        self.interaction_hyper = MS_HGNN_hyper(embedding_dim=model_dim, h_dim=model_dim, mlp_dim=64, bottleneck_dim=model_dim,
                                            batch_norm=0, nmp_layers=1, actor_number=1)

        self.interaction_hyper2 = MS_HGNN_hyper(embedding_dim=model_dim, h_dim=model_dim, mlp_dim=64, bottleneck_dim=model_dim,
                                            batch_norm=0, nmp_layers=1, actor_number=1)

        self.interaction_hyper3 = MS_HGNN_hyper(embedding_dim=model_dim, h_dim=model_dim, mlp_dim=64, bottleneck_dim=model_dim,
                                            batch_norm=0, nmp_layers=1, actor_number=1)
        
        if len(self.hyper_scales) == 0:
            self.out_dim = model_dim
        elif len(self.hyper_scales) == 1:
            self.out_dim = 2 * model_dim
        elif len(self.hyper_scales) == 2:
            self.out_dim = 3 * model_dim
        else:
            self.out_dim = 4 * model_dim

    def reset(self, num_tracks, max_num_neigh, device):
        pass

    def forward(self, ftraj_input, prev_positions, curr_positions):
        # final_feature = torch.zeros_like(ftraj_input)
        # return final_feature.view(-1, self.out_dim)
        # ftraj_input = ftraj_input.detach()
        # ftraj_input = torch.nan_to_num(ftraj_input)
        # query_input = F.normalize(ftraj_input,p=2,dim=2)
        # feat_corr = torch.matmul(query_input,query_input.permute(0,2,1))
        ftraj_inter,_ = self.interaction(ftraj_input)
        if len(self.hyper_scales) > 0:
            ftraj_inter_hyper,_ = self.interaction_hyper(ftraj_input,feat_corr)
        if len(self.hyper_scales) > 1:
            ftraj_inter_hyper2,_ = self.interaction_hyper2(ftraj_input,feat_corr)
        if len(self.hyper_scales) > 2:
            ftraj_inter_hyper3,_ = self.interaction_hyper3(ftraj_input,feat_corr)

        if len(self.hyper_scales) == 0:
            final_feature = ftraj_inter
        if len(self.hyper_scales) == 1:
            final_feature = torch.cat((ftraj_inter,ftraj_inter_hyper),dim=-1)
        elif len(self.hyper_scales) == 2:
            final_feature = torch.cat((ftraj_inter,ftraj_inter_hyper,ftraj_inter_hyper2),dim=-1)
        elif len(self.hyper_scales) == 3:
            final_feature = torch.cat((ftraj_inter,ftraj_inter_hyper,ftraj_inter_hyper2,ftraj_inter_hyper3),dim=-1)
        return final_feature.view(-1, self.out_dim)
