import math
import numpy as np
import torch
import torch.nn as nn

class SocialNCE():
    '''
        Social NCE: Contrastive Learning of Socially-aware Motion Representations (https://arxiv.org/abs/2012.11717)
    '''
    def __init__(self, obs_length, pred_length, head_projection, encoder_sample, temperature, horizon, sampling):

        # problem setting
        self.obs_length = obs_length
        self.pred_length = pred_length

        # nce models
        self.head_projection = head_projection
        self.encoder_sample = encoder_sample

        # nce loss
        self.criterion = nn.CrossEntropyLoss()

        # nce param
        self.temperature = temperature
        self.horizon = horizon

        # sampling param
        self.noise_local = 0.05 #TODO maybe 0.1
        self.min_seperation = 0.2 # #TODO increase this ? (uncomfortable zone is up to 20[cm])
        self.agent_zone = self.min_seperation * torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0], [0.707, 0.707], [0.707, -0.707], [-0.707, 0.707], [-0.707, -0.707], [0.0, 0.0]])

        self.sampling = sampling #by maxime
    def spatial(self, batch_scene, batch_split, batch_feat):
        '''
            Social NCE with spatial samples, i.e., samples are locations at a specific time of the future
            Input:
                batch_scene: coordinates of agents in the scene, tensor of shape [obs_length + pred_length, total num of agents in the batch, 2]
                batch_split: index of scene split in the batch, tensor of shape [batch_size + 1]
                batch_feat: encoded features of observations, tensor of shape [pred_length, scene, feat_dim]
                                                                                            ^ #person maybe ?
            Output:
                loss: social nce loss
        '''

        # -----------------------------------------------------
        #               Visualize Trajectories 
        #       (Use this block to visualize the raw data)
        # -----------------------------------------------------

        for i in range(batch_split.shape[0] - 1):
            traj_primary = batch_scene[:, batch_split[i]] # [time, 2]
            traj_neighbor = batch_scene[:, batch_split[i]+1:batch_split[i+1]] # [time, num, 2]
            plot_scene(traj_primary, traj_neighbor, fname='scene_{:d}.png'.format(i))
        # import pdb; pdb.set_trace() # --> to do an embedded breakpoint with Python (without PyCharm debugger)
        
        # #####################################################
        #           TODO: fill the following code
        # #####################################################

        # hint from navigation repo : https://github.com/vita-epfl/social-nce-crowdnav/blob/main/crowd_nav/snce/contrastive.py
        # hint from forecasting repo: https://github.com/YuejiangLIU/social-nce-trajectron-plus-plus/blob/master/trajectron/snce/contrastive.py

        # -----------------------------------------------------
        #               Contrastive Sampling 
        # -----------------------------------------------------

        # batch_split : (8) (ID of the start of the scene and of the person of interrest)
        # batch_scene : ( time x persons (personsOfInterrest and neighboor ) x coordinate)
        #                (21)  x 40 x 2
        # traj_primary: 21x2 (time x coordinate)
        # traj_neighbor: 21x3x2 (time x persons x coordinate)

        (sample_pos, sample_neg)= self._sampling_spatial(batch_scene, batch_split)

        # -----------------------------------------------------
        #              Lower-dimensional Embedding 
        # -----------------------------------------------------
        # 12x40x8                             12x40x128
        interestsID = batch_split[0:-1]
        emb_obsv = self.head_projection(batch_feat[:, interestsID, :]) #TODO should not he whole batch
        query = nn.functional.normalize(emb_obsv, dim=2) #TODO might not be dim 1


        emb_pos = self.encoder_sample(sample_pos.float()) #todo: maybe implemented a validity mask
        emb_neg = self.encoder_sample(torch.tensor(sample_neg).float())
        key_pos = nn.functional.normalize(emb_pos, dim=2)
        key_neg = nn.functional.normalize(emb_neg, dim=3)

        # -----------------------------------------------------
        #                   Compute Similarity 
        # -----------------------------------------------------
        # similarity
        #12x40x8   12x8x1x8
        sim_pos = (query[:, :, None, :] * key_pos[:, :, None,:]).sum(dim=-1)
        sim_neg = (query[:,:, None, :] * key_neg).sum(dim=-1)
        # logits
        logits_3dim = torch.cat([sim_pos, sim_neg], dim=2) / self.temperature

        #TODO: warning really unsure about this
        #the logits need to be 2 dimensionnal, at the moement they are 3
        logits=logits_3dim.reshape((logits.shape[0], -1))

        # -----------------------------------------------------
        #                       NCE Loss
        # -----------------------------------------------------
        labels = torch.zeros(logits.size(0), dtype=torch.long)
        loss = self.criterion(logits, labels)
        print(f"the contrast loss is {loss} ")
        return loss

    def event(self, batch_scene, batch_split, batch_feat):
        '''
            Social NCE with event samples, i.e., samples are spatial-temporal events at various time steps of the future
        '''
        raise ValueError("Optional")

    def _sampling_spatial(self, batch_scene, batch_split):
        # "_" indicates that this is a private function that we can only access from the class
        # batch_split : 9 (maybe the ID of the persons we want to select)
        # batch_scene : ( time x persons x coordinate)

        #gt_future : (time x person x coord)
        gt_future = batch_scene[self.obs_length: self.obs_length+self.pred_length]



        # #####################################################
        #           TODO: fill the following code
        # #####################################################

        # -----------------------------------------------------
        #                  Positive Samples
        # -----------------------------------------------------
        #cf paper equ. 7
        #ground truth + N(0, c_e * I )

        #positive sample (time x persons x coordinate)

        c_e = self.noise_local
        #for main interrests only
        sample_pos = gt_future[:, batch_split[0:-1], :] + np.random.multivariate_normal([0,0], np.array([[c_e, 0], [0, c_e]])) #TODO, maybe diff noise for each person

        # for everyone
        # sample_pos = gt_future[:, :, :] + np.random.multivariate_normal([0,0], np.array([[c_e, 0], [0, c_e]]))

        # -----------------------------------------------------
        #                  Negative Samples
        # -----------------------------------------------------
        # cf paper fig 4b,
        '''
        # negative sample for neighboor only
        personInterest = batch_split[0:-1]
        neighboors = np.ones(gt_future.shape[1])
        neighboors[personInterest]=0
        neighboorsID = np.argwhere(neighboors==1)
        sceneNeighboors= gt_future[:, neighboorsID, :]

        #12 x 30 x 9 x 2
                    time x primary x allsample x coord
        # should be 12 x 10 x 32 x2
        sample_neg = sceneNeighboors + self.agent_zone[None, None, :, :] + np.random.multivariate_normal([0,0], np.array([[c_e, 0], [0, c_e]]))
        
        '''
        nDirection = self.agent_zone.shape[0]
        nMaxNeighboor= 12 #TODO re-tune

        #dim: (#time_step_predicted, #personOfIntereset, #neigboor for ThisPerson of interest*#direction, #coordinate)
        sample_neg = np.empty((gt_future.shape[0], batch_split.shape[0] - 1, nDirection*nMaxNeighboor,2))
        sample_neg[:] = np.NaN
        for i in range(batch_split.shape[0] - 1):
            traj_primary = gt_future[:, batch_split[i]]
            traj_neighbor = gt_future[:, batch_split[i]+1:batch_split[i+1]] # [time, numb neigboor, 2]

            #(#time_step_predicted, #neigboor for ThisPerson of interest, #direction, #coordinate)
            #                                   12x5x2                                  9x2
            negSampleNonSqueezed = traj_neighbor[:, :, None,:] + self.agent_zone[None, None, :, :] + np.random.multivariate_normal([0,0], np.array([[c_e, 0], [0, c_e]]))

            negSampleSqueezed = negSampleNonSqueezed.reshape((negSampleNonSqueezed.shape[0], -1, negSampleNonSqueezed.shape[3]))


            
            #only fill the first part, leave the Nan after
            sample_neg[:, i, 0:negSampleSqueezed.shape[1], :] = negSampleSqueezed


    # negative sample for everyone
        #the position          # the direction to look around     #some noise
        # sample_neg = gt_future[:,:,None,:] + self.agent_zone[None, None, :, :] + np.random.multivariate_normal([0,0], np.array([[c_e, 0], [0, c_e]]))

        # -----------------------------------------------------
        #       Remove negatives that are too hard (optional)
        # -----------------------------------------------------

        # -----------------------------------------------------
        #       Remove negatives that are too easy (optional)
        # -----------------------------------------------------

        return sample_pos, sample_neg

class EventEncoder(nn.Module):
    '''
        Event encoder that maps an sampled event (location & time) to the embedding space
    '''
    def __init__(self, hidden_dim, head_dim):

        super(EventEncoder, self).__init__()
        self.temporal = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.spatial = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, head_dim)
        )

    def forward(self, state, time):
        emb_state = self.spatial(state)
        emb_time = self.temporal(time)
        out = self.encoder(torch.cat([emb_time, emb_state], axis=-1))
        return out

class SpatialEncoder(nn.Module):
    '''
        Spatial encoder that maps an sampled location to the embedding space
    '''
    def __init__(self, hidden_dim, head_dim):
        super(SpatialEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, head_dim)
        )

    def forward(self, state):
        return self.encoder(state)

class ProjHead(nn.Module):
    '''
        Nonlinear projection head that maps the extracted motion features to the embedding space
    '''
    def __init__(self, feat_dim, hidden_dim, head_dim):
        super(ProjHead, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, head_dim)
        )

    def forward(self, feat):
        return self.head(feat)

def plot_scene(primary, neighbor, fname):
    '''
        Plot raw trajectories
    '''
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig = plt.figure(frameon=False)
    fig.set_size_inches(16, 9)
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(primary[:, 0], primary[:, 1], 'k-')
    for i in range(neighbor.size(1)):
        ax.plot(neighbor[:, i, 0], neighbor[:, i, 1], 'b-.')

    ax.set_aspect('equal')
    plt.grid()
    plt.savefig(fname, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
