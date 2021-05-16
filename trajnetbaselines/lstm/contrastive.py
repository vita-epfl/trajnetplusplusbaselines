import math
import numpy as np
import torch
import torch.nn as nn


class SocialNCE():
    """
        Social NCE: Contrastive Learning of Socially-aware Motion Representations (https://arxiv.org/abs/2012.11717)
    """

    def __init__(self, obs_length, pred_length, head_projection, encoder_sample,
                 temperature, horizon, sampling):

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
        self.noise_local = 0.05  # TODO maybe 0.1
        self.min_seperation = 0.2  # #TODO increase this ? (uncomfortable zone is up to 20[cm])
        self.agent_zone = self.min_seperation * torch.tensor(
            [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0], [0.707, 0.707],
             [0.707, -0.707], [-0.707, 0.707], [-0.707, -0.707], [0.0, 0.0]])

        self.sampling = sampling  # by maxime

    def spatial(self, batch_scene, batch_split, batch_feat):
        """
            Social NCE with spatial samples, i.e., samples are locations at a specific time of the future
            Input:                                                                  ( 9       +    12   )
                batch_scene: coordinates of agents in the scene, tensor of shape [obs_length + pred_length, total num of agents in the batch, 2]
                batch_split: index of scene split in the batch, tensor of shape [batch_size + 1]
                batch_feat: encoded features of observations, tensor of shape [pred_length, scene, feat_dim]
                                                                                            ^ #person maybe ?
            Output:
                loss: social nce loss
        """

        if torch.any(torch.isnan(batch_feat)):
            print("WAAAAARNING ! there is NaN in the batch_feat!")

        # -----------------------------------------------------
        #               Visualize Trajectories 
        #       (Use this block to visualize the raw data)
        # -----------------------------------------------------

        # for i in range(batch_split.shape[0] - 1):
        #     traj_primary = batch_scene[:, batch_split[i]] # [time, 2]
        #     traj_neighbour = batch_scene[:, batch_split[i]+1:batch_split[i+1]] # [time, num, 2]
        #     plot_scene(traj_primary, traj_neighbour, fname='scene_{:d}.png'.format(i))
        # import pdb; pdb.set_trace() # --> to do an embedded breakpoint with Python (without PyCharm debugger)

        # #####################################################
        #           TODO: fill the following code
        # #####################################################

        # hint from navigation repo : https://github.com/vita-epfl/social-nce-crowdnav/blob/main/crowd_nav/snce/contrastive.py
        # hint from forecasting repo: https://github.com/YuejiangLIU/social-nce-trajectron-plus-plus/blob/master/trajectron/snce/contrastive.py

        # -----------------------------------------------------
        #               Contrastive Sampling 
        # -----------------------------------------------------

        # batch_split : (8) (ID of the start of the scene and of the person of interest)
        # batch_scene : (time x persons (i.e. personsOfInterest and neighbours) x coordinate)
        #                (21)  x 40 x 2
        # traj_primary: 21x2 (time x coordinate)
        # traj_neighbour: 21x3x2 (time x persons x coordinate)

        (sample_pos, sample_neg) = self._sampling_spatial(batch_scene, batch_split)  # TODO pytorch tensor instead

        # visualisation
        visualize = 1
        if visualize:
            for i in range(batch_split.shape[0] - 1):  # for each scene
                """
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt

                fig = plt.figure(frameon=False)
                fig.set_size_inches(16, 9)
                ax = fig.add_subplot(1, 1, 1)

                # ax.plot(primary[:, 0], primary[:, 1], 'k-')
                # for i in range(neighbor.size(1)):
                #     ax.plot(neighbor[:, i, 0], neighbor[:, i, 1], 'b-.')

                # Displaying the positions of the person of interest
                # TODO: Remove next lines of comments once you have understood why we had a visualization problem here!
                # True position
                # False ❌:
                # ax.scatter(batch_scene[self.obs_length, i, 0],
                #            batch_scene[self.obs_length, i, 1],
                #            label="person of interest true pos")
                # Correct ✅:
                ax.scatter(batch_scene[self.obs_length, batch_split[i], 0],
                           batch_scene[self.obs_length, batch_split[i], 1],
                           label="person of interest true pos")
                # Positive sample
                ax.scatter(sample_pos[i, 0], sample_pos[i, 1],
                           label="positive sample")

                # Displaying the position of the neighbours
                # True position
                ax.scatter(batch_scene[self.obs_length, batch_split[i] + 1:batch_split[i + 1], 0].view(-1),
                           batch_scene[self.obs_length, batch_split[i] + 1:batch_split[i + 1], 1].view(-1),
                           label="neighbours true pos")
                # Negative sample
                ax.scatter(sample_neg[i, :, 0].view(-1),
                           sample_neg[i, :, 1].view(-1),
                           label="negative sample")

                ax.legend()
                ax.set_aspect('equal')
                ax.set_xlim(-7, 7)
                ax.set_ylim(-7, 7)
                plt.grid()
                fname = 'sampling_scene_{:d}.png'.format(i)
                plt.savefig(fname, bbox_inches='tight', pad_inches=0)
                plt.close(fig)
                print(f'displayed samples {i}')
                """
        # -----------------------------------------------------
        #              Lower-dimensional Embedding
        # -----------------------------------------------------
        # 12x40x8                             12x40x128
        interestsID = batch_split[0:-1]
        emb_obsv = self.head_projection(batch_feat[self.obs_length, interestsID, :])  # TODO should not the whole batch
        query = nn.functional.normalize(emb_obsv, dim=-1)  # TODO might not be dim 1

        # Embedding is not necessarily a dimension reduction process! Here we
        # want to find a way to compute the similarity btw. the motion features
        # (for this we have to increase the number of features!)
        # sample_neg: 8x108x2
        mask_normal_space = torch.isnan(sample_neg)
        sample_neg[torch.isnan(sample_neg)] = 0
        # key_neg : 8x108x8
        emb_pos = self.encoder_sample(
            sample_pos)  # TODO cast to pytorch first #todo: maybe implemented a validity mask
        emb_neg = self.encoder_sample(sample_neg)
        key_pos = nn.functional.normalize(emb_pos, dim=-1)
        key_neg = nn.functional.normalize(emb_neg, dim=-1)

        # -----------------------------------------------------
        #                   Compute Similarity 
        # -----------------------------------------------------
        # similarity
        # 12x40x8   12x8x1x8
        sim_pos = (query[:, None, :] * key_pos[:, None, :]).sum(dim=-1)
        sim_neg = (query[:, None, :] * key_neg).sum(dim=-1)

        # 8x108
        mask_new_space = torch.logical_and(mask_normal_space[:, :, 0],
                                           mask_normal_space[:, :, 1])
        sim_neg[mask_new_space] = -10

        logits = torch.cat([sim_pos, sim_neg], dim=-1) / self.temperature  # Warning! Pos and neg samples are concatenated!

        # -----------------------------------------------------
        #                       NCE Loss
        # -----------------------------------------------------
        labels = torch.zeros(logits.size(0), dtype=torch.long)
        loss = self.criterion(logits, labels)
        #print(f"the contrast loss is {loss}")
        return loss

    def event(self, batch_scene, batch_split, batch_feat):
        """
            Social NCE with event samples, i.e., samples are spatial-temporal events at various time steps of the future
        """
        (sample_pos, sample_neg) = self._sampling_event(batch_scene, batch_split)


        # -----------------------------------------------------
        #              Lower-dimensional Embedding
        # -----------------------------------------------------

        interestsID = batch_split[0:-1]
        emb_obsv = self.head_projection(batch_feat[self.obs_length, interestsID, :])
        query = nn.functional.normalize(emb_obsv, dim=-1)

        # Embedding is not necessarily a dimension reduction process! Here we
        # want to find a way to compute the similarity btw. the motion features
        # (for this we have to increase the number of features!)
        # sample_neg: 8x108x2
        mask_normal_space = torch.isnan(sample_neg)

        sample_neg[torch.isnan(sample_neg)] = 0
        # key_neg : 8x108x8
        emb_pos = self.encoder_sample(sample_pos)
        emb_neg = self.encoder_sample(sample_neg)
        key_pos = nn.functional.normalize(emb_pos, dim=-1)
        key_neg = nn.functional.normalize(emb_neg, dim=-1)

        # -----------------------------------------------------
        #                   Compute Similarity
        # -----------------------------------------------------
        # similarity
        # 12x40x8   12x8x1x8
        sim_pos = (query[:, None, :] * key_pos[:, None, :]).sum(dim=-1)
        sim_neg = (query[:, None, :] * key_neg).sum(dim=-1)

        # 8x108
        mask_new_space = torch.logical_and(mask_normal_space[:, :, 0],
                                           mask_normal_space[:, :, 1])
        sim_neg[mask_new_space] = -10

        logits = torch.cat([sim_pos, sim_neg], dim=-1) / self.temperature  # Warning! Pos and neg samples are concatenated!

        # -----------------------------------------------------
        #                       NCE Loss
        # -----------------------------------------------------
        labels = torch.zeros(logits.size(0), dtype=torch.long)
        loss = self.criterion(logits, labels)
        #print(f"the contrast loss is {loss}")
        return loss


    def _sampling_event(self, batch_scene, batch_split):

        gt_future = batch_scene[self.obs_length: self.obs_length+self.pred_length]

        #positive sample
        c_e = self.noise_local
        # Retrieving the location of the pedestrians of interest only
        personOfInterestLocation = gt_future[:, batch_split[0:-1], :]  # (persons of interest x coordinates) --> for instance: 8 x 2
        noise_pos = np.random.multivariate_normal([0, 0], np.array([[c_e, 0], [0, c_e]]), (self.pred_length, 8))  # (2,)
        #                      8 x 2                   1 x 2
        # sample_pos = personOfInterestLocation + noise.reshape(1, 2) # TODO, maybe diff noise for each person (/!\ --> apparently not necessary finally, according to Liu)
        #                      8 x 2             (2,)
        sample_pos = personOfInterestLocation + noise_pos



        #_______negative sample____________
        nDirection = self.agent_zone.shape[0]
        nMaxNeighbour = 50 # TODO re-tune

        # sample_neg: (#persons of interest, #neigboor for this person of interest * #directions, #coordinates)
        # --> for instance: 8 x 12*9 x 2 = 8 x 108 x 2
        sample_neg = np.empty(
            (self.pred_length, batch_split.shape[0] - 1, nDirection * nMaxNeighbour, 2))
        sample_neg[:] = np.NaN  # populating sample_neg with NaN values
        for i in range(batch_split.shape[0] - 1):

            traj_neighbour = gt_future[:, batch_split[i] + 1:batch_split[i + 1]]  # (number of neigbours x coordinates) --> for instance: 3 x 2

            noise_neg = np.random.multivariate_normal([0, 0], np.array([[c_e, 0], [0, c_e]]), (self.pred_length, traj_neighbour.shape[1], self.agent_zone.shape[0])) # (2,)
            # negSampleNonSqueezed: (time x number of neighbours x directions x coordinates)
            #                            12x 3 x 1 x 2                     12x 3 x 9 x 2                (12,3,9,2)
            negSampleNonSqueezed = traj_neighbour[:,:, None, :] + self.agent_zone[None, None, :, :] + noise_neg

            # negSampleSqueezed: (time x number of neighbours * directions x coordinates)
            negSampleSqueezed = negSampleNonSqueezed.reshape((self.pred_length,-1, negSampleNonSqueezed.shape[-1]))

            # Filling only the first part in the second dimension of sample_neg (leaving the rest as NaN values)
            sample_neg[:, i, 0:negSampleSqueezed.shape[1], :] = negSampleSqueezed

        sample_pos = sample_pos.float()
        sample_neg = torch.tensor(sample_neg).float()
        return sample_pos, sample_neg


    def _sampling_spatial(self, batch_scene, batch_split):
        # "_" indicates that this is a private function that we can only access from the class
        # batch_split : 9 (ID of the persons we want to select (except the last element which marks the end of the batch))
        # batch_scene : (time x persons x coordinates) --> for instance: 21 x 39 x 2

        # gt_future : (time x person x coord)
        # gt_future = batch_scene[self.obs_length: self.obs_length+self.pred_length]

        # Selecting only the first pred sample (i.e. the prediction for the first timestamp)
        # (persons x coordinates) --> gt_future is for instance of size 39 x 2
        gt_future = batch_scene[self.obs_length]
        # Note: Since the first 9 frames of the scene correspond to observations
        # and since Python uses zero-based indexing, the first location prediction
        # sample (i.e. the 10th element in batch_scene) is accessed as
        # "batch_scene[9]" (i.e. "batch_scene[self.obs_length]")

        # #####################################################
        #           TODO: fill the following code
        # #####################################################

        # -----------------------------------------------------
        #                  Positive Samples
        # -----------------------------------------------------
        # cf. equ. 7 in paper "Social NCE: Contrastive Learning of Socially-aware
        # Motion Representations" (https://arxiv.org/abs/2012.11717):

        # positive sample ≡ ground truth + N(0, c_e * I )
        # positive sample: (persons of interest x coordinates)

        c_e = self.noise_local
        # Retrieving the location of the pedestrians of interest only
        personOfInterestLocation = gt_future[batch_split[0:-1], :]  # (persons of interest x coordinates) --> for instance: 8 x 2
        noise_pos = np.random.multivariate_normal([0, 0], np.array([[c_e, 0], [0, c_e]]), (personOfInterestLocation.shape[0]))  # (2,)
        #                      8 x 2                   1 x 2
        # sample_pos = personOfInterestLocation + noise.reshape(1, 2) # TODO, maybe diff noise for each person (/!\ --> apparently not necessary finally, according to Liu)
        #                      8 x 2             (2,)
        sample_pos = personOfInterestLocation + noise_pos

              # Retrieving the location of all pedestrians
        # sample_pos = gt_future[:, :, :] + np.random.multivariate_normal([0,0], np.array([[c_e, 0], [0, c_e]]))

        # -----------------------------------------------------
        #                  Negative Samples
        # -----------------------------------------------------
        # cf. fig 4b and eq. 6 in paper "Social NCE: Contrastive Learning of
        # Socially-aware Motion Representations" (https://arxiv.org/abs/2012.11717):

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
        nMaxNeighbour = 80  # TODO re-tune

        # sample_neg: (#persons of interest, #neigboor for this person of interest * #directions, #coordinates)
        # --> for instance: 8 x 12*9 x 2 = 8 x 108 x 2
        sample_neg = np.empty(
            (batch_split.shape[0] - 1, nDirection * nMaxNeighbour, 2))
        sample_neg[:] = np.NaN  # populating sample_neg with NaN values
        for i in range(batch_split.shape[0] - 1):
            # traj_primary = gt_future[batch_split[i]]
            traj_neighbour = gt_future[batch_split[i] + 1:batch_split[i + 1]]  # (number of neigbours x coordinates) --> for instance: 3 x 2

            noise_neg = np.random.multivariate_normal([0, 0], np.array([[c_e, 0], [0, c_e]]), (traj_neighbour.shape[0], self.agent_zone.shape[0])) # (2,)
            # negSampleNonSqueezed: (number of neighbours x directions x coordinates) --> for instance: 3 x 9 x 2
            #                            3 x 1 x 2                     1 x 9 x 2                (2,)
            negSampleNonSqueezed = traj_neighbour[:, None, :] + self.agent_zone[None, :, :] + noise_neg

            # negSampleSqueezed: (number of neighbours * directions x coordinates) --> for instance: 27 x 2
            negSampleSqueezed = negSampleNonSqueezed.reshape((-1, negSampleNonSqueezed.shape[2]))

            # Filling only the first part in the second dimension of sample_neg (leaving the rest as NaN values)
            sample_neg[i, 0:negSampleSqueezed.shape[0], :] = negSampleSqueezed

        # negative sample for everyone
        # the position          # the direction to look around     #some noise
        # sample_neg = gt_future[:,:,None,:] + self.agent_zone[None, None, :, :] + np.random.multivariate_normal([0,0], np.array([[c_e, 0], [0, c_e]]))

        # -----------------------------------------------------
        #       Remove negatives that are too hard (optional)
        # -----------------------------------------------------

        # -----------------------------------------------------
        #       Remove negatives that are too easy (optional)
        # -----------------------------------------------------

        sample_pos = sample_pos.float()
        sample_neg = torch.tensor(sample_neg).float()
        return sample_pos, sample_neg


class EventEncoder(nn.Module):
    """
        Event encoder that maps an sampled event (location & time) to the embedding space
    """

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
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, head_dim)
        )

    def forward(self, state, time):
        emb_state = self.spatial(state)
        emb_time = self.temporal(time)
        out = self.encoder(torch.cat([emb_time, emb_state], axis=-1))
        return out


class SpatialEncoder(nn.Module):
    """
        Spatial encoder that maps a sampled location to the embedding space
    """

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
    """
        Nonlinear projection head that maps the extracted motion features to the embedding space
    """

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
    """
        Plot raw trajectories
    """
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