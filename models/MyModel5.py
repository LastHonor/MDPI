import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
from torch.nn import functional as F

import ipdb

_st = ipdb.set_trace


### Dynamic Particle Interaction Networks

class RelationEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RelationEncoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_size),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        '''
        Args:
            x: [n_relations, input_size]
        Returns:
            [n_relations, output_size]
        '''
        return self.model(x)


class ParticleEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ParticleEncoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_size),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        '''
        Args:
            x: [n_particles, input_size]
        Returns:
            [n_particles, output_size]
        '''
        # print(x.size())
        return self.model(x)


class Propagator(nn.Module):
    def __init__(self, input_size, output_size, residual=False):
        super(Propagator, self).__init__()

        self.residual = residual

        self.linear = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, res=None):
        '''
        Args:
            x: [n_relations/n_particles, input_size]
        Returns:
            [n_relations/n_particles, output_size]
        '''
        if self.residual:
            x = self.relu(self.linear(x) + res)
        else:
            x = self.relu(self.linear(x))

        return x


class ParticlePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ParticlePredictor, self).__init__()

        self.linear_0 = nn.Linear(input_size, hidden_size)
        self.linear_1 = nn.Linear(hidden_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        '''
        Args:
            x: [n_particles, input_size]
        Returns:
            [n_particles, output_size]
        '''
        x = self.relu(self.linear_0(x))
        x = self.relu(self.linear_1(x))

        return self.linear_2(x)


class TransformerEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, heads, dropout, use_gpu, small_batch=256):
        super(TransformerEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(input_size, dim_feedforward=hidden_size, nhead=heads,
                                                   batch_first=True, dropout=dropout, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # self.q_linear = nn.Linear(input_size, hidden_size)
        # self.k_linear = nn.Linear(input_size, hidden_size)
        # self.v_linear = nn.Linear(input_size, hidden_size)
        # self.feedforward = nn.Linear(hidden_size, hidden_size)
        # self.relu = nn.ReLU(inplace=True)
        self.use_gpu = use_gpu
        self.small_batch = small_batch

    def forward(self, particles_x, root_x):
        """
        Args:
            particles_x: [n_particles, input_size]
            root_x: [1, input_size]
        Returns:
            output: [1, output_size]
            k: [n_particles, input_size]
        """

        # n_particles = particles_x.size(0) + 1

        x = torch.cat((particles_x, root_x), dim=0)
        x = x.unsqueeze(0)
        # q = self.q_linear(x)
        # k = self.k_linear(x)
        # v = self.v_linear(x)
        #
        # score = F.softmax(torch.mm(q, k.T), dim=1)  # [particles_n, particles_n(softmax)]
        #
        # va = score.view(n_particles, n_particles, 1) * v.view(n_particles, 1, -1)
        #
        # va = va.sum(1)
        #
        # o = self.relu(self.feedforward(va))

        o = self.encoder(x)[0]

        return o


class TransformerInteractor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, heads, dropout, use_gpu, small_batch=256):
        super(TransformerInteractor, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(input_size, dim_feedforward=hidden_size, nhead=heads,
                                                   batch_first=True, dropout=dropout, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # self.q_linear = nn.Linear(input_size, hidden_size)
        # self.k_linear = nn.Linear(input_size, hidden_size)
        # self.v_linear = nn.Linear(input_size, hidden_size)
        # self.feedforward = nn.Linear(hidden_size, hidden_size)
        # self.relu = nn.ReLU(inplace=True)
        self.use_gpu = use_gpu
        self.small_batch = small_batch

    def forward(self, x):
        """
        Args:
            x: [n_object, input_size]
        Returns:
            output: [n_object, output_size]
        """
        x = x.unsqueeze(0)
        # n_object = x.size(0)
        #
        # q = self.q_linear(x)
        # k = self.k_linear(x)
        # v = self.v_linear(x)
        #
        # score = F.softmax(torch.mm(q, k.T), dim=1)  # [n_object, n_object]
        #
        # va = score.view(n_object, n_object, 1) * v.view(n_object, 1, -1)
        #
        # va = va.sum(1)
        #
        # o = self.feedforward(va)

        o = self.encoder(x).squeeze(0)

        return o


class FluidDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, heads, dropout, use_gpu, small_batch=256):
        super(FluidDecoder, self).__init__()
        # encoder_layer = nn.TransformerEncoderLayer(input_size, dim_feedforward=input_size, nhead=heads,
        #                                            batch_first=True, dropout=dropout, norm_first=True)
        # self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.q_linear = nn.Linear(input_size * 2, 1)
        self.v_linear = nn.Linear(input_size, hidden_size)
        self.feedforward = nn.Linear(hidden_size, hidden_size)
        # self.relu = nn.ReLU(inplace=True)
        self.use_gpu = use_gpu
        self.small_batch = small_batch

    def forward(self, particles_k, particles_v, root_x):
        """
        Args:
            particles_k: [n_particles, input_size]
            particles_v: [n_particles, input_size]
            root_x: [1, input_size]
        Returns:
            output: [1, output_size]
            k: [n_particles, input_size]
        """

        n_particles = particles_k.size(0)
        input_size = root_x.size(1)

        x = torch.broadcast_to(root_x, (n_particles, input_size))
        x = torch.cat((x, particles_k), 1)

        scale = self.q_linear(x)  # (n_particles, 1)

        particles_v = self.v_linear(particles_v)

        o = particles_v * scale

        o = self.feedforward(o)

        return o


class MyModel5(nn.Module):
    def __init__(self, args, stat, phases_dict, residual=False, use_gpu=False):

        super(MyModel5, self).__init__()

        self.args = args

        state_dim = args.state_dim
        attr_dim = args.attr_dim
        relation_dim = args.relation_dim
        nf_particle = args.nf_particle
        nf_relation = args.nf_relation
        nf_effect = args.nf_effect

        self.nf_effect = args.nf_effect

        self.stat = stat
        self.use_gpu = use_gpu
        self.residual = residual
        self.quat_offset = torch.FloatTensor([1., 0., 0., 0.])
        if use_gpu:
            self.quat_offset = self.quat_offset.cuda()

        self.n_stages = args.n_stages
        self.n_stages_types = args.n_stages_types
        self.dt = args.dt * args.training_fpt

        if use_gpu:

            # make sure std_v and std_p is ok
            for item in range(2):
                for idx in range(3):
                    if stat[item][idx, 1] == 0:
                        stat[item][idx, 1] = 1

            self.pi = Variable(torch.FloatTensor([np.pi])).cuda()
            self.dt = Variable(torch.FloatTensor([self.dt])).cuda()
            self.mean_v = Variable(torch.FloatTensor(stat[1][:, 0])).cuda()  # velocity
            self.std_v = Variable(torch.FloatTensor(stat[1][:, 1])).cuda()  # velocity
            self.mean_p = Variable(torch.FloatTensor(stat[0][:3, 0])).cuda()  # position
            self.std_p = Variable(torch.FloatTensor(stat[0][:3, 1])).cuda()  # position
        else:
            self.pi = Variable(torch.FloatTensor([np.pi]))
            self.dt = Variable(torch.FloatTensor(self.dt))
            self.mean_v = Variable(torch.FloatTensor(stat[1][:, 0]))
            self.std_v = Variable(torch.FloatTensor(stat[1][:, 1]))
            self.mean_p = Variable(torch.FloatTensor(stat[0][:3, 0]))
            self.std_p = Variable(torch.FloatTensor(stat[0][:3, 1]))

        # # (1) particle attr (2) state
        # self.particle_encoder_list = nn.ModuleList()
        # for i in range(args.n_stages):
        #     # print(attr_dim + state_dim * 2)
        #     self.particle_encoder_list.append(
        #         ParticleEncoder(attr_dim + state_dim * 2, nf_particle, nf_effect))
        #
        # # (1) sender attr (2) receiver attr (3) state receiver (4) state_diff (5) relation attr
        # self.relation_encoder_list = nn.ModuleList()
        # for i in range(args.n_stages):
        #     self.relation_encoder_list.append(RelationEncoder(
        #         2 * attr_dim + 4 * state_dim + relation_dim,
        #         nf_relation, nf_relation))
        #
        # # (1) relation encode (2) sender effect (3) receiver effect
        # self.relation_propagator_list = nn.ModuleList()
        # for i in range(args.n_stages):
        #     self.relation_propagator_list.append(Propagator(nf_relation + 2 * nf_effect, nf_effect))
        #
        # # (1) particle encode (2) particle effect
        # self.particle_propagator_list = nn.ModuleList()
        # for i in range(args.n_stages):
        #     self.particle_propagator_list.append(Propagator(2 * nf_effect, nf_effect, self.residual))

        num_layers = args.trans_num_layers
        num_heads = args.trans_num_heads
        dropout = args.trans_dropout

        self.particle_encoder = ParticleEncoder(state_dim, nf_particle, nf_effect)
        # self.tran_encoder = TransformerEncoder(nf_effect, nf_effect, num_layers, num_heads, dropout, use_gpu)
        self.tran_interactor = TransformerInteractor(nf_effect, nf_effect, num_layers, num_heads, dropout, use_gpu)
        # self.tran_decoder = FluidDecoder(nf_effect, nf_effect, num_layers, num_heads, dropout, use_gpu)

        # self.lstm_h_decoder = nn.Sequential(
        #     nn.Linear(nf_effect * 2, nf_effect),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(nf_effect, nf_effect),
        #     nn.ReLU(inplace=True)
        # )

        # (1) set particle effect
        self.rigid_particle_predictor = ParticlePredictor(nf_effect, nf_effect, 7)  # predict rigid motion
        self.fluid_particle_predictor = ParticlePredictor(nf_effect, nf_effect, args.position_dim)

    def save_local(self, args, path_name):
        def foo(args):
            return locals()

        output = foo(args)
        output["pi"] = self.pi.cpu().numpy()
        output["dt"] = self.dt.cpu().numpy()
        output["mean_v"] = self.mean_v.cpu().numpy()
        output["std_v"] = self.std_v.cpu().numpy()
        output["mean_p"] = self.mean_p.cpu().numpy()
        output["std_p"] = self.std_p.cpu().numpy()

        with open(path_name, "wb") as f:
            pickle.dump(output, f)

    def load_local(self, path_name):
        with open(path_name, "rb") as f:
            output = pickle.load(f)
        if self.use_gpu:
            self.pi = Variable(torch.FloatTensor(output["pi"])).cuda()
            self.dt = Variable(torch.FloatTensor(output["dt"])).cuda()
            self.mean_v = Variable(torch.FloatTensor(output["mean_v"])).cuda()  # velocity
            self.std_v = Variable(torch.FloatTensor(output["std_v"])).cuda()  # velocity
            self.mean_p = Variable(torch.FloatTensor(output["mean_p"])).cuda()  # position
            self.std_p = Variable(torch.FloatTensor(output["std_p"])).cuda()  # position
        else:
            self.pi = Variable(torch.FloatTensor(output["pi"]))
            self.dt = Variable(torch.FloatTensor(output["dt"]))
            self.mean_v = Variable(torch.FloatTensor(output["mean_v"]))  # velocity
            self.std_v = Variable(torch.FloatTensor(output["std_v"]))  # velocity
            self.mean_p = Variable(torch.FloatTensor(output["mean_p"]))  # position
            self.std_p = Variable(torch.FloatTensor(output["std_p"]))  # position
        return vars(output["args"])

    def rotation_matrix_from_quaternion(self, params):
        # params dim - 4: w, x, y, z

        if self.use_gpu:
            one = Variable(torch.ones(1, 1)).cuda()
            zero = Variable(torch.zeros(1, 1)).cuda()
        else:
            one = Variable(torch.ones(1, 1))
            zero = Variable(torch.zeros(1, 1))

        # multiply the rotation matrix from the right-hand side
        # the matrix should be the transpose of the conventional one

        # Reference
        # http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/index.htm

        params = params / torch.norm(params)
        w, x, y, z = params[0].view(1, 1), params[1].view(1, 1), params[2].view(1, 1), params[3].view(1, 1)

        rot = torch.cat((
            torch.cat((one - y * y * 2 - z * z * 2, x * y * 2 + z * w * 2, x * z * 2 - y * w * 2), 1),
            torch.cat((x * y * 2 - z * w * 2, one - x * x * 2 - z * z * 2, y * z * 2 + x * w * 2), 1),
            torch.cat((x * z * 2 + y * w * 2, y * z * 2 - x * w * 2, one - x * x * 2 - y * y * 2), 1)), 0)

        return rot

    def forward(self, attr, state, Rr, Rs, Ra, Rr_idxs, n_particles, node_r_idx, node_s_idx, pstep, rels_types,
                instance_idx, phases_dict, verbose=0):
        """
        attr: #nodes x attr_dim
        state: #nodes x state_dim


        """

        # # calculate particle encoding
        # if self.use_gpu:
        #     particle_effect = Variable(torch.zeros((attr.size(0), self.nf_effect)).cuda())
        # else:
        #     particle_effect = Variable(torch.zeros((attr.size(0), self.nf_effect)))
        #
        # # add offset to center-of-mass for rigids to attr
        # if self.use_gpu:
        #     offset = Variable(torch.zeros((attr.size(0), state.size(1))).cuda())
        # else:
        #     offset = Variable(torch.zeros((attr.size(0), state.size(1))))
        #
        # for i in range(len(instance_idx) - 1):
        #     st, ed = instance_idx[i], instance_idx[i + 1]
        #     if phases_dict['material'][i] == 'rigid':
        #         c = torch.mean(state[st:ed], dim=0)
        #         offset[st:ed] = state[st:ed] - c
        # attr = torch.cat([attr, offset], 1)
        #
        # n_stage = len(Rr)
        # assert (len(rels_types) == n_stage)
        #
        # # print([rels_type in self.n_stages_types for rels_type in rels_types])
        #
        # assert (False not in [rels_type in self.n_stages_types for rels_type in rels_types])
        # # compute features for all nodes first
        #
        # attr_state = torch.cat([attr, state], 1)

        encoded_particles = self.particle_encoder(state)

        particle_effect = self.tran_interactor(encoded_particles)

        pred = []
        # ex. fliudFall instance_idx[0, 189] means there is only one object state[0:190]
        # ex. boxBath [0, 64, 1024], instance=["cube", "fluid"], material=["rigid", "fluid"]
        # particle effect: 1032 x 200
        # ex. FluidShake: [0, 570], fluid
        for i in range(len(instance_idx) - 1):
            st, ed = instance_idx[i], instance_idx[i + 1]

            if phases_dict['material'][i] == 'rigid':
                t = self.rigid_particle_predictor(torch.mean(particle_effect[st:ed], 0)).view(-1)

                R = self.rotation_matrix_from_quaternion(t[:4] + self.quat_offset)
                b = t[4:] * self.std_p

                p_0 = state[st:ed, :3] * self.std_p + self.mean_p
                c = torch.mean(p_0, dim=0)  # center
                p_1 = torch.mm(p_0 - c, R) + b + c
                v = (p_1 - p_0) / self.dt
                pred.append((v - self.mean_v) / self.std_v)
            elif phases_dict['material'][i] in ['fluid', 'cloth']:
                pred.append(self.fluid_particle_predictor(particle_effect[st:ed]))

        pred = torch.cat(pred, 0)

        if verbose:
            print("pred:", pred.size())

        return pred
