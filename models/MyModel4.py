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
    def __init__(self, input_size, num_layers, heads, dropout, use_gpu, small_batch=256):
        super(TransformerEncoder, self).__init__()
        # encoder_layer = nn.TransformerEncoderLayer(input_size, dim_feedforward=input_size, nhead=heads,
        #                                            batch_first=True, dropout=dropout, norm_first=True)
        # self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.q_linear = nn.Linear(input_size, input_size)
        self.k_linear = nn.Linear(input_size, input_size)
        self.v_linear = nn.Linear(input_size, input_size)
        self.feedforward = nn.Linear(input_size, input_size)
        # self.relu = nn.ReLU(inplace=True)
        self.use_gpu = use_gpu
        self.small_batch = small_batch

    def forward(self, x):
        """
        Args:
            x: [n_particles, input_size]
            x_his: [n_particles, sequence_length, input_size]
        Returns:
            [n_particles, output_size]
        """

        n_particles = x.size(0)

        q = self.q_linear(x[-1:])
        k = self.k_linear(x[:-1])
        v = self.v_linear(x[:-1])

        a = F.softmax(torch.mm(q, k.T), dim=1)  # [particles_n, particles_n]

        va = a.view(n_particles - 1, 1) * v.view(n_particles - 1, -1)

        va = va.sum(0)

        o = self.feedforward(va)

        return o


class MyModel4(nn.Module):
    def __init__(self, args, stat, phases_dict, residual=False, use_gpu=False):

        super(MyModel4, self).__init__()

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

        # (1) particle attr (2) state
        self.particle_encoder_list = nn.ModuleList()
        for i in range(args.n_stages):
            # print(attr_dim + state_dim * 2)
            self.particle_encoder_list.append(
                ParticleEncoder(attr_dim + state_dim * 2, nf_particle, nf_effect))

        # (1) sender attr (2) receiver attr (3) state receiver (4) state_diff (5) relation attr
        self.relation_encoder_list = nn.ModuleList()
        for i in range(args.n_stages):
            self.relation_encoder_list.append(RelationEncoder(
                2 * attr_dim + 4 * state_dim + relation_dim,
                nf_relation, nf_relation))

        # (1) relation encode (2) sender effect (3) receiver effect
        self.relation_propagator_list = nn.ModuleList()
        for i in range(args.n_stages):
            self.relation_propagator_list.append(Propagator(nf_relation + 2 * nf_effect, nf_effect))

        # (1) particle encode (2) particle effect
        self.particle_propagator_list = nn.ModuleList()
        for i in range(args.n_stages):
            self.particle_propagator_list.append(Propagator(2 * nf_effect, nf_effect, self.residual))

        num_layers = args.trans_num_layers
        num_heads = args.trans_num_heads
        dropout = args.trans_dropout

        self.attention = TransformerEncoder(nf_effect, num_layers, num_heads, dropout, use_gpu)

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

        # calculate particle encoding
        if self.use_gpu:
            particle_effect = Variable(torch.zeros((attr.size(0), self.nf_effect)).cuda())
        else:
            particle_effect = Variable(torch.zeros((attr.size(0), self.nf_effect)))

        # add offset to center-of-mass for rigids to attr
        if self.use_gpu:
            offset = Variable(torch.zeros((attr.size(0), state.size(1))).cuda())
        else:
            offset = Variable(torch.zeros((attr.size(0), state.size(1))))

        for i in range(len(instance_idx) - 1):
            st, ed = instance_idx[i], instance_idx[i + 1]
            if phases_dict['material'][i] == 'rigid':
                c = torch.mean(state[st:ed], dim=0)
                offset[st:ed] = state[st:ed] - c
        attr = torch.cat([attr, offset], 1)

        n_stage = len(Rr)
        assert (len(rels_types) == n_stage)

        # print([rels_type in self.n_stages_types for rels_type in rels_types])

        assert (False not in [rels_type in self.n_stages_types for rels_type in rels_types])
        # compute features for all nodes first

        attr_state = torch.cat([attr, state], 1)

        root_leaf_idx_dict = {node_r_idx[idx_list_i].item(): node_r_idx[idx_list_i + 2] for idx_list_i in
                              range(1, len(node_r_idx), 3)}

        for stage_id in range(self.n_stages):
            """
            node_r_idx, node_s_idx: global idx
            Rr: #within_group_nodes x # rels, place 1/0 at location where there is relations
            """
            stage_name = self.n_stages_types[stage_id]
            if verbose:
                print("=== Stage", stage_id, ":", stage_name)

            rel_idx_of_stage = [i for i, x in enumerate(rels_types) if x == stage_name]
            if len(rel_idx_of_stage) == 0:
                continue

            attr_state_rs = []
            attr_state_ss = []
            attr_state_r_rels = []
            attr_state_s_rels = []
            node_r_idxs = []
            node_s_idxs = []
            Rrp_global_idxs = []
            Rsp_global_idxs = []
            Rr_mat_idx = []
            Ras = []

            current_rel_startidx = 0
            current_r_startidx = 0

            for s in rel_idx_of_stage:
                Rrp = Rr[s].t()
                Rsp = Rs[s].t()
                Rr_idx = Rr_idxs[s]

                # receiver_attr, sender_attr
                attr_state_r = attr_state[node_r_idx[s]]
                attr_state_s = attr_state[node_s_idx[s]]
                # print("Rrp-attr_state_r", Rrp.shape, attr_state_r.shape)
                attr_state_r_rel = Rrp.mm(attr_state_r)  ## (#rels x #group_nodes) x (#group_nodes x #attr_dim)
                attr_state_s_rel = Rsp.mm(attr_state_s)

                Rrp_global_idx = Rrp.mv(torch.from_numpy(node_r_idx[s].astype(np.float32)).cuda())
                Rsp_global_idx = Rsp.mv(torch.from_numpy(node_s_idx[s].astype(np.float32)).cuda())
                nrels = Rrp.shape[0]
                nrs = node_r_idx[s].shape[0]
                global_Rr_idx = Rr_idx + torch.from_numpy(
                    np.array([[current_r_startidx], [current_rel_startidx]]).astype(np.int32)).cuda()
                Rr_mat_idx.append(global_Rr_idx)
                # range(current_rel_startidx, current_rel_startidx + nrels)
                current_rel_startidx += nrels
                current_r_startidx += nrs

                attr_state_rs.append(attr_state_r)
                attr_state_ss.append(attr_state_s)
                attr_state_r_rels.append(attr_state_r_rel)
                attr_state_s_rels.append(attr_state_s_rel)
                node_r_idxs.append(node_r_idx[s])
                node_s_idxs.append(node_s_idx[s])
                Rrp_global_idxs.append(Rrp_global_idx)  # .cpu().numpy().astype(int)
                Rsp_global_idxs.append(Rsp_global_idx)
                Ras.append(Ra[s])
            s = 10000000

            attr_state_rs = torch.cat(attr_state_rs, axis=0)
            attr_state_ss = torch.cat(attr_state_ss, axis=0)
            attr_state_r_rels = torch.cat(attr_state_r_rels, axis=0)
            attr_state_s_rels = torch.cat(attr_state_s_rels, axis=0)
            node_r_idxs = np.concatenate(node_r_idxs)
            node_s_idxs = np.concatenate(node_s_idxs)
            Rrp_global_idxs = torch.cat(Rrp_global_idxs).cpu().numpy().astype(int)
            Rsp_global_idxs = torch.cat(Rsp_global_idxs).cpu().numpy().astype(int)
            Rr_mat_idx = torch.cat(Rr_mat_idx, axis=1)
            Ras = torch.cat(Ras, axis=0)

            # build a matrix of #rels x #receivers
            nrels = Rrp_global_idxs.size
            nreceivers = node_r_idxs.shape[0]
            Rr_merged = torch.sparse.FloatTensor(
                Rr_mat_idx.type(torch.cuda.LongTensor), torch.ones((nrels)).cuda(), torch.Size([nreceivers, nrels]))

            # Rrs = torch.sparse.FloatTensor(
            #        torch.stack([Rrp_global_idxs, Rsp_global_idxs], axis=0).int, torch.ones((Rrp_global_idxs.size(0))).cuda(), torch.Size([attr.size(0), attr.size(0)]))
            # particle encode
            if verbose:
                print('attr_state_r', attr_state_rs.shape)
            particle_encode = self.particle_encoder_list[stage_id](attr_state_rs)
            # calculate relation encoding
            relation_encode = self.relation_encoder_list[stage_id](
                torch.cat([attr_state_r_rels, attr_state_s_rels, Ras], 1))
            if verbose:
                print("relation encode:", relation_encode.size())

            # use the pstep for the first relationship of type stage_name
            first_idx = rel_idx_of_stage[0]
            psteps = pstep[first_idx]
            for i in range(psteps):
                if verbose:
                    print("pstep", i)
                    print("Receiver index range", np.min(node_r_idxs), np.max(node_r_idxs))
                    print("Sender index range", np.min(node_s_idxs), np.max(node_s_idxs))

                effect_p_r = particle_effect[node_r_idxs]

                receiver_effect = particle_effect[Rrp_global_idxs]
                sender_effect = particle_effect[Rsp_global_idxs]

                # calculate relation effect
                effect_rel = self.relation_propagator_list[stage_id](
                    torch.cat([relation_encode, receiver_effect, sender_effect], 1))

                if verbose:
                    print("relation effect:", effect_rel.size())

                # calculate particle effect by aggregating relation effect
                effect_p_r_agg = Rr_merged.mm(effect_rel)

                # calculate particle effect
                effect_p = self.particle_propagator_list[stage_id](
                    torch.cat([particle_encode, effect_p_r_agg], 1),
                    res=effect_p_r)
                if verbose:
                    print("particle effect:", effect_p.size())

                particle_effect[node_r_idxs] = effect_p

        # LSTM Cell
        # particles_c, particles_h = self.lstm_cell(particles_c, particles_h, particle_effect)
        # if particles_c is None:
        #     particles_c = torch.zeros_like(particle_effect, device='cuda')
        # else:
        #     particles_c = particles_c.detach()

        # particle_effect = self.lstm_h_decoder(torch.cat([particles_c, particle_effect], dim=1))
        # particle_effect = particles_h

        pred = []
        # ex. fliudFall instance_idx[0, 189] means there is only one object state[0:190]
        # ex. boxBath [0, 64, 1024], instance=["cube", "fluid"], material=["rigid", "fluid"]
        # particle effect: 1032 x 200
        # ex. FluidShake: [0, 570], fluid
        for i in range(len(instance_idx) - 1):
            st, ed = instance_idx[i], instance_idx[i + 1]
            root_idx = list(root_leaf_idx_dict.keys())[i]

            if phases_dict['material'][i] == 'rigid':

                attention_particle_effect = self.attention(
                    torch.cat((particle_effect[st:ed], particle_effect[root_idx: root_idx + 1]), 0))
                t = self.rigid_particle_predictor(attention_particle_effect).view(-1)

                R = self.rotation_matrix_from_quaternion(t[:4] + self.quat_offset)
                b = t[4:] * self.std_p

                p_0 = state[st:ed, :3] * self.std_p + self.mean_p
                c = torch.mean(p_0, dim=0)  # center
                p_1 = torch.mm(p_0 - c, R) + b + c
                v = (p_1 - p_0) / self.dt
                pred.append((v - self.mean_v) / self.std_v)
            elif phases_dict['material'][i] in ['fluid', 'cloth']:
                pred.append(self.fluid_particle_predictor(attention_particle_effect))

        pred = torch.cat(pred, 0)

        if verbose:
            print("pred:", pred.size())

        return pred
