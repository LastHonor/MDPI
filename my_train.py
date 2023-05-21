import os
import sys
import random
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

import multiprocessing as mp

import tqdm
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import ipdb

st = ipdb.set_trace

from models import DPINet, DPINet2, GNS, GNSRigid, GNSRigidH, \
    MyModel0, MyModel1, MyModel2, MyModel3, MyModel4, MyModel5, MyGNSRigid
from my_data import PhysicsFleXDataset, collate_fn

from utils import count_parameters, get_query_dir

TEST = False
MEMORY_LENGTH = 10


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pstep', type=int, default=2)
    parser.add_argument('--n_rollout', type=int, default=0)
    parser.add_argument('--time_step', type=int, default=0)
    parser.add_argument('--time_step_clip', type=int, default=0)
    parser.add_argument('--dt', type=float, default=1. / 60.)
    parser.add_argument('--training_fpt', type=float, default=1)

    parser.add_argument('--trans_num_layers', type=int, default=1)
    parser.add_argument('--trans_num_heads', type=int, default=4)
    parser.add_argument('--trans_dropout', type=float, default=0.0)

    parser.add_argument('--nf_relation', type=int, default=300)
    parser.add_argument('--nf_particle', type=int, default=200)
    parser.add_argument('--nf_effect', type=int, default=200)
    parser.add_argument('--model_name', default='DPINet2')
    parser.add_argument('--floor_cheat', type=int, default=0)
    parser.add_argument('--env', default='')
    parser.add_argument('--train_valid_ratio', type=float, default=0.9)
    parser.add_argument('--outf', default='files')
    parser.add_argument('--dataf', default='data')
    parser.add_argument('--statf', default="")
    parser.add_argument('--noise_std', type=float, default='0')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--gen_stat', type=int, default=0)

    parser.add_argument('--subsample_particles', type=int, default=1)

    parser.add_argument('--log_per_iter', type=int, default=1000)
    parser.add_argument('--ckp_per_epoch', type=int, default=1)
    parser.add_argument('--eval', type=int, default=0)
    parser.add_argument('--augment_worldcoord', type=int, default=0)

    parser.add_argument('--verbose_data', type=int, default=0)
    parser.add_argument('--verbose_model', type=int, default=0)

    parser.add_argument('--n_instance', type=int, default=0)
    parser.add_argument('--n_stages', type=int, default=0)
    parser.add_argument('--n_his', type=int, default=0)

    parser.add_argument('--n_epoch', type=int, default=100)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--forward_times', type=int, default=2)

    parser.add_argument('--resume_epoch', type=int, default=0)
    parser.add_argument('--resume_iter', type=int, default=0)
    parser.add_argument("--resume_best_loss", type=float, default=np.inf)
    parser.add_argument("--resume_eval_first", type=float, default=0)

    # shape state:
    # [x, y, z, x_last, y_last, z_last, quat(4), quat_last(4)]
    parser.add_argument('--shape_state_dim', type=int, default=14)

    # object attributes:
    parser.add_argument('--attr_dim', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)

    # object state:
    parser.add_argument('--state_dim', type=int, default=0)
    parser.add_argument('--position_dim', type=int, default=0)

    # relation attr:
    parser.add_argument('--relation_dim', type=int, default=0)

    parser.add_argument('--memory_release', type=int, default=0)

    args = parser.parse_args()

    phases_dict = dict()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    data_root = get_query_dir("dpi_data_dir")
    out_root = get_query_dir("out_dir")
    if args.env == "TDWdominoes":
        args.n_rollout = None  # how many data
        # don't use, determined by data

        # object states:
        # [x, y, z, xdot, ydot, zdot]  [3 positions, 3 velocities]
        args.state_dim = 6
        args.position_dim = 3
        args.dt = 0.01

        # object attr:
        # [rigid, fluid, root_0]
        args.attr_dim = 3

        # relation attr:
        # [none]
        args.relation_dim = 1

        args.n_instance = -1
        args.time_step = 301  # ??
        args.time_step_clip = 0
        args.n_stages = 4
        args.n_stages_types = ["leaf-leaf", "leaf-root", "root-root", "root-leaf"]

        args.neighbor_radius = 0.08

        phases_dict = dict()  # load from data
        # ["root_num"] = [[]]
        # phases_dict["instance"] = ["fluid"]
        # phases_dict["material"] = ["fluid"]
        args.outf = args.outf.strip()

        args.outf = os.path.join(out_root, 'dump/dump_TDWdominoes/' + args.outf).replace("\\", "/")


    else:
        raise AssertionError("Unsupported env")

    writer = SummaryWriter(os.path.join(args.outf, "log").replace("\\", "/"))
    data_root = os.path.join(data_root, "train").replace("\\", "/")
    args.data_root = data_root
    if "," in args.dataf:
        # list of folder
        args.dataf = [os.path.join(data_root, tmp.strip()).replace("\\", "/") for tmp in args.dataf.split(",") if
                      tmp != ""]

    else:
        args.dataf = args.dataf.strip()
        if "/" in args.dataf:
            args.dataf = 'data/' + args.dataf
        else:  # only prefix
            args.dataf = 'data/' + args.dataf + '_' + args.env
        os.system('mkdir ' + args.dataf.replace("/", "\\"))
    os.system('mkdir ' + args.outf.replace("/", "\\"))

    # generate data
    datasets = {phase: PhysicsFleXDataset(
        args, phase, phases_dict, args.verbose_data) for phase in ['train', 'valid']}

    for phase in ['train', 'valid']:
        datasets[phase].load_data(args.env)

    use_gpu = torch.cuda.is_available()
    assert use_gpu

    # import torch.multiprocessing

    # torch.multiprocessing.set_sharing_strategy('file_system')
    dataloaders = {x: torch.utils.data.DataLoader(
        datasets[x], batch_size=args.batch_size,
        shuffle=True if x == 'train' else False,
        num_workers=args.num_workers,
        collate_fn=collate_fn)
        for x in ['train', 'valid']}

    # define propagation network
    if args.env == "TDWdominoes":
        if args.model_name == "DPINet":
            model = DPINet(args, datasets['train'].stat, phases_dict, residual=True, use_gpu=use_gpu)
        elif args.model_name == "DPINet2":
            """
            original DPI, but don't apply different fc for different objects?
            originla dpi only has one object, so they are trying to apply different fc for
            different type of relation.
            But I have several objects, do many relations are actually the same
            to do: add relationship type
            """
            model = DPINet2(args, datasets['train'].stat, phases_dict, residual=True, use_gpu=use_gpu)
        elif args.model_name == "MyModel1":
            "LSTM without shortcut"
            model = MyModel1(args, datasets['train'].stat, phases_dict, residual=True, use_gpu=use_gpu)
        elif args.model_name == "MyModel2":
            "LSTM with shortcut"
            model = MyModel2(args, datasets['train'].stat, phases_dict, residual=True, use_gpu=use_gpu)
        elif args.model_name == "MyModel3":
            "Transformer Encoder"
            model = MyModel3(args, datasets['train'].stat, phases_dict, residual=True, use_gpu=use_gpu)
        elif args.model_name == "MyModel4":
            ""
            model = MyModel4(args, datasets['train'].stat, phases_dict, residual=True, use_gpu=use_gpu)
        elif args.model_name == "MyModel0":
            ""
            model = MyModel0(args, datasets['train'].stat, phases_dict, residual=True, use_gpu=use_gpu)
        elif args.model_name == "MyModel5":
            ""
            model = MyModel5(args, datasets['train'].stat, phases_dict, residual=True, use_gpu=use_gpu)
        elif args.model_name == "GNS":
            """
            deep mind model, hierarchy, use only relative information
            """
            args.pstep = 10
            args.n_stages = 1
            args.noise_std = 3e-4
            args.n_stages_types = ["leaf-leaf"]

            model = GNS(args, datasets['train'].stat, phases_dict, residual=True, use_gpu=use_gpu)

        elif args.model_name == "GNSRigid":
            """
            deep mind model, hierarchy, use only relative information
            """
            args.pstep = 3
            args.n_stages = 1
            args.noise_std = 3e-4
            args.n_stages_types = ["leaf-leaf"]

            model = GNSRigid(args, datasets['train'].stat, phases_dict, residual=True, use_gpu=use_gpu)

        elif args.model_name == "MyGNSRigid":
            """
            deep mind model, hierarchy, use only relative information
            """
            args.pstep = 3
            args.n_stages = 1
            args.noise_std = 3e-4
            args.n_stages_types = ["leaf-leaf"]

            model = MyGNSRigid(args, datasets['train'].stat, phases_dict, residual=True, use_gpu=use_gpu)

        elif args.model_name == "GNSRigidH":
            """
            DPI2, use only relative information
            """
            args.noise_std = 3e-4

            model = GNSRigidH(args, datasets['train'].stat, phases_dict, residual=True, use_gpu=use_gpu)
        else:
            raise ValueError(f"no such model {args.model_name} for env {args.env}")
    else:
        model = DPINet(args, datasets['train'].stat, phases_dict, residual=True, use_gpu=use_gpu)
    if use_gpu:
        model = model.cuda()
    print("Number of parameters: %d" % count_parameters(model))

    opt_mode = "dpi"

    if opt_mode == "dpi":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
        # optimizer = optim.SGD(model.parameters(), lr=args.lr/100)
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=10, verbose=True)
    elif opt_mode == "gns":
        min_lr = 1e-6
        lr = tf.train.exponential_decay(optimizer,
                                        decay_steps=int(5e6),
                                        decay_rate=0.1) + min_lr
        optimizer = tf.train.AdamOptimizer(learning_rate=args.lr - min_lr)

    if args.resume_epoch > 0 or args.resume_iter > 0:
        # load local parameters
        args_load = model.load_local(os.path.join(args.outf, "args_stat.pkl").replace("\\", "/"))
        args_current = vars(args)

        exempt_list = ["dataf", "lr", "num_workers", "resume_epoch", "resume_iter", "resume_best_loss", "resume_eval_first"]

        for key in args_load:
            if key in exempt_list:
                continue

            assert (args_load[key] == args_current[
                key]), f"{key} is mismatched in loaded args and current args: {args_load[key]} vs {args_current[key]}"

        # check args_load
        model_path = os.path.join(args.outf,
                                  'net_epoch_ckp.pth').replace(
            "\\", "/")
        print("Loading saved ckp from %s" % model_path)
        # torch.save(model.state_dict(), model_path)
        # checkpoint = torch.load(model_path)
        # model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # schedular.load_state_dict(checkpoint['scheduler_state_dict'])

        checkpoint = torch.load(model_path)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            model.load_state_dict(torch.load(model_path))

    # criterion
    criterionMSE = nn.MSELoss()

    # optimizer

    optimizer.zero_grad()
    if use_gpu:
        model = model.cuda()
        # criterionMSE = criterionMSE.cuda()

    # save args, stat
    model.save_local(args, os.path.join(args.outf, "args_stat.pkl").replace("\\", "/"))

    st_epoch = args.resume_epoch if args.resume_epoch > 0 else 0
    best_valid_loss = args.resume_best_loss
    train_iter = 0
    current_loss = 0

    max_nparticles = 0

    # dummy_log_file_name = \
    #     f'dummyLog/{args.model_name}-{args.dataf[0].split("/")[-1]}-{args.nf_relation}-{args.nf_particle}-{args.trans_num_layers}-{args.trans_num_heads}.csv'
    if args.outf.lower().split("/")[-1] == 'draft':
        dummy_log_file_name = \
            f'dummyLog/{args.model_name}-{args.dataf[0].split("/")[-1]}-draft.csv'
    else:
        dummy_log_file_name = \
            f'dummyLog/{args.model_name}-{args.dataf[0].split("/")[-1]}.csv'
    with open(dummy_log_file_name, "w") as dummy_log_file:
        dummy_log_file.write('total_loss_tr,half_loss1_tr,half_loss2_tr,'
                             'total_loss_va,half_loss1_va,half_loss2_va,second_per_sample\n')

    # for i, data in enumerate(dataloaders[phase]):
    #     attr_list, state_list, rels_list, n_particles_list, n_shapes_list, instance_idx_list, label_list, \
    #     phases_dict_current = data
    #     tensor_list = []
    #     data_list = []
    #     sequence_length = len(attr_list)
    #     # sequence_length = 1
    #
    #     for time_step_i in range(sequence_length):
    #         attr, state, rels, n_particles, n_shapes, instance_idx, label = \
    #             attr_list[time_step_i], state_list[time_step_i], rels_list[time_step_i], \
    #             n_particles_list[time_step_i], n_shapes_list[time_step_i], \
    #             instance_idx_list[time_step_i], label_list[time_step_i]
    #
    #         if n_particles > max_nparticles:
    #             max_nparticles = n_particles
    #
    #         Ra, node_r_idx, node_s_idx, pstep, rels_types = rels[3], rels[4], rels[5], rels[6], rels[7]
    #
    #         Rr, Rs, Rr_idxs = [], [], []
    #         for j in range(len(rels[0])):
    #             Rr_idx, Rs_idx, values = rels[0][j], rels[1][j], rels[2][j]
    #
    #             Rr_idxs.append(Rr_idx)
    #             Rr.append(torch.sparse.FloatTensor(
    #                 Rr_idx, values, torch.Size([node_r_idx[j].shape[0], Ra[j].size(0)])))
    #
    #             Rs.append(torch.sparse.FloatTensor(
    #                 Rs_idx, values, torch.Size([node_s_idx[j].shape[0], Ra[j].size(0)])))
    #
    #         tensor = [attr, state, Rr, Rs, Ra, Rr_idxs, label]
    #         data = [n_particles, node_r_idx, node_s_idx, pstep, rels_types, instance_idx]
    #         tensor_list.append(tensor)
    #         data_list.append(data)
    #
    #     # print("data prep:", time.time() - start_time)
    #     if use_gpu:
    #         for time_step_i in range(len(tensor_list)):
    #             for d in range(len(tensor_list[time_step_i])):
    #                 if type(tensor_list[time_step_i][d]) == list:
    #                     for t in range(len(tensor_list[time_step_i][d])):
    #                         tensor_list[time_step_i][d][t] = Variable(tensor_list[time_step_i][d][t].cuda(),
    #                                                                   requires_grad=False)
    #                 else:
    #                     tensor_list[time_step_i][d] = Variable(tensor_list[time_step_i][d].cuda(), requires_grad=False)
    #     else:
    #         for time_step_i in range(len(tensor_list)):
    #             for d in range(len(tensor_list[time_step_i])):
    #                 if type(tensor_list[time_step_i][d]) == list:
    #                     for t in range(len(tensor_list[time_step_i][d])):
    #                         tensor_list[time_step_i][d][t] = Variable(tensor_list[time_step_i][d][t], requires_grad=False)
    #                 else:
    #                     tensor_list[time_step_i][d] = Variable(tensor_list[time_step_i][d], requires_grad=False)
    #     break
    # for epoch in range(0, 200):
    all_time_cost = 0
    iter_count = 0
    total_iters = (args.n_epoch - st_epoch) * (len(dataloaders['train']) + len(dataloaders['valid']))
    for epoch in range(st_epoch, args.n_epoch):
        phases = ['train', 'valid'] if args.eval == 0 else ['valid']
        # for phase in phases[:1]:
        for phase in phases:
            if args.resume_eval_first and epoch == st_epoch and phase == 'train':
                continue
            # import time

            model.train(phase == 'train')

            losses = 0

            # max_length = 0
            # max_n_particles = 0img
            # max_i = 0
            loss_count = 0
            half_loss_count1 = 0
            half_loss_count2 = 0
            total_loss = 0
            half_loss1 = 0
            half_loss2 = 0
            single_epoch_start_time = time.time()
            s_time1 = time.time()

            for i, data in enumerate(dataloaders[phase]):
                # if i <= 625:
                #     continue
                # for _ in range(1):
                iter_count += 1

                attr_list, state_list, rels_list, n_particles_list, n_shapes_list, instance_idx_list, label_list, \
                phases_dict_current = data

                current_loss = 0

                sequence_length = len(attr_list)

                # if sequence_length > max_length:
                #     max_length = sequence_length
                # if len(state_list[0]) > max_n_particles:
                #     max_n_particles = len(state_list[0])
                #     max_i = i
                #
                # if max_n_particles == 7329:
                #     break

                # continue
                tensor_list = []
                data_list = []
                for time_step_i in range(sequence_length):
                    attr, state, rels, n_particles, n_shapes, instance_idx, label = \
                        attr_list[time_step_i], state_list[time_step_i], rels_list[time_step_i], \
                        n_particles_list[time_step_i], n_shapes_list[time_step_i], \
                        instance_idx_list[time_step_i], label_list[time_step_i]

                    # if n_particles > max_nparticles:
                    #     max_nparticles = n_particles
                    # print(n_particles)

                    Ra, node_r_idx, node_s_idx, pstep, rels_types = rels[3], rels[4], rels[5], rels[6], rels[7]

                    Rr, Rs, Rr_idxs = [], [], []
                    for j in range(len(rels[0])):
                        Rr_idx, Rs_idx, values = rels[0][j], rels[1][j], rels[2][j]

                        Rr_idxs.append(Rr_idx)
                        Rr.append(torch.sparse.FloatTensor(
                            Rr_idx, values, torch.Size([node_r_idx[j].shape[0], Ra[j].size(0)])))

                        Rs.append(torch.sparse.FloatTensor(
                            Rs_idx, values, torch.Size([node_s_idx[j].shape[0], Ra[j].size(0)])))

                    tensor = [attr, state, Rr, Rs, Ra, Rr_idxs, label]
                    data = [n_particles, node_r_idx, node_s_idx, pstep, rels_types, instance_idx]
                    tensor_list.append(tensor)
                    data_list.append(data)

                if use_gpu:
                    for time_step_i in range(sequence_length):
                        for d in range(len(tensor_list[time_step_i])):
                            if type(tensor_list[time_step_i][d]) == list:
                                for t in range(len(tensor_list[time_step_i][d])):
                                    tensor_list[time_step_i][d][t] = Variable(tensor_list[time_step_i][d][t].cuda(),
                                                                              requires_grad=False)
                            else:
                                tensor_list[time_step_i][d] = Variable(tensor_list[time_step_i][d].cuda(),
                                                                       requires_grad=False)
                else:
                    for time_step_i in range(sequence_length):
                        for d in range(len(tensor_list[time_step_i])):
                            if type(tensor_list[time_step_i][d]) == list:
                                for t in range(len(tensor_list[time_step_i][d])):
                                    tensor_list[time_step_i][d][t] = Variable(tensor_list[time_step_i][d][t],
                                                                              requires_grad=False)
                            else:
                                tensor_list[time_step_i][d] = Variable(tensor_list[time_step_i][d], requires_grad=False)
                s_time0 = time.time()
                # torch.cuda.empty_cache()

                # if not os.path.exists('pause,txt'):
                #     with open('pause.txt', 'w') as f:
                #         f.write('a')
                wait_time = 0
                while True:
                    try:
                        with open('pause.txt', 'r') as f:
                            pause = f.read()
                        if len(pause) > 0:
                            # print('\rContinue...', end='')
                            break
                    except FileNotFoundError:
                        pass
                    print('\rPausing...', end='')
                    if wait_time == 0:
                        torch.cuda.empty_cache()
                    time.sleep(1)
                    single_epoch_start_time += 1
                    wait_time += 1

                # repeat_times = 1 if args.model_name != "MyModel1" else int(sequence_length / MEMORY_LENGTH)
                # for _ in range(repeat_times):

                particles_c = None
                particles_h = None
                particles_effects = None

                sampled_sequence_index = np.arange(sequence_length)
                # if sequence_length <= MEMORY_LENGTH or args.model_name != "MyModel1" or phase != 'train':
                #     pass
                # else:
                #     start_idx = np.random.randint(sequence_length - MEMORY_LENGTH + 1)
                #     end_idx = start_idx + MEMORY_LENGTH
                #     sampled_sequence_index = sampled_sequence_index[start_idx: end_idx]

                if 'Drape' in [dataf_i.split("/")[-1] for dataf_i in args.dataf]:
                    cut_count = np.random.randint(3, 9)
                    # cut_count = 3
                else:
                    # if 'GNS' in args.model_name:
                    #     cut_count = np.random.randint(15, 51)
                    # else:
                    cut_count = np.random.randint(3, 11)

                with torch.set_grad_enabled(phase == 'train'):
                    # st_time = time.time()
                    # try:
                    for time_step_i in sampled_sequence_index:
                        # print(time_step_i)
                        attr, state, Rr, Rs, Ra, Rr_idxs, label = tensor_list[time_step_i]
                        n_particles, node_r_idx, node_s_idx, pstep, rels_types, instance_idx = data_list[time_step_i]
                        if args.model_name in ["MyModel1", "MyModel2", "MyModel0", "MyGNSRigid"]:
                            predicted, particles_c, particles_h = model(
                                attr, state, Rr, Rs, Ra, Rr_idxs, n_particles,
                                node_r_idx, node_s_idx, pstep, rels_types,
                                instance_idx, phases_dict_current, particles_c, particles_h, args.verbose_model)
                            # if args.model_name in ["MyModel1", 'MyModel0'] and phase == 'train':
                            if phase == 'train':
                                cut_count -= 1
                                if cut_count == 0:
                                    cut_count = np.random.randint(3, 11)
                                    particles_c = particles_c.detach()
                                    particles_h = particles_h.detach()

                        elif args.model_name in ["MyModel3"]:
                            predicted, particles_effects = model(
                                attr, state, Rr, Rs, Ra, Rr_idxs, n_particles,
                                node_r_idx, node_s_idx, pstep, rels_types,
                                instance_idx, phases_dict_current, particles_effects, args.verbose_model)
                        else:
                            predicted = model(
                                attr, state, Rr, Rs, Ra, Rr_idxs, n_particles,
                                node_r_idx, node_s_idx, pstep, rels_types,
                                instance_idx, phases_dict_current, args.verbose_model)
                        # except RuntimeError:
                        #     print(f"bad----n: {len(state)}, len: {sequence_length}")
                        #     break

                        loss = criterionMSE(predicted, label)  # / sequence_length  # / args.forward_times
                        with torch.no_grad():
                            loss_count += 1
                            if phase == 'train':
                                # try:
                                # loss.backward(retain_graph=True)  # i < sequence_length - 1)
                                loss.backward(retain_graph=args.model_name in ["MyModel1", "MyModel2", "MyGNSRigid"] and time_step_i != sampled_sequence_index[-1])  # i < sequence_length - 1)
                                if args.memory_release:
                                    for _ in range(5):
                                        torch.cuda.empty_cache()
                                # except RuntimeError:
                                #     print(f"bad----n: {len(state)}, len: {sequence_length}")
                                #     train_iter -= 1
                                #     break
                            total_loss += np.sqrt(loss.item())
                            current_loss += np.sqrt(loss.item()) / sequence_length
                            if time_step_i < sequence_length / 2:
                                half_loss1 += np.sqrt(loss.item())
                                half_loss_count1 += 1
                            else:
                                half_loss2 += np.sqrt(loss.item())
                                half_loss_count2 += 1
                # current_loss = np.sqrt(loss.item())  # * sequence_length)  # * args.forward_times)
                e_time0 = time.time()

                if phase == 'train':
                    optimizer.step()
                    optimizer.zero_grad()
                time_cost = int(time.time() - single_epoch_start_time)
                time_remain = int(time_cost / (i + 1) * (len(dataloaders[phase]) - (i + 1)))

                print(f'\r{phase}: [{epoch}/{args.n_epoch}][{i}/{len(dataloaders[phase])}], '
                      f'CurrentLoss: {current_loss: .06f}, '
                      f'AverageLoss: {total_loss / loss_count: .06f}, '
                      f'TimeCost: {time_cost: d}s',
                      f'TimeRemain: {time_remain: d}s, '
                      f'TimeCostPerSample: {int(time_cost / (i + 1)): d}s',
                      end=""
                      )

                e_time1 = time.time()
                print(f", GraphCost/TimeCost: {int(e_time0 - s_time0): d}/{int(e_time1 - s_time1): d}", end='')
                s_time1 = time.time()

                if TEST:
                    break
            total_loss = total_loss / loss_count
            half_loss1 = half_loss1 / loss_count
            half_loss2 = half_loss2 / loss_count
            # half_loss1 = half_loss1 / half_loss_count1
            # half_loss2 = half_loss2 / half_loss_count2
            if phase == "train":
                lr = get_lr(optimizer)
                writer.add_scalar(f'lr', lr, epoch)
            # writer.add_histogram(f'{phase}/label_x', label[:, 0], train_iter)
            # writer.add_histogram(f'{phase}/label_y', label[:, 1], train_iter)
            # writer.add_histogram(f'{phase}/label_z', label[:, 2], train_iter)
            # writer.add_histogram(f'{phase}/predicted_x', predicted[:, 0], train_iter)
            # writer.add_histogram(f'{phase}/predicted_y', predicted[:, 1], train_iter)
            # writer.add_histogram(f'{phase}/predicted_z', predicted[:, 2], train_iter)
            writer.add_scalar(f'{phase}/total_loss', total_loss, epoch)
            writer.add_scalar(f'{phase}/half_loss1', half_loss1, epoch)
            writer.add_scalar(f'{phase}/half_loss2', half_loss2, epoch)
            # previous_run_time = time.time()

            if phase == 'train' and epoch % args.ckp_per_epoch == 0:
                torch.save({'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict()},
                           '%s/net_epoch_%d.pth' % (args.outf, epoch))
                torch.save({'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict()},
                           '%s/net_epoch_ckp.pth' % args.outf)

            # print("total time:", time.time() - previous_run_time)
            if phase == 'train':
                scheduler.step(total_loss)
            if total_loss < best_valid_loss and phase == 'valid':
                best_valid_loss = total_loss
                torch.save({'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict()},
                           '%s/net_best.pth' % args.outf)
            # print('%s [%d/%d] Loss: %.4f, Best valid: %.4f' %
            #       (phase, epoch, args.n_epoch, losses, best_valid_loss))
            all_time_cost += int(time.time() - single_epoch_start_time)

            time_cost = all_time_cost
            time_cost_per_sample = int(time_cost / iter_count)
            time_remain = time_cost_per_sample * (total_iters - iter_count)
            print(f'\r{phase}: [{epoch}/{args.n_epoch}], '
                  f'AverageLoss: {total_loss: .06f}, '
                  f'BestValidLoss:{best_valid_loss: .06f}, '
                  f'TimeCost: {time_cost: d}s, '
                  f'TimeRemain: {time_remain: d}s, '
                  f'TimeCostPerSample: {time_cost_per_sample: d}s'
                  )
            with open(dummy_log_file_name, "a") as dummy_log_file:
                if phase == 'train':
                    dummy_log_file.write(f"{total_loss},{half_loss1},{half_loss2},")
                else:
                    dummy_log_file.write(f"{total_loss},{half_loss1},{half_loss2},{time_cost_per_sample}\n")

        if TEST:
            break


if __name__ == '__main__':
    main()
