import os

# for dataf in ['Collide', 'Roll', 'Drop', 'Drape', 'Dominoes', 'Contain', 'Link', 'Support'][::-1][1:]:
# for dataf in ['Collide', 'Roll'][1:]:
#     for model_name in ['MyModel1', 'DPINet2']:
#         os.system('python my_eval_vis.py '
#                   '--env TDWdominoes '
#                   f'--model_name {model_name} '
#                   '--training_fpt 3 '
#                   '--mode "test" '
#                   '--floor_cheat 1 '
#                   '--test_training_data_processing 1 '
#                   f'--modelf "Single{dataf}-{model_name}" '
#                   f'--dataf {dataf}')

# for model_dataf in ['Collide', 'Roll', 'Drop', 'Drape', 'Dominoes', 'Contain', 'Link', 'Support'][:3]:
#     for dataf in ['Collide', 'Roll', 'Drop', 'Drape', 'Dominoes', 'Contain', 'Link', 'Support']:
#         if model_dataf == 'Drape' and dataf != 'Drape':
#             continue
#         if model_dataf != 'Drape' and dataf == 'Drape':
#             continue
#         if model_dataf == 'Collide' and dataf in ['Collide', 'Roll', 'Drop', 'Dominoes']:
#             continue
#         for model in ["MyModel2", "DPINet2"]:
#             os.system('python my_eval.py '
#                       '--env TDWdominoes '
#                       f"--model_name {model} "
#                       '--training_fpt 3 '
#                       '--mode "test" '
#                       '--floor_cheat 1 '
#                       '--test_training_data_processing 1 '
#                       f"--modelf Single{model_dataf}-{model} "
#                       f'--dataf {dataf}')


# model_names = ["MyModel2", "DPINet2"]
model_names = ['GNSRigid', 'MyGNSRigid']
datafs = ['Collide', 'Roll', 'Drop', 'Drape', 'Dominoes', 'Contain', 'Link', 'Support']
for model in model_names:
    for dataf in datafs:

# model = model_names[0]
# for dataf in ['Link', 'Drape', 'Support']:
# model_dataf = datafs[-2]
# dataf = datafs[-1]
        model_dataf = dataf
# epoch = -1
# for epoch in range(9, -1, -1):
#         os.system('python my_eval.py '
#                   '--env TDWdominoes '
#                   f"--model_name {model} "
#                   '--training_fpt 3 '
#                   '--mode "test" '
#                   '--floor_cheat 1 '
#                   '--test_training_data_processing 0 '
#                   f"--modelf Single{model_dataf}-{model} "
#                   f"--epoch {epoch} "
#                   f'--dataf {dataf} ')
#               # f'--epoch 5 ')
        os.system('python my_eval.py '
                      '--env TDWdominoes '
                      f"--model_name {model} "
                      '--training_fpt 3 '
                      '--mode "test" '
                      '--floor_cheat 1 '
                      '--test_training_data_processing 0 '
                      f"--modelf Single{model_dataf}-{model} "
                      f"--epoch {-1} "
                      f'--dataf {dataf} ')
                      # f'--epoch 5 ')
