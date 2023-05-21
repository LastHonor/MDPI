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


# for model_dataf in ['Collide', 'Roll', 'Drop', 'Drape', 'Dominoes', 'Contain', 'Link', 'Support']:
#     for dataf in ['Collide', 'Roll', 'Drop', 'Drape', 'Dominoes', 'Contain', 'Link', 'Support']:
#         if model_dataf == 'Drape' and dataf != 'Drape':
#             continue
#         if model_dataf != 'Drape' and dataf == 'Drape':
#             continue
#         if model_dataf == 'Collide' and dataf in ['Collide', 'Roll']:
#             continue
#         if model_dataf in ['Collide', 'Roll', 'Drop']:
#             for model in ["MyModel2", "DPINet2"]:
#                 os.system('python my_eval_vis.py '
#                           '--env TDWdominoes '
#                           f"--model_name {model} "
#                           '--training_fpt 3 '
#                           '--mode "test" '
#                           '--floor_cheat 1 '
#                           # '--test_training_data_processing 0 '
#                           f"--modelf Single{model_dataf}-{model} "
#                           # f"--modelf Single{dataf}-draft "
#                           f'--dataf {dataf} '
#                           '--saveavi 0 '
#                           '--gt_only 0'
#                           # '--gt_only 1'
#                           )
#         else:
#             os.system('python my_eval_vis.py '
#                       '--env TDWdominoes '
#                       f"--model_name DPINet2 "
#                       '--training_fpt 3 '
#                       '--mode "test" '
#                       '--floor_cheat 1 '
#                       # '--test_training_data_processing 0 '
#                       # f"--modelf Single{dataf}-{model} "
#                       f"--modelf {dataf}-draft "
#                       f'--dataf {dataf} '
#                       '--saveavi 0 '
#                       # '--gt_only 0'
#                       '--gt_only 1'
#                       )

model_names = ["MyModel2", "DPINet2"]
# model_names = ['GNSRigid', 'MyGNSRigid']
datafs = ['Collide', 'Roll', 'Drop', 'Drape', 'Dominoes', 'Contain', 'Link', 'Support']

model = model_names[0]
model_dataf = datafs[0]
dataf = model_dataf

# for model in model_names:
# for model_dataf in ['Collide']:
#     dataf = model_dataf

os.system('python my_eval_vis.py '
          '--env TDWdominoes '
          f"--model_name {model} "
          '--training_fpt 3 '
          '--mode "test" '
          '--floor_cheat 1 '
          # '--test_training_data_processing 0 '
          f"--modelf Single{model_dataf}-{model} "
          "--epoch -1 "
          # f"--modelf Single{dataf}-draft "
          f'--dataf {dataf} '
          '--saveavi 0 '
          '--gt_only 0'
          # '--gt_only 1'
          )