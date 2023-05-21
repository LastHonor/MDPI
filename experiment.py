import os

# for dataf in ['Collide', 'Roll', 'Drop', 'Drape', 'Dominoes', 'Contain', 'Link', 'Support'][::-1][1:]:
for dataf in ['Collide']:
    for model_name in ['MyModel5']:
        os.system('python my_train.py '
                  '--env TDWdominoes '
                  f'--model_name {model_name} '
                  '--training_fpt 3 '
                  '--floor_cheat 1 '
                  f'--dataf "{dataf}," '
                  f'--outf "Single{dataf}-{model_name}" '
                  '--nf_relation 300 '
                  '--nf_particle 200 '
                  '--nf_effect 200 '
                  '--n_epoch 100 '
                  '--num_workers 1 '
                  '--trans_num_layers 1 '
                  '--trans_num_heads 1 '
                  '--trans_dropout 0.0 ')


# model_name = 'MyModel2'
# dataf = 'Support'
# os.system('python my_train.py '
#                   '--env TDWdominoes  '
#                   f'--model_name {model_name} '
#                   '--training_fpt 3 '
#                   '--floor_cheat 1 '
#                   f'--dataf "{dataf}," '
#                   f'--outf "Single{dataf}-{model_name}" '
#                   '--nf_relation 300 '
#                   '--nf_particle 200 '
#                   '--nf_effect 200 '
#                   '--n_epoch 10')
