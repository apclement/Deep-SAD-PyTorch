
FOLD = 0
train_sample_pct = '01'
# input layer size
#TODO define it properly
#154 + 30 topics
N_features = 160 + 30 + 2 #TODO check the origin of the +&
nfeature_map = {0: 160, 1: 159, 2: 159, 3: 159, 4: 160, 5: 159, 6: 161, 7: 159, 8: 161, 9: 158}

rep_dim = 2
max_batches = 145#150#145#750 #1400 #735 #755 #780 #4994 #830#650#1500

nbatches01_map = {0: 150, 1: 145, 2: 735, 3: 790, 4: 714, 5: 720, 6: 1500, 7: 1445, 8: 1550, 9: 150}

nbatches_map = {0: 780, 1: 750, 2: 735, 3: 790, 4: 714, 5: 720, 6: 1500, 7: 1445, 8: 1550, 9: 1570} 
