import sys
if '../../src/MMTransE' not in sys.path:
    sys.path.append('../../src/MMTransE')

from MMTransE import MMTransE

model = MMTransE(dim=75, save_dir='model_MMtransE_person_15k.bin')
model.Train_MT(epochs=400, save_every_epochs=100, languages=['en', 'fr'], graphs=['../../data/WK3l-15k/en_fr/P_en_v5.csv','../../data/WK3l-15k/en_fr/P_fr_v5.csv'], intersect_graph='../../data/WK3l-15k/en_fr/P_en_fr_v5', save_dirs = ['model_en.bin','model_fr.bin'], rate=0.01, split_rate=True, L1_flag=False)
model.save('model_MMtransE_person_15k.bin')
