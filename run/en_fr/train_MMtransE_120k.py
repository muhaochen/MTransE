import sys
if '../../src/MMTransE' not in sys.path:
    sys.path.append('../../src/MMTransE')

from MMTransE import MMTransE

model = MMTransE(dim=100, save_dir='model_MMtransE_person_120k.bin')
model.Train_MT(epochs=400, save_every_epochs=100, languages=['en', 'fr'], graphs=['../../data/WK3l-120k/en_fr/P_en_v5_120k.csv','../../data/WK3l-120k/en_fr/P_fr_v5_120k.csv'], intersect_graph='../../data/WK3l-120k/en_fr/P_en_fr_v5_120k.csv', save_dirs = ['model_en.bin','model_fr.bin'], rate=0.01, split_rate=True, L1_flag=False)
model.save('model_MMtransE_person_120k.bin')
