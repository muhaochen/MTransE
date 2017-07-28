import sys
if '../../src/MMTransE' not in sys.path:
    sys.path.append('../../src/MMTransE')

from MMTransE import MMTransE

model = MMTransE(dim=50, save_dir='model_MMtransE_person_cn.bin')
model.Train_MT(epochs=400, save_every_epochs=100, languages=['en', 'fr'], graphs=['../../data/CN3l/en_fr/C_en_f.csv','../../data/CN3l/en_fr/C_fr.csv'], intersect_graph='../../data/CN3l/en_fr/C_en_fr.csv', save_dirs = ['model_en.bin','model_fr.bin'], rate=0.001, split_rate=True, L1_flag=False)
model.save('model_MMtransE_person_cn.bin')
