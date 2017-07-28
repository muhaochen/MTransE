import sys
if '../../src/MMTransE' not in sys.path:
    sys.path.append('../../src/MMTransE')

from MMTransE import MMTransE

model = MMTransE(dim=50, save_dir='model_MMtransE_person_cn_ed.bin')
model.Train_MT(epochs=400, save_every_epochs=100, languages=['en', 'de'], graphs=['../../data/CN3l/en_de/C_en_d.csv','../../data/CN3l/en_de/C_de.csv'], intersect_graph='../../data/CN3l/en_de/C_en_de.csv', save_dirs = ['model_en_ed.bin','model_de_ed.bin'], rate=0.001, split_rate=True, L1_flag=False)
model.save('model_MMtransE_person_cn_ed.bin')
