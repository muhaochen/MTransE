import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src/MMTransE'))

from MMTransE import MMTransE

model = MMTransE(dim=50, save_dir=os.path.join(os.path.dirname(__file__), 'model_MMtransE_cn.bin'))
model.Train_MT(epochs=400, save_every_epochs=100, languages=['en', 'fr'], graphs=[os.path.join(os.path.dirname(__file__), '../../data/CN3l/en_fr/C_en_f.csv'),os.path.join(os.path.dirname(__file__), '../../data/CN3l/en_fr/C_fr.csv')], intersect_graph=os.path.join(os.path.dirname(__file__), '../../data/CN3l/en_fr/C_en_fr.csv'), save_dirs = ['model_en.bin','model_fr.bin'], rate=0.001, split_rate=True, L1_flag=False)
model.save(os.path.join(os.path.dirname(__file__), 'model_MMtransE_cn.bin'))
