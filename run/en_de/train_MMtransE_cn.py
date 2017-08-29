import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src/MMTransE'))

from MMTransE import MMTransE

model = MMTransE(dim=50, save_dir=os.path.join(os.path.dirname(__file__), 'model_MMtransE_cn_ed.bin'))
model.Train_MT(epochs=400, save_every_epochs=100, languages=['en', 'de'], graphs=[os.path.join(os.path.dirname(__file__), '../../data/CN3l/en_de/C_en_d.csv'),os.path.join(os.path.dirname(__file__), '../../data/CN3l/en_de/C_de.csv')], intersect_graph=os.path.join(os.path.dirname(__file__), '../../data/CN3l/en_de/C_en_de.csv'), save_dirs = ['model_en_ed.bin','model_de_ed.bin'], rate=0.001, split_rate=True, L1_flag=False)
model.save(os.path.join(os.path.dirname(__file__), 'model_MMtransE_cn_ed.bin'))
