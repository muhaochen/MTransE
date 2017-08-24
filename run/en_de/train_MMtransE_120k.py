import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src/MMTransE'))

from MMTransE import MMTransE

model = MMTransE(dim=100, save_dir=os.path.join(os.path.dirname(__file__), 'model_MMtransE_person_120k_ed.bin'))
model.Train_MT(epochs=400, save_every_epochs=100, languages=['en', 'de'], graphs=[os.path.join(os.path.dirname(__file__), '../../data/WK3l-120k/en_de/P_en_v6_120k.csv'), os.path.join(os.path.dirname(__file__), '../../data/WK3l-120k/en_de/P_de_v6_120k.csv')], intersect_graph=os.path.join(os.path.dirname(__file__), '../../data/WK3l-120k/en_de/P_en_de_v6_120k.csv'), save_dirs = ['model_en_ed.bin','model_de_ed.bin'], rate=0.01, split_rate=True, L1_flag=False)
model.save(os.path.join(os.path.dirname(__file__), 'model_MMtransE_person_120k_ed.bin'))
