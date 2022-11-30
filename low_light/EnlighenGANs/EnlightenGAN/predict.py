import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
# from util.visualizer import Visualizer
from pdb import set_trace as st
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
# visualizer = Visualizer(opt)
# create website
# web_dir = os.path.join("./ablation/", opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
# webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

# test
print(len(dataset))
for i, data in enumerate(dataset):
    # print("current data", data)
    print("data size:", data.keys())
    model.set_input(data)
    visuals = model.predict()
    img_path = model.get_image_paths()
    print('process image... %s' % img_path)
    print("len: ", len(visuals))
    print(visuals['real_A'].shape)
    print(visuals['fake_B'].shape)
    # visualizer.save_images(webpage, visuals, img_path)
    print("done")
    im = Image.fromarray(visuals['fake_B'])
    img_name = img_path[0].split('/')[-1]
    im.save("./enlighten_images/" + img_name)
# plt.imshow(visuals['real_A'])
# np.save("fake_image",visuals['fake_B'])
    

# webpage.save()
# python scripts/script.py --predict
