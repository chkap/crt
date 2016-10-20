
import numpy as np


def load_data():
    path = '/home/chenkai/workspace/caffe_model/vgg16_D/VGG_16_layers.npy'
    save_path = '/home/chenkai/workspace/caffe_model/vgg16_D/VGG_16_layers_py3.npz'
    data_dict = np.load(path).item()
    kw_data = {}
    for item in data_dict:
        tmp = data_dict[item]
        scope = str(item)
        for name, data in tmp.iteritems():
            key = '{:s}/{:s}'.format(scope, name)
            kw_data[key] = data
    np.savez(save_path, **kw_data)


if __name__ == '__main__':
    load_data()
