import torch
from d2l import torch as d2l
from ResNet50 import resnet50
from Utils.DataLoader import load_data
from matplotlib import pyplot as plt
from Utils.Trainer import train_with_graph

type = 'smoke'
net = resnet50(num_classes=2)
lr, num_epochs, batch_size = 0.05, 12, 32
train_dir = '../../Data/data/'+ type + '_train'
test_dir = '../../Data/data/' + type + '_test'
train_iter, test_iter = load_data(train_dir, test_dir, batch_size, resize=224)
train_l,train_acc,test_acc = train_with_graph(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
model_path = './saved_models/'+ type + '/' + type + f'_{train_l:.3f}_{train_acc:.3f}'f'_{test_acc:.3f}' + '.pkl'
fig_path = './saved_figs/'+ type + '/' + type + f'_{train_l:.3f}_{train_acc:.3f}'f'_{test_acc:.3f}' + '.png'
torch.save(net.state_dict(),model_path)
plt.savefig(fig_path)
plt.show()