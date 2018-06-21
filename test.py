


from dataset import *

from torch.utils.data import DataLoader
from torch.autograd import Variable
from skimage import io
from matplotlib import pyplot as plt

from models.resnet import *


cifar100_test = DataLoader(CIFAR100Test(g_cifar_100_path), batch_size=10, shuffle=True, num_workers=2)


net = ResNet101()
net.load_state_dict(torch.load('resnet101.pt'))
net.eval()

correct = 0
total = 0

for (label, image) in cifar100_test:
    image = image.permute(0, 3, 1, 2).float()
    output = net(Variable(image))
    #print(output)
    #print(output.max())
    #output = output.squeeze()

    print(output)
    _, res = output.max(1)
    print(res)
    #print(label)
    #print(res)
    correct += res.eq(Variable(label)).sum().data[0]
    total +=  output.size(0)
    print(correct / total)

    break
    #res = output.max(0)[1]
    #for i in res.data:
    #    if i != 0:
    #        print(i)
   # _, predict = outputs.max(0)
   # correct += output
    #print(label)
    #print(image)
#    image = image.squeeze()
#
#
#    print(type(image))
#    print(image.size())
#    image = image.permute(2, 0, 1)
#    print(image.size())
#    image = image.permute(1, 2, 0)
#    print(image.size())
#    io.imshow(image.numpy())
#    plt.show()
#