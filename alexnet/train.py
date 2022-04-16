import os
import json
import sys

import torch
import torch.nn as nn
# from attr import validate
# from sklearn.utils import shuffle
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim

from tqdm import tqdm

from model import AlexNet


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)

    # transform 中定义了对数据集变换的操作
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),   # cannot 224, must (224, 224)
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    }

    # set the path of dataset
    data_root = "./" # get data root path
    image_path = os.path.join(data_root, "flower_data") # flower dataset path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)


    batch_size = 32
    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    train_dataset = datasets.ImageFolder(root=image_path + "./train",
                                            transform=data_transform["train"])
    train_num = len(train_dataset)

    flower_list =train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)


    # 好像是在windows 环境上只能set 0
    # nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    # print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=batch_size, shuffle=True,
                                                num_workers=0)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, 'val'),
                                                transform=data_transform['val'])
    val_num = len(validate_dataset)

    # 如果要做数据集测试的话，batch_size=4, shuffle=True
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                batch_size=batch_size, shuffle=False,
                                                num_workers=0)
                                    
    print("using {} images for training, {} images for validation.".format(train_num, val_num))


    # # 检查验证集的代码
    # test_data_iter = iter(validate_loader)
    # test_image, test_label = test_data_iter.next()

    # def imshow(img):
    #     img = img / 2 + 0.5     # unnormalize
    #     npimg = img.numpy()
    #     plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #     plt.show()

    # print(' '.join('%5s' % cla_dict[test_label[j].item()] for j in range(4)))
    # imshow(utils.make_grid(test_image))

    net = AlexNet(num_classes=5, init_weights=True)

    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    # pata = list(net.parameters()) # 调试用的，用来查看模型的参数
    optimizer = optim.Adam(net.parameters(), lr=2e-4)   # 学习率是调试过的

    save_path = './AlexNet.pth'
    best_acc = 0.0
    epoch_num = 20
    # train_num = len(train_dataset)

    for epoch in range(epoch_num):
        # 模型的mode {train, eval} 会管理模型的 dropout和batchnorm层
        net.train()
        running_loss = 0.0
        # 用来预测，训练时间
        # t1 = time.perf_counter()
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
        
            # 梯度数据清空
            optimizer.zero_grad()
            outputs = net(images.to(device))
        
            # 损失函数定义，返回的是tensor
            loss = loss_function(outputs, labels.to(device))
            # 反向传播
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            
            # # print train process
            # rate = (step + 1) / len(train_loader)
            # a = "*" * int(rate * 50)
            # b = "." * int((1 - rate) * 50)

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epoch_num, loss)


        # print()
        # print(time.perf_counter()-t1)
        print(step)

        # validation part
        net.eval()
        acc = 0.0   # accumulate accurate num / epoch
        with torch.no_grad():
            for data_set in validate_loader:
                test_images, test_labels = data_set
                outputs = net(test_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += (predict_y == test_labels.to(device)).sum().item()
                # acc += torch.eq(predict_y, test_labels.to(device)).sum().item


            accurate_test = acc / val_num
            print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
                        (epoch + 1, running_loss / step, acc / val_num))
            
            if accurate_test > best_acc:
                best_acc = accurate_test
                torch.save(net.state_dict(), save_path)
        
    
        # 上面 running_loss / step 和 train_num 相同嘛
    print('Finished Training')


if __name__ == '__main__':
    main()