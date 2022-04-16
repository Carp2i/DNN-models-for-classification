import os
import sys
import json
# from sklearn.utils import shuffle

import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm

from model import vgg

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # transform
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) 
    }

    # dataset
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../")) # get data root path
    image_path = os.path.join(data_root, "data", "flower_data") # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                        transform=data_transform["train"])
    train_num = len(train_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    
    # json 的方法记得再看下 22/4/15
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 27
    # 设置 num_workers 的数量，Linux系统下
    # win 用户只能setting 0
    
    # nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    # print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=batch_size, shuffle=True,
                                                num_workers=0)
    val_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                        transform=data_transform["val"])
    val_num = len(val_dataset)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=batch_size, shuffle=False,
                                            num_workers=0)
    print("using {} images for training, {} images for validation.".format(train_num,
                                                                            val_num))
    # test_data_iter = iter(val_loader)
    # test_image, test_label = test_data_iter.next()

    # 下面是Fashion-MNIST的展示架构，不支持这边

    # def imshow(img):
    #     img = img / 2 + 0.5     # unnormalize
    #     npimg = img.numpy()
    #     plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #     plt.show

    # # print labels
    # print(' '.join('%5s' % classes[test_label[j]] for j in range(4)))
    # # show images
    # imshow(torchvision.utils.make_grid(test_image))

    model_name = "vgg16"
    net = vgg(model_name=model_name, num_classes=5, init_weights=True)
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    
    num_epoch = 30
    best_acc = 0.0
    save_path = './Net.pth'.format(model_name)
    train_steps = len(train_loader)
    for epoch in range(num_epoch):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                    num_epoch,
                                                                    loss)
        # validation
        net.eval()
        acc = 0.0   # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
            
        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f val_accuracy: %.3f' %
                (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finishing Training')


if __name__ == '__main__':
    main()