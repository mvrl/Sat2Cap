from torchvision.datasets import ImageFolder
import torchvision
import torchgeo
import torch

def get_dataloader(dataset_name='resisc', train_size=0.3, batch_size=16, num_workers=2):
        #configure dataset
    if dataset_name.lower()=='resisc':
        print('Loading RESISC45 dataset')
        dataset_path = '/home/a.dhakal/active/user_a.dhakal/datasets/RESISC45/NWPU-RESISC45'
        dataset = ImageFolder(dataset_path, transform=torchvision.transforms.Compose([
                    torchvision.transforms.Resize((224,224), interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
                    torchvision.transforms.ToTensor()
                    ]))
    elif dataset_name.lower()=='eurosat':
        print('Loading EuroSat dataset')
        dataset_path = '/home/a.dhakal/active/user_a.dhakal/datasets/eurosat/2750'
        dataset = ImageFolder(dataset_path, transform=torchvision.transforms.Compose([
                    torchvision.transforms.Resize((224,224), interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
                    torchvision.transforms.ToTensor()
                    ]))
    elif dataset_name.lower()=='ucmer':
        print('Loading UCMER dataset')
        dataset_path = '/home/a.dhakal/active/user_a.dhakal/datasets/UCMerced_LandUse/Images'
        dataset = ImageFolder(dataset_path, transform=torchvision.transforms.Compose([
                    torchvision.transforms.Resize((224,224), interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
                    torchvision.transforms.ToTensor()
                    ]))
    else:
        raise ValueError('Invalid Dataset Name')

    # total_length = len(dataset)
    # train_len = int(train_size*total_length)
    # test_len = total_length-train_len
    val_size = 0.1
    test_size = 1-val_size-train_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size,val_size, test_size])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers=num_workers,
                                                    pin_memory=True, persistent_workers=True, drop_last=True)
    
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size, shuffle=False, num_workers=num_workers,
                                                    pin_memory=True, persistent_workers=True, drop_last=True)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle=False, num_workers=num_workers,
                                                    pin_memory=True, persistent_workers=True, drop_last=True)
    
    num_classes = len(dataset.classes)
    
    return (train_dataloader,val_dataloader, test_dataloader, num_classes)