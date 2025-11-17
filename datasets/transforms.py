import torchvision.transforms as T

def get_transforms(input_size=50):
    return T.Compose([
        T.Resize((input_size, input_size)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ToTensor()
    ])
