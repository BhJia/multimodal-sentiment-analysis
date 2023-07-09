import os
from torchvision import transforms

transform=transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def create_exp_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    print('Experiment dir : {}'.format(path))


def count_parameters(model):
    total_num = sum(p.numel() for p in model.parameters())
    return total_num

def num_to_tag(num):
    return 'negative' if num==0 else 'neutral' if num==1 else 'positive'

def test_output(save_path, guids, predictions):
    output_file=os.path.join(save_path,"output.txt")
    with open(output_file, 'w') as f:
        f.writelines("guid,tag")
        f.write('\r\n')
        for id, (guid, pred) in enumerate(zip(guids,predictions)):
            f.writelines(str(guid) + "," + num_to_tag(pred))
            f.write('\r\n')
    f.close()