import torch
from torchsummary import summary
from net.models import deeplabv3plus

def compare_models(model1, model2):
    # Get state dictionaries of both models
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()
    
    # Check if the keys are the same
    if state_dict1.keys() != state_dict2.keys():
        print("Models have different sets of parameters")
        return False
    
    # Check if the values are the same for each parameter
    for key in state_dict1.keys():
        if not torch.equal(state_dict1[key], state_dict2[key]):
            print(f"Parameter '{key}' is different between the two models")
            return False
    
    print("Models are the same")
    return True


model1 = deeplabv3plus(num_classes=3)
model1.cuda()
model1.load_state_dict(torch.load('models/deeplabv3plus_xception_VOC2012_epoch46_all.pth'), strict = False)

# print("model1")
# print(summary(model1, input_size = (3, 224, 224)))

model2 = deeplabv3plus(num_classes=3)
model2.cuda()
pretrained_dict = torch.load('models/deeplabv3plus_xception_VOC2012_epoch46_all.pth')
net_dict = model2.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in net_dict) and (v.shape == net_dict[k].shape)}
net_dict.update(pretrained_dict)
model2.load_state_dict(net_dict)
# print("model2")
# print(summary(model2, input_size = (3, 224, 224)))

model3 = deeplabv3plus(num_classes=3)
model3.cuda()
pretrained_dict = torch.load('models/DR_CoarseSN_9/CoarseSN_e227.pth')
net_dict = model3.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in net_dict) and (v.shape == net_dict[k].shape)}
net_dict.update(pretrained_dict)
model3.load_state_dict(net_dict)
# print("model3")
# print(summary(model3, input_size = (3, 224, 224)))

model4 = deeplabv3plus(num_classes=3)
model4.cuda()
model4.load_state_dict(torch.load('models/DR_CoarseSN_9/CoarseSN_e227.pth'))

compare_models(model3, model4)
