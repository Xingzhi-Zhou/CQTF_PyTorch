from CQTF_vgg_small import CQTF_VGG_SMALL
from CQTF_resnet18 import CQTF_ResNet18
from CQTF_resnet9 import CQTF_ResNet9

model_zoo = {
    'CQTF_VGG_SMALL': CQTF_VGG_SMALL,
    'CQTF_ResNet9': CQTF_ResNet9,
    'CQTF_ResNet18': CQTF_ResNet18,
}

def get_model(name):
    model_class = model_zoo.get(name)
    if model_class is None:
        raise ValueError(f"Model {name} not found")
    return model_class