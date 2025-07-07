from CQTF_vgg_small import CQTF_VGG_SMALL

model_zoo = {
    'CQTF_VGG_SMALL': CQTF_VGG_SMALL,
}

def get_model(name):
    model_class = model_zoo.get(name)
    if model_class is None:
        raise ValueError(f"Model {name} not found")
    return model_class