// Canonical display names for models
const MODEL_DISPLAY_NAMES = {
    densenet121:  'DenseNet121',
    vit:          'ViT',
    resnet50:     'ResNet50',
    vgg16:        'VGG16',
    efficientnet: 'EfficientNet',
    segresnet:    'SegResNet',
    swinunetr:    'SwinUNETR',
    unet:         'UNet',
    custom:       'Custom',
};

function getDisplayName(name) {
    return MODEL_DISPLAY_NAMES[(name || '').toLowerCase()] || name;
}
