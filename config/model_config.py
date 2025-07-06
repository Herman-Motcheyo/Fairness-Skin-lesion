import tensorflow as tf

from tf.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tf.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tf.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tf.keras.applications.densenet import preprocess_input as densenet_preprocess
from tf.keras.application.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tf.keras.applications.xception import preprocess_input as xceptopn_preprocess

from PIL import Image

def get_model_config():
    """
    Retourne la configuration pour chaque modèle avec ses paramètres spécifiques.
    """
    model_configs = {
        'xception': {
            'create_function': 'create_multimodal_xception',
            'model_name': 'MultimodalXception',
            'preprocess_func': xceptopn_preprocess,
            'input_size': (224, 224)  
        },
        'mobilenetv2': {
            'create_function': 'create_multimodal_mobilenet',
            'model_name': 'MultimodalMobileNet',
            'preprocess_func': mobilenet_preprocess,
            'input_size': (224, 224)  
        },
        'resnet': {
            'create_function': 'create_multimodal_resnet',
            'model_name': 'MultimodalResNet',
            'preprocess_func': resnet_preprocess,
            'input_size': (224, 224)  
        },
        'vgg': {
            'create_function': 'create_multimodal_vgg',
            'model_name': 'MultimodalVGG',
            'preprocess_func': vgg_preprocess,
            'input_size': (224, 224)  
        },
        'efficientnet': {
            'create_function': 'create_multimodal_efficientnet',
            'model_name': 'MultimodalEfficientNet',
            'preprocess_func': efficientnet_preprocess,
            'input_size': (224, 224) 
        },
        'densenet': {
            'create_function': 'create_multimodal_densenet',
            'model_name': 'MultimodalDenseNet',
            'preprocess_func': densenet_preprocess,
            'input_size': (224, 224)  
        }
    }
    return model_configs