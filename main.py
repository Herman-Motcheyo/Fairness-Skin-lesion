from config import config
from utils.load_data import load_skin_data
import numpy as np
from PIL import Image

from multimodal_cnn import MultimodalCNNFramework
from stratified_cross_validator import StratifiedCrossValidator
from config.model_config import train_multiple_models


lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}
    
df = load_skin_data(
        dataset_dir=config.DATASET_DIR,
        metadata_path=config.METADATA_PATH,
        lesion_type_dict=lesion_type_dict
    )
    
    
print(f"Dataframe shape: {df.shape}")
    
print(df['path'].isnull().sum(), "images non trouvées dans le dossier")

NUMBER_OF_FOLDS = 10

validator = StratifiedCrossValidator(
    n_splits=NUMBER_OF_FOLDS,
    test_size=0.2,  # 20% pour test
    val_size=0.1    # 10% pour la validation
)

# Simple prepration des données
data = validator.prepare_data(df, target_col='dx')

# Creation des splits pour la validation
splits = validator.create_stratified_splits_finale(data, 'dx')

# attributes à discretiser
column_name_to_discretize =  [ 'sex', 'localization']

mul_cnn = MultimodalCNNFramework(
        input_shape=(224, 224, 3), 
        num_classes=7, 
        batch_size=32, 
        epochs=2, 
        learning_rate=0.001
    )


histories = train_multiple_models(
    framework=mul_cnn,
    validator=validator,
    df=df,
    column_name_to_discretize=column_name_to_discretize,
    models_to_train=['xception','mobilenetv2','resnet', 'vgg', 'efficientnet', 'densenet'],
    NUMBER_OF_FOLDS=1,
    save_dir='histories'
)
