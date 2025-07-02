from config import config
from utils.load_data import load_skin_data


def main():
    # Load the skin lesion data
    lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}
    
    skin_df = load_skin_data(
        dataset_dir=config.DATASET_DIR,
        metadata_path=config.METADATA_PATH,
        lesion_type_dict=lesion_type_dict
    )
    
    # Display the first few rows of the dataframe
    print(skin_df.head())
    
    # Display the shape of the dataframe
    print(f"Dataframe shape: {skin_df.shape}")
    
if __name__ == "__main__":
    main()