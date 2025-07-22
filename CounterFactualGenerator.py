import pandas as pd
import numpy as np

class SkinCancerCounterfactualGenerator:
    def __init__(self, lesion_type_dict):
        
        # Contraintes biologiques par type de lésion
        self.lesion_constraints = lesion_type_dict

    def is_biologically_valid(self, age, sex, location, lesion_type):
        """Vérifie si la combinaison est biologiquement plausible"""
        constraints = self.lesion_constraints.get(lesion_type, {})
        
        # Vérifier l'âge
        age_range = constraints.get('age_range', (0, 100))
        if not (age_range[0] <= age <= age_range[1]):
            return False
            
        # Vérifier la localisation
        common_locs = constraints.get('common_locations', [])
        if common_locs and location not in common_locs:
            return False
            
        return True

    def generate_counterfactual_metadata(self, original_row, target_class, change_type='sex'):
        """Génère une métadonnée contrefactuelle biologiquement valide"""
        cf_row = original_row.copy()
        
        if change_type == 'sex':
            cf_row = self._change_sex(cf_row, target_class)
        elif change_type == 'age':
            cf_row = self._change_age(cf_row, target_class)
        elif change_type == 'location':
            cf_row = self._change_location(cf_row, target_class)
        elif change_type == 'mixed':
            cf_row = self._change_mixed(cf_row, target_class)
            
        return cf_row

    def _change_sex(self, row, target_class):
        """Change le sexe en respectant les contraintes biologiques"""
        current_sex = row['sex']
        new_sex = 'female' if current_sex == 'male' else 'male'
        
        # Vérifier si le changement est biologiquement valide
        if self.is_biologically_valid(row['age'], new_sex, row['localization'], target_class):
            row['sex'] = new_sex
        else:
            # Si pas valide, ajuster l'âge pour rendre valide
            constraints = self.lesion_constraints.get(target_class, {})
            age_range = constraints.get('age_range', (row['age'], row['age']))
            row['age'] = np.random.randint(age_range[0], age_range[1] + 1)
            row['sex'] = new_sex
            
        return row

    def _change_age(self, row, target_class):
        """Change l'âge selon les contraintes du type de lésion"""
        constraints = self.lesion_constraints.get(target_class, {})
        age_range = constraints.get('age_range', (20, 80))
        
        # Éviter de garder le même âge
        current_age = row['age']
        possible_ages = [a for a in range(age_range[0], age_range[1] + 1) if abs(a - current_age) > 5]
        
        if possible_ages:
            row['age'] = np.random.choice(possible_ages)
        else:
            row['age'] = np.random.randint(age_range[0], age_range[1] + 1)
            
        return row

    def _change_location(self, row, target_class):
        """Change la localisation selon les contraintes du type de lésion"""
        constraints = self.lesion_constraints.get(target_class, {})
        common_locations = constraints.get('common_locations', ['torso', 'arm', 'leg', 'face'])
        
        current_location = row['localization']
        possible_locations = [loc for loc in common_locations if loc != current_location]
        
        if possible_locations:
            row['localization'] = np.random.choice(possible_locations)
            
        return row

    def _change_mixed(self, row, target_class):
        """Change plusieurs attributs de façon cohérente"""
        # Commencer par changer la classe cible
        changes = np.random.choice(['sex', 'age', 'location'], size=2, replace=False)
        
        for change in changes:
            if change == 'sex':
                row = self._change_sex(row, target_class)
            elif change == 'age':
                row = self._change_age(row, target_class)
            elif change == 'location':
                row = self._change_location(row, target_class)
                
        return row