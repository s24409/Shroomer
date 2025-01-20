import json
from pathlib import Path

class SpeciesLookup:
    def __init__(self):
        data_dir = Path(__file__).parent.parent / 'data'
        with open(data_dir / 'species_info.json', 'r', encoding='utf-8') as f:
            self.species_dict = json.load(f)
    
    def get_species_info(self, class_id):
        """Get species information from class ID"""
        return self.species_dict.get(str(class_id))
    
    def format_species_info(self, class_id):
        """Format species information for display"""
        info = self.get_species_info(class_id)
        if not info:
            return "Species information not found"
            
        return f"""
Species Information:
------------------
Scientific Name: {info['scientific_name']}
Common Name: {info['common_name']}
Taxonomy:
  Kingdom: {info['kingdom']}
  Phylum: {info['phylum']}
  Class: {info['class']}
  Order: {info['order']}
  Family: {info['family']}
  Genus: {info['genus']}
"""
