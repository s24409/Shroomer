import pandas as pd
import json

def process_mushroom_metadata(csv_path):
    df = pd.read_csv(csv_path)
    
    species_dict = {}
    
    unique_species = df[['scientificName', 'species', 'class_id']].drop_duplicates()
    
    for _, row in unique_species.iterrows():
        class_id = int(row['class_id'])
        
        if class_id <= 1604:
            species_info = {
                'scientific_name': row['scientificName'],
                'common_name': row['species'],
                'class_id': class_id
            }
            
            first_occurrence = df[df['class_id'] == class_id].iloc[0]
            species_info.update({
                'kingdom': first_occurrence['kingdom'],
                'phylum': first_occurrence['phylum'],
                'family': first_occurrence['family'],
                'genus': first_occurrence['genus']
            })
            
            species_dict[class_id] = species_info
    
    output_path = 'species_info.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(species_dict, f, ensure_ascii=False, indent=2)
    
    return species_dict

def get_species_info(class_id, species_dict):
    """Helper function to get species info from class ID"""
    return species_dict.get(class_id, None)

if __name__ == "__main__":
    csv_path = "DF20-metadata/DF20-public_test_metadata_PROD-2.csv"
    species_dict = process_mushroom_metadata(csv_path)
    
    print("Example species info for class_id 1:")
    print(json.dumps(get_species_info(1, species_dict), indent=2))
