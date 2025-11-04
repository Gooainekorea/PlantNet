"""
파일명: matadata_tool.py
설명: 데이터 인덱싱을 위한 메타데이터 처리
해당 파일을 실행하면 .json .csv 파일이 각각 생성됩니다.
"""
import pandas as pd
import os

# config
base_input_path = '' # 기본 입력경로 
input_path = f'{base_input_path}plantnet_300K/' # 데이터 파일 경로
output_path = f'{base_input_path}output_data/' # 출력결과 저장 경로
plantnet_metadata_path = f'{input_path}plantnet300K_metadata.json' # 메타 데이터 파일 경로
species_idx_path = f'{input_path}class_idx_to_species_id.json'#종 id 파일 경로
species_name_path = f'{input_path}plantnet300K_species_id_2_name.json'#학명 파일 경로

# 출력 및 모델 저장 경로가 없으면 자동으로 생성
metadata_dir = os.path.join(output_path, 'metadata')
os.makedirs(output_path, exist_ok=True)
os.makedirs(metadata_dir, exist_ok=True)
print(f"출력 폴더가 생성되었습니다: {output_path}")
print(f"메타데이터 저장 폴더가 생성되었습니다: {metadata_dir}")

# 메타 데이터 처리
metadata = pd.read_json(plantnet_metadata_path)
metadata = metadata.transpose() # w전치
metadata = metadata.reset_index() # 열이름을 id로 변환
metadata = metadata.rename(columns={"index": "id"})
metadata.to_csv(f'{output_path}metadata/metadata.csv', index=False)
metadata.to_json(f'{output_path}metadata/metadata.json', index=False)

# 인덱스 데이터 처리
species_idx = pd.read_json(species_idx_path, orient='index')
species_idx = species_idx.reset_index()
species_idx = species_idx.rename(columns={"index": "species_idx", 0: "species_id"})
species_idx.to_csv(f'{output_path}metadata/species_idx.csv', index=True)
species_idx.to_json(f'{output_path}metadata/species_idx.json', orient='records', index=False)
species_idx.to_json(f'{output_path}metadata/species_idx.json', orient='split', index=False)

# 학명 데이터 처리
species_names = pd.read_json(species_name_path, orient="index")
species_names = species_names.reset_index()
species_names.index.name = 'species_idx'
species_names = species_names.rename(columns={"index": "species_id", 0: "species_name"})
# species_names.to_json(f'{output_path}metadata/species_names.json', orient='index')
species_names.to_csv(f'{output_path}metadata/species_names.csv', index=True)
species_names.to_json(f'{output_path}metadata/species_names.json', orient='records', index=False)
species_names.to_json(f'{output_path}metadata/species_names.json', orient='split', index=False)

# 인덱스 - id - 학명
# idx_name_mapping = pd.merge(species_idx, species_names, on='species_id', how='left')
# species_names = species_names.reset_index()
# idx_name_mapping.to_csv(f'{output_path}metadata/species_idx_name_mapping.csv', index=True)
# idx_name_mapping.to_json(f'{output_path}metadata/species_idx_name_mapping.json', index=False)

print("메타데이터 파일이 생성되었습니다.")
