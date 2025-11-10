import io
import json
import re
import pathlib
import torch
import torchvision.transforms as transforms
import pandas as pd
import PIL.Image
import wikipediaapi
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from torchvision.models import alexnet, AlexNet_Weights
from collections import OrderedDict
from googletrans import Translator

# config

# 현재 파일의 위치를 기준으로 경로 설정
current_path = pathlib.Path(__file__).parent
metadata_path = current_path / 'metadata'
best_model_path = current_path / 'model' / 'best_model.pth'

models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 앱 시작 시 모델 및 메타데이터 로드
    print("Loading model and metadata...")
    try:
        # 사용 가능한 장치(GPU/CPU) 자동 감지
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using CUDA device.")
        else:
            device = torch.device("cpu")
            print("Using CPU device.")
        models['device'] = device

        # Load species names first to determine the number of classes and for later lookup
        species_names_df = pd.read_csv(metadata_path / 'species_names.csv')
        models['species_names_df'] = species_names_df  # Store the entire DataFrame
        num_classes = len(species_names_df)
        
        # Load model
        model = alexnet(weights=AlexNet_Weights.DEFAULT)
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes)
        
        checkpoint = torch.load(best_model_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Remove 'module.' prefix if it exists (from DataParallel saving)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        
        model.load_state_dict(new_state_dict)
        model.to(device) # 모델을 감지된 장치로 이동
        model.eval()

        models['plant_model'] = model

        # Wikipedia and Translator API 객체 추가
        models['wiki_en'] = wikipediaapi.Wikipedia(user_agent='PlantNetApp/1.0', language='en')
        models['wiki_ko'] = wikipediaapi.Wikipedia(user_agent='PlantNetApp/1.0', language='ko')
        models['translator'] = Translator()

        print("Model and metadata loaded successfully.")
    except Exception as e:
        print(f"Error loading model or metadata: {e}")
        
    yield
    # 앱 종료 시 정리
    models.clear()
    print("Cleaned up resources.")

# FastAPI 앱 생성
app = FastAPI(lifespan=lifespan)

# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def transform_image(image_bytes):
    # 정규화
    device = models.get('device', torch.device('cpu'))
    image = PIL.Image.open(io.BytesIO(image_bytes)).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # 텐서를 모델과 동일한 장치로 이동
    return transform(image).unsqueeze(0).to(device)

def get_prediction(tensor):
    #위키
    device = models.get('device', torch.device('cpu'))
    model = models.get('plant_model')
    wiki_en = models.get('wiki_en')
    wiki_ko = models.get('wiki_ko')
    translator = models.get('translator')

    if model is None or wiki_en is None or wiki_ko is None or translator is None:
        raise HTTPException(status_code=503, detail="Model or external APIs are not loaded")

    with torch.no_grad():
        tensor = tensor.to(device) # 텐서가 올바른 장치에 있는지 확인
        output = model(tensor)
        _, predicted = torch.max(output, 1)
        class_idx = predicted.item()

        species_info_df = models.get('species_names_df')
        scientific_name = "Unknown"
        if species_info_df is not None:
            predicted_row = species_info_df[species_info_df['species_idx'] == class_idx]
            if not predicted_row.empty:
                scientific_name = predicted_row['species_name'].iloc[0]

        common_name, family, description = "N/A", "N/A", "정보를 찾을 수 없습니다."

        if scientific_name != "Unknown":
            try:
                # 검색을 위해 학명에서 앞 두 단어(속명, 종명)만 사용
                search_name = ' '.join(scientific_name.split()[:2])
                en_page = wiki_en.page(search_name)

                if not en_page.exists():
                    description = f"'{scientific_name}'에 대한 위키백과 페이지를 찾을 수 없습니다."
                else:
                    page_to_use = en_page
                    use_translation = True
                    # 한국어 버전 확인
                    if 'ko' in en_page.langlinks:
                        ko_page_title = en_page.langlinks['ko'].title
                        ko_page = wiki_ko.page(ko_page_title)
                        if ko_page.exists():
                            page_to_use = ko_page
                            use_translation = False # 한국어 페이지가 있으므로 번역 안함
                    
                    description = page_to_use.summary
                    common_name = page_to_use.title
                    content = page_to_use.text

                    # 한국어 페이지가 없을 경우, 영어 설명을 번역
                    if use_translation and description:
                        try:
                            translated = translator.translate(description, src='en', dest='ko')
                            description = translated.text
                            common_name = translator.translate(common_name, src='en', dest='ko').text
                        except Exception as trans_error:
                            # 번역 실패 시 영어 원문을 그대로 사용하고, 콘솔에 로그를 남깁니다.
                            print(f"Translation failed: {trans_error}") # 번역 실패 시 영어 원문 사용

                    # 과(family) 정보 파싱 (한국어 및 영어 모두 고려) # 계속 안나와서 확인했더니 family 클래스가 많고 개수가 변동되는것 같아 뺌
                    # family_match = re.search(r"\|\s*(?:family|과)\s*=\s*\[\[([^\]]+)\]\]", content, re.IGNORECASE)
                    # if family_match:
                    #     family = family_match.group(1).replace("과 (생물)", "").strip()
                    # else:
                    #     # 요약 정보에서 "과:" 또는 "Family:" 패턴으로 한번 더 찾아보기
                    #     family_match_summary = re.search(r"(?:Family|과)\s*:\s*(\w+)", description, re.IGNORECASE)
                    #     if family_match_summary:
                    #         family = family_match_summary.group(1)

            except Exception as e:
                description = f"정보를 검색하는 중 오류가 발생했습니다: {e}"

        return {
            "scientificName": scientific_name,
            "commonName": common_name,
            "family": family,
            "description": description
        }

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """이미지 파일을 받아 식물 분석 결과를 반환하는 엔드포인트"""
    try:
        image_bytes = await file.read()
        tensor = transform_image(image_bytes)
        prediction = get_prediction(tensor)
        return prediction
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {str(e)}")
