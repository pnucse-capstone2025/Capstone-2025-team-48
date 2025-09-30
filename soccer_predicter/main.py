#python -m uvicorn main:app --reload
#start index5.html


import cv2
import numpy as np
from ultralytics import YOLO
import tensorflow as tf
import sys
from PIL import ImageFont, ImageDraw, Image
import pickle
from collections import deque, Counter
import os
import pandas as pd
from pathlib import Path
import uuid

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# --- ⚠️ 모델 및 파일 경로 설정 ⚠️ ---
YOLO_MODEL_PATH = 'runs/train/soccer_players_v13/weights/best.pt'
CLASSIFIER_MODEL_PATH = 'soccer_player_classifier_5_classes_efficient.h5' # ⚠️ 5개 클래스로 학습된 새 모델 경로로 변경해야 할 수 있습니다.
FONT_PATH = 'font/NanumSquareR.ttf'

MODEL_DIR = 'stratige/saved_model'
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, 'tactics_lstm_model.h5')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.pkl')

# --- ⚙️ 하이퍼파라미터 및 설정 변경 ---
# ✅ 1. 클래스 5개로 확장
CLASS_NAMES = ['심판', '서울', '전북', '제주', '대구']
SEQUENCE_LENGTH = 60
MAX_PLAYERS = 22 # LSTM 입력은 22명으로 유지
PRE_ANALYSIS_FRAMES = 15 # 자동 팀 탐지를 위해 분석할 초기 프레임 수 (30fps 기준 5초)
IMG_SIZE = (224, 224)

# --- FastAPI App 및 디렉토리 설정 ---
app = FastAPI()
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
UPLOADS_DIR = STATIC_DIR / "uploads"
RESULTS_DIR = STATIC_DIR / "results"
UPLOADS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
jobs = {}

# --- 모델 로드 ---
print("Loading all models and preprocessors...")
try:
    font = ImageFont.truetype(FONT_PATH, 20)
    yolo_model = YOLO(YOLO_MODEL_PATH)
    classifier_model = tf.keras.models.load_model(CLASSIFIER_MODEL_PATH, compile=False)
    lstm_model = tf.keras.models.load_model(LSTM_MODEL_PATH, compile=False)
    with open(SCALER_PATH, 'rb') as f: scaler = pickle.load(f)
    with open(ENCODER_PATH, 'rb') as f: encoder = pickle.load(f)
    print("All models loaded successfully.")
except Exception as e:
    sys.exit(f"Error loading models: {e}")


# --- 헬퍼 함수 ---
def preprocess_image_for_classifier(img):
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized_img = cv2.resize(rgb_img, IMG_SIZE)
    return tf.keras.applications.mobilenet_v2.preprocess_input(np.expand_dims(resized_img, axis=0))

# --- 메인 영상 처리 함수 ---
def process_video_file(job_id: str, input_path: str, output_video_path: str, output_csv_path: str):
    try:
        player_classifications = {}
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened(): raise IOError("Could not open video file.")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0: raise ValueError("Video has no frames.")

        # --- ✅ 2. 자동 팀 탐지 (Pre-analysis Phase) ---
        print(f"Job {job_id}: Starting pre-analysis to detect teams...")
        team_counts = Counter()
        pre_analysis_ids = set()
        for i in range(PRE_ANALYSIS_FRAMES):
            ret, frame = cap.read()
            if not ret: break
            
            results = yolo_model.track(frame, persist=True, tracker='bytetrack.yaml', classes=[0], verbose=False)
            if results[0].boxes.id is not None:
                for box, track_id in zip(results[0].boxes.xyxy.cpu(), results[0].boxes.id.int().cpu().tolist()):
                    if track_id in pre_analysis_ids: continue # 이미 식별된 선수는 카운트에서 제외
                    
                    x1, y1, x2, y2 = [int(c) for c in box]
                    player_crop = frame[y1:y2, x1:x2]
                    if player_crop.size > 0:
                        processed_img = preprocess_image_for_classifier(player_crop)
                        predictions = classifier_model.predict(processed_img, verbose=0)
                        label = CLASS_NAMES[np.argmax(predictions)]
                        
                        if label != '심판':
                            team_counts[label] += 1
                            pre_analysis_ids.add(track_id)

        # 가장 많이 등장한 2개 팀을 선정
        detected_teams = [team for team, count in team_counts.most_common(2)]
        if len(detected_teams) < 2:
            raise ValueError(f"Could not detect two distinct teams. Detected: {detected_teams}")
        
        active_classes = detected_teams + ['심판']
        team_a, team_b = detected_teams[0], detected_teams[1]
        print(f"Job {job_id}: Teams detected -> {team_a} vs {team_b}. Starting main analysis...")

        # --- 메인 분석을 위해 비디오를 처음으로 되돌림 ---
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # --- 메인 분석 (Main Analysis Phase) ---
        sequence_data = deque(maxlen=SEQUENCE_LENGTH)
        predicted_tactic = "Analyzing..."
        position_data_for_csv = []
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame_count += 1
            results = yolo_model.track(frame, persist=True, tracker='bytetrack.yaml', classes=[0], verbose=False)
            
            current_frame_coords = []
            
            # --- ✅ 3. 추적 인원 제한 로직 ---
            team_a_tracked = 0
            team_b_tracked = 0
            referee_tracked = 0

            if results[0].boxes.id is not None:
                output_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(output_pil)

                for box, track_id in zip(results[0].boxes.xyxy.cpu(), results[0].boxes.id.int().cpu().tolist()):
                    label = player_classifications.get(track_id)
                    if label is None: # 처음 보는 선수면 분류
                        x1, y1, x2, y2 = [int(c) for c in box]
                        player_crop = frame[y1:y2, x1:x2]
                        if player_crop.size > 0:
                            processed_img = preprocess_image_for_classifier(player_crop)
                            predictions = classifier_model.predict(processed_img, verbose=0)
                            label = CLASS_NAMES[np.argmax(predictions)]
                            player_classifications[track_id] = label
                    
                    if label in active_classes:
                        # 인원수 제한 체크
                        if label == team_a and team_a_tracked >= 10: continue
                        if label == team_b and team_b_tracked >= 10: continue
                        if label == '심판' and referee_tracked >= 1: continue

                        # 인원수 증가
                        if label == team_a: team_a_tracked += 1
                        elif label == team_b: team_b_tracked += 1
                        elif label == '심판': referee_tracked += 1

                        # 시각화 및 데이터 수집
                        x1, y1, x2, y2 = [int(c) for c in box]
                        if "서울" in label: color = (255, 0, 0)
                        elif "전북" in label: color = (0, 128, 0)
                        elif "제주" in label: color = (255, 165, 0)
                        elif "대구" in label: color = (0, 191, 255)
                        else: color = (0, 255, 255) # 심판
                        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                        draw.text((x1, y1 - 25), f"ID {track_id}: {label}", font=font, fill=color)

                        if label != '심판':
                            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                            current_frame_coords.append([cx, cy])
                            position_data_for_csv.append([frame_count, track_id, label, cx, cy])
                
                frame_with_boxes = cv2.cvtColor(np.array(output_pil), cv2.COLOR_RGB2BGR)
            else:
                frame_with_boxes = frame.copy()


            # LSTM 전술 분석
            scaled_coords = scaler.transform(np.array(current_frame_coords)) if len(current_frame_coords) > 0 else np.array([])
            padded_coords = np.zeros((MAX_PLAYERS, 2))
            data_len = min(len(scaled_coords), MAX_PLAYERS)
            if data_len > 0: padded_coords[:data_len] = scaled_coords[:data_len]
            sequence_data.append(padded_coords)

            if len(sequence_data) == SEQUENCE_LENGTH:
                input_data = np.expand_dims(np.array(sequence_data), axis=0)
                prediction = lstm_model.predict(input_data, verbose=0)
                predicted_tactic = encoder.inverse_transform([np.argmax(prediction)])[0]
                            # --- ✨ 여기가 수정/추가된 부분입니다! ✨ ---
                # 1. 모델이 원래 학습했던 기준 팀 이름을 정의합니다.
                #    (만약 다른 팀 이름으로 학습했다면 이 부분을 수정하세요)
                predicted_class = encoder.inverse_transform([np.argmax(prediction)])[0]
                original_team_a = '서울'
                original_team_b = '전북'

                # 2. 예측된 전술 이름에서 기준 팀 이름을 현재 탐지된 팀 이름으로 교체합니다.
                #    team_a와 team_b는 영상 앞부분에서 자동 탐지된 팀입니다.
                dynamic_tactic = predicted_class.replace(original_team_a, team_a)
                dynamic_tactic = dynamic_tactic.replace(original_team_b, team_b)
                
                predicted_tactic = dynamic_tactic
                                    # --- ✨ 수정 완료 ✨ ---

            # 최종 프레임에 전술 텍스트 그리기
            final_pil = Image.fromarray(cv2.cvtColor(frame_with_boxes, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(final_pil)
            text_font = ImageFont.truetype(FONT_PATH, 24)
            tactic_text = f"Tactic: {predicted_tactic}"
            text_bbox = draw.textbbox((15, 15), tactic_text, font=text_font)
            draw.rectangle((text_bbox[0] - 5, text_bbox[1] - 5, text_bbox[2] + 5, text_bbox[3] + 5), fill="black")
            draw.text((15, 15), tactic_text, font=text_font, fill="white")
            final_frame = cv2.cvtColor(np.array(final_pil), cv2.COLOR_RGB2BGR)

            video_writer.write(final_frame)
            jobs[job_id]['progress'] = (frame_count / total_frames) * 100

        cap.release()
        video_writer.release()
        
        df = pd.DataFrame(position_data_for_csv, columns=['frame', 'id', 'team', 'x', 'y'])
        df.to_csv(output_csv_path, index=False)
        
        jobs[job_id].update({
            'status': 'done', 'progress': 100,
            'result_video_url': f"/static/results/{Path(output_video_path).name}",
            'result_csv_url': f"/static/results/{Path(output_csv_path).name}"
        })
        print(f"Job {job_id} finished successfully.")

    except Exception as e:
        print(f"Error processing video for job {job_id}: {e}")
        jobs[job_id].update({'status': 'error', 'message': str(e)})


# --- API Endpoints (기존과 동일) ---
@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    file_id = f"{uuid.uuid4()}"
    file_path = UPLOADS_DIR / f"{file_id}{Path(file.filename).suffix}"
    with open(file_path, "wb") as buffer: buffer.write(await file.read())
    return {"file_id": file_id, "video_url": f"/static/uploads/{file_path.name}"}

@app.post("/api/analyze")
async def analyze_video(background_tasks: BackgroundTasks, file_id: str = Form(...)):
    job_id = str(uuid.uuid4())
    uploaded_file = next((f for f in UPLOADS_DIR.iterdir() if f.stem == file_id), None)
    if not uploaded_file: raise HTTPException(404, "File not found.")
    
    output_video_path = str(RESULTS_DIR / f"result_{file_id}.mp4")
    output_csv_path = str(RESULTS_DIR / f"result_{file_id}.csv")
    jobs[job_id] = {"status": "processing", "progress": 0}
    background_tasks.add_task(process_video_file, job_id, str(uploaded_file), output_video_path, output_csv_path)
    return {"job_id": job_id}

@app.get("/api/status/{job_id}")
async def get_job_status(job_id: str):
    if not (job := jobs.get(job_id)): raise HTTPException(404, "Job not found.")
    return job

@app.get("/api/result/{job_id}")
async def get_job_result(job_id: str):
    job = jobs.get(job_id)
    if not job or job.get('status') != 'done': raise HTTPException(404, "Result not ready.")
    return {"video_url": job.get('result_video_url'), "csv_url": job.get('result_csv_url')}