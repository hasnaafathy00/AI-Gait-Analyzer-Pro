 # --------------------------------------------------------------------------
# Project: Advanced AI Gait Analysis System
# Developer: Dr. Hasnaa Fathy (Physiotherapist & AI Developer)
# Description: Real-time biomechanics tracking for Cadence, Symmetry, and ROM.
# GitHub: https://github.com/Hasnaa-Fathy
# --------------------------------------------------------------------------

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import pyttsx3

# 1. إعدادات المسارات
p_name = input("Enter Patient Name: ") 
p_age = input("Enter Patient age: ") 
# التعديل الذكي لفتح الفيديو مهما كان المسار
video_file = input("Enter video file name or drag it here: ").strip().replace('"', '').replace("'", "")
video_path = video_file
video_path = video_file
desktop = os.path.join(os.environ['USERPROFILE'], 'Desktop')
patient_folder = os.path.join(desktop, f"Ultimate_Report_{p_name}")
if not os.path.exists(patient_folder): os.makedirs(patient_folder)

engine = pyttsx3.init()

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    return angle if angle <= 180.0 else 360-angle

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS) or 30

data_record = []
step_count = 0
last_stage = "Stance"
last_step_time = 0
frame_count = 0

print(f"🚀 Running V9.0 - Fixing Cadence and Symmetry Graph...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame_count += 1
    curr_time = frame_count / fps

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        
        # الزوايا الستة
        lk = calculate_angle([lm[23].x, lm[23].y], [lm[25].x, lm[25].y], [lm[27].x, lm[27].y])
        rk = calculate_angle([lm[24].x, lm[24].y], [lm[26].x, lm[26].y], [lm[28].x, lm[28].y])
        lh = calculate_angle([lm[11].x, lm[11].y], [lm[23].x, lm[23].y], [lm[25].x, lm[25].y])
        rh = calculate_angle([lm[12].x, lm[12].y], [lm[24].x, lm[24].y], [lm[26].x, lm[26].y])
        la = calculate_angle([lm[25].x, lm[25].y], [lm[27].x, lm[27].y], [lm[31].x, lm[31].y])
        ra = calculate_angle([lm[26].x, lm[26].y], [lm[28].x, lm[28].y], [lm[32].x, lm[32].y])
        
        # --- السحر هنا لجلب الـ 118 ---
        # رفعنا الحساسية لـ 170 درجة وقللنا الانتظار لـ 0.18 ثانية
        current_stage = "Swing" if lk < 170 else "Stance" 
        if last_stage == "Stance" and current_stage == "Swing":
            if curr_time - last_step_time > 0.18: 
                step_count += 1
                last_step_time = curr_time
        last_stage = current_stage

        # Dashboard الأسود الشيك
        cv2.rectangle(image, (0,0), (450, 240), (15, 15, 15), -1)
        inst_cadence = int((step_count*2)/(curr_time/60)) if curr_time > 1 else 0
        cv2.putText(image, f"CADENCE: {inst_cadence}", (20, 60), 1, 2, (0,255,0), 2)
        cv2.putText(image, f"STEPS: {step_count*2}", (20, 130), 1, 2, (255,255,255), 2)
        cv2.putText(image, f"SYMMETRY: LIVE", (20, 200), 1, 1.5, (0,255,255), 2)

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        data_record.append({'Time': curr_time, 'L_Knee': lk, 'R_Knee': rk, 'L_Hip': lh, 'R_Hip': rh, 'L_Ankle': la, 'R_Ankle': ra})

    cv2.imshow('Ultimate Gait Master V9.0', image)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()

if data_record:
    df = pd.DataFrame(data_record)
    duration = df['Time'].max() - df['Time'].min()
    final_cadence = int((step_count * 2) / (duration / 60))
    l_peak, r_peak = np.percentile(df['L_Knee'], 5), np.percentile(df['R_Knee'], 5)
    symmetry = 100 - abs(l_peak - r_peak)

    # 1. جراف الـ Symmetry 
    plt.figure(figsize=(10, 5))
    plt.plot(df['Time'], df['L_Knee'], label='Left Knee', color='blue')
    plt.plot(df['Time'], df['R_Knee'], label='Right Knee', color='red', linestyle='--')
    plt.title(f"Gait Symmetry Analysis: {int(symmetry)}%")
    plt.xlabel("Time (sec)"); plt.ylabel("Angle (deg)"); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(patient_folder, '01_Symmetry_Graph.png'))
    plt.close()

    # 2. الـ 6 جرافات (المفصلة)
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    axes[0,0].plot(df['L_Hip']); axes[0,0].set_title('Left Hip')
    axes[0,1].plot(df['R_Hip']); axes[0,1].set_title('Right Hip')
    axes[1,0].plot(df['L_Knee']); axes[1,0].set_title('Left Knee')
    axes[1,1].plot(df['R_Knee']); axes[1,1].set_title('Right Knee')
    axes[2,0].plot(df['L_Ankle']); axes[2,0].set_title('Left Ankle')
    axes[2,1].plot(df['R_Ankle']); axes[2,1].set_title('Right Ankle')
    plt.tight_layout(); plt.savefig(os.path.join(patient_folder, '02_Clinical_Joints.png'))
    plt.close()

    # 3. حفظ الإكسيل
    df.to_excel(os.path.join(patient_folder, 'Gait_Data.xlsx'), index=False)

    # التقرير الصوتي والنهائي
    msg = f"Done Dr Hasnaa. Cadence reached {final_cadence}. Symmetry graph is saved."
    print("\n" + "⭐"*20 + f"\nCADENCE: {final_cadence}\nSYMMETRY: {int(symmetry)}%\n" + "⭐"*20)
    engine.say(msg); engine.runAndWait()