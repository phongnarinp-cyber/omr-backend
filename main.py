from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64
from pyzbar.pyzbar import decode

app = FastAPI()

# อนุญาตให้ Google Apps Script เรียกใช้งาน API นี้ได้
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_choice_label(index):
    # ปรับจูนให้รองรับ A, B, C, D (สามารถตั้งค่ารับจากภายนอกได้ในอนาคต)
    return ["A", "B", "C", "D", "E"][index]

@app.post("/scan")
async def scan_omr(file: UploadFile = File(...)):
    try:
        # 1. โหลดรูปภาพ
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        src = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 2. อ่าน QR Code ทันทีด้วย pyzbar (แม่นยำกว่า JS 100 เท่า)
        qr_data = None
        decoded_objects = decode(src)
        for obj in decoded_objects:
            qr_text = obj.data.decode("utf-8")
            if "OMR|" in qr_text:
                qr_data = qr_text.split('|')[1]
                break

        # 3. แปลงเป็นภาพสีเทาและปรับแสง (CLAHE) ทะลุเงาดำ
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5)

        # 4. หากรอบกระดาษ (Contour) แบบ Area > 50000 
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        paper_contour = None
        max_area = 0
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 50000:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                if len(approx) == 4 and area > max_area:
                    max_area = area
                    paper_contour = approx

        if paper_contour is None:
            return {"status": "error", "message": "หากรอบกระดาษ 4 มุมไม่พบ โปรดถ่ายให้เห็นขอบกระดาษชัดเจน"}

        # จัดเรียง 4 มุม (Top-Left, Top-Right, Bottom-Right, Bottom-Left)
        pts = paper_contour.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        # 5. ดึงภาพให้ตรงเป๊ะ (Perspective Warp) ขนาด 800x1213
        maxWidth, maxHeight = 800, 1213
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(gray, M, (maxWidth, maxHeight))
        
        # 6. ตัดภาพหัวกระดาษและแปลงเป็น Base64 ส่งกลับไป
        header_roi = warped[0:260, 0:800]
        _, buffer = cv2.imencode('.jpg', header_roi)
        header_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')

        # 7. ทำ Binarization เตรียมตรวจรอยฝน
        warped_thresh = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 12)

        # ==========================================
        # สแกนรหัสนักเรียน (Static Grid)
        # ==========================================
        ID_START_X, ID_STEP_X = 345, 26.5
        ID_START_Y, ID_STEP_Y = 247, 23.5
        student_id = ""
        
        for c in range(5):
            cx = int(ID_START_X + (c * ID_STEP_X))
            col_bubbles = []
            for r in range(10):
                cy = int(ID_START_Y + (r * ID_STEP_Y))
                rx, ry = max(0, cx - 10), max(0, cy - 10)
                roi = warped_thresh[ry:ry+20, rx:rx+20]
                ink_density = cv2.mean(roi)[0]
                col_bubbles.append((str(r), ink_density))
            
            # เรียงหาจุดที่ดำที่สุด
            col_bubbles.sort(key=lambda x: x[1], reverse=True)
            darkest = col_bubbles[0]
            lightest = col_bubbles[-1]
            
            if darkest[1] - lightest[1] > 15 and darkest[1] > 20:
                student_id += darkest[0]
            else:
                student_id += "?"

        # ==========================================
        # สแกนข้อสอบ 100 ข้อ (Static Grid)
        # ==========================================
        EXAM_START_Y, EXAM_STEP_Y = 485, 23.5
        EXAM_COLS_START_X = [95, 295, 495, 695]
        EXAM_STEP_X = 26.3
        
        raw_answers = []
        num_cols = 4
        rows_per_col = 25
        num_choices = 4 # รองรับ A B C D (สามารถปรับปรุงเพื่อรับค่าแบบไดนามิกได้)

        for col in range(num_cols):
            col_start_x = EXAM_COLS_START_X[col]
            for r in range(rows_per_col):
                cy = int(EXAM_START_Y + (r * EXAM_STEP_Y))
                row_bubbles = []
                for c in range(num_choices):
                    cx = int(col_start_x + (c * EXAM_STEP_X))
                    rx, ry = max(0, cx - 10), max(0, cy - 10)
                    roi = warped_thresh[ry:ry+20, rx:rx+20]
                    ink_density = cv2.mean(roi)[0]
                    row_bubbles.append((get_choice_label(c), ink_density))
                
                row_bubbles.sort(key=lambda x: x[1], reverse=True)
                darkest = row_bubbles[0]
                second_darkest = row_bubbles[1]
                lightest = row_bubbles[-1]
                
                if darkest[1] - lightest[1] < 15:
                    raw_answers.append("BLANK")
                elif darkest[1] - second_darkest[1] < 10 and second_darkest[1] > 25:
                    raw_answers.append("MULTIPLE")
                else:
                    raw_answers.append(darkest[0])

        return {
            "status": "success",
            "exam_id": qr_data,
            "student_id": student_id,
            "raw_answers": raw_answers,
            "header_image": header_b64
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}
