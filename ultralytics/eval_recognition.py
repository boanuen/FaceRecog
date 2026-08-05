"""Đo ĐỘ CHÍNH XÁC THẬT trên test set: nhận diện qua DB embedding vs nhãn gốc.
Dùng GT box đo riêng phần NHẬN DIỆN"""
import os, glob, warnings
warnings.filterwarnings("ignore")
import cv2, torch
from recognizer import FaceRecognizer, BASE_DIR

NAMES=['nghia','quan','son','tri','tui']
rec=FaceRecognizer()   # load DB đã tạo
def y2xy(cx,cy,bw,bh,W,H): return (int((cx-bw/2)*W),int((cy-bh/2)*H),int((cx+bw/2)*W),int((cy+bh/2)*H))

# gom (true_id, embedding) trên test
data=[]
for lbl in glob.glob(os.path.join(BASE_DIR,"test","labels","*.txt")):
    with open(lbl) as f: line=f.readline().split()
    if len(line)<5: continue
    cid=int(line[0]); stem=os.path.splitext(os.path.basename(lbl))[0]
    p=os.path.join(BASE_DIR,"test","images",stem+".jpg")
    if not os.path.isfile(p): continue
    im=cv2.imread(p); H,W=im.shape[:2]
    box=y2xy(*map(float,line[1:5]),W,H)
    v=rec._embed_crop(im,box)
    if v is not None: data.append((cid,v))
print(f"Đã embed {len(data)}/127 mặt test\n")

names=rec.db_names
for thr in [0.25,0.30,0.35,0.40,0.45]:
    correct=stranger=wrong=0
    conf={t:{'nghia':0,'quan':0,'son':0,'tri':0,'tui':0,'LẠ':0} for t in NAMES}
    for cid,v in data:
        name,score=rec._match(v,thr)
        true=NAMES[cid]
        if name is None: stranger+=1; conf[true]['LẠ']+=1
        elif name==true: correct+=1; conf[true][name]+=1
        else: wrong+=1; conf[true][name]+=1
    n=len(data)
    print(f"thr={thr}:  đúng {correct}/{n} ({100*correct/n:.1f}%) | sai {wrong} | gọi nhầm-lạ {stranger}")

# chi tiết ở ngưỡng 0.35
print("\n── Ma trận nhầm ở thr=0.35 (hàng=thật, cột=đoán) ──")
thr=0.35
conf={t:{c:0 for c in NAMES+['LẠ']} for t in NAMES}
for cid,v in data:
    name,score=rec._match(v,thr); true=NAMES[cid]
    conf[true][name if name else 'LẠ']+=1
print("        "+"  ".join(f"{c:>5}" for c in NAMES+['LẠ']))
for t in NAMES:
    print(f"{t:>6}  "+"  ".join(f"{conf[t][c]:5d}" for c in NAMES+['LẠ']))
