from __future__ import annotations
import argparse, csv
import cv2, numpy as np
from qr_onboarding.pipeline import QRReader

def distortions(image):
    out=[('clean',image)]
    lr=cv2.resize(image,None,fx=0.35,fy=0.35,interpolation=cv2.INTER_AREA); lr=cv2.resize(lr,(image.shape[1], image.shape[0]),interpolation=cv2.INTER_NEAREST); out.append(('lowres',lr))
    out.append(('blur',cv2.GaussianBlur(image,(7,7),1.8)))
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY); noise=np.clip(gray.astype(np.int16)+np.random.normal(0,18,gray.shape),0,255).astype(np.uint8); out.append(('noise',cv2.cvtColor(noise,cv2.COLOR_GRAY2BGR)))
    h,w=image.shape[:2]; src=np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]]); dst=np.float32([[18,25],[w-40,0],[w-10,h-15],[0,h-30]]); out.append(('perspective',cv2.warpPerspective(image,cv2.getPerspectiveTransform(src,dst),(w,h))))
    combo=cv2.GaussianBlur(lr,(5,5),1.2); out.append(('combined',np.clip(combo*0.5,0,255).astype('uint8')))
    return out

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('image'); parser.add_argument('--private-key'); parser.add_argument('--csv', default='benchmark_results.csv'); args=parser.parse_args()
    image=cv2.imread(args.image); reader=QRReader(private_key=args.private_key); rows=[]
    for name,distorted in distortions(image):
        result=reader.scan_image(distorted); rows.append({'scenario':name,'success':result.success,'decoder':result.decoder or '','stage':result.stage or '','payload_kind':result.parsed_payload.payload_kind if result.parsed_payload else '','hint':result.quality.operator_hint if result.quality else ''})
    with open(args.csv,'w',newline='',encoding='utf-8') as fp:
        writer=csv.DictWriter(fp, fieldnames=list(rows[0].keys())); writer.writeheader(); writer.writerows(rows)
    [print(r) for r in rows]
if __name__=='__main__': main()
