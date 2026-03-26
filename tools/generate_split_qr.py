from __future__ import annotations
import argparse
from pathlib import Path
import qrcode
from qr_onboarding.payload_codecs import encode_x25519_cbor_v1
from qr_onboarding.split_qr import chunk_texts

def _write_qr(text: str, out_path: Path, level: str='M') -> None:
    ec=getattr(qrcode.constants, f'ERROR_CORRECT_{level.upper()}')
    qr=qrcode.QRCode(error_correction=ec, box_size=8, border=4)
    qr.add_data(text)
    qr.make(fit=True)
    qr.make_image(fill_color='black', back_color='white').save(out_path)

def main()->int:
    p=argparse.ArgumentParser()
    p.add_argument('--session-id', default='session-demo')
    p.add_argument('--output-dir', default='assets/generated/split')
    p.add_argument('--public-key')
    p.add_argument('--with-parity', action='store_true')
    args=p.parse_args()
    out=Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload=b'{"ssid":"LabWiFi","registration-id":"rid-demo","registration-token":"rtk-demo"}'
    if args.public_key:
        payload=encode_x25519_cbor_v1({'ssid':'LabWiFi','registration-id':'rid-demo','registration-token':'rtk-demo'}, args.public_key)
    for idx,text in enumerate(chunk_texts(payload, session_id=args.session_id, max_chunk_bytes=96, with_parity=args.with_parity)):
        _write_qr(text, out/f'chunk_{idx:02d}.png')
        (out/f'chunk_{idx:02d}.txt').write_text(text, encoding='utf-8')
    return 0

if __name__=='__main__':
    raise SystemExit(main())
