from __future__ import annotations
import argparse, json
from pathlib import Path
import segno
from qr_onboarding.crypto_utils import generate_demo_keypair
from qr_onboarding.payload_codecs import encode_cbor_v1, encode_json_v1, encode_x25519_cbor_v1, encode_x25519_raw_json_v1

def make_qr(payload: bytes|str, path: Path, level: str='M')->None:
    qr=segno.make(payload, mode='byte', error=level, micro=False); qr.save(path, scale=12, border=4, dark='black', light='white')

def main()->int:
    parser=argparse.ArgumentParser(); parser.add_argument('--output-dir', default='assets/generated'); args=parser.parse_args(); out=Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    demo={'ssid':'my;"shiny";ssid','psk':'"""\\;;;buzzz;;;\\"""','CC':'US','registration-id':'118f61fdf9f04ff75405a8dc','registration-token':'02c0d822-c0a5-45b8-9db3-c222fdb4390b'}
    kp=generate_demo_keypair(); (out/'demo_private_key.hex').write_text(kp.private_key_hex, encoding='utf-8'); (out/'demo_public_key.hex').write_text(kp.public_key_hex, encoding='utf-8'); (out/'demo_payload.json').write_text(json.dumps(demo, ensure_ascii=False, indent=2), encoding='utf-8')
    payloads={'json_v1': encode_json_v1(demo),'cbor_v1': encode_cbor_v1(demo),'x25519_json_v1': encode_x25519_raw_json_v1(demo, kp.public_key_hex),'x25519_cbor_v1': encode_x25519_cbor_v1(demo, kp.public_key_hex),'wifi_direct':'WIFI:T:WPA;S:my_shiny_ssid;P:correct-horse-battery-staple;;','secure_bootstrap_uri':'https://onboard.example/setup?rid=118f61fdf9f04ff75405a8dc&rtk=02c0d822-c0a5-45b8-9db3-c222fdb4390b'}
    for name,data in payloads.items():
        for level in ['L','M','Q','H']: make_qr(data, out/f'{name}_{level}.png', level)
        (out/f'{name}.hex' if isinstance(data,bytes) else out/f'{name}.txt').write_text(data.hex() if isinstance(data,bytes) else data, encoding='utf-8')
    print(f'Generated assets in {out}'); return 0
if __name__=='__main__': raise SystemExit(main())
