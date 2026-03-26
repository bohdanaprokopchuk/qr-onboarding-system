from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import cv2
import uvicorn

from .camera import LinuxCameraSource, RaspberryPiCameraSource
from .cloud_service import CloudBootstrapService, DeviceRecord, build_cloud_bootstrap_app
from .enhanced_pipeline import EnhancedQRSystem
from .evaluation import StatisticalEvaluationLoop
from .overlay import draw_overlay
from .pipeline import QRReader
from .provisioning import CloudBootstrapClient, NmcliWifiAdapter, ProvisioningManager, RetryPolicy, WpaCliWifiAdapter
from .web_api import WebApiSystemContext, build_onboarding_web_app


def _add_common_args(p):
    p.add_argument('--private-key')
    p.add_argument('--no-opencv-fallback', action='store_true')


def _add_enhanced_args(p):
    p.add_argument('--stats-path', default='pipeline_stats.json')
    p.add_argument('--calibration-warmup', type=int, default=40)
    p.add_argument('--adapt-after', type=int, default=15)


def command_image(args):
    reader = QRReader(private_key=args.private_key, try_opencv_text_fallback=not args.no_opencv_fallback)
    result = reader.scan_path(args.input)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    if args.output and Path(args.input).exists():
        img = cv2.imread(args.input)
        if img is not None:
            cv2.imwrite(args.output, draw_overlay(img, result))
    return 0 if result.success else 2


def _iter_paths(root: Path, pattern: str, recursive: bool = False):
    if recursive or '**' in pattern:
        return sorted([path for path in root.rglob(pattern.replace('**/', '').replace('**\\', '')) if path.is_file()])
    return sorted([path for path in root.glob(pattern) if path.is_file()])


def command_batch(args):
    root = Path(args.input_dir)
    if not root.exists():
        print(json.dumps({'success': False, 'error': f'Input directory not found: {root}'}, ensure_ascii=False, indent=2))
        return 2
    reader = QRReader(private_key=args.private_key, try_opencv_text_fallback=not args.no_opencv_fallback)
    rows = []
    for path in _iter_paths(root, args.pattern, recursive=getattr(args, 'recursive', False)):
        start = perf_counter()
        r = reader.scan_path(path)
        rows.append({
            'file': str(path),
            'success': r.success,
            'decoder': r.decoder or '',
            'stage': r.stage or '',
            'payload_kind': r.parsed_payload.payload_kind if r.parsed_payload else '',
            'hint': r.quality.operator_hint if r.quality else '',
            'decode_ms': round((perf_counter() - start) * 1000.0, 2),
            'error': r.error or '',
        })
    with open(args.csv, 'w', newline='', encoding='utf-8') as fp:
        fieldnames = list(rows[0].keys()) if rows else ['file', 'success', 'decoder', 'stage', 'payload_kind', 'hint', 'decode_ms', 'error']
        w = csv.DictWriter(fp, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    summary = {
        'files': len(rows),
        'successes': sum(1 for row in rows if row['success']),
        'csv': args.csv,
        'rows': rows,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _prov(args):
    backend = WpaCliWifiAdapter(interface=args.interface) if args.backend == 'wpa_cli' else NmcliWifiAdapter()
    cloud = CloudBootstrapClient(args.cloud_url) if args.cloud_url else None
    return ProvisioningManager(backend, cloud_client=cloud, retry_policy=RetryPolicy(attempts=args.retries, backoff_seconds=args.backoff))


def _camera_source(args):
    if args.camera_stack == 'picamera2':
        return RaspberryPiCameraSource(width=args.width, height=args.height, fps=args.fps)
    return LinuxCameraSource(device=args.device, width=args.width, height=args.height, fps=args.fps)


def _enhanced_system(args, provisioning: bool = False) -> EnhancedQRSystem:
    return EnhancedQRSystem(
        private_key=args.private_key,
        provisioning_manager=_prov(args) if provisioning else None,
        stats_path=args.stats_path,
        calibration_warmup=args.calibration_warmup,
        adapt_after=args.adapt_after,
    )


def command_enhanced_image(args):
    system = _enhanced_system(args, provisioning=args.provision)
    img = cv2.imread(args.input)
    if img is None:
        print(json.dumps({
            'success': False,
            'error': f'Failed to load image: {args.input}',
            'calibration': system.calibration_status(),
            'stats': system.pipeline_stats_summary(),
        }, ensure_ascii=False, indent=2))
        return 2
    result = system.scan_image(img)
    print(json.dumps({'calibration': system.calibration_status(), 'stats': system.pipeline_stats_summary(), 'result': result.to_dict()}, ensure_ascii=False, indent=2))
    return 0 if result.success else 2


def command_enhanced_camera(args):
    system = _enhanced_system(args, provisioning=args.provision)
    source = _camera_source(args)
    source.open()
    print(system.calibration_status())
    try:
        while True:
            frame, decision = source.read_adaptive()
            result = system.scan_stream_frame(frame, None if decision is None else decision.to_dict())
            base = result.base_result
            display = draw_overlay(frame, base) if base is not None else frame
            cv2.imshow('qr-onboarding-research', display)
            key = cv2.waitKey(1) & 0xFF
            if result.success or result.partial_success:
                print(json.dumps({'calibration': system.calibration_status(), 'result': result.to_dict()}, ensure_ascii=False, indent=2))
            if key in (27, ord('q')):
                break
    finally:
        source.release()
        cv2.destroyAllWindows()
    return 0


def command_evaluate(args):
    root = Path(args.input_dir)
    if not root.exists():
        print(json.dumps({'success': False, 'error': f'Input directory not found: {root}'}, ensure_ascii=False, indent=2))
        return 2
    system = _enhanced_system(args)
    loop = StatisticalEvaluationLoop(system)
    dataset = []
    for path in _iter_paths(root, args.pattern, recursive=getattr(args, 'recursive', False)):
        img = cv2.imread(str(path))
        if img is None:
            continue
        dataset.append({'label': path.name, 'image': img, 'expected_substring': args.expected_substring})
    report = loop.run(dataset)
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 0 if report.total else 2


def command_pipeline_stats(args):
    system = _enhanced_system(args)
    print(json.dumps({'calibration': system.calibration_status(), 'stats': system.pipeline_stats_summary()}, ensure_ascii=False, indent=2))
    return 0


def command_bootstrap_service(args):
    service = CloudBootstrapService(ttl_seconds=args.ttl)
    service.store.add_device(DeviceRecord(registration_id=args.registration_id, registration_token=args.registration_token, wifi={'ssid': args.ssid, 'psk': args.psk, 'security': args.security}, metadata={'demo': True}))
    uvicorn.run(build_cloud_bootstrap_app(service), host=args.host, port=args.port)
    return 0


def command_web_api(args):
    context = WebApiSystemContext(EnhancedQRSystem(private_key=args.private_key))
    uvicorn.run(build_onboarding_web_app(context), host=args.host, port=args.port)
    return 0


def command_desktop_console(args):
    from .desktop_console import main as desktop_console_main
    return desktop_console_main()


def build_parser():
    p = argparse.ArgumentParser(description='Research-grade QR onboarding system for Linux and Raspberry Pi cameras')
    sub = p.add_subparsers(dest='command', required=True)
    p1 = sub.add_parser('image'); p1.add_argument('input'); p1.add_argument('--output'); _add_common_args(p1); p1.add_argument('--csv', default='batch_results.csv'); p1.set_defaults(func=command_image)
    p2 = sub.add_parser('batch'); p2.add_argument('input_dir'); p2.add_argument('--pattern', default='*.png'); p2.add_argument('--recursive', action='store_true'); p2.add_argument('--csv', default='batch_results.csv'); _add_common_args(p2); p2.set_defaults(func=command_batch)
    pe = sub.add_parser('enhanced-image'); pe.add_argument('input'); pe.add_argument('--provision', action='store_true'); pe.add_argument('--cloud-url'); pe.add_argument('--backend', default='nmcli', choices=['nmcli', 'wpa_cli']); pe.add_argument('--interface', default='wlan0'); pe.add_argument('--retries', type=int, default=3); pe.add_argument('--backoff', type=float, default=1.0); _add_common_args(pe); _add_enhanced_args(pe); pe.set_defaults(func=command_enhanced_image)
    pc = sub.add_parser('enhanced-camera'); pc.add_argument('--device', default=0); pc.add_argument('--camera-stack', default='v4l2', choices=['v4l2', 'picamera2']); pc.add_argument('--width', type=int, default=1280); pc.add_argument('--height', type=int, default=720); pc.add_argument('--fps', type=int, default=30); pc.add_argument('--provision', action='store_true'); pc.add_argument('--cloud-url'); pc.add_argument('--backend', default='nmcli', choices=['nmcli', 'wpa_cli']); pc.add_argument('--interface', default='wlan0'); pc.add_argument('--retries', type=int, default=3); pc.add_argument('--backoff', type=float, default=1.0); _add_common_args(pc); _add_enhanced_args(pc); pc.set_defaults(func=command_enhanced_camera)
    pv = sub.add_parser('evaluate'); pv.add_argument('input_dir'); pv.add_argument('--pattern', default='*.png'); pv.add_argument('--recursive', action='store_true'); pv.add_argument('--expected-substring'); _add_common_args(pv); _add_enhanced_args(pv); pv.set_defaults(func=command_evaluate)
    ps = sub.add_parser('pipeline-stats'); _add_common_args(ps); _add_enhanced_args(ps); ps.set_defaults(func=command_pipeline_stats)
    pb = sub.add_parser('bootstrap-service'); pb.add_argument('--host', default='127.0.0.1'); pb.add_argument('--port', type=int, default=8001); pb.add_argument('--ttl', type=int, default=300); pb.add_argument('--registration-id', default='rid-demo'); pb.add_argument('--registration-token', default='rtk-demo'); pb.add_argument('--ssid', default='DemoLabWiFi'); pb.add_argument('--psk', default='DemoPassword123'); pb.add_argument('--security', default='WPA-PSK'); pb.set_defaults(func=command_bootstrap_service)
    pw = sub.add_parser('web-api'); pw.add_argument('--host', default='127.0.0.1'); pw.add_argument('--port', type=int, default=8010); _add_common_args(pw); pw.set_defaults(func=command_web_api)
    pd = sub.add_parser('desktop-console'); pd.set_defaults(func=command_desktop_console)
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == '__main__':
    raise SystemExit(main())
