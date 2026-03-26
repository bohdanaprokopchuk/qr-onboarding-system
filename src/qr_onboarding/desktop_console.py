from __future__ import annotations

import csv
import json
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

from .camera import LinuxCameraSource
from .enhanced_pipeline import EnhancedQRSystem
from .payload_codecs import (
    armor_binary_payload,
    encode_cbor_v1,
    encode_json_v1,
    encode_x25519_cbor_v1,
    encode_x25519_raw_json_v1,
    payload_is_text_friendly,
)
from .qr_generation import build_binary_payload_qr, build_split_qr_pngs, build_text_qr
from .split_qr import chunk_texts


PAYLOAD_TEMPLATES: dict[str, dict[str, Any]] = {
    'Wi-Fi onboarding': {
        'ssid': 'LabWiFi',
        'psk': 'DemoPassword123',
        'CC': 'UA',
        'registration-id': 'rid-demo',
        'registration-token': 'rtk-demo',
    },
    'Compact registration': {
        'ssid': 'L',
        'registration-id': 'rid-x1',
        'registration-token': 'tk-x1',
    },
    'Provisioning bundle': {
        'ssid': 'FactoryFloor',
        'psk': 'Provisioning2026',
        'CC': 'UA',
        'registration-id': 'rid-factory-001',
        'registration-token': 'rtk-factory-001',
        'bootstrap-url': 'http://127.0.0.1:8001',
        'device-id': 'edge-demo-001',
    },
    'Consent token': {
        'subject': 'device-owner',
        'purpose': 'wifi-bootstrap',
        'ttl_seconds': 300,
        'metadata': {'channel': 'desktop-ui'},
    },
}


STATUS_STYLES = {
    'success': ('Success', '#0f7b4d'),
    'partial': ('Partial', '#b36a00'),
    'error': ('Failed', '#b3261e'),
    'idle': ('Ready', '#475467'),
}


def _select_payload_bytes(payload: dict, encrypted: bool, public_key_hex: Optional[str], payload_codec: str) -> tuple[bytes, str]:
    codec = (payload_codec or 'auto').lower()
    if codec not in {'auto', 'json', 'cbor'}:
        raise ValueError('payload codec must be auto, json, or cbor')
    if encrypted and not public_key_hex:
        raise ValueError('public key is required for encrypted mode')
    if encrypted:
        options = {
            'json': encode_x25519_raw_json_v1(payload, public_key_hex),
            'cbor': encode_x25519_cbor_v1(payload, public_key_hex),
        }
    else:
        options = {'json': encode_json_v1(payload), 'cbor': encode_cbor_v1(payload)}
    chosen = min(options.items(), key=lambda item: len(item[1])) if codec == 'auto' else (codec, options[codec])
    return chosen[1], chosen[0]


def _png_bytes_to_image(data: bytes) -> Image.Image:
    arr = np.frombuffer(data, dtype=np.uint8)
    decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError('failed to decode generated PNG bytes')
    return Image.fromarray(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB))


def _make_montage(images: list[Image.Image], tile_size: int = 260, columns: int = 2) -> Image.Image:
    columns = max(1, min(columns, len(images) if images else 1))
    rows = (len(images) + columns - 1) // columns
    canvas = Image.new('RGB', (columns * tile_size, rows * tile_size), 'white')
    for idx, image in enumerate(images):
        thumb = image.copy()
        thumb.thumbnail((tile_size - 12, tile_size - 12))
        x = (idx % columns) * tile_size + 6
        y = (idx // columns) * tile_size + 6
        canvas.paste(thumb, (x, y))
    return canvas


def _resize_for_preview(image: Image.Image, max_size: tuple[int, int]) -> Image.Image:
    out = image.copy()
    out.thumbnail(max_size)
    return out


def _pretty_json(data: Any) -> str:
    try:
        return json.dumps(data, ensure_ascii=False, indent=2)
    except Exception:
        return str(data)


def _status_word(result_dict: dict[str, Any]) -> str:
    if result_dict.get('success'):
        return 'SUCCESS'
    if result_dict.get('partial_success'):
        return 'PARTIAL SUCCESS'
    return 'FAILED'


def _method_explanation(result_dict: dict[str, Any]) -> tuple[str, str]:
    base = result_dict.get('base_result') or {}
    payload = base.get('parsed_payload') or {}
    quality = base.get('quality') or {}
    attempts = base.get('attempts') or []
    notes = result_dict.get('notes') or []
    stage = result_dict.get('enhancement_stage') or base.get('stage') or 'unknown'
    decoder = base.get('decoder')
    used: list[str] = []
    why: list[str] = []
    used_seen: set[str] = set()
    why_seen: set[str] = set()

    def add_used(line: str) -> None:
        if line and line not in used_seen:
            used.append(f'• {line}')
            used_seen.add(line)

    def add_why(line: str) -> None:
        if line and line not in why_seen:
            why.append(f'• {line}')
            why_seen.add(line)

    add_used(f'Overall verdict: {_status_word(result_dict)}.')
    if decoder:
        add_used(f'Decoder selected for the final read: {decoder}.')
    if stage == 'direct' and base.get('success'):
        add_used('Direct decoding was sufficient, so no heavier recovery stage was needed.')
    elif stage and stage != 'unknown':
        add_used(f'The decisive recovery stage was {stage}.')
    if result_dict.get('roi_used'):
        add_used('ROI tracking was used to focus decoding on the most relevant QR region.')
    if result_dict.get('camera_adaptation'):
        add_used('Adaptive camera metadata was present for this frame and was included in the decision context.')
    if stage == 'multi-frame-fusion':
        add_used('Multi-frame fusion contributed to the successful decode across consecutive frames.')
    if payload.get('payload_kind') == 'split-chunk':
        add_used('Split-QR parsing was used for this case, so the scan was interpreted as one chunk of a larger transport sequence.')
    if result_dict.get('split_progress'):
        add_used(f"Split-QR assembly state: {result_dict.get('split_progress')}.")
    if result_dict.get('partial_success'):
        add_used('This is a partial success because the current chunk was captured, but the full assembled payload is not complete yet.')

    successful_attempts = [a for a in attempts if a.get('success')]
    if successful_attempts:
        stages = ', '.join(sorted({f"{a.get('stage')} via {a.get('decoder')}" for a in successful_attempts if a.get('stage') and a.get('decoder')}))
        if stages:
            add_used(f'Successful attempt trace: {stages}.')
    elif attempts:
        tried = ', '.join(sorted({f"{a.get('stage')} via {a.get('decoder')}" for a in attempts if a.get('stage') and a.get('decoder')}))
        if tried:
            add_used(f'Attempted methods before the final outcome: {tried}.')

    note_map = [
        ('screen-like moire or watermark artifacts detected', 'Screen-like moire or watermark patterns were detected, so decoder robustness against display artifacts mattered here.'),
        ('low-light detected', 'Low-light conditions were detected, so illumination-aware handling was relevant for this case.'),
        ('low sharpness detected', 'Reduced sharpness was detected, so blur-tolerant recovery mattered for this case.'),
        ('small QR projection detected', 'The QR footprint was small in the image, so scale-sensitive recovery was important.'),
        ('oversized QR crop detected', 'The QR occupied too much of the frame, which can damage locator geometry and require crop-aware handling.'),
        ('glare or low local contrast detected', 'Local contrast issues or glare were detected, so contrast-sensitive recovery was relevant.'),
        ('no locator match, trying distance or soft-focus strategy', 'The locator was weak, so the pipeline switched toward distance or soft-focus recovery logic.'),
        ('using calibrated thresholds', 'Calibrated thresholds were active, so the decision logic was adapted to observed image statistics rather than fixed constants.'),
        ('Decode succeeded after online pipeline switch:', 'An online pipeline switch changed the preprocessing strategy and directly enabled the successful decode.'),
        ('Direct decode succeeded', 'The original image quality was sufficient for a direct read, so no additional recovery cost was required.'),
        ('Decode recovered via multi-frame fusion', 'Temporal fusion across frames contributed to the recovery, which is useful in unstable live-camera conditions.'),
        ('Decode recovered inside tracked ROI', 'Tracking the QR region reduced the search area and helped the system concentrate on the most informative pixels.'),
        ('Provisioning pipeline executed', 'The decoded payload was strong enough to continue into the provisioning stage.'),
        ('Parity reconstruction recovered missing split chunk', 'Parity information allowed the system to reconstruct a missing split chunk and finish assembly.'),
    ]
    for note in notes:
        matched = False
        for prefix, line in note_map:
            if note.startswith(prefix):
                add_why(line)
                matched = True
                break
        if not matched:
            add_why(note[0].upper() + note[1:] + ('' if note.endswith('.') else '.'))

    if quality.get('operator_hint'):
        add_why(f"Operator-facing quality hint: {quality.get('operator_hint')}.")

    if not why:
        if result_dict.get('success'):
            add_why('The available evidence was sufficient for a successful final decode.')
        elif result_dict.get('partial_success'):
            add_why('The scan produced a usable intermediate result, but full completion still requires additional chunks or frames.')
        else:
            add_why('The available evidence was not strong enough for a complete decode under the current conditions.')

    return '\n'.join(used), '\n'.join(why)


@dataclass
class GeneratedPreview:
    image: Image.Image
    log_text: str
    saved_paths: list[str]


class ScrollableTabFrame:
    def __init__(self, parent, tk, ttk, background: str) -> None:
        self.tk = tk
        self.ttk = ttk
        self.outer = ttk.Frame(parent)
        self.canvas = tk.Canvas(self.outer, background=background, highlightthickness=0, bd=0)
        self.v_scroll = ttk.Scrollbar(self.outer, orient='vertical', command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.v_scroll.set)
        self.v_scroll.pack(side='right', fill='y')
        self.canvas.pack(side='left', fill='both', expand=True)
        self.body = ttk.Frame(self.canvas)
        self._window_id = self.canvas.create_window((0, 0), window=self.body, anchor='nw')
        self.body.bind('<Configure>', self._sync_scrollregion)
        self.canvas.bind('<Configure>', self._sync_width)
        self.body.bind('<Enter>', self._bind_mousewheel)
        self.body.bind('<Leave>', self._unbind_mousewheel)

    def pack(self, **kwargs) -> None:
        self.outer.pack(**kwargs)

    def _sync_scrollregion(self, _event=None) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox('all'))

    def _sync_width(self, event) -> None:
        self.canvas.itemconfigure(self._window_id, width=event.width)

    def _bind_mousewheel(self, _event=None) -> None:
        self.canvas.bind_all('<MouseWheel>', self._on_mousewheel)
        self.canvas.bind_all('<Button-4>', self._on_mousewheel)
        self.canvas.bind_all('<Button-5>', self._on_mousewheel)

    def _unbind_mousewheel(self, _event=None) -> None:
        self.canvas.unbind_all('<MouseWheel>')
        self.canvas.unbind_all('<Button-4>')
        self.canvas.unbind_all('<Button-5>')

    def _on_mousewheel(self, event) -> None:
        if getattr(event, 'num', None) == 4:
            self.canvas.yview_scroll(-1, 'units')
            return
        if getattr(event, 'num', None) == 5:
            self.canvas.yview_scroll(1, 'units')
            return
        delta = getattr(event, 'delta', 0)
        if delta:
            self.canvas.yview_scroll(int(-delta / 120), 'units')


class DesktopConsole:
    def __init__(self) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.root = tk.Tk()
        self.root.title('QR Onboarding Research Studio')
        self.root.geometry('1500x930')
        self.root.minsize(1180, 780)

        self._configure_style()

        self.system = EnhancedQRSystem(stats_path='pipeline_stats.json')
        self.temp_dir = Path(tempfile.mkdtemp(prefix='qr_onboarding_desktop_'))
        self.results_dir = self.temp_dir / 'results'
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.camera_source: Optional[LinuxCameraSource] = None
        self.camera_thread: Optional[threading.Thread] = None
        self.camera_running = False
        self.dataset_thread: Optional[threading.Thread] = None
        self.dataset_running = False

        self.current_photo = None
        self.current_scan_photo = None
        self.current_camera_photo = None
        self.last_scan_path: Optional[Path] = None
        self.generated_preview: Optional[GeneratedPreview] = None
        self.last_dataset_rows: list[dict[str, Any]] = []
        self.last_dataset_csv: Optional[Path] = None
        self.camera_counters = {'frames': 0, 'success': 0, 'partial': 0, 'fail': 0}
        self.last_scan_result: dict[str, Any] | None = None
        self.last_camera_result: dict[str, Any] | None = None
        self.last_camera_signature = ''
        self.camera_preview_interval_s = 0.06
        self.camera_decode_interval_s = 0.18
        self.camera_text_refresh_interval_s = 0.35

        self._build_ui()
        self._reset_scan_view()
        self._reset_camera_view()

    def _configure_style(self) -> None:
        ttk = self.ttk
        style = ttk.Style(self.root)
        try:
            style.theme_use('clam')
        except Exception:
            pass
        bg = '#f5f7fb'
        panel = '#ffffff'
        text = '#111827'
        muted = '#667085'
        accent = '#2457f5'
        border = '#d9e0ee'
        self.palette = {'bg': bg, 'panel': panel, 'text': text, 'muted': muted, 'accent': accent, 'border': border}
        self.root.configure(bg=bg)
        style.configure('.', background=bg, foreground=text, font=('Segoe UI', 10))
        style.configure('TFrame', background=bg)
        style.configure('Panel.TFrame', background=panel)
        style.configure('Card.TLabelframe', background=panel, bordercolor=border, relief='solid')
        style.configure('Card.TLabelframe.Label', background=panel, foreground=text, font=('Segoe UI', 11, 'bold'))
        style.configure('TLabel', background=bg, foreground=text)
        style.configure('Muted.TLabel', background=bg, foreground=muted)
        style.configure('CardTitle.TLabel', background=panel, foreground=text, font=('Segoe UI', 11, 'bold'))
        style.configure('Hero.TLabel', background=bg, foreground=text, font=('Segoe UI', 24, 'bold'))
        style.configure('Section.TLabel', background=bg, foreground=muted, font=('Segoe UI', 10))
        style.configure('TButton', padding=(10, 6))
        style.configure('Accent.TButton', padding=(12, 8), background=accent, foreground='white')
        style.map('Accent.TButton', background=[('active', '#1b43c5')])
        style.configure('TNotebook', background=bg, borderwidth=0)
        style.configure('TNotebook.Tab', padding=(14, 9), font=('Segoe UI', 10, 'bold'))
        style.configure('Treeview', rowheight=26)
        style.configure('Treeview.Heading', font=('Segoe UI', 10, 'bold'))

    def _build_ui(self) -> None:
        tk, ttk = self.tk, self.ttk
        root = self.root

        shell = ttk.Frame(root, padding=16)
        shell.pack(fill='both', expand=True)

        header = ttk.Frame(shell)
        header.pack(fill='x', pady=(0, 12))
        ttk.Label(header, text='QR Onboarding Research Studio', style='Hero.TLabel').pack(anchor='w')
        ttk.Label(
            header,
            text='A polished desktop dashboard for scan validation, QR generation, and live camera demos with explicit method explanations.',
            style='Section.TLabel',
        ).pack(anchor='w', pady=(6, 0))

        status_row = ttk.Frame(shell)
        status_row.pack(fill='x', pady=(0, 12))
        self.hero_status = tk.StringVar(value='Ready to scan an image, generate a QR, or test the live camera.')
        self.hero_stats = tk.StringVar(value='Compact UI mode enabled · live camera updates are throttled for smoother preview')
        ttk.Label(status_row, textvariable=self.hero_status, style='CardTitle.TLabel').pack(anchor='w')
        ttk.Label(status_row, textvariable=self.hero_stats, style='Muted.TLabel').pack(anchor='w', pady=(4, 0))

        notebook = ttk.Notebook(shell)
        notebook.pack(fill='both', expand=True)
        self.scan_tab = ttk.Frame(notebook, padding=8)
        self.generate_tab = ttk.Frame(notebook, padding=8)
        self.camera_tab = ttk.Frame(notebook, padding=8)
        notebook.add(self.scan_tab, text='Scan studio')
        notebook.add(self.generate_tab, text='Generate studio')
        notebook.add(self.camera_tab, text='Camera live')

        self._build_scan_tab()
        self._build_generate_tab()
        self._build_camera_tab()
        self.root.protocol('WM_DELETE_WINDOW', self._on_close)

    def _make_card(self, parent, title: str):
        frame = self.ttk.LabelFrame(parent, text=title, style='Card.TLabelframe', padding=12)
        return frame

    def _new_metric_vars(self, keys: list[str]) -> dict[str, Any]:
        return {key: self.tk.StringVar(value='—') for key in keys}

    def _build_metric_grid(self, parent, metrics: dict[str, Any], columns: int = 2) -> None:
        items = list(metrics.items())
        for idx, (key, var) in enumerate(items):
            frame = self.ttk.Frame(parent, style='Panel.TFrame')
            frame.grid(row=idx // columns, column=(idx % columns), sticky='nsew', padx=6, pady=6)
            self.ttk.Label(frame, text=key, style='Muted.TLabel').pack(anchor='w')
            self.ttk.Label(frame, textvariable=var, style='CardTitle.TLabel', wraplength=230, justify='left').pack(anchor='w', pady=(2, 0))
        for col in range(columns):
            parent.columnconfigure(col, weight=1)

    def _build_text_panel(self, parent, *, height: int = 10, background: str = '#fbfcfe'):
        text = self.tk.Text(parent, wrap='word', font=('Consolas', 10), height=height, relief='flat', background=background)
        scroll = self.ttk.Scrollbar(parent, orient='vertical', command=text.yview)
        text.configure(yscrollcommand=scroll.set)
        text.pack(side='left', fill='both', expand=True)
        scroll.pack(side='right', fill='y')
        return text

    def _make_scrollable_body(self, parent):
        scrollable = ScrollableTabFrame(parent, self.tk, self.ttk, self.palette['bg'])
        scrollable.pack(fill='both', expand=True)
        return scrollable.body

    def _build_result_tabs(self, parent):
        notebook = self.ttk.Notebook(parent)
        notebook.pack(fill='both', expand=True)
        payload_tab = self.ttk.Frame(notebook, padding=8)
        why_tab = self.ttk.Frame(notebook, padding=8)
        methods_tab = self.ttk.Frame(notebook, padding=8)
        details_tab = self.ttk.Frame(notebook, padding=8)
        notebook.add(payload_tab, text='Decoded payload')
        notebook.add(why_tab, text='Why it happened')
        notebook.add(methods_tab, text='Methods')
        notebook.add(details_tab, text='Technical details')
        return (
            self._build_text_panel(payload_tab, height=14),
            self._build_text_panel(why_tab, height=14),
            self._build_text_panel(methods_tab, height=14),
            self._build_text_panel(details_tab, height=14),
        )

    def _payload_display_text(self, result_dict: dict[str, Any]) -> str:
        base = result_dict.get('base_result') or {}
        payload = base.get('parsed_payload') or {}
        normalized = payload.get('normalized') if isinstance(payload, dict) else None
        sections: list[str] = []
        if isinstance(normalized, dict) and normalized:
            sections.append(_pretty_json(normalized))
        elif payload:
            sections.append(_pretty_json(payload))
        decoded_text = base.get('decoded_text')
        if decoded_text and decoded_text.strip():
            if not sections or decoded_text.strip() not in sections[0]:
                sections.append('Decoded text\n' + decoded_text.strip())
        assembled = result_dict.get('assembled')
        if assembled:
            sections.append('Assembly\n' + _pretty_json(assembled))
        provisioned = result_dict.get('provisioned')
        if provisioned:
            sections.append('Provisioning\n' + _pretty_json(provisioned))
        if result_dict.get('split_progress'):
            sections.append('Split progress\n' + str(result_dict.get('split_progress')))
        if not sections:
            error = result_dict.get('error') or base.get('error')
            if error:
                return f'No decoded payload available yet.\n\nReason: {error}'
            return 'No decoded payload available yet.'
        return '\n\n'.join(sections)

    def _compact_result_json(self, result_dict: dict[str, Any]) -> str:
        base = result_dict.get('base_result') or {}
        quality = base.get('quality') or {}
        payload = base.get('parsed_payload') or {}
        compact = {
            'success': result_dict.get('success', False),
            'partial_success': result_dict.get('partial_success', False),
            'error': result_dict.get('error'),
            'decoder': base.get('decoder'),
            'stage': result_dict.get('enhancement_stage') or base.get('stage'),
            'scenario': result_dict.get('scenario'),
            'payload_kind': payload.get('payload_kind'),
            'normalized_payload': payload.get('normalized'),
            'decoded_text': base.get('decoded_text'),
            'split_progress': result_dict.get('split_progress'),
            'notes': result_dict.get('notes') or [],
            'quality': {
                'brightness': quality.get('mean_brightness'),
                'contrast': quality.get('contrast_stddev'),
                'sharpness': quality.get('laplacian_variance'),
                'operator_hint': quality.get('operator_hint'),
            },
            'roi_used': result_dict.get('roi_used', False),
            'camera_adaptation': result_dict.get('camera_adaptation'),
        }
        return _pretty_json(compact)

    def _pipeline_summary_line(self) -> str:
        try:
            calibrator = self.system.calibrator
            calibration_text = 'ready' if calibrator.is_ready else f'warm-up {calibrator.frames_collected}/{calibrator.warmup_frames}'
        except Exception:
            calibration_text = 'unavailable'
        try:
            summary = self.system.pipeline_stats_summary() or {}
        except Exception:
            summary = {}
        scenario_count = len(summary)
        best_stage = None
        best_wins = -1
        for _scenario, data in summary.items():
            for stage_name, stage_stats in (data.get('stage_stats') or {}).items():
                wins = int(stage_stats.get('wins', 0))
                if wins > best_wins:
                    best_wins = wins
                    best_stage = stage_name
        if best_stage:
            wins_label = 'wins' if best_wins != 1 else 'win'
            return f'Calibration: {calibration_text} · Scenarios tracked: {scenario_count} · Top recovery stage: {best_stage} ({best_wins} {wins_label})'
        return f'Calibration: {calibration_text} · Scenarios tracked: {scenario_count} · No recovery statistics yet.'

    def _build_json_panel(self, parent, title: str):
        frame = self._make_card(parent, title)
        text = self.tk.Text(frame, wrap='word', font=('Consolas', 10), height=10, relief='flat', background='#fbfcfe')
        scroll = self.ttk.Scrollbar(frame, orient='vertical', command=text.yview)
        text.configure(yscrollcommand=scroll.set)
        text.pack(side='left', fill='both', expand=True)
        scroll.pack(side='right', fill='y')
        return frame, text

    def _build_scan_tab(self) -> None:
        tk, ttk = self.tk, self.ttk
        body = self._make_scrollable_body(self.scan_tab)
        container = ttk.PanedWindow(body, orient='horizontal')
        container.pack(fill='both', expand=True)
        left = ttk.Frame(container)
        right = ttk.Frame(container)
        container.add(left, weight=3)
        container.add(right, weight=4)

        controls = self._make_card(left, 'Scan image')
        controls.pack(fill='x', pady=(0, 10))
        button_row = ttk.Frame(controls, style='Panel.TFrame')
        button_row.pack(fill='x')
        ttk.Button(button_row, text='Open image…', width=16, style='Accent.TButton', command=self.open_and_scan_image).pack(side='left')
        ttk.Button(button_row, text='Rescan current', width=14, command=self.rescan_last_image).pack(side='left', padx=(8, 0))
        ttk.Button(button_row, text='Copy payload', width=14, command=lambda: self._copy_text_widget(self.scan_payload_text)).pack(side='left', padx=(8, 0))
        ttk.Button(button_row, text='Copy details', width=14, command=lambda: self._copy_text_widget(self.scan_json_text)).pack(side='left', padx=(8, 0))
        self.scan_path_var = tk.StringVar(value='No image selected')
        ttk.Label(controls, textvariable=self.scan_path_var, style='Muted.TLabel', wraplength=440, justify='left').pack(anchor='w', pady=(10, 0))

        preview_card = self._make_card(left, 'Preview')
        preview_card.pack(fill='both', expand=True)
        preview_card.rowconfigure(0, weight=1)
        preview_card.columnconfigure(0, weight=1)
        self.scan_image_label = tk.Label(preview_card, bg='#fbfcfe', bd=0, anchor='center', text='Open an image to preview the QR area.', fg=self.palette['muted'], font=('Segoe UI', 11))
        self.scan_image_label.grid(row=0, column=0, sticky='nsew')

        self.scan_status_label = ttk.Label(right, text='Waiting for image', foreground=STATUS_STYLES['idle'][1], font=('Segoe UI', 15, 'bold'))
        self.scan_status_label.pack(anchor='w')
        self.scan_status_hint = tk.StringVar(value='Choose an image and the decoded payload will appear here.')
        ttk.Label(right, textvariable=self.scan_status_hint, style='Muted.TLabel', wraplength=900, justify='left').pack(anchor='w', pady=(4, 10))

        self.scan_metric_vars = self._new_metric_vars(['Result', 'Decoder', 'Stage', 'Payload', 'Progress', 'Hint'])
        metric_card = self._make_card(right, 'Scan summary')
        metric_card.pack(fill='x', pady=(0, 10))
        self._build_metric_grid(metric_card, self.scan_metric_vars, columns=3)

        info_card = self._make_card(right, 'Scan output')
        info_card.pack(fill='both', expand=True)
        self.scan_payload_text, self.scan_reason_text, self.scan_methods_text, self.scan_json_text = self._build_result_tabs(info_card)

    def _build_generate_tab(self) -> None:
        tk, ttk = self.tk, self.ttk
        container = ttk.PanedWindow(self.generate_tab, orient='horizontal')
        container.pack(fill='both', expand=True)
        left = ttk.Frame(container)
        right = ttk.Frame(container)
        container.add(left, weight=3)
        container.add(right, weight=4)

        controls = self._make_card(left, 'Generation controls')
        controls.pack(fill='x', pady=(0, 10))
        top = ttk.Frame(controls, style='Panel.TFrame')
        top.pack(fill='x', pady=(0, 8))
        ttk.Label(top, text='Template', style='Muted.TLabel').pack(side='left')
        self.template_var = tk.StringVar(value='Wi-Fi onboarding')
        template_box = ttk.Combobox(top, textvariable=self.template_var, values=list(PAYLOAD_TEMPLATES.keys()), state='readonly', width=24)
        template_box.pack(side='left', padx=(10, 24))
        template_box.bind('<<ComboboxSelected>>', lambda _e: self.load_selected_template())
        ttk.Label(top, text='Payload codec', style='Muted.TLabel').pack(side='left')
        self.codec_var = tk.StringVar(value='auto')
        ttk.Combobox(top, textvariable=self.codec_var, values=['auto', 'json', 'cbor'], width=10, state='readonly').pack(side='left', padx=(10, 24))
        ttk.Label(top, text='Session id', style='Muted.TLabel').pack(side='left')
        self.session_var = tk.StringVar(value='desktop-session')
        ttk.Entry(top, textvariable=self.session_var, width=20).pack(side='left', padx=(10, 0))

        mid = ttk.Frame(controls, style='Panel.TFrame')
        mid.pack(fill='x', pady=(0, 8))
        ttk.Label(mid, text='Public key for encrypted mode', style='Muted.TLabel').pack(side='left')
        self.public_key_var = tk.StringVar()
        ttk.Entry(mid, textvariable=self.public_key_var).pack(side='left', fill='x', expand=True, padx=(10, 0))

        options = ttk.Frame(controls, style='Panel.TFrame')
        options.pack(fill='x', pady=(0, 10))
        self.encrypted_var = tk.BooleanVar(value=False)
        self.compat_var = tk.BooleanVar(value=True)
        self.split_var = tk.BooleanVar(value=False)
        self.parity_var = tk.BooleanVar(value=True)
        self.chunk_size_var = tk.IntVar(value=180)
        ttk.Checkbutton(options, text='Encrypted', variable=self.encrypted_var).pack(side='left', padx=(0, 14))
        ttk.Checkbutton(options, text='Compatibility armor', variable=self.compat_var).pack(side='left', padx=(0, 14))
        ttk.Checkbutton(options, text='Split QR', variable=self.split_var).pack(side='left', padx=(0, 14))
        ttk.Checkbutton(options, text='Parity recovery', variable=self.parity_var).pack(side='left', padx=(0, 14))
        ttk.Label(options, text='Chunk size', style='Muted.TLabel').pack(side='left', padx=(10, 0))
        ttk.Spinbox(options, from_=32, to=512, increment=4, textvariable=self.chunk_size_var, width=8).pack(side='left', padx=(10, 0))

        self.payload_text = tk.Text(left, height=16, wrap='word', font=('Consolas', 11), relief='flat', background='#fbfcfe')
        self.payload_text.pack(fill='both', expand=True, pady=(0, 10))
        self.load_selected_template()

        actions = ttk.Frame(left)
        actions.pack(fill='x')
        ttk.Button(actions, text='Generate QR', style='Accent.TButton', command=self.generate_qr).pack(side='left')
        ttk.Button(actions, text='Save preview', command=self.save_current_generated_preview).pack(side='left', padx=(8, 0))
        ttk.Button(actions, text='Copy payload JSON', command=lambda: self._copy_text_widget(self.payload_text)).pack(side='left', padx=(8, 0))

        self.generate_status_label = ttk.Label(right, text='Ready to generate', foreground=STATUS_STYLES['idle'][1], font=('Segoe UI', 15, 'bold'))
        self.generate_status_label.pack(anchor='w')
        self.generate_status_hint = tk.StringVar(value='Generate a QR and preview the final image or chunk set here.')
        ttk.Label(right, textvariable=self.generate_status_hint, style='Muted.TLabel', wraplength=820).pack(anchor='w', pady=(4, 10))

        preview_card = self._make_card(right, 'Generated preview')
        preview_card.pack(fill='both', expand=True, pady=(0, 10))
        self.generated_label = ttk.Label(preview_card)
        self.generated_label.pack(fill='both', expand=True)

        self.generate_metric_vars = self._new_metric_vars(['Result', 'Mode', 'Codec', 'Payload bytes', 'Chunk count', 'Saved files'])
        metric_card = self._make_card(right, 'Generation summary')
        metric_card.pack(fill='x', pady=(0, 10))
        self._build_metric_grid(metric_card, self.generate_metric_vars, columns=3)

        lower = ttk.PanedWindow(right, orient='horizontal')
        lower.pack(fill='both', expand=True)
        methods_card = self._make_card(lower, 'Methods used and why')
        log_card = self._make_card(lower, 'Generation log')
        lower.add(methods_card, weight=2)
        lower.add(log_card, weight=3)
        self.generate_methods_text = self.tk.Text(methods_card, wrap='word', font=('Consolas', 10), height=10, relief='flat', background='#fbfcfe')
        self.generate_methods_text.pack(fill='both', expand=True)
        self.generate_log = self.tk.Text(log_card, wrap='word', font=('Consolas', 10), height=10, relief='flat', background='#fbfcfe')
        self.generate_log.pack(fill='both', expand=True)
        self._set_text(self.generate_log, 'No QR generated yet.')
        self._set_text(self.generate_methods_text, 'No method summary yet.')

    def _build_camera_tab(self) -> None:
        tk, ttk = self.tk, self.ttk
        body = self._make_scrollable_body(self.camera_tab)
        container = ttk.PanedWindow(body, orient='horizontal')
        container.pack(fill='both', expand=True)
        left = ttk.Frame(container)
        right = ttk.Frame(container)
        container.add(left, weight=3)
        container.add(right, weight=4)

        controls = self._make_card(left, 'Camera')
        controls.pack(fill='x', pady=(0, 10))
        form = ttk.Frame(controls, style='Panel.TFrame')
        form.pack(fill='x')
        ttk.Label(form, text='Device', style='Muted.TLabel').grid(row=0, column=0, sticky='w')
        self.camera_device_var = tk.StringVar(value='0')
        ttk.Entry(form, textvariable=self.camera_device_var, width=10).grid(row=0, column=1, sticky='w', padx=(10, 18))
        ttk.Label(form, text='Private key', style='Muted.TLabel').grid(row=1, column=0, sticky='w', pady=(10, 0))
        self.camera_private_key_var = tk.StringVar()
        ttk.Entry(form, textvariable=self.camera_private_key_var).grid(row=1, column=1, columnspan=3, sticky='ew', padx=(10, 0), pady=(10, 0))
        form.columnconfigure(3, weight=1)

        action_row = ttk.Frame(controls, style='Panel.TFrame')
        action_row.pack(fill='x', pady=(10, 0))
        ttk.Button(action_row, text='Start camera', width=16, style='Accent.TButton', command=self.start_camera).pack(side='left')
        ttk.Button(action_row, text='Stop camera', width=16, command=self.stop_camera).pack(side='left', padx=(8, 0))
        ttk.Button(action_row, text='Copy payload', width=14, command=lambda: self._copy_text_widget(self.camera_payload_text)).pack(side='left', padx=(8, 0))
        ttk.Button(action_row, text='Copy details', width=14, command=lambda: self._copy_text_widget(self.camera_json_text)).pack(side='left', padx=(8, 0))

        self.camera_status_var = tk.StringVar(value='Camera stopped. Live decode now refreshes at a stable lower rate to keep the preview smooth.')
        ttk.Label(controls, textvariable=self.camera_status_var, style='Muted.TLabel', wraplength=460, justify='left').pack(anchor='w', pady=(10, 0))

        preview_card = self._make_card(left, 'Live preview')
        preview_card.pack(fill='both', expand=True)
        preview_card.rowconfigure(0, weight=1)
        preview_card.columnconfigure(0, weight=1)
        self.camera_image_label = tk.Label(preview_card, bg='#fbfcfe', bd=0, anchor='center', text='Start the camera to see the live preview.', fg=self.palette['muted'], font=('Segoe UI', 11))
        self.camera_image_label.grid(row=0, column=0, sticky='nsew')

        self.camera_status_label = ttk.Label(right, text='Camera stopped', foreground=STATUS_STYLES['idle'][1], font=('Segoe UI', 15, 'bold'))
        self.camera_status_label.pack(anchor='w')
        self.camera_status_hint = tk.StringVar(value='The latest decoded payload will appear here during live scanning.')
        ttk.Label(right, textvariable=self.camera_status_hint, style='Muted.TLabel', wraplength=900, justify='left').pack(anchor='w', pady=(4, 10))

        self.camera_metric_vars = self._new_metric_vars(['Result', 'Frames', 'Hits', 'Misses', 'Decoder', 'Stage', 'Payload', 'Progress'])
        camera_metrics_card = self._make_card(right, 'Live summary')
        camera_metrics_card.pack(fill='x', pady=(0, 10))
        self._build_metric_grid(camera_metrics_card, self.camera_metric_vars, columns=4)

        info_card = self._make_card(right, 'Live decode output')
        info_card.pack(fill='both', expand=True)
        self.camera_payload_text, self.camera_reason_text, self.camera_methods_text, self.camera_json_text = self._build_result_tabs(info_card)

    def _build_dataset_tab(self) -> None:
        tk, ttk = self.tk, self.ttk
        top = self._make_card(self.dataset_tab, 'Dataset benchmark runner')
        top.pack(fill='x', pady=(0, 10))
        row = ttk.Frame(top, style='Panel.TFrame')
        row.pack(fill='x')
        self.dataset_folder_var = tk.StringVar(value='')
        self.dataset_pattern_var = tk.StringVar(value='*.png')
        self.dataset_recursive_var = tk.BooleanVar(value=True)
        ttk.Button(row, text='Choose folder…', style='Accent.TButton', command=self.choose_dataset_folder).pack(side='left')
        ttk.Entry(row, textvariable=self.dataset_folder_var).pack(side='left', fill='x', expand=True, padx=(10, 10))
        ttk.Label(row, text='Pattern', style='Muted.TLabel').pack(side='left')
        ttk.Entry(row, textvariable=self.dataset_pattern_var, width=12).pack(side='left', padx=(10, 10))
        ttk.Checkbutton(row, text='Recursive', variable=self.dataset_recursive_var).pack(side='left', padx=(0, 10))
        ttk.Button(row, text='Run benchmark', command=self.run_dataset_benchmark).pack(side='left')
        ttk.Button(row, text='Export CSV', command=self.export_dataset_csv).pack(side='left', padx=(8, 0))

        self.dataset_status_var = tk.StringVar(value='Select a dataset folder to benchmark success rate, enhancement stages, and difficult cases.')
        ttk.Label(top, textvariable=self.dataset_status_var, style='Muted.TLabel', wraplength=1200).pack(anchor='w', pady=(10, 0))

        mid = ttk.PanedWindow(self.dataset_tab, orient='horizontal')
        mid.pack(fill='both', expand=True)
        left = ttk.Frame(mid)
        right = ttk.Frame(mid)
        mid.add(left, weight=3)
        mid.add(right, weight=4)

        summary_card = self._make_card(left, 'Folder summary')
        summary_card.pack(fill='both', expand=True, pady=(0, 10))
        self.dataset_tree = ttk.Treeview(summary_card, columns=('folder', 'total', 'success', 'partial', 'rate'), show='headings', height=12)
        for col, width in [('folder', 220), ('total', 70), ('success', 80), ('partial', 80), ('rate', 80)]:
            self.dataset_tree.heading(col, text=col.title())
            self.dataset_tree.column(col, width=width, anchor='w' if col == 'folder' else 'center')
        self.dataset_tree.pack(fill='both', expand=True)

        failures_card = self._make_card(left, 'Most important failures')
        failures_card.pack(fill='both', expand=True)
        self.dataset_failures_text = tk.Text(failures_card, wrap='word', font=('Consolas', 10), relief='flat', background='#fbfcfe')
        self.dataset_failures_text.pack(fill='both', expand=True)

        self.dataset_metric_vars = self._new_metric_vars(['Files', 'Success rate', 'Partial rate', 'Avg decode ms', 'Best stage', 'Hardest folder'])
        metric_card = self._make_card(right, 'Benchmark overview')
        metric_card.pack(fill='x', pady=(0, 10))
        self._build_metric_grid(metric_card, self.dataset_metric_vars, columns=3)

        raw_card, self.dataset_json_text = self._build_json_panel(right, 'Benchmark details JSON')
        raw_card.pack(fill='both', expand=True)

    def _set_text(self, widget, text: str) -> None:
        widget.delete('1.0', 'end')
        widget.insert('1.0', text)

    def _copy_text_widget(self, widget) -> None:
        text = widget.get('1.0', 'end').strip()
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.hero_status.set('Copied current panel contents to clipboard.')

    def _set_status_label(self, label_widget, mode: str, hint_var, hint_text: str) -> None:
        mode = mode if mode in STATUS_STYLES else 'idle'
        title, color = STATUS_STYLES[mode]
        label_widget.configure(text=title, foreground=color)
        hint_var.set(hint_text)

    def _result_summary(self, result_dict: dict[str, Any]) -> tuple[str, dict[str, str], str, str, str]:
        base = result_dict.get('base_result') or {}
        payload = (base.get('parsed_payload') or {})
        quality = (base.get('quality') or {})
        status = 'idle'
        verdict = _status_word(result_dict)
        hint_text = 'No result yet.'
        if result_dict.get('success'):
            status = 'success'
            hint_text = 'Payload decoded successfully.'
        elif result_dict.get('partial_success'):
            status = 'partial'
            hint_text = 'One chunk was captured, but the full payload is not complete yet.'
        elif result_dict.get('error'):
            status = 'error'
            hint_text = result_dict.get('error') or 'Decode failed.'
        metrics = {
            'Result': verdict,
            'Decoder': base.get('decoder') or '—',
            'Stage': result_dict.get('enhancement_stage') or base.get('stage') or '—',
            'Payload': payload.get('payload_kind') or '—',
            'Progress': result_dict.get('split_progress') or '—',
            'Hint': quality.get('operator_hint') or hint_text,
        }
        methods_text, reason_text = _method_explanation(result_dict)
        return status, metrics, self._payload_display_text(result_dict), self._compact_result_json(result_dict), methods_text, reason_text

    def _reset_scan_view(self) -> None:
        self.scan_status_label.configure(text='Waiting for image', foreground=STATUS_STYLES['idle'][1])
        self.scan_status_hint.set('Choose an image and the decoded payload will appear here.')
        for var in self.scan_metric_vars.values():
            var.set('—')
        self._set_text(self.scan_payload_text, 'Decoded payload will appear here after a successful scan.')
        self._set_text(self.scan_reason_text, 'Why the result happened will appear here.')
        self._set_text(self.scan_methods_text, 'The method trace will appear here.')
        self._set_text(self.scan_json_text, 'Compact technical details will appear here after a scan.')

    def _reset_camera_view(self) -> None:
        self.camera_status_label.configure(text='Camera stopped', foreground=STATUS_STYLES['idle'][1])
        self.camera_status_hint.set('Start the camera and the latest decoded payload will appear here.')
        for var in self.camera_metric_vars.values():
            var.set('—')
        self._set_text(self.camera_payload_text, 'Live decoded payload will appear here.')
        self._set_text(self.camera_reason_text, 'Why the latest result happened will appear here.')
        self._set_text(self.camera_methods_text, 'The method trace for the latest result will appear here.')
        self._set_text(self.camera_json_text, 'Compact technical details will appear here during live scanning.')

    def load_selected_template(self) -> None:
        payload = PAYLOAD_TEMPLATES.get(self.template_var.get(), PAYLOAD_TEMPLATES['Wi-Fi onboarding'])
        self.payload_text.delete('1.0', 'end')
        self.payload_text.insert('1.0', _pretty_json(payload))

    def generate_qr(self) -> None:
        try:
            payload = json.loads(self.payload_text.get('1.0', 'end').strip())
            raw, codec = _select_payload_bytes(payload, self.encrypted_var.get(), self.public_key_var.get().strip() or None, self.codec_var.get())
            session_id = self.session_var.get().strip() or 'desktop-session'
            chunk_size = int(self.chunk_size_var.get())
            saved_paths: list[str] = []
            used_text_armor = self.compat_var.get() and not self.split_var.get() and not payload_is_text_friendly(raw)
            if self.split_var.get():
                chunks = chunk_texts(raw, session_id, max_chunk_bytes=chunk_size, with_parity=self.parity_var.get())
                pngs = build_split_qr_pngs(chunks)
                images = []
                split_dir = self.temp_dir / f'generated_split_{session_id}'
                split_dir.mkdir(parents=True, exist_ok=True)
                for idx, data in enumerate(pngs):
                    path = split_dir / f'{session_id}_chunk_{idx:02d}.png'
                    path.write_bytes(data)
                    saved_paths.append(str(path))
                    images.append(_png_bytes_to_image(data))
                preview = _make_montage(images, columns=2)
                payload_preview = f'{len(chunks)} QR chunks prepared'
                log_payload = {
                    'success': True,
                    'split': True,
                    'chunk_count': len(chunks),
                    'session_id': session_id,
                    'payload_size_bytes': len(raw),
                    'payload_codec': codec,
                    'with_parity': self.parity_var.get(),
                    'saved_paths': saved_paths,
                    'chunk_texts': chunks,
                }
                mode = 'Split QR'
                chunk_count = str(len(chunks))
            else:
                qr_text = armor_binary_payload(raw) if used_text_armor else None
                png = build_text_qr(qr_text) if qr_text is not None else build_binary_payload_qr(raw)
                path = self.temp_dir / f'generated_{int(time.time())}.png'
                path.write_bytes(png)
                saved_paths.append(str(path))
                preview = _png_bytes_to_image(png)
                payload_preview = (qr_text[:120] if qr_text else raw[:48].hex())
                log_payload = {
                    'success': True,
                    'split': False,
                    'payload_size_bytes': len(raw),
                    'payload_codec': codec,
                    'compatibility_text': used_text_armor,
                    'saved_paths': saved_paths,
                    'preview': payload_preview,
                }
                mode = 'Single QR'
                chunk_count = '1'
            log_text = _pretty_json(log_payload)
            self.generated_preview = GeneratedPreview(preview, log_text, saved_paths)
            self._display_pil_image(self.generated_label, preview, 'current_photo', (720, 720))
            method_lines = [
                '• Result: SUCCESS.',
                f"• Payload codec used: {codec.upper()}.",
                f"• Transport mode: {'split QR sequence' if self.split_var.get() else 'single QR image'}.",
                f"• Encryption: {'enabled' if self.encrypted_var.get() else 'disabled'}.",
                f"• Compatibility armor: {'enabled' if used_text_armor else 'not used'}.",
            ]
            if self.split_var.get():
                method_lines.append('• Split transport was chosen to reduce per-symbol density and improve robustness for larger payloads.')
                method_lines.append(f"• Parity recovery is {'enabled' if self.parity_var.get() else 'disabled'}, so one missing chunk {'can' if self.parity_var.get() else 'cannot'} be reconstructed during assembly.")
            else:
                method_lines.append('• Single-QR transport was sufficient because the payload fit into one symbol under the selected encoding settings.')
            self._set_text(self.generate_log, log_text)
            self._set_text(self.generate_methods_text, '\n'.join(method_lines))
            self.generate_metric_vars['Result'].set('SUCCESS')
            self.generate_metric_vars['Mode'].set(mode)
            self.generate_metric_vars['Codec'].set(codec)
            self.generate_metric_vars['Payload bytes'].set(str(len(raw)))
            self.generate_metric_vars['Chunk count'].set(chunk_count)
            self.generate_metric_vars['Saved files'].set(str(len(saved_paths)))
            self._set_status_label(self.generate_status_label, 'success', self.generate_status_hint, 'SUCCESS · QR assets generated successfully. Save the preview or use the PNG files for demos.')
            self.hero_status.set(f'Generated {mode.lower()} using {codec.upper()} encoding.')
        except Exception as exc:
            self._set_status_label(self.generate_status_label, 'error', self.generate_status_hint, f'QR generation failed: {exc}')
            self._set_text(self.generate_log, f'QR generation failed:\n{exc}')
            self.hero_status.set('Generation failed. Fix the payload or options and try again.')

    def _display_pil_image(self, label, image: Image.Image, attr_name: str, max_size: tuple[int, int]) -> None:
        label.update_idletasks()
        width = label.winfo_width()
        height = label.winfo_height()
        dynamic_size = (max_size[0], max_size[1])
        if width > 80 and height > 80:
            dynamic_size = (max(width - 12, 220), max(height - 12, 220))
        preview = _resize_for_preview(image, dynamic_size)
        photo = ImageTk.PhotoImage(preview)
        setattr(self, attr_name, photo)
        label.configure(image=photo, text='')

    def save_current_generated_preview(self) -> None:
        if self.generated_preview is None:
            self._set_text(self.generate_log, 'Nothing to save yet. Generate a QR first.')
            return
        from tkinter import filedialog
        default_name = 'generated_qr.png'
        target = filedialog.asksaveasfilename(
            title='Save generated QR preview',
            defaultextension='.png',
            initialfile=default_name,
            filetypes=[('PNG image', '*.png')],
        )
        if not target:
            return
        self.generated_preview.image.save(target)
        self._set_text(self.generate_log, self.generated_preview.log_text + f'\n\nPreview saved to: {target}')
        self.hero_status.set(f'Generated preview saved to {target}.')

    def open_and_scan_image(self) -> None:
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title='Choose image',
            filetypes=[('Images', '*.png *.jpg *.jpeg *.bmp *.webp'), ('All files', '*.*')],
        )
        if not path:
            return
        self.last_scan_path = Path(path)
        self.scan_path_var.set(path)
        self._scan_path(Path(path))

    def rescan_last_image(self) -> None:
        if self.last_scan_path is None:
            self._set_text(self.scan_json_text, 'No image selected yet.')
            return
        self._scan_path(self.last_scan_path)

    def _scan_path(self, path: Path) -> None:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            self._set_status_label(self.scan_status_label, 'error', self.scan_status_hint, f'Failed to open image: {path}')
            self._set_text(self.scan_json_text, f'Failed to open image: {path}')
            return
        result = self.system.scan_image(image)
        result_dict = result.to_dict()
        display = image.copy()
        polygon = ((result_dict.get('base_result') or {}).get('polygon'))
        if polygon:
            pts = np.asarray(polygon, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(display, [pts], True, (26, 115, 232), 3)
        preview = Image.fromarray(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
        self._display_pil_image(self.scan_image_label, preview, 'current_scan_photo', (720, 720))
        status, metrics, payload_text, raw_text, methods_text, reason_text = self._result_summary(result_dict)
        self.last_scan_result = result_dict
        hint = metrics.get('Hint', 'Scan complete.')
        title_map = {'success': 'Decoded successfully', 'partial': 'Partial live result', 'error': 'No payload decoded', 'idle': 'Waiting for image'}
        self.scan_status_label.configure(text=title_map.get(status, 'Scan result'), foreground=STATUS_STYLES.get(status, STATUS_STYLES['idle'])[1])
        self.scan_status_hint.set(hint)
        for key, var in self.scan_metric_vars.items():
            var.set(metrics.get(key, '—'))
        self._set_text(self.scan_payload_text, payload_text)
        self._set_text(self.scan_reason_text, reason_text)
        self._set_text(self.scan_methods_text, methods_text)
        self._set_text(self.scan_json_text, raw_text)
        self.hero_status.set(f"Scanned {path.name} · {metrics.get('Result', '—')}")
        self.hero_stats.set(self._pipeline_summary_line())

    def start_camera(self) -> None:
        if self.camera_running:
            return
        try:
            device_value = self.camera_device_var.get().strip()
            device: int | str = int(device_value) if device_value.isdigit() else device_value
            self.system = EnhancedQRSystem(private_key=self.camera_private_key_var.get().strip() or None, stats_path='pipeline_stats.json')
            self.camera_source = LinuxCameraSource(device=device, width=960, height=540, fps=24)
            self.camera_source.open()
            self.camera_running = True
            self.camera_counters = {'frames': 0, 'success': 0, 'partial': 0, 'fail': 0}
            self.camera_status_var.set(f'Camera running on device {self.camera_source.device} ({self.camera_source.backend_name or "auto"}). Live decode is intentionally throttled to keep the preview smoother.')
            self.camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
            self.camera_thread.start()
            self.hero_status.set('Camera stream started.')
            self.last_camera_result = None
            self.last_camera_signature = ''
        except Exception as exc:
            self.camera_running = False
            self.camera_source = None
            self.camera_status_var.set(f'Failed to start camera: {exc}')
            self._set_status_label(self.camera_status_label, 'error', self.camera_status_hint, str(exc))

    def stop_camera(self) -> None:
        self.camera_running = False
        if self.camera_source is not None:
            try:
                self.camera_source.release()
            except Exception:
                pass
        self.camera_source = None
        self.camera_status_var.set('Camera stopped. You can restart it at any time.')
        self.camera_status_label.configure(text='Camera stopped', foreground=STATUS_STYLES['idle'][1])
        self.camera_status_hint.set('Start the camera and the latest decoded payload will appear here.')
        self.hero_status.set('Camera stream stopped.')

    def _camera_loop(self) -> None:
        last_preview_push = 0.0
        last_decode_time = 0.0
        last_text_push = 0.0
        latest_result_dict: dict[str, Any] | None = None
        latest_display: np.ndarray | None = None
        while self.camera_running and self.camera_source is not None:
            try:
                frame, decision = self.camera_source.read_adaptive()
                now = time.monotonic()
                should_decode = (now - last_decode_time) >= self.camera_decode_interval_s or latest_result_dict is None
                refresh_text = False
                if should_decode:
                    result = self.system.scan_stream_frame(frame, None if decision is None else decision.to_dict())
                    latest_result_dict = result.to_dict()
                    last_decode_time = now
                    self.camera_counters['frames'] += 1
                    if result.success:
                        self.camera_counters['success'] += 1
                    elif result.partial_success:
                        self.camera_counters['partial'] += 1
                    else:
                        self.camera_counters['fail'] += 1
                    signature = json.dumps({
                        'success': latest_result_dict.get('success'),
                        'partial_success': latest_result_dict.get('partial_success'),
                        'error': latest_result_dict.get('error'),
                        'decoded_text': ((latest_result_dict.get('base_result') or {}).get('decoded_text')),
                        'stage': latest_result_dict.get('enhancement_stage') or ((latest_result_dict.get('base_result') or {}).get('stage')),
                        'split_progress': latest_result_dict.get('split_progress'),
                    }, ensure_ascii=False, sort_keys=True)
                    refresh_text = result.success or result.partial_success or signature != self.last_camera_signature or (now - last_text_push) >= self.camera_text_refresh_interval_s
                    if refresh_text:
                        self.last_camera_signature = signature
                        last_text_push = now
                    display = frame.copy()
                    if result.base_result and result.base_result.polygon:
                        pts = np.asarray(result.base_result.polygon, dtype=np.int32).reshape((-1, 1, 2))
                        cv2.polylines(display, [pts], True, (0, 180, 0), 2)
                    latest_display = display
                else:
                    latest_display = frame
                if latest_display is not None and latest_result_dict is not None and (now - last_preview_push) >= self.camera_preview_interval_s:
                    self.root.after(0, self._update_camera_preview, latest_display, latest_result_dict, refresh_text)
                    last_preview_push = now
            except Exception as exc:
                self.root.after(0, self.camera_status_var.set, f'Camera error: {exc}')
                self.root.after(0, self.stop_camera)
                break
            time.sleep(0.005)

    def _update_camera_preview(self, frame: np.ndarray, result_dict: dict[str, Any], refresh_text: bool = True) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._display_pil_image(self.camera_image_label, Image.fromarray(rgb), 'current_camera_photo', (720, 720))
        status, metrics, payload_text, raw_text, methods_text, reason_text = self._result_summary(result_dict)
        self.last_camera_result = result_dict
        self.camera_metric_vars['Result'].set(metrics.get('Result', '—'))
        self.camera_metric_vars['Frames'].set(str(self.camera_counters['frames']))
        hits = self.camera_counters['success'] + self.camera_counters['partial']
        self.camera_metric_vars['Hits'].set(str(hits))
        self.camera_metric_vars['Misses'].set(str(self.camera_counters['fail']))
        self.camera_metric_vars['Decoder'].set(metrics.get('Decoder', '—'))
        self.camera_metric_vars['Stage'].set(metrics.get('Stage', '—'))
        self.camera_metric_vars['Payload'].set(metrics.get('Payload', '—'))
        self.camera_metric_vars['Progress'].set(metrics.get('Progress', '—'))
        title_map = {'success': 'Live decode success', 'partial': 'Chunk captured', 'error': 'No payload decoded', 'idle': 'Camera waiting'}
        self.camera_status_label.configure(text=title_map.get(status, 'Live decode'), foreground=STATUS_STYLES.get(status, STATUS_STYLES['idle'])[1])
        self.camera_status_hint.set(metrics.get('Hint', 'Live decode updated.'))
        if refresh_text:
            self._set_text(self.camera_payload_text, payload_text)
            self._set_text(self.camera_reason_text, reason_text)
            self._set_text(self.camera_methods_text, methods_text)
            self._set_text(self.camera_json_text, raw_text)
        self.hero_stats.set(self._pipeline_summary_line())

    def choose_dataset_folder(self) -> None:
        from tkinter import filedialog
        folder = filedialog.askdirectory(title='Choose dataset folder')
        if folder:
            self.dataset_folder_var.set(folder)

    def _iter_dataset_paths(self, root: Path, pattern: str, recursive: bool) -> list[Path]:
        iterator = root.rglob(pattern) if recursive else root.glob(pattern)
        return sorted([path for path in iterator if path.is_file()])

    def run_dataset_benchmark(self) -> None:
        if self.dataset_running:
            return
        folder = Path(self.dataset_folder_var.get().strip())
        pattern = self.dataset_pattern_var.get().strip() or '*.png'
        recursive = self.dataset_recursive_var.get()
        if not folder.exists():
            self.dataset_status_var.set('Dataset folder does not exist.')
            return
        self.dataset_running = True
        self.dataset_status_var.set('Benchmark running. Large datasets may take a moment...')
        self.hero_status.set('Dataset benchmark running...')
        self.dataset_thread = threading.Thread(target=self._dataset_worker, args=(folder, pattern, recursive), daemon=True)
        self.dataset_thread.start()

    def _dataset_worker(self, folder: Path, pattern: str, recursive: bool) -> None:
        system = EnhancedQRSystem(stats_path='pipeline_stats.json')
        paths = self._iter_dataset_paths(folder, pattern, recursive)
        rows: list[dict[str, Any]] = []
        summaries: dict[str, dict[str, Any]] = {}
        for idx, path in enumerate(paths, start=1):
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            start = perf_counter()
            if img is None:
                row = {
                    'file': str(path), 'folder': path.parent.name, 'success': False, 'partial_success': False,
                    'scenario': '', 'stage': '', 'payload_kind': '', 'decode_ms': 0.0, 'error': 'Failed to open image',
                }
            else:
                result = system.scan_image(img)
                result_dict = result.to_dict()
                base = result_dict.get('base_result') or {}
                payload = base.get('parsed_payload') or {}
                row = {
                    'file': str(path),
                    'folder': path.parent.name,
                    'success': bool(result_dict.get('success')),
                    'partial_success': bool(result_dict.get('partial_success')),
                    'scenario': result_dict.get('scenario') or '',
                    'stage': result_dict.get('enhancement_stage') or base.get('stage') or '',
                    'payload_kind': payload.get('payload_kind') or '',
                    'decode_ms': round((perf_counter() - start) * 1000.0, 2),
                    'error': result_dict.get('error') or base.get('error') or '',
                }
            bucket = summaries.setdefault(row['folder'], {'total': 0, 'success': 0, 'partial': 0, 'sum_ms': 0.0, 'stage_counts': {}, 'fails': []})
            bucket['total'] += 1
            bucket['success'] += int(row['success'])
            bucket['partial'] += int(row['partial_success'])
            bucket['sum_ms'] += row['decode_ms']
            if row['stage']:
                bucket['stage_counts'][row['stage']] = bucket['stage_counts'].get(row['stage'], 0) + 1
            if not row['success']:
                bucket['fails'].append(Path(row['file']).name)
            rows.append(row)
            if idx % 5 == 0 or idx == len(paths):
                self.root.after(0, self.dataset_status_var.set, f'Benchmark running... {idx}/{len(paths)} files processed')
        self.root.after(0, self._dataset_worker_done, folder, rows, summaries)

    def _dataset_worker_done(self, folder: Path, rows: list[dict[str, Any]], summaries: dict[str, dict[str, Any]]) -> None:
        self.dataset_running = False
        self.last_dataset_rows = rows
        self.last_dataset_csv = self.results_dir / f'dataset_benchmark_{int(time.time())}.csv'
        with self.last_dataset_csv.open('w', newline='', encoding='utf-8') as fp:
            fieldnames = list(rows[0].keys()) if rows else ['file', 'folder', 'success', 'partial_success', 'scenario', 'stage', 'payload_kind', 'decode_ms', 'error']
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        for item in self.dataset_tree.get_children():
            self.dataset_tree.delete(item)
        total = len(rows)
        successes = sum(1 for row in rows if row['success'])
        partials = sum(1 for row in rows if row['partial_success'])
        avg_ms = sum(row['decode_ms'] for row in rows) / total if total else 0.0
        best_stage_counts: dict[str, int] = {}
        hardest_folder = '—'
        hardest_rate = 101.0
        failure_lines: list[str] = []
        for folder_name, bucket in sorted(summaries.items()):
            rate = (bucket['success'] / bucket['total'] * 100.0) if bucket['total'] else 0.0
            partial_rate = (bucket['partial'] / bucket['total'] * 100.0) if bucket['total'] else 0.0
            self.dataset_tree.insert('', 'end', values=(folder_name, bucket['total'], bucket['success'], bucket['partial'], f'{rate:.1f}%'))
            if rate < hardest_rate:
                hardest_rate = rate
                hardest_folder = folder_name
            for stage, count in bucket['stage_counts'].items():
                best_stage_counts[stage] = best_stage_counts.get(stage, 0) + count
            if bucket['fails']:
                failure_lines.append(f'{folder_name}: ' + ', '.join(bucket['fails'][:6]))
                if len(bucket['fails']) > 6:
                    failure_lines[-1] += ' …'
        best_stage = max(best_stage_counts.items(), key=lambda item: item[1])[0] if best_stage_counts else '—'
        self.dataset_metric_vars['Files'].set(str(total))
        self.dataset_metric_vars['Success rate'].set(f'{(successes / total * 100.0) if total else 0.0:.1f}%')
        self.dataset_metric_vars['Partial rate'].set(f'{(partials / total * 100.0) if total else 0.0:.1f}%')
        self.dataset_metric_vars['Avg decode ms'].set(f'{avg_ms:.1f}')
        self.dataset_metric_vars['Best stage'].set(best_stage)
        self.dataset_metric_vars['Hardest folder'].set(hardest_folder)
        self._set_text(self.dataset_failures_text, '\n'.join(failure_lines) if failure_lines else 'No important failures recorded.')
        details = {
            'dataset_root': str(folder),
            'csv_path': str(self.last_dataset_csv),
            'total': total,
            'successes': successes,
            'partials': partials,
            'success_rate': (successes / total * 100.0) if total else 0.0,
            'avg_decode_ms': avg_ms,
            'best_stage': best_stage,
            'hardest_folder': hardest_folder,
            'folder_summaries': {
                name: {
                    'total': bucket['total'],
                    'success': bucket['success'],
                    'partial': bucket['partial'],
                    'avg_decode_ms': bucket['sum_ms'] / bucket['total'] if bucket['total'] else 0.0,
                    'stage_counts': bucket['stage_counts'],
                    'sample_failures': bucket['fails'][:10],
                }
                for name, bucket in sorted(summaries.items())
            },
        }
        self._set_text(self.dataset_json_text, _pretty_json(details))
        self.dataset_status_var.set(f'Benchmark finished. CSV saved to {self.last_dataset_csv}')
        self.hero_status.set('Dataset benchmark finished successfully.')

    def export_dataset_csv(self) -> None:
        if not self.last_dataset_csv or not self.last_dataset_csv.exists():
            self.dataset_status_var.set('No benchmark CSV is available yet.')
            return
        from tkinter import filedialog
        target = filedialog.asksaveasfilename(title='Export benchmark CSV', defaultextension='.csv', initialfile=self.last_dataset_csv.name, filetypes=[('CSV file', '*.csv')])
        if not target:
            return
        Path(target).write_bytes(self.last_dataset_csv.read_bytes())
        self.dataset_status_var.set(f'Benchmark CSV exported to {target}')
        self.hero_status.set('Dataset CSV exported.')

    def _on_close(self) -> None:
        self.stop_camera()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> int:
    DesktopConsole().run()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
