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
    'idle': ('Idle', '#475467'),
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


@dataclass
class GeneratedPreview:
    image: Image.Image
    log_text: str
    saved_paths: list[str]


class DesktopConsole:
    def __init__(self) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.root = tk.Tk()
        self.root.title('QR Onboarding Research Studio')

        self._wrap_labels: list[tuple[Any, int]] = []
        self._responsive_pairs: list[tuple[Any, Any, Any]] = []
        self._layout_mode = 'unknown'
        self._configure_after_id = None
        self._configure_window()
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
        self._image_sources: dict[str, tuple[Image.Image, tuple[int, int]]] = {}
        self._last_camera_result: dict[str, Any] | None = None
        self._last_camera_polygon: list[list[int]] | None = None
        self._last_camera_ui_ts = 0.0
        self._last_camera_preview_ts = 0.0
        self._last_camera_decode_ts = 0.0
        self.camera_decode_fps = 7
        self.camera_processing_max_dim = 720

        self._build_ui()
        self._reset_scan_view()
        self._reset_camera_view()
        self.root.bind('<Configure>', self._on_window_configure, add='+')

    def _configure_window(self) -> None:
        self.root.update_idletasks()
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()

        width = min(max(1100, int(screen_w * 0.94)), screen_w)
        height = min(max(700, int(screen_h * 0.90)), screen_h)

        x = max(0, (screen_w - width) // 2)
        y = max(0, (screen_h - height) // 2)

        self.root.geometry(f'{width}x{height}+{x}+{y}')
        self.root.minsize(980, 640)

        try:
            self.root.state('zoomed')
        except Exception:
            pass

    def _register_wrap_label(self, widget, padding: int = 40) -> None:
        self._wrap_labels.append((widget, padding))

    def _update_wrap_labels(self) -> None:
        width = max(640, self.root.winfo_width())
        for widget, padding in self._wrap_labels:
            try:
                widget.configure(wraplength=max(320, width - padding))
            except Exception:
                pass

    def _register_responsive_pair(self, parent, left, right) -> None:
        self._responsive_pairs.append((parent, left, right))

    def _apply_responsive_layout(self) -> None:
        width = max(980, self.root.winfo_width())

        if self._layout_mode == 'wide':
            mode = 'stacked' if width < 1380 else 'wide'
        elif self._layout_mode == 'stacked':
            mode = 'wide' if width > 1500 else 'stacked'
        else:
            mode = 'stacked' if width < 1450 else 'wide'

        if mode == self._layout_mode:
            return

        self._layout_mode = mode
        for parent, left, right in self._responsive_pairs:
            left.grid_forget()
            right.grid_forget()

            if mode == 'stacked':
                parent.columnconfigure(0, weight=1)
                parent.columnconfigure(1, weight=0)
                left.grid(row=0, column=0, sticky='nsew', padx=0, pady=(0, 10))
                right.grid(row=1, column=0, sticky='nsew')
            else:
                parent.columnconfigure(0, weight=5)
                parent.columnconfigure(1, weight=5)
                left.grid(row=0, column=0, sticky='nsew', padx=(0, 8), pady=0)
                right.grid(row=0, column=1, sticky='nsew')

    def _on_window_configure(self, _event=None) -> None:
        if self._configure_after_id is not None:
            try:
                self.root.after_cancel(self._configure_after_id)
            except Exception:
                pass
        self._configure_after_id = self.root.after(50, self._apply_pending_window_updates)

    def _apply_pending_window_updates(self) -> None:
        self._configure_after_id = None
        self._update_wrap_labels()
        self._apply_responsive_layout()

    def _configure_style(self) -> None:
        ttk = self.ttk
        style = ttk.Style(self.root)
        try:
            style.theme_use('clam')
        except Exception:
            pass

        bg = '#f5f7fb'
        panel = '#ffffff'
        text_color = '#111827'
        muted = '#667085'
        accent = '#2457f5'
        border = '#d9e0ee'

        self.palette = {
            'bg': bg,
            'panel': panel,
            'text': text_color,
            'muted': muted,
            'accent': accent,
            'border': border,
        }

        self.root.configure(bg=bg)

        style.configure('.', background=bg, foreground=text_color, font=('Segoe UI', 9))
        style.configure('TFrame', background=bg)
        style.configure('Panel.TFrame', background=panel)

        style.configure(
            'Card.TLabelframe',
            background=panel,
            bordercolor=border,
            relief='solid',
            borderwidth=1,
            padding=8,
        )
        style.configure(
            'Card.TLabelframe.Label',
            background=panel,
            foreground=text_color,
            font=('Segoe UI', 10, 'bold'),
        )

        style.configure('TLabel', background=bg, foreground=text_color)
        style.configure('Muted.TLabel', background=bg, foreground=muted, font=('Segoe UI', 9))
        style.configure('CardTitle.TLabel', background=panel, foreground=text_color, font=('Segoe UI', 10, 'bold'))
        style.configure('Hero.TLabel', background=bg, foreground=text_color, font=('Segoe UI', 20, 'bold'))
        style.configure('Section.TLabel', background=bg, foreground=muted, font=('Segoe UI', 10))

        button_bg = '#ffffff'
        button_hover = '#eef4ff'
        button_pressed = '#dbe7ff'
        button_hover_border = '#8ba6ff'
        button_pressed_border = '#5b7cff'

        style.configure(
            'TButton',
            padding=(14, 8),
            background=button_bg,
            foreground=text_color,
            borderwidth=1,
            relief='solid',
            focusthickness=0,
            focuscolor=button_bg,
            anchor='center',
            font=('Segoe UI', 10, 'bold'),
        )
        style.map(
            'TButton',
            background=[
                ('disabled', '#f5f7fb'),
                ('pressed', button_pressed),
                ('active', button_hover),
            ],
            foreground=[
                ('disabled', '#98a2b3'),
                ('pressed', text_color),
                ('active', text_color),
            ],
            bordercolor=[
                ('disabled', border),
                ('pressed', button_pressed_border),
                ('active', button_hover_border),
            ],
            lightcolor=[
                ('disabled', '#f5f7fb'),
                ('pressed', button_pressed),
                ('active', button_hover),
            ],
            darkcolor=[
                ('disabled', '#f5f7fb'),
                ('pressed', button_pressed),
                ('active', button_hover),
            ],
            relief=[
                ('disabled', 'solid'),
                ('pressed', 'sunken'),
                ('active', 'solid'),
            ],
        )

        style.configure(
            'Accent.TButton',
            padding=(14, 8),
            background=accent,
            foreground='white',
            borderwidth=1,
            relief='solid',
            focusthickness=0,
            focuscolor=accent,
            anchor='center',
            font=('Segoe UI', 10, 'bold'),
        )
        style.map(
            'Accent.TButton',
            background=[
                ('disabled', '#a7b8f8'),
                ('pressed', '#163eb8'),
                ('active', '#2f64ff'),
            ],
            foreground=[
                ('disabled', '#eef2ff'),
                ('pressed', 'white'),
                ('active', 'white'),
            ],
            bordercolor=[
                ('disabled', '#a7b8f8'),
                ('pressed', '#163eb8'),
                ('active', '#2f64ff'),
            ],
            lightcolor=[
                ('disabled', '#a7b8f8'),
                ('pressed', '#163eb8'),
                ('active', '#2f64ff'),
            ],
            darkcolor=[
                ('disabled', '#a7b8f8'),
                ('pressed', '#163eb8'),
                ('active', '#2f64ff'),
            ],
            relief=[
                ('disabled', 'solid'),
                ('pressed', 'sunken'),
                ('active', 'solid'),
            ],
        )

        style.configure('TNotebook', background=bg, borderwidth=0)
        style.configure(
            'TNotebook.Tab',
            padding=(14, 8),
            font=('Segoe UI', 9, 'bold'),
            focuscolor=panel,
        )
        style.map(
            'TNotebook.Tab',
            background=[('selected', panel), ('active', '#eef4ff')],
            foreground=[('selected', text_color), ('active', text_color)],
            lightcolor=[('selected', panel), ('active', '#eef4ff')],
            darkcolor=[('selected', panel), ('active', '#eef4ff')],
        )

        style.configure('Treeview', rowheight=24)
        style.configure('Treeview.Heading', font=('Segoe UI', 9, 'bold'))

    def _apply_button_hover(self, button) -> None:
        try:
            button.configure(cursor='hand2')
        except Exception:
            pass

        def _on_enter(_event):
            try:
                if 'disabled' not in button.state():
                    button.state(['active'])
            except Exception:
                pass

        def _on_leave(_event):
            try:
                button.state(['!active', '!pressed'])
            except Exception:
                pass

        button.bind('<Enter>', _on_enter, add='+')
        button.bind('<Leave>', _on_leave, add='+')

    def _enhance_interactions(self, widget=None) -> None:
        widget = widget or self.root
        try:
            children = widget.winfo_children()
        except Exception:
            return

        for child in children:
            if child.winfo_class() == 'TButton':
                self._apply_button_hover(child)
            elif child.winfo_class() == 'TNotebook':
                try:
                    child.configure(takefocus=False)
                except Exception:
                    pass
            self._enhance_interactions(child)

    def _build_ui(self) -> None:
        tk, ttk = self.tk, self.ttk
        root = self.root

        shell = ttk.Frame(root, padding=10)
        shell.pack(fill='both', expand=True)
        shell.columnconfigure(0, weight=1)
        shell.rowconfigure(2, weight=1)

        header = ttk.Frame(shell)
        header.grid(row=0, column=0, sticky='ew', pady=(0, 6))

        hero = ttk.Label(header, text='QR Onboarding Research Studio', style='Hero.TLabel')
        hero.pack(anchor='w')
        self._register_wrap_label(hero, padding=60)

        subtitle = ttk.Label(
            header,
            text='Desktop dashboard for scan validation, QR generation, and live camera demos with a cleaner responsive layout.',
            style='Section.TLabel',
            justify='left',
        )
        subtitle.pack(anchor='w', pady=(4, 0))
        self._register_wrap_label(subtitle, padding=60)

        status_row = ttk.Frame(shell)
        status_row.grid(row=1, column=0, sticky='ew', pady=(0, 8))

        self.hero_status = tk.StringVar(value='Ready for scan, generation, or camera demo.')
        self.hero_stats = tk.StringVar(value='Calibration: warm-up pending · Pipeline stats: no history yet')

        self.hero_status_label = ttk.Label(
            status_row,
            textvariable=self.hero_status,
            style='CardTitle.TLabel',
            justify='left',
        )
        self.hero_status_label.pack(anchor='w')
        self._register_wrap_label(self.hero_status_label, padding=70)

        self.hero_stats_label = ttk.Label(
            status_row,
            textvariable=self.hero_stats,
            style='Muted.TLabel',
            justify='left',
        )
        self.hero_stats_label.pack(anchor='w', pady=(2, 0))
        self._register_wrap_label(self.hero_stats_label, padding=70)

        notebook = ttk.Notebook(shell, takefocus=False)
        notebook.grid(row=2, column=0, sticky='nsew')

        self.scan_tab = self._make_tab(notebook, 'Scan studio')
        self.generate_tab = self._make_tab(notebook, 'Generate studio')
        self.camera_tab = self._make_tab(notebook, 'Camera live')

        self._build_scan_tab()
        self._build_generate_tab()
        self._build_camera_tab()
        self._enhance_interactions()
        self._update_hero_stats()
        self._update_wrap_labels()
        self._apply_responsive_layout()

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
            self.ttk.Label(frame, textvariable=var, style='CardTitle.TLabel', wraplength=190, justify='left').pack(anchor='w', pady=(2, 0))
        for col in range(columns):
            parent.columnconfigure(col, weight=1)

    def _make_tab(self, notebook, title: str):
        frame = self.ttk.Frame(notebook, padding=6)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        notebook.add(frame, text=title)
        return frame

    def _build_json_panel(self, parent, title: str):
        frame = self._make_card(parent, title)
        body = self.ttk.Frame(frame, style='Panel.TFrame')
        body.pack(fill='both', expand=True)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)

        text = self.tk.Text(
            body,
            wrap='none',
            font=('Consolas', 9),
            height=6,
            relief='flat',
            background='#fbfcfe',
        )
        yscroll = self.ttk.Scrollbar(body, orient='vertical', command=text.yview)
        xscroll = self.ttk.Scrollbar(body, orient='horizontal', command=text.xview)
        text.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)

        text.grid(row=0, column=0, sticky='nsew')
        yscroll.grid(row=0, column=1, sticky='ns')
        xscroll.grid(row=1, column=0, sticky='ew')
        return frame, text

    def _build_scrolled_text(self, parent, height: int = 6, wrap: str = 'word'):
        body = self.ttk.Frame(parent, style='Panel.TFrame')
        body.pack(fill='both', expand=True)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)

        text = self.tk.Text(
            body,
            wrap=wrap,
            font=('Consolas', 9),
            height=height,
            relief='flat',
            background='#fbfcfe',
        )
        yscroll = self.ttk.Scrollbar(body, orient='vertical', command=text.yview)
        text.configure(yscrollcommand=yscroll.set)

        text.grid(row=0, column=0, sticky='nsew')
        yscroll.grid(row=0, column=1, sticky='ns')
        return text

    def _prepare_preview_label(self, label, attr_name: str, max_size: tuple[int, int], empty_text: str) -> None:
        label.configure(text=empty_text, anchor='center', justify='center')
        label.bind('<Configure>', lambda _event, lab=label, attr=attr_name, size=max_size: self._refresh_image_label(lab, attr, size))

    def _refresh_image_label(self, label, attr_name: str, max_size: tuple[int, int]) -> None:
        source = self._image_sources.get(attr_name)
        if not source:
            return

        image, fallback_size = source
        max_w = max_size[0] if max_size else fallback_size[0]
        max_h = max_size[1] if max_size else fallback_size[1]

        width = max(120, label.winfo_width() - 16)
        height = max(120, label.winfo_height() - 16)

        target = (
            min(max_w, width),
            min(max_h, height),
        )

        preview = _resize_for_preview(image, target)
        photo = ImageTk.PhotoImage(preview)
        setattr(self, attr_name, photo)
        label.configure(image=photo, text='')

    def _update_hero_stats(self) -> None:
        calibration = self.system.calibration_status()
        stats = self.system.pipeline_stats_summary() or {}
        top_stage = '—'
        top_wins = 0
        scenario_count = len(stats) if isinstance(stats, dict) else 0
        if isinstance(stats, dict):
            for scenario_payload in stats.values():
                wins = (scenario_payload or {}).get('wins') or {}
                for stage, count in wins.items():
                    if int(count) > top_wins:
                        top_stage = stage
                        top_wins = int(count)
        tail = f'Top recovery stage: {top_stage} ({top_wins} wins)' if top_wins else 'Top recovery stage: no wins yet'
        self.hero_stats.set(f'Calibration: {calibration} · Scenarios tracked: {scenario_count} · {tail}')

    def _build_scan_tab(self) -> None:
        tk, ttk = self.tk, self.ttk

        self.scan_tab.columnconfigure(0, weight=1)
        self.scan_tab.columnconfigure(1, weight=1)
        self.scan_tab.rowconfigure(0, weight=1)
        self.scan_tab.rowconfigure(1, weight=1)

        left = ttk.Frame(self.scan_tab)
        right = ttk.Frame(self.scan_tab)
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)
        right.rowconfigure(2, weight=1)
        right.columnconfigure(0, weight=1)

        self._register_responsive_pair(self.scan_tab, left, right)
        left.grid(row=0, column=0, sticky='nsew', padx=(0, 8))
        right.grid(row=0, column=1, sticky='nsew')

        scan_controls = self._make_card(left, 'Image input')
        scan_controls.grid(row=0, column=0, sticky='ew', pady=(0, 8))

        actions = ttk.Frame(scan_controls, style='Panel.TFrame')
        actions.pack(fill='x')
        buttons = [
            ('Open image…', 'Accent.TButton', self.open_and_scan_image),
            ('Rescan', 'TButton', self.rescan_last_image),
            ('Copy full JSON', 'TButton', lambda: self._copy_text_widget(self.scan_json_text)),
            ('Copy payload only', 'TButton', lambda: self._copy_text_widget(self.scan_payload_text)),
        ]
        for idx, (title, style, command) in enumerate(buttons):
            btn = ttk.Button(actions, text=title, style=style, command=command, takefocus=False)
            btn.grid(row=idx // 2, column=idx % 2, sticky='ew', padx=(0 if idx % 2 == 0 else 6, 0), pady=(0, 6))
        actions.columnconfigure(0, weight=1)
        actions.columnconfigure(1, weight=1)

        self.scan_path_var = tk.StringVar(value='No image selected')
        scan_path = ttk.Label(scan_controls, textvariable=self.scan_path_var, style='Muted.TLabel', justify='left')
        scan_path.pack(anchor='w', pady=(6, 0))
        self._register_wrap_label(scan_path, padding=120)

        preview_card = self._make_card(left, 'Scan preview')
        preview_card.grid(row=1, column=0, sticky='nsew')
        preview_card.columnconfigure(0, weight=1)
        preview_card.rowconfigure(0, weight=1)

        self.scan_image_label = ttk.Label(preview_card)
        self.scan_image_label.grid(row=0, column=0, sticky='nsew')
        self._prepare_preview_label(
            self.scan_image_label,
            'current_scan_photo',
            (760, 520),
            'Scanned image preview will appear here.',
        )

        status_card = self._make_card(right, 'Scan status')
        status_card.grid(row=0, column=0, sticky='ew', pady=(0, 8))
        self.scan_status_label = ttk.Label(
            status_card,
            text='Idle',
            foreground=STATUS_STYLES['idle'][1],
            font=('Segoe UI', 10, 'bold'),
        )
        self.scan_status_label.pack(anchor='w')

        self.scan_status_hint = tk.StringVar(value='Open an image to decode and inspect payload structure.')
        hint = ttk.Label(status_card, textvariable=self.scan_status_hint, style='Muted.TLabel', justify='left')
        hint.pack(anchor='w', pady=(4, 0))
        self._register_wrap_label(hint, padding=120)

        self.scan_metric_vars = self._new_metric_vars([
            'Decoder', 'Stage', 'Scenario', 'Payload kind', 'Split progress',
            'Brightness', 'Contrast', 'Sharpness', 'Operator hint'
        ])
        metric_card = self._make_card(right, 'Decoded summary')
        metric_card.grid(row=1, column=0, sticky='ew', pady=(0, 8))
        self._build_metric_grid(metric_card, self.scan_metric_vars, columns=2)

        details = ttk.Notebook(right, takefocus=False)
        details.grid(row=2, column=0, sticky='nsew')

        notes_tab = ttk.Frame(details, padding=4)
        payload_tab = ttk.Frame(details, padding=4)
        json_tab = ttk.Frame(details, padding=4)
        details.add(notes_tab, text='Notes')
        details.add(payload_tab, text='Payload')
        details.add(json_tab, text='Full JSON')

        notes_card = self._make_card(notes_tab, 'Notes and recovery log')
        notes_card.pack(fill='both', expand=True)
        self.scan_notes_text = self._build_scrolled_text(notes_card, height=7, wrap='word')

        payload_card = self._make_card(payload_tab, 'Normalized payload')
        payload_card.pack(fill='both', expand=True)
        self.scan_payload_text = self._build_scrolled_text(payload_card, height=7, wrap='word')

        json_card, self.scan_json_text = self._build_json_panel(json_tab, 'Full enhanced result JSON')
        json_card.pack(fill='both', expand=True)

    def _build_generate_tab(self) -> None:
        tk, ttk = self.tk, self.ttk

        self.generate_tab.columnconfigure(0, weight=1)
        self.generate_tab.columnconfigure(1, weight=1)
        self.generate_tab.rowconfigure(0, weight=1)
        self.generate_tab.rowconfigure(1, weight=1)

        left = ttk.Frame(self.generate_tab)
        right = ttk.Frame(self.generate_tab)
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)
        right.rowconfigure(2, weight=1)
        right.columnconfigure(0, weight=1)

        self._register_responsive_pair(self.generate_tab, left, right)
        left.grid(row=0, column=0, sticky='nsew', padx=(0, 8))
        right.grid(row=0, column=1, sticky='nsew')

        controls = self._make_card(left, 'Generation controls')
        controls.grid(row=0, column=0, sticky='ew', pady=(0, 8))

        form = ttk.Frame(controls, style='Panel.TFrame')
        form.pack(fill='x')
        for col in (1, 3):
            form.columnconfigure(col, weight=1)

        ttk.Label(form, text='Template', style='Muted.TLabel').grid(row=0, column=0, sticky='w', padx=(0, 8), pady=(0, 6))
        self.template_var = tk.StringVar(value='Wi-Fi onboarding')
        template_box = ttk.Combobox(form, textvariable=self.template_var, values=list(PAYLOAD_TEMPLATES.keys()), state='readonly')
        template_box.grid(row=0, column=1, sticky='ew', pady=(0, 6))
        template_box.bind('<<ComboboxSelected>>', lambda _e: self.load_selected_template())

        ttk.Label(form, text='Payload codec', style='Muted.TLabel').grid(row=0, column=2, sticky='w', padx=(16, 8), pady=(0, 6))
        self.codec_var = tk.StringVar(value='auto')
        ttk.Combobox(form, textvariable=self.codec_var, values=['auto', 'json', 'cbor'], state='readonly', width=12).grid(row=0, column=3, sticky='ew', pady=(0, 6))

        ttk.Label(form, text='Session id', style='Muted.TLabel').grid(row=1, column=0, sticky='w', padx=(0, 8), pady=(0, 6))
        self.session_var = tk.StringVar(value='desktop-session')
        ttk.Entry(form, textvariable=self.session_var).grid(row=1, column=1, sticky='ew', pady=(0, 6))

        ttk.Label(form, text='Chunk size', style='Muted.TLabel').grid(row=1, column=2, sticky='w', padx=(16, 8), pady=(0, 6))
        self.chunk_size_var = tk.IntVar(value=180)
        ttk.Spinbox(form, from_=32, to=512, increment=4, textvariable=self.chunk_size_var, width=10).grid(row=1, column=3, sticky='w', pady=(0, 6))

        ttk.Label(form, text='Public key for encrypted mode', style='Muted.TLabel').grid(row=2, column=0, sticky='w', padx=(0, 8))
        self.public_key_var = tk.StringVar()
        ttk.Entry(form, textvariable=self.public_key_var).grid(row=2, column=1, columnspan=3, sticky='ew')

        options = ttk.Frame(controls, style='Panel.TFrame')
        options.pack(fill='x', pady=(8, 0))
        self.encrypted_var = tk.BooleanVar(value=False)
        self.compat_var = tk.BooleanVar(value=True)
        self.split_var = tk.BooleanVar(value=False)
        self.parity_var = tk.BooleanVar(value=True)

        ttk.Checkbutton(options, text='Encrypted', variable=self.encrypted_var).grid(row=0, column=0, sticky='w', padx=(0, 16), pady=(0, 4))
        ttk.Checkbutton(options, text='Compatibility armor', variable=self.compat_var).grid(row=0, column=1, sticky='w', padx=(0, 16), pady=(0, 4))
        ttk.Checkbutton(options, text='Split QR', variable=self.split_var).grid(row=1, column=0, sticky='w')
        ttk.Checkbutton(options, text='Parity recovery', variable=self.parity_var).grid(row=1, column=1, sticky='w')

        payload_card = self._make_card(left, 'Payload JSON')
        payload_card.grid(row=1, column=0, sticky='nsew', pady=(0, 8))
        self.payload_text = self._build_scrolled_text(payload_card, height=7, wrap='word')
        self.load_selected_template()

        actions = self._make_card(left, 'Actions')
        actions.grid(row=2, column=0, sticky='ew')
        actions_row = ttk.Frame(actions, style='Panel.TFrame')
        actions_row.pack(fill='x')

        action_buttons = [
            ('Generate QR', 'Accent.TButton', self.generate_qr),
            ('Scan generated QR', 'TButton', self.scan_generated_preview),
            ('Save preview', 'TButton', self.save_current_generated_preview),
            ('Copy payload JSON', 'TButton', lambda: self._copy_text_widget(self.payload_text)),
        ]
        for idx, (title, style, command) in enumerate(action_buttons):
            ttk.Button(actions_row, text=title, style=style, command=command, takefocus=False).grid(
                row=idx // 2,
                column=idx % 2,
                sticky='ew',
                padx=(0 if idx % 2 == 0 else 6, 0),
                pady=(0, 6),
            )
        actions_row.columnconfigure(0, weight=1)
        actions_row.columnconfigure(1, weight=1)

        status_card = self._make_card(right, 'Generation status')
        status_card.grid(row=0, column=0, sticky='ew', pady=(0, 8))
        self.generate_status_label = ttk.Label(
            status_card,
            text='Idle',
            foreground=STATUS_STYLES['idle'][1],
            font=('Segoe UI', 10, 'bold'),
        )
        self.generate_status_label.pack(anchor='w')

        self.generate_status_hint = tk.StringVar(value='Generate a QR to see preview, encoding decisions, and verification results.')
        hint = ttk.Label(status_card, textvariable=self.generate_status_hint, style='Muted.TLabel', justify='left')
        hint.pack(anchor='w', pady=(4, 0))
        self._register_wrap_label(hint, padding=120)

        preview_card = self._make_card(right, 'Generated preview')
        preview_card.grid(row=1, column=0, sticky='nsew', pady=(0, 8))
        preview_card.columnconfigure(0, weight=1)
        preview_card.rowconfigure(0, weight=1)

        self.generated_label = ttk.Label(preview_card)
        self.generated_label.grid(row=0, column=0, sticky='nsew')
        self._prepare_preview_label(
            self.generated_label,
            'current_photo',
            (760, 520),
            'Generated QR preview will appear here.',
        )

        details = ttk.Notebook(right, takefocus=False)
        details.grid(row=2, column=0, sticky='nsew')

        summary_tab = ttk.Frame(details, padding=4)
        verification_tab = ttk.Frame(details, padding=4)
        log_tab = ttk.Frame(details, padding=4)
        details.add(summary_tab, text='Summary')
        details.add(verification_tab, text='Verification scan')
        details.add(log_tab, text='Generation log')

        self.generate_metric_vars = self._new_metric_vars(['Mode', 'Codec', 'Payload bytes', 'Chunk count', 'Saved files', 'Preview summary'])
        metric_card = self._make_card(summary_tab, 'Generation summary')
        metric_card.pack(fill='both', expand=True)
        self._build_metric_grid(metric_card, self.generate_metric_vars, columns=2)

        verify_card, self.generate_verify_text = self._build_json_panel(verification_tab, 'Verification result')
        verify_card.pack(fill='both', expand=True)
        self._set_text(self.generate_verify_text, 'Generate a QR and use Scan generated QR to validate it inside the app.')

        log_card, self.generate_log = self._build_json_panel(log_tab, 'Generation log')
        log_card.pack(fill='both', expand=True)
        self._set_text(self.generate_log, 'No QR generated yet.')

    def _build_camera_tab(self) -> None:
        tk, ttk = self.tk, self.ttk

        self.camera_tab.columnconfigure(0, weight=1)
        self.camera_tab.columnconfigure(1, weight=1)
        self.camera_tab.rowconfigure(0, weight=1)
        self.camera_tab.rowconfigure(1, weight=1)

        left = ttk.Frame(self.camera_tab)
        right = ttk.Frame(self.camera_tab)

        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)

        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=0)
        right.rowconfigure(1, weight=0)
        right.rowconfigure(2, weight=1, minsize=220)

        self._register_responsive_pair(self.camera_tab, left, right)
        left.grid(row=0, column=0, sticky='nsew', padx=(0, 8))
        right.grid(row=0, column=1, sticky='nsew')

        controls = self._make_card(left, 'Camera controls')
        controls.grid(row=0, column=0, sticky='ew', pady=(0, 8))

        form = ttk.Frame(controls, style='Panel.TFrame')
        form.pack(fill='x')
        form.columnconfigure(1, weight=1)

        ttk.Label(form, text='Device', style='Muted.TLabel').grid(
            row=0, column=0, sticky='w', padx=(0, 8), pady=(0, 6)
        )
        self.camera_device_var = tk.StringVar(value='0')
        ttk.Entry(form, textvariable=self.camera_device_var, width=12).grid(
            row=0, column=1, sticky='ew', pady=(0, 6)
        )

        ttk.Label(form, text='Private key', style='Muted.TLabel').grid(
            row=1, column=0, sticky='w', padx=(0, 8), pady=(0, 6)
        )
        self.camera_private_key_var = tk.StringVar()
        ttk.Entry(form, textvariable=self.camera_private_key_var).grid(
            row=1, column=1, sticky='ew', pady=(0, 6)
        )

        buttons = ttk.Frame(controls, style='Panel.TFrame')
        buttons.pack(fill='x', pady=(8, 0))
        buttons.columnconfigure(0, weight=1, uniform='camera_actions')
        buttons.columnconfigure(1, weight=1, uniform='camera_actions')

        ttk.Button(
            buttons,
            text='Start camera',
            style='Accent.TButton',
            command=self.start_camera,
            takefocus=False,
        ).grid(row=0, column=0, sticky='ew', padx=(0, 6))

        ttk.Button(
            buttons,
            text='Stop',
            command=self.stop_camera,
            takefocus=False,
        ).grid(row=0, column=1, sticky='ew')

        self.camera_status_var = tk.StringVar(value='Camera stopped.')
        status_line = ttk.Label(
            controls,
            textvariable=self.camera_status_var,
            style='Muted.TLabel',
            justify='left',
        )
        status_line.pack(anchor='w', pady=(6, 0))
        self._register_wrap_label(status_line, padding=120)

        preview_card = self._make_card(left, 'Live frame preview')
        preview_card.grid(row=1, column=0, sticky='nsew')
        preview_card.columnconfigure(0, weight=1)
        preview_card.rowconfigure(0, weight=1)

        self.camera_image_label = ttk.Label(preview_card)
        self.camera_image_label.grid(row=0, column=0, sticky='nsew')
        self._prepare_preview_label(
            self.camera_image_label,
            'current_camera_photo',
            (760, 520),
            'Camera preview will appear here after startup.',
        )

        status_card = self._make_card(right, 'Live status')
        status_card.grid(row=0, column=0, sticky='ew', pady=(0, 8))
        self.camera_status_label = ttk.Label(
            status_card,
            text='Idle',
            foreground=STATUS_STYLES['idle'][1],
            font=('Segoe UI', 10, 'bold'),
        )
        self.camera_status_label.pack(anchor='w')

        self.camera_status_hint = tk.StringVar(
            value='Start the camera to inspect live decoding and recovery behavior.'
        )
        hint = ttk.Label(
            status_card,
            textvariable=self.camera_status_hint,
            style='Muted.TLabel',
            justify='left',
        )
        hint.pack(anchor='w', pady=(4, 0))
        self._register_wrap_label(hint, padding=120)

        self.camera_metric_vars = self._new_metric_vars([
            'Frames', 'Successes', 'Partial',
            'Fails', 'Decoder', 'Stage',
            'Scenario', 'Split progress', 'Calibration'
        ])

        camera_metrics_card = self._make_card(right, 'Live metrics')
        camera_metrics_card.grid(row=1, column=0, sticky='ew', pady=(0, 8))
        self._build_metric_grid(camera_metrics_card, self.camera_metric_vars, columns=3)

        details = ttk.Notebook(right, takefocus=False)
        details.grid(row=2, column=0, sticky='nsew')

        notes_tab = ttk.Frame(details, padding=4)
        json_tab = ttk.Frame(details, padding=4)

        notes_tab.columnconfigure(0, weight=1)
        notes_tab.rowconfigure(0, weight=1)
        json_tab.columnconfigure(0, weight=1)
        json_tab.rowconfigure(0, weight=1)

        details.add(notes_tab, text='Notes')
        details.add(json_tab, text='Latest JSON')

        notes_card = self._make_card(notes_tab, 'Live notes')
        notes_card.grid(row=0, column=0, sticky='nsew')
        self.camera_notes_text = self._build_scrolled_text(notes_card, height=10, wrap='word')

        json_card, self.camera_json_text = self._build_json_panel(json_tab, 'Latest JSON')
        json_card.grid(row=0, column=0, sticky='nsew')
    def _build_dataset_tab(self) -> None:
        tk, ttk = self.tk, self.ttk
        top = self._make_card(self.dataset_tab, 'Dataset benchmark runner')
        top.pack(fill='x', pady=(0, 10))
        row = ttk.Frame(top, style='Panel.TFrame')
        row.pack(fill='x')
        self.dataset_folder_var = tk.StringVar(value='')
        self.dataset_pattern_var = tk.StringVar(value='*.png')
        self.dataset_recursive_var = tk.BooleanVar(value=True)
        ttk.Button(row, text='Choose folder…', style='Accent.TButton', command=self.choose_dataset_folder, takefocus=False).pack(side='left')
        ttk.Entry(row, textvariable=self.dataset_folder_var).pack(side='left', fill='x', expand=True, padx=(10, 10))
        ttk.Label(row, text='Pattern', style='Muted.TLabel').pack(side='left')
        ttk.Entry(row, textvariable=self.dataset_pattern_var, width=12).pack(side='left', padx=(10, 10))
        ttk.Checkbutton(row, text='Recursive', variable=self.dataset_recursive_var).pack(side='left', padx=(0, 10))
        ttk.Button(row, text='Run benchmark', command=self.run_dataset_benchmark, takefocus=False).pack(side='left')
        ttk.Button(row, text='Export CSV', command=self.export_dataset_csv, takefocus=False).pack(side='left', padx=(8, 0))

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

    def _set_text(self, widget, text: str, scroll_to: str = 'start') -> None:
        widget.delete('1.0', 'end')
        widget.insert('1.0', text)
        if scroll_to == 'end':
            widget.see('end')
        else:
            widget.see('1.0')
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
        notes = result_dict.get('notes') or []
        hint = quality.get('operator_hint') or '—'
        status = 'idle'
        hint_text = 'No result yet.'
        if result_dict.get('success'):
            status = 'success'
            hint_text = 'Final payload assembled and ready for inspection.'
        elif result_dict.get('partial_success'):
            status = 'partial'
            hint_text = 'A split chunk was captured; more pieces are still needed.'
        elif result_dict.get('error'):
            status = 'error'
            hint_text = result_dict.get('error') or 'Decode failed.'
        metrics = {
            'Decoder': base.get('decoder') or '—',
            'Stage': result_dict.get('enhancement_stage') or base.get('stage') or '—',
            'Scenario': result_dict.get('scenario') or '—',
            'Payload kind': payload.get('payload_kind') or '—',
            'Split progress': result_dict.get('split_progress') or '—',
            'Brightness': f"{quality.get('mean_brightness', 0):.1f}" if quality.get('mean_brightness') is not None else '—',
            'Contrast': f"{quality.get('contrast_stddev', 0):.1f}" if quality.get('contrast_stddev') is not None else '—',
            'Sharpness': f"{quality.get('laplacian_variance', 0):.1f}" if quality.get('laplacian_variance') is not None else '—',
            'Operator hint': hint,
        }
        notes_text = '\n'.join(f'• {note}' for note in notes) if notes else 'No notes.'
        payload_text = _pretty_json(payload.get('normalized') or payload or {})
        raw_text = _pretty_json(result_dict)
        return status, metrics, notes_text, payload_text, raw_text

    def _reset_scan_view(self) -> None:
        self._set_status_label(self.scan_status_label, 'idle', self.scan_status_hint, 'Open an image to decode and inspect payload structure.')
        for var in self.scan_metric_vars.values():
            var.set('—')
        self._set_text(self.scan_notes_text, 'No notes yet.')
        self._set_text(self.scan_payload_text, '{}')
        self._set_text(self.scan_json_text, 'Choose an image to decode QR payloads and inspect enhanced pipeline output here.')

    def _reset_camera_view(self) -> None:
        self._set_status_label(self.camera_status_label, 'idle', self.camera_status_hint, 'Start the camera to inspect live decoding, ROI tracking, and split chunk collection.')
        for var in self.camera_metric_vars.values():
            var.set('—')
        self._set_text(self.camera_notes_text, 'No camera notes yet.')
        self._set_text(self.camera_json_text, 'Start the camera to see live results.')

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
            self._display_pil_image(self.generated_label, preview, 'current_photo', (900, 760))
            self._set_text(self.generate_log, log_text)
            self.generate_metric_vars['Mode'].set(mode)
            self.generate_metric_vars['Codec'].set(codec)
            self.generate_metric_vars['Payload bytes'].set(str(len(raw)))
            self.generate_metric_vars['Chunk count'].set(chunk_count)
            self.generate_metric_vars['Saved files'].set(str(len(saved_paths)))
            self.generate_metric_vars['Preview summary'].set(payload_preview[:64] + ('…' if len(payload_preview) > 64 else ''))
            self._set_status_label(self.generate_status_label, 'success', self.generate_status_hint, 'QR assets generated successfully. Save the preview, send it to Scan studio, or review the verification panel below.')
            self.hero_status.set(f'Generated {mode.lower()} using {codec.upper()} encoding.')
            self.scan_generated_preview(auto=True)
        except Exception as exc:
            self._set_status_label(self.generate_status_label, 'error', self.generate_status_hint, f'QR generation failed: {exc}')
            self._set_text(self.generate_log, f'QR generation failed\n{exc}')
            self._set_text(self.generate_verify_text, 'Verification not available because generation failed.')
            self.hero_status.set('Generation failed. Fix the payload or options and try again.')

    def _display_pil_image(self, label, image: Image.Image, attr_name: str, max_size: tuple[int, int]) -> None:
        self._image_sources[attr_name] = (image.copy(), max_size)
        self._refresh_image_label(label, attr_name, max_size)

    def scan_generated_preview(self, auto: bool = False) -> None:
        if self.generated_preview is None or not self.generated_preview.saved_paths:
            self._set_text(self.generate_verify_text, 'Nothing to scan yet. Generate a QR first.')
            if not auto:
                self._set_status_label(self.generate_status_label, 'idle', self.generate_status_hint, 'Generate a QR first, then run an internal verification scan.')
            return
        try:
            rows: list[dict[str, Any]] = []
            last_success_result: dict[str, Any] | None = None
            for idx, path_str in enumerate(self.generated_preview.saved_paths, start=1):
                image = cv2.imread(path_str, cv2.IMREAD_COLOR)
                if image is None:
                    rows.append({'file': path_str, 'success': False, 'error': 'Failed to open generated PNG'})
                    continue
                result = self.system.scan_image(image)
                result_dict = result.to_dict()
                base = result_dict.get('base_result') or {}
                payload = base.get('parsed_payload') or {}
                rows.append({
                    'file': path_str,
                    'success': bool(result_dict.get('success')),
                    'partial_success': bool(result_dict.get('partial_success')),
                    'stage': result_dict.get('enhancement_stage') or base.get('stage') or '',
                    'decoder': base.get('decoder') or '',
                    'payload_kind': payload.get('payload_kind') or '',
                    'error': result_dict.get('error') or base.get('error') or '',
                })
                if result_dict.get('success'):
                    last_success_result = result_dict
            summary = {
                'files_tested': len(rows),
                'successes': sum(1 for row in rows if row.get('success')),
                'partials': sum(1 for row in rows if row.get('partial_success')),
                'failures': sum(1 for row in rows if not row.get('success')),
                'rows': rows,
                'best_result': last_success_result,
            }
            self._set_text(self.generate_verify_text, _pretty_json(summary))
            if last_success_result is not None and self.generated_preview.saved_paths:
                self.last_scan_path = Path(self.generated_preview.saved_paths[0])
                self.scan_path_var.set(str(self.last_scan_path))
                self._scan_path(self.last_scan_path)
            if not auto:
                self.hero_status.set('Generated QR scanned internally for verification.')
        except Exception as exc:
            self._set_text(self.generate_verify_text, f'Generated QR verification failed\n{exc}')
            if not auto:
                self._set_status_label(self.generate_status_label, 'error', self.generate_status_hint, f'Internal verification failed: {exc}')

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
        self._display_pil_image(self.scan_image_label, preview, 'current_scan_photo', (900, 760))
        status, metrics, notes_text, payload_text, raw_text = self._result_summary(result_dict)
        self._set_status_label(self.scan_status_label, status, self.scan_status_hint, metrics.get('Operator hint', 'Scan complete.'))
        for key, var in self.scan_metric_vars.items():
            var.set(metrics.get(key, '—'))
        self._set_text(self.scan_notes_text, notes_text)
        self._set_text(self.scan_payload_text, payload_text)
        self._set_text(self.scan_json_text, raw_text)
        self.hero_status.set(f'Scanned {path.name}.')
        self._update_hero_stats()

    def start_camera(self) -> None:
        if self.camera_running:
            return
        try:
            device_value = self.camera_device_var.get().strip()
            device: int | str = int(device_value) if device_value.isdigit() else device_value

            capture_width, capture_height, capture_fps = 960, 540, 24

            self.system = EnhancedQRSystem(
                private_key=self.camera_private_key_var.get().strip() or None,
                stats_path='pipeline_stats.json',
            )
            self.camera_source = LinuxCameraSource(
                device=device,
                width=capture_width,
                height=capture_height,
                fps=capture_fps,
            )
            self.camera_source.open()
            self.camera_running = True
            self.camera_counters = {'frames': 0, 'success': 0, 'partial': 0, 'fail': 0}
            self._last_camera_result = None
            self._last_camera_polygon = None
            self._last_camera_ui_ts = 0.0
            self._last_camera_preview_ts = 0.0
            self._last_camera_decode_ts = 0.0
            self.camera_status_var.set(
                f'Camera running on device {self.camera_source.device} '
                f'({self.camera_source.backend_name or "auto"}) at {capture_width}×{capture_height}.'
            )
            self.camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
            self.camera_thread.start()
            self.hero_status.set('Camera stream started.')
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
        self.camera_status_var.set('Camera stopped.')
        self.hero_status.set('Camera stream stopped.')

    @staticmethod
    def _rescale_polygon(polygon: Any, scale_x: float, scale_y: float) -> list[list[int]] | None:
        if polygon is None:
            return None
        arr = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        if arr.size == 0:
            return None
        arr[:, 0] *= scale_x
        arr[:, 1] *= scale_y
        return [[int(round(x)), int(round(y))] for x, y in arr]

    def _camera_processing_frame(self, frame: np.ndarray) -> tuple[np.ndarray, float, float]:
        max_dim = max(480, int(self.camera_processing_max_dim))
        height, width = frame.shape[:2]
        longest = max(height, width)
        if longest <= max_dim:
            return frame, 1.0, 1.0
        scale = max_dim / float(longest)
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        resized = cv2.resize(
            frame,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=interpolation,
        )
        return resized, width / float(resized.shape[1]), height / float(resized.shape[0])

    def _update_camera_preview_only(self, frame: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._display_pil_image(self.camera_image_label, Image.fromarray(rgb), 'current_camera_photo', (760, 520))

    def _camera_loop(self) -> None:
        preview_interval = 1.0 / 20.0
        ui_interval = 1.0 / 8.0

        while self.camera_running and self.camera_source is not None:
            try:
                frame, decision = self.camera_source.read_adaptive()
                result_dict = self._last_camera_result or {
                    'success': False,
                    'partial_success': False,
                    'notes': ['Waiting for first processed frame...'],
                    'base_result': {},
                }
                now = time.time()

                should_decode = self._last_camera_result is None
                if not should_decode:
                    decode_fps = max(2, int(self.camera_decode_fps))
                    should_decode = (now - self._last_camera_decode_ts) >= (1.0 / float(decode_fps))

                if should_decode:
                    self._last_camera_decode_ts = now
                    processing_frame, scale_x, scale_y = self._camera_processing_frame(frame)
                    result = self.system.scan_stream_frame(processing_frame, None if decision is None else decision.to_dict())
                    result_dict = result.to_dict()
                    polygon = ((result_dict.get('base_result') or {}).get('polygon'))
                    if polygon is not None and (scale_x != 1.0 or scale_y != 1.0):
                        polygon = self._rescale_polygon(polygon, scale_x, scale_y)
                        if result_dict.get('base_result') is not None:
                            result_dict['base_result']['polygon'] = polygon
                    self._last_camera_result = result_dict
                    self._last_camera_polygon = polygon
                    self.camera_counters['frames'] += 1
                    if result.success:
                        self.camera_counters['success'] += 1
                    elif result.partial_success:
                        self.camera_counters['partial'] += 1
                    else:
                        self.camera_counters['fail'] += 1

                display = frame.copy()
                if self._last_camera_polygon:
                    pts = np.asarray(self._last_camera_polygon, dtype=np.int32).reshape((-1, 1, 2))
                    color = (0, 180, 0) if result_dict.get('success') else (26, 115, 232)
                    cv2.polylines(display, [pts], True, color, 2)

                if (now - self._last_camera_preview_ts) >= preview_interval:
                    self._last_camera_preview_ts = now
                    self.root.after(0, self._update_camera_preview_only, display)

                if should_decode and (now - self._last_camera_ui_ts) >= ui_interval:
                    self._last_camera_ui_ts = now
                    self.root.after(0, self._update_camera_preview, display, result_dict)
            except Exception as exc:
                self.root.after(0, self.camera_status_var.set, f'Camera error: {exc}')
                self.root.after(0, self.stop_camera)
                break
            time.sleep(0.004)

    def _update_camera_preview(self, frame: np.ndarray, result_dict: dict[str, Any]) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._display_pil_image(self.camera_image_label, Image.fromarray(rgb), 'current_camera_photo', (760, 520))
        status, metrics, notes_text, _payload_text, raw_text = self._result_summary(result_dict)
        calibration = self.system.calibration_status()
        self.camera_metric_vars['Frames'].set(str(self.camera_counters['frames']))
        self.camera_metric_vars['Successes'].set(str(self.camera_counters['success']))
        self.camera_metric_vars['Partial'].set(str(self.camera_counters['partial']))
        self.camera_metric_vars['Fails'].set(str(self.camera_counters['fail']))
        self.camera_metric_vars['Decoder'].set(metrics.get('Decoder', '—'))
        self.camera_metric_vars['Stage'].set(metrics.get('Stage', '—'))
        self.camera_metric_vars['Scenario'].set(metrics.get('Scenario', '—'))
        self.camera_metric_vars['Split progress'].set(metrics.get('Split progress', '—'))
        self.camera_metric_vars['Calibration'].set(calibration)
        self._set_status_label(self.camera_status_label, status, self.camera_status_hint, metrics.get('Operator hint', 'Live decode updated.'))
        self._set_text(self.camera_notes_text, notes_text, scroll_to='start')
        self._set_text(self.camera_json_text, raw_text, scroll_to='start')
        self._update_hero_stats()
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
