from __future__ import annotations

import base64
import json

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .cloud_service import CloudBootstrapService
from .consent import ConsentManager
from .enhanced_pipeline import EnhancedQRSystem
from .payload_codecs import (
    armor_binary_payload,
    encode_cbor_v1,
    encode_json_v1,
    encode_x25519_cbor_v1,
    encode_x25519_raw_json_v1,
    payload_is_text_friendly,
)
from .payload_optimizer import PayloadComplexityController
from .qr_generation import build_binary_payload_qr, build_split_qr_pngs, build_text_qr
from .split_qr import chunk_texts


class GenerateQRRequest(BaseModel):
    payload: dict
    encrypted: bool = False
    public_key_hex: str | None = None
    payload_codec: str = 'auto'
    compatibility_text: bool = True
    split: bool = False
    session_id: str = 'web-session'
    max_chunk_bytes: int = 180
    with_parity: bool = True


class OptimizePayloadRequest(BaseModel):
    payload: dict


class ConsentIssueRequest(BaseModel):
    subject: str
    purpose: str
    ttl_seconds: int = 300
    metadata: dict = {}


class ConsentVerifyRequest(BaseModel):
    payload: dict


class RegisterBootstrapRequest(BaseModel):
    registration_id: str
    registration_token: str
    wifi: dict
    metadata: dict = {}


class ProvisionFromPayloadRequest(BaseModel):
    payload: dict
    device_id: str = 'web-api-device'


class WebApiSystemContext:
    def __init__(self, system: EnhancedQRSystem, consent_manager: ConsentManager | None = None, bootstrap_service: CloudBootstrapService | None = None):
        self.system = system
        self.consent_manager = consent_manager
        self.bootstrap_service = bootstrap_service


def _select_payload_bytes(req: GenerateQRRequest) -> tuple[bytes, str]:
    codec = (req.payload_codec or 'auto').lower()
    if codec not in {'auto', 'json', 'cbor'}:
        raise HTTPException(status_code=400, detail='payload_codec must be one of: auto, json, cbor')
    if req.encrypted and not req.public_key_hex:
        raise HTTPException(status_code=400, detail='public_key_hex is required for encrypted payloads')
    if req.encrypted:
        options = {'json': encode_x25519_raw_json_v1(req.payload, req.public_key_hex), 'cbor': encode_x25519_cbor_v1(req.payload, req.public_key_hex)}
    else:
        options = {'json': encode_json_v1(req.payload), 'cbor': encode_cbor_v1(req.payload)}
    chosen = min(options.items(), key=lambda item: len(item[1])) if codec == 'auto' else (codec, options[codec])
    return chosen[1], chosen[0]


def _ui_html() -> str:
    demo_payload = json.dumps(
        {
            'ssid': 'LabWiFi',
            'psk': 'DemoPassword123',
            'CC': 'UA',
            'registration-id': 'rid-demo',
            'registration-token': 'rtk-demo',
        },
        ensure_ascii=False,
        indent=2,
    )
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>QR Onboarding Research Studio</title>
  <style>
    :root {{
      --bg: #0b1220;
      --panel: #121a2b;
      --card: #182235;
      --muted: #9fb0d1;
      --text: #f3f7ff;
      --accent: #4c7dff;
      --accent-2: #17b26a;
      --warn: #f79009;
      --border: #24324a;
      --shadow: 0 18px 44px rgba(2, 10, 28, 0.35);
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: Inter, Segoe UI, system-ui, sans-serif; background: linear-gradient(180deg, #08111e, #0d1626 34%, #10192b); color: var(--text); }}
    .shell {{ max-width: 1440px; margin: 0 auto; padding: 28px; }}
    .hero {{ display: grid; grid-template-columns: 1.35fr 1fr; gap: 18px; align-items: stretch; }}
    .panel {{ background: rgba(18, 26, 43, 0.96); border: 1px solid var(--border); border-radius: 24px; box-shadow: var(--shadow); }}
    .hero-copy {{ padding: 30px; }}
    .eyebrow {{ display: inline-flex; align-items: center; gap: 8px; padding: 8px 14px; border-radius: 999px; background: rgba(76, 125, 255, 0.14); color: #b8cbff; font-size: 13px; font-weight: 700; letter-spacing: .02em; }}
    h1 {{ margin: 18px 0 12px; font-size: 40px; line-height: 1.05; }}
    .lead {{ color: var(--muted); font-size: 16px; line-height: 1.65; max-width: 760px; }}
    .hero-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin-top: 22px; }}
    .mini {{ background: rgba(255, 255, 255, 0.04); border: 1px solid rgba(159, 176, 209, 0.15); border-radius: 18px; padding: 14px; }}
    .mini .k {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .06em; }}
    .mini .v {{ font-size: 22px; font-weight: 700; margin-top: 8px; }}
    .hero-side {{ padding: 24px; display: grid; gap: 12px; }}
    .status-pill {{ display: inline-flex; align-items: center; gap: 10px; background: rgba(23, 178, 106, 0.14); color: #7be0ae; padding: 10px 14px; border-radius: 14px; font-weight: 700; }}
    .tabs {{ display: flex; gap: 10px; margin: 22px 0 18px; flex-wrap: wrap; }}
    .tab-btn {{ border: 1px solid var(--border); background: rgba(24, 34, 53, 0.9); color: var(--text); padding: 11px 15px; border-radius: 14px; cursor: pointer; font-weight: 600; }}
    .tab-btn.active {{ background: var(--accent); border-color: transparent; }}
    .tab {{ display: none; }}
    .tab.active {{ display: block; }}
    .grid-2 {{ display: grid; grid-template-columns: 1.1fr .9fr; gap: 18px; }}
    .card {{ padding: 22px; }}
    h2 {{ margin: 0 0 18px; font-size: 22px; }}
    h3 {{ margin: 0 0 12px; font-size: 17px; }}
    textarea, input[type="text"], input[type="number"], input[type="file"] {{ width: 100%; border-radius: 14px; border: 1px solid var(--border); background: #0e1727; color: var(--text); padding: 12px 14px; font: inherit; }}
    textarea {{ min-height: 260px; resize: vertical; line-height: 1.5; }}
    label {{ display: block; margin-bottom: 8px; color: var(--muted); font-weight: 600; }}
    .row {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }}
    .row-3 {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; }}
    .checks {{ display: flex; gap: 18px; flex-wrap: wrap; margin-top: 10px; }}
    .checks label {{ display: inline-flex; gap: 8px; align-items: center; margin: 0; color: var(--text); }}
    .actions {{ display: flex; gap: 10px; margin-top: 16px; flex-wrap: wrap; }}
    button.primary {{ background: var(--accent); color: white; border: 0; border-radius: 14px; padding: 12px 16px; font: inherit; font-weight: 700; cursor: pointer; }}
    button.secondary {{ background: transparent; color: var(--text); border: 1px solid var(--border); border-radius: 14px; padding: 12px 16px; font: inherit; font-weight: 700; cursor: pointer; }}
    .metrics {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin-top: 14px; }}
    .metric {{ background: rgba(255,255,255,0.04); border: 1px solid rgba(159,176,209,0.15); border-radius: 16px; padding: 14px; }}
    .metric .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .06em; }}
    .metric .value {{ font-size: 18px; font-weight: 700; margin-top: 8px; line-height: 1.35; word-break: break-word; }}
    .result-box {{ min-height: 460px; background: #0e1727; border: 1px solid var(--border); border-radius: 18px; padding: 16px; overflow: auto; }}
    pre {{ margin: 0; white-space: pre-wrap; word-break: break-word; font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 13px; line-height: 1.5; }}
    .image-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; margin-top: 18px; }}
    .image-grid img {{ width: 100%; border-radius: 18px; border: 1px solid var(--border); background: white; }}
    .muted {{ color: var(--muted); }}
    .status-line {{ margin-bottom: 12px; font-weight: 700; }}
    .status-success {{ color: #7be0ae; }}
    .status-partial {{ color: #ffcc7c; }}
    .status-fail {{ color: #ff9f95; }}
    .status-idle {{ color: #d4ddf0; }}
    @media (max-width: 1120px) {{
      .hero, .grid-2, .row, .row-3 {{ grid-template-columns: 1fr; }}
      .metrics {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
    @media (max-width: 720px) {{
      .shell {{ padding: 16px; }}
      h1 {{ font-size: 30px; }}
      .metrics {{ grid-template-columns: 1fr; }}
      .hero-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div class="panel hero-copy">
        <div class="eyebrow">Research-grade QR workflow</div>
        <h1>QR Onboarding Research Studio</h1>
        <p class="lead">A browser UI for image scanning, QR generation, split-chunk workflows, and payload optimization on top of the enhanced onboarding pipeline.</p>
        <div class="hero-grid">
          <div class="mini"><div class="k">Modes</div><div class="v">Scan + Generate</div></div>
          <div class="mini"><div class="k">Split support</div><div class="v">Parity-ready</div></div>
          <div class="mini"><div class="k">Output</div><div class="v">Visual + JSON</div></div>
        </div>
      </div>
      <div class="panel hero-side">
        <div class="status-pill">API online · /scan/image · /qr/generate · /payload/optimize</div>
        <div class="mini"><div class="k">Use this UI for</div><div class="v">Demo, validation, screenshots, and manual inspection</div></div>
        <div class="mini"><div class="k">Best practice</div><div class="v">Start with the compact payload, then compare with split QR + parity</div></div>
        <div class="mini"><div class="k">Why it matters</div><div class="v">You can show practical robustness, not just raw JSON endpoints</div></div>
      </div>
    </section>

    <div class="tabs">
      <button class="tab-btn active" data-tab="scan">Scan image</button>
      <button class="tab-btn" data-tab="generate">Generate QR</button>
      <button class="tab-btn" data-tab="optimize">Optimize payload</button>
    </div>

    <section class="tab active" id="tab-scan">
      <div class="grid-2">
        <div class="panel card">
          <h2>Upload image for enhanced scan</h2>
          <label for="scan-file">Image file</label>
          <input id="scan-file" type="file" accept="image/*" />
          <div class="actions">
            <button id="scan-submit" class="primary">Run scan</button>
            <button id="scan-reset" class="secondary">Clear</button>
          </div>
          <div class="metrics" id="scan-metrics"></div>
        </div>
        <div class="panel card">
          <h2>Scan result</h2>
          <div id="scan-status" class="status-line status-idle">Idle</div>
          <div class="result-box"><pre id="scan-json">Upload a PNG or photo to inspect the enhanced result.</pre></div>
        </div>
      </div>
    </section>

    <section class="tab" id="tab-generate">
      <div class="grid-2">
        <div class="panel card">
          <h2>Payload and transport options</h2>
          <label for="payload">Payload JSON</label>
          <textarea id="payload">{demo_payload}</textarea>
          <div class="row">
            <div>
              <label for="payload-codec">Payload codec</label>
              <input id="payload-codec" type="text" value="auto" />
            </div>
            <div>
              <label for="session-id">Session id</label>
              <input id="session-id" type="text" value="web-session" />
            </div>
          </div>
          <div class="row">
            <div>
              <label for="max-chunk-bytes">Max chunk bytes</label>
              <input id="max-chunk-bytes" type="number" value="180" min="32" max="512" />
            </div>
            <div>
              <label for="public-key">Public key hex for encrypted mode</label>
              <input id="public-key" type="text" value="" />
            </div>
          </div>
          <div class="checks">
            <label><input id="encrypted" type="checkbox" /> Encrypted</label>
            <label><input id="compatibility-text" type="checkbox" checked /> Compatibility armor</label>
            <label><input id="split" type="checkbox" /> Split QR</label>
            <label><input id="with-parity" type="checkbox" checked /> Parity recovery</label>
          </div>
          <div class="actions">
            <button id="generate-submit" class="primary">Generate QR</button>
            <button id="payload-copy" class="secondary">Copy payload JSON</button>
          </div>
        </div>
        <div class="panel card">
          <h2>Generated output</h2>
          <div id="generate-status" class="status-line status-idle">Idle</div>
          <div class="metrics" id="generate-metrics"></div>
          <div class="image-grid" id="generate-images"></div>
          <div class="result-box" style="margin-top:16px"><pre id="generate-json">Generated output will appear here.</pre></div>
        </div>
      </div>
    </section>

    <section class="tab" id="tab-optimize">
      <div class="grid-2">
        <div class="panel card">
          <h2>Payload complexity optimizer</h2>
          <p class="muted">Use the same payload JSON to compare more compact variants and justify scientific claims around payload complexity control.</p>
          <div class="actions">
            <button id="optimize-submit" class="primary">Optimize payload</button>
          </div>
          <div class="result-box" style="margin-top:16px"><pre id="optimize-json">Optimization report will appear here.</pre></div>
        </div>
        <div class="panel card">
          <h2>Why this tab matters</h2>
          <div class="mini"><div class="k">For research</div><div class="v">Show that compact payload design improves practical QR robustness</div></div>
          <div class="mini" style="margin-top:12px;"><div class="k">For demo</div><div class="v">Compare the best variant, a dense single QR, and a split + parity transport</div></div>
        </div>
      </div>
    </section>
  </div>
  <script>
    const $ = (id) => document.getElementById(id);
    const tabs = document.querySelectorAll('.tab-btn');
    tabs.forEach((btn) => btn.addEventListener('click', () => {{
      tabs.forEach((b) => b.classList.remove('active'));
      document.querySelectorAll('.tab').forEach((tab) => tab.classList.remove('active'));
      btn.classList.add('active');
      $('tab-' + btn.dataset.tab).classList.add('active');
    }}));

    function setMetrics(targetId, data) {{
      const root = $(targetId);
      root.innerHTML = '';
      Object.entries(data).forEach(([key, value]) => {{
        const div = document.createElement('div');
        div.className = 'metric';
        div.innerHTML = `<div class="label">${{key}}</div><div class="value">${{value ?? '—'}}</div>`;
        root.appendChild(div);
      }});
    }}

    function statusClass(mode) {{
      if (mode === 'success') return 'status-success';
      if (mode === 'partial') return 'status-partial';
      if (mode === 'fail') return 'status-fail';
      return 'status-idle';
    }}

    function setStatus(node, mode, text) {{
      node.className = 'status-line ' + statusClass(mode);
      node.textContent = text;
    }}

    function getPayloadJson() {{
      return JSON.parse($('payload').value);
    }}

    $('payload-copy').addEventListener('click', async () => {{
      await navigator.clipboard.writeText($('payload').value);
    }});

    $('scan-submit').addEventListener('click', async () => {{
      const file = $('scan-file').files[0];
      if (!file) {{
        setStatus($('scan-status'), 'fail', 'Choose an image first.');
        return;
      }}
      setStatus($('scan-status'), 'idle', 'Scanning image...');
      const form = new FormData();
      form.append('file', file);
      const response = await fetch('/scan/image', {{ method: 'POST', body: form }});
      const data = await response.json();
      $('scan-json').textContent = JSON.stringify(data, null, 2);
      const base = data.base_result || data;
      const payload = (base.parsed_payload || {{}});
      const quality = (base.quality || {{}});
      const state = data.success ? 'success' : (data.partial_success ? 'partial' : 'fail');
      setStatus($('scan-status'), state, state === 'success' ? 'Final payload decoded successfully.' : state === 'partial' ? 'Split chunk captured; more pieces needed.' : (data.error || 'Decode failed.'));
      setMetrics('scan-metrics', {{
        Decoder: base.decoder || '—',
        Stage: data.enhancement_stage || base.stage || '—',
        Scenario: data.scenario || '—',
        Payload: payload.payload_kind || '—',
        Split: data.split_progress || '—',
        Brightness: quality.mean_brightness != null ? quality.mean_brightness.toFixed(1) : '—',
        Contrast: quality.contrast_stddev != null ? quality.contrast_stddev.toFixed(1) : '—',
        Sharpness: quality.laplacian_variance != null ? quality.laplacian_variance.toFixed(1) : '—',
        Hint: quality.operator_hint || '—',
      }});
    }});

    $('scan-reset').addEventListener('click', () => {{
      $('scan-file').value = '';
      $('scan-json').textContent = 'Upload a PNG or photo to inspect the enhanced result.';
      $('scan-metrics').innerHTML = '';
      setStatus($('scan-status'), 'idle', 'Idle');
    }});

    $('generate-submit').addEventListener('click', async () => {{
      let payload;
      try {{
        payload = getPayloadJson();
      }} catch (err) {{
        setStatus($('generate-status'), 'fail', 'Payload JSON is invalid.');
        return;
      }}
      setStatus($('generate-status'), 'idle', 'Generating QR assets...');
      $('generate-images').innerHTML = '';
      const body = {{
        payload,
        encrypted: $('encrypted').checked,
        public_key_hex: $('public-key').value || null,
        payload_codec: $('payload-codec').value || 'auto',
        compatibility_text: $('compatibility-text').checked,
        split: $('split').checked,
        session_id: $('session-id').value || 'web-session',
        max_chunk_bytes: Number($('max-chunk-bytes').value || 180),
        with_parity: $('with-parity').checked,
      }};
      const response = await fetch('/qr/generate', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify(body),
      }});
      const data = await response.json();
      $('generate-json').textContent = JSON.stringify(data, null, 2);
      if (!response.ok) {{
        setStatus($('generate-status'), 'fail', data.detail || 'Generation failed.');
        return;
      }}
      setStatus($('generate-status'), 'success', data.split ? 'Split QR assets generated successfully.' : 'Single QR generated successfully.');
      const metrics = data.split ? {{
        Mode: 'Split QR',
        Chunks: data.chunk_count,
        Codec: data.encoding?.payload_codec || '—',
        Transport: data.encoding?.transport || '—',
        Parity: $('with-parity').checked ? 'enabled' : 'off',
        Preview: 'Chunk set ready for sequential demo',
      }} : {{
        Mode: 'Single QR',
        PayloadBytes: data.payload_size_bytes,
        Codec: data.encoding?.payload_codec || '—',
        Transport: data.encoding?.transport || '—',
        Armor: data.encoding?.compatibility_text ? 'on' : 'off',
        Preview: data.payload_preview || '—',
      }};
      setMetrics('generate-metrics', metrics);
      const images = data.split ? data.chunk_png_base64 : [data.png_base64];
      images.forEach((b64, idx) => {{
        const img = document.createElement('img');
        img.src = `data:image/png;base64,${{b64}}`;
        img.alt = data.split ? `QR chunk ${{idx + 1}}` : 'QR preview';
        $('generate-images').appendChild(img);
      }});
    }});

    $('optimize-submit').addEventListener('click', async () => {{
      let payload;
      try {{
        payload = getPayloadJson();
      }} catch (err) {{
        $('optimize-json').textContent = 'Payload JSON is invalid.';
        return;
      }}
      $('optimize-json').textContent = 'Optimizing payload...';
      const response = await fetch('/payload/optimize', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ payload }}),
      }});
      const data = await response.json();
      $('optimize-json').textContent = JSON.stringify(data, null, 2);
    }});
  </script>
</body>
</html>
"""


def build_onboarding_web_app(context: WebApiSystemContext | None = None) -> FastAPI:
    context = context or WebApiSystemContext(EnhancedQRSystem())
    app = FastAPI(title='Embedded QR Onboarding Web API', version='3.4')
    app.state.context = context

    @app.get('/health')
    def health():
        return {'ok': True}

    @app.get('/', response_class=HTMLResponse)
    def root_ui():
        return HTMLResponse(_ui_html())

    @app.get('/ui', response_class=HTMLResponse)
    def ui_alias():
        return HTMLResponse(_ui_html())

    @app.post('/scan/image')
    async def scan_image(file: UploadFile = File(...)):
        blob = await file.read()
        arr = np.frombuffer(blob, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail='Invalid image')
        result = context.system.scan_image(img)
        return {
            **result.to_dict(),
            'calibration': context.system.calibration_status(),
            'stats': context.system.pipeline_stats_summary(),
        }

    @app.post('/qr/generate')
    def generate_qr(req: GenerateQRRequest):
        raw, codec = _select_payload_bytes(req)
        used_text_armor = req.compatibility_text and not req.split and not payload_is_text_friendly(raw)
        qr_text = armor_binary_payload(raw) if used_text_armor else None
        if req.split:
            chunks = chunk_texts(raw, req.session_id, req.max_chunk_bytes, with_parity=req.with_parity)
            pngs = build_split_qr_pngs(chunks)
            return {
                'split': True,
                'chunk_count': len(chunks),
                'chunk_sizes': [len(x) for x in pngs],
                'chunk_texts': chunks,
                'chunk_png_base64': [base64.b64encode(x).decode('ascii') for x in pngs],
                'encoding': {
                    'encrypted': req.encrypted,
                    'payload_codec': codec,
                    'compatibility_text': False,
                    'transport': 'split-chunks',
                    'with_parity': req.with_parity,
                },
            }
        png = build_text_qr(qr_text) if qr_text is not None else build_binary_payload_qr(raw)
        preview = qr_text if qr_text is not None else (raw.decode('utf-8', errors='ignore')[:160] if payload_is_text_friendly(raw) else f'binary:{len(raw)} bytes')
        return {
            'split': False,
            'png_size': len(png),
            'png_base64': base64.b64encode(png).decode('ascii'),
            'payload_size_bytes': len(raw),
            'payload_preview': preview[:160],
            'encoding': {
                'encrypted': req.encrypted,
                'payload_codec': codec,
                'compatibility_text': used_text_armor,
                'transport': 'single-qr',
            },
        }

    @app.post('/payload/optimize')
    def payload_optimize(req: OptimizePayloadRequest):
        controller = PayloadComplexityController()
        variants = [variant.to_dict() for variant in controller.variants(req.payload)]
        return {'best': variants[0], 'variants': variants}

    @app.post('/consent/issue')
    def consent_issue(req: ConsentIssueRequest):
        if context.consent_manager is None:
            raise HTTPException(status_code=500, detail='Consent manager not configured')
        return context.consent_manager.issue(req.subject, req.purpose, req.ttl_seconds, req.metadata).to_payload()

    @app.post('/consent/verify')
    def consent_verify(req: ConsentVerifyRequest):
        if context.consent_manager is None:
            raise HTTPException(status_code=500, detail='Consent manager not configured')
        return context.consent_manager.verify(req.payload)

    @app.post('/bootstrap/register')
    def bootstrap_register(req: RegisterBootstrapRequest):
        if context.bootstrap_service is None:
            raise HTTPException(status_code=500, detail='Bootstrap service not configured')
        return context.bootstrap_service.register_device(req.registration_id, req.registration_token, req.wifi, req.metadata)

    @app.post('/provision/from-payload')
    def provision_from_payload(req: ProvisionFromPayloadRequest):
        manager = context.system.provisioning_manager
        if manager is None:
            raise HTTPException(status_code=500, detail='Provisioning manager not configured')
        return manager.provision(req.payload, req.device_id)

    return app
