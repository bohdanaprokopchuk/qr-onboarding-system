from fastapi.testclient import TestClient

from qr_onboarding.cloud_service import CloudBootstrapService
from qr_onboarding.consent import ConsentManager
from qr_onboarding.crypto_utils import generate_demo_keypair
from qr_onboarding.enhanced_pipeline import EnhancedQRSystem
from qr_onboarding.provisioning import ProvisioningManager
from qr_onboarding.web_api import WebApiSystemContext, build_onboarding_web_app


class FakeWifiBackend:
    def connect(self, ssid, password, security='WPA-PSK', hidden=False): return {'ok': True, 'ssid': ssid, 'password': password, 'security': security, 'hidden': hidden}
    def current_connection(self): return {'ok': True, 'ssid': 'demo'}


class FakeCloudClient:
    def create_session(self, registration_id, registration_token): return {'session_token': 'tok-1', 'expires_at': 4102444800.0}
    def fetch_context(self, session_token): return {'wifi': {'ssid': 'Campus', 'psk': 'pw'}}
    def acknowledge(self, session_token, device_id): return {'ok': True, 'device_id': device_id}


def test_web_api_utility_endpoints():
    manager = ProvisioningManager(FakeWifiBackend(), cloud_client=FakeCloudClient())
    context = WebApiSystemContext(system=EnhancedQRSystem(provisioning_manager=manager), consent_manager=ConsentManager('secret'), bootstrap_service=CloudBootstrapService(secret='s' * 32))
    client = TestClient(build_onboarding_web_app(context))
    qr_resp = client.post('/qr/generate', json={'payload': {'ssid': 'A', 'registration-id': 'rid', 'registration-token': 'tok-1234567890', 'psk': 'SecretPass123'}, 'split': True, 'session_id': 's1', 'max_chunk_bytes': 12})
    assert qr_resp.status_code == 200 and qr_resp.json()['chunk_count'] >= 2
    assert len(qr_resp.json()['chunk_png_base64']) == qr_resp.json()['chunk_count']
    consent = client.post('/consent/issue', json={'subject': 'user', 'purpose': 'bootstrap'})
    verify = client.post('/consent/verify', json={'payload': consent.json()})
    assert verify.json()['ok'] is True
    reg = client.post('/bootstrap/register', json={'registration_id': 'rid', 'registration_token': 'rtk', 'wifi': {'ssid': 'Campus', 'psk': 'pw'}, 'metadata': {}})
    assert reg.status_code == 200
    prov = client.post('/provision/from-payload', json={'payload': {'registration-id': 'rid', 'registration-token': 'rtk'}})
    assert prov.status_code == 200 and prov.json()['wifi']['ok'] is True


def test_web_api_generates_text_armored_encrypted_qr():
    kp = generate_demo_keypair()
    client = TestClient(build_onboarding_web_app(WebApiSystemContext(system=EnhancedQRSystem(private_key=kp.private_key_hex))))
    resp = client.post('/qr/generate', json={'payload': {'ssid': 'Secure', 'registration-id': 'rid-sec'}, 'encrypted': True, 'public_key_hex': kp.public_key_hex, 'payload_codec': 'auto'})
    body = resp.json()
    assert resp.status_code == 200
    assert body['encoding']['encrypted'] is True
    assert body['encoding']['compatibility_text'] is True
    assert body['png_base64']
    assert body['payload_size_bytes'] > 0
