import time
from fastapi.testclient import TestClient
from qr_onboarding.cloud_service import CloudBootstrapService, DeviceRecord, build_cloud_bootstrap_app
from qr_onboarding.provisioning import ProvisioningManager
class FakeWifiBackend:
    def __init__(self): self.calls=[]
    def connect(self, ssid, password, security='WPA-PSK', hidden=False): self.calls.append((ssid,password,security,hidden)); return {'ok':True,'ssid':ssid,'backend':'fake'}
    def current_connection(self): return {'ok':True}
class FakeCloudClient:
    def create_session(self, registration_id, registration_token): assert registration_id=='rid-2' and registration_token=='rtk-2'; return {'session_token':'token-123','expires_at':time.time()+60}
    def fetch_context(self, session_token): assert session_token=='token-123'; return {'wifi':{'ssid':'FieldWiFi','psk':'pass-2'}}
    def acknowledge(self, session_token, device_id): return {'ok':True,'session_token':session_token,'device_id':device_id}
def test_cloud_service_issue_and_acknowledge():
    service=CloudBootstrapService(secret='secret-secret-secret-secret-123456', ttl_seconds=60); service.store.add_device(DeviceRecord(registration_id='rid-demo', registration_token='rtk-demo', wifi={'ssid':'DemoWiFi','psk':'pass'})); client=TestClient(build_cloud_bootstrap_app(service)); token=client.post('/bootstrap/session', json={'registration_id':'rid-demo','registration_token':'rtk-demo'}).json()['session_token']; ctx=client.get('/bootstrap/context', headers={'Authorization':f'Bearer {token}'}); assert ctx.json()['wifi']['ssid']=='DemoWiFi'; ack=client.post('/bootstrap/ack', headers={'Authorization':f'Bearer {token}'}, json={'device_id':'dev1'}); assert ack.json()['ok'] is True
def test_provisioning_manager_end_to_end_with_client_stub():
    wifi=FakeWifiBackend(); manager=ProvisioningManager(wifi_backend=wifi, cloud_client=FakeCloudClient()); result=manager.provision({'registration-id':'rid-2','registration-token':'rtk-2'}); assert result['context']['wifi']['ssid']=='FieldWiFi' and wifi.calls[0][0]=='FieldWiFi' and result['ack']['ok'] is True
