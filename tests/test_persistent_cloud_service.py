from fastapi.testclient import TestClient
from qr_onboarding.cloud_service import CloudBootstrapService, PersistentCloudBootstrapStore, build_cloud_bootstrap_app

def test_persistent_cloud_bootstrap_flow(tmp_path):
    store = PersistentCloudBootstrapStore(str(tmp_path / 'bootstrap.db'))
    service = CloudBootstrapService(secret='d'*32, ttl_seconds=120, store=store)
    client = TestClient(build_cloud_bootstrap_app(service))
    assert client.post('/bootstrap/register', json={'registration_id': 'rid-1', 'registration_token': 'rtk-1', 'wifi': {'ssid': 'Lab', 'psk': 'secret'}, 'metadata': {'owner': 'tester'}}).status_code == 200
    session = client.post('/bootstrap/session', json={'registration_id': 'rid-1', 'registration_token': 'rtk-1'})
    token = session.json()['session_token']
    context = client.get('/bootstrap/context', headers={'Authorization': f'Bearer {token}'})
    assert context.status_code == 200 and context.json()['wifi']['ssid'] == 'Lab'
    ack = client.post('/bootstrap/ack', json={'device_id': 'dev-1'}, headers={'Authorization': f'Bearer {token}'})
    assert ack.status_code == 200
    acks = client.get('/bootstrap/acks')
    assert len(acks.json()['acks']) == 1
