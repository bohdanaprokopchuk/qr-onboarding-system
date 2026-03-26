from qr_onboarding.consent import ConsentManager
from qr_onboarding.persistence import SQLiteBootstrapStore

def test_consent_roundtrip(tmp_path):
    store = SQLiteBootstrapStore(tmp_path / 'store.db')
    manager = ConsentManager('secret', store=store)
    record = manager.issue('user-1', 'wifi-bootstrap', ttl_seconds=60, metadata={'scope': 'demo'})
    result = manager.verify(record.to_payload())
    assert result['ok'] is True
    saved = store.get_consent(record.consent_id)
    assert saved is not None
    assert saved['verified_at'] is not None
