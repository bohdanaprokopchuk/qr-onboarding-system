from qr_onboarding.crypto_utils import generate_demo_keypair
from qr_onboarding.payload_codecs import classify_text_payload, decode_versioned_payload, encode_cbor_v1, encode_chunk_text, encode_x25519_cbor_v1
def test_cbor_roundtrip_and_split_chunk_classification():
    payload={'ssid':'LabWiFi','registration-id':'rid-1','registration-token':'rtk-1'}; encoded=encode_cbor_v1(payload); parsed=decode_versioned_payload(encoded); assert parsed.normalized['ssid']=='LabWiFi'; chunk=encode_chunk_text(b'abc123','sess-1',0,2); cp=classify_text_payload(chunk); assert cp.payload_kind=='split-chunk' and cp.normalized['total']==2
def test_x25519_cbor_roundtrip_with_fallback_crypto():
    kp=generate_demo_keypair(); payload={'ssid':'SecureWiFi','registration-id':'rid-2','registration-token':'rtk-2'}; encoded=encode_x25519_cbor_v1(payload, kp.public_key_hex); parsed=decode_versioned_payload(encoded, private_key=kp.private_key_hex); assert parsed.normalized['ssid']=='SecureWiFi'
