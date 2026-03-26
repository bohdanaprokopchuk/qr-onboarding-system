from qr_onboarding.crypto_utils import generate_demo_keypair
from qr_onboarding.payload_codecs import armor_binary_payload, decode_versioned_payload, dearmor_binary_payload, encode_cbor_v1, encode_json_v1, encode_x25519_cbor_v1, encode_x25519_raw_json_v1


def test_roundtrip():
    payload = {'ssid': 'demo', 'psk': 'secret1234', 'CC': 'US', 'registration-id': 'abcd1234'}; kp = generate_demo_keypair()
    assert decode_versioned_payload(encode_json_v1(payload)).normalized['ssid'] == 'demo'
    assert decode_versioned_payload(encode_cbor_v1(payload)).normalized['ssid'] == 'demo'
    assert decode_versioned_payload(encode_x25519_raw_json_v1(payload, kp.public_key_hex), private_key=kp.private_key_hex).normalized['ssid'] == 'demo'
    assert decode_versioned_payload(encode_x25519_cbor_v1(payload, kp.public_key_hex), private_key=kp.private_key_hex).normalized['ssid'] == 'demo'


def test_armored_binary_payload_roundtrip():
    payload = encode_cbor_v1({'ssid': 'armored-demo', 'registration-id': 'rid-arm'})
    armored = armor_binary_payload(payload)
    assert dearmor_binary_payload(armored) == payload
    assert decode_versioned_payload(armored.encode('utf-8')).normalized['ssid'] == 'armored-demo'
