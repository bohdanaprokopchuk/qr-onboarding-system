from qr_onboarding.split_qr import SplitQRAssembler, chunk_texts
def test_split_qr_chunk_assembly_roundtrip():
    payload=b'hello split qr system'*20; texts=chunk_texts(payload,'session-xyz',max_chunk_bytes=40); assembler=SplitQRAssembler(); assembled=None
    for text in reversed(texts): assembled=assembler.add_chunk_text(text) or assembled
    assert assembled is not None and assembled.payload==payload and assembled.total==len(texts)
