from __future__ import annotations

import base64
import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .payload_codecs import PayloadError, encode_chunk_text, parse_chunk_text


PARITY_INDEX_SENTINEL = 'xor-parity'


def _xor_parity(chunks: List[bytes]) -> bytes:
    if not chunks:
        raise PayloadError('At least one chunk is required for parity calculation')
    width = max(len(chunk) for chunk in chunks)
    out = bytearray(width)
    for chunk in chunks:
        for i in range(width):
            out[i] ^= chunk[i] if i < len(chunk) else 0
    return bytes(out)


@dataclass
class SplitChunk:
    session_id: str
    index: int
    total: int
    digest: str
    crc32: str
    chunk_bytes: bytes

    @property
    def is_parity(self) -> bool:
        return self.index == self.total

    @classmethod
    def from_text(cls, text: str) -> 'SplitChunk':
        return cls(**parse_chunk_text(text))

    def to_text(self) -> str:
        return encode_chunk_text(self.chunk_bytes, self.session_id, self.index, self.total, payload_digest=self.digest)


@dataclass
class XorParityBlock:
    lengths: list[int]
    parity_bytes: bytes

    def encode(self) -> bytes:
        payload = {
            'kind': PARITY_INDEX_SENTINEL,
            'lengths': self.lengths,
            'parity': base64.urlsafe_b64encode(self.parity_bytes).decode('ascii'),
        }
        return json.dumps(payload, separators=(',', ':')).encode('utf-8')

    @classmethod
    def decode(cls, data: bytes) -> 'XorParityBlock':
        try:
            obj = json.loads(data.decode('utf-8'))
        except Exception as exc:
            raise PayloadError('Invalid XOR parity chunk payload') from exc
        if obj.get('kind') != PARITY_INDEX_SENTINEL:
            raise PayloadError('Chunk does not contain XOR parity metadata')
        lengths = obj.get('lengths')
        parity_b64 = obj.get('parity')
        if not isinstance(lengths, list) or not all(isinstance(item, int) and item >= 0 for item in lengths):
            raise PayloadError('Invalid parity lengths metadata')
        if not isinstance(parity_b64, str):
            raise PayloadError('Invalid parity payload encoding')
        try:
            parity_bytes = base64.urlsafe_b64decode(parity_b64.encode('ascii'))
        except Exception as exc:
            raise PayloadError('Invalid parity base64 payload') from exc
        return cls(lengths=lengths, parity_bytes=parity_bytes)


@dataclass
class AssembledPayload:
    session_id: str
    total: int
    digest: str
    payload: bytes
    recovered_indices: list[int] = field(default_factory=list)
    used_parity: bool = False


@dataclass
class SplitProgress:
    session_id: str
    total: int
    collected: int
    missing_data: list[int]
    have_parity: bool
    can_recover: bool
    done: bool

    def status_line(self) -> str:
        blocks = ''.join('■' if idx not in self.missing_data else '□' for idx in range(self.total))
        suffix = []
        if self.have_parity:
            suffix.append('parity')
        if self.can_recover and not self.done:
            suffix.append('recoverable')
        if self.done:
            suffix.append('complete')
        return f'Split-QR {blocks} {self.collected}/{self.total}' + (f' ({", ".join(suffix)})' if suffix else '')


@dataclass
class SessionState:
    total: int
    digest: str
    data_chunks: Dict[int, SplitChunk] = field(default_factory=dict)
    parity_chunk: SplitChunk | None = None


@dataclass
class SplitQRAssembler:
    sessions: Dict[str, SessionState] = field(default_factory=dict)

    def add_chunk_text(self, text: str) -> Optional[AssembledPayload]:
        return self.add_chunk(SplitChunk.from_text(text))

    def add_chunk(self, chunk: SplitChunk) -> Optional[AssembledPayload]:
        state = self.sessions.get(chunk.session_id)
        if state is None:
            state = SessionState(total=chunk.total, digest=chunk.digest)
            self.sessions[chunk.session_id] = state
        if state.total != chunk.total:
            raise PayloadError('Conflicting total count for split QR session')
        if state.digest != chunk.digest:
            raise PayloadError('Conflicting digest for split QR session')

        if chunk.is_parity:
            state.parity_chunk = chunk
        else:
            if chunk.index < 0 or chunk.index >= chunk.total:
                raise PayloadError('Chunk index out of range for split QR payload')
            state.data_chunks[chunk.index] = chunk

        return self._assemble_if_ready(chunk.session_id, state)

    def _assemble_if_ready(self, session_id: str, state: SessionState) -> Optional[AssembledPayload]:
        if len(state.data_chunks) == state.total:
            return self._assemble_full(session_id, state, used_parity=False, recovered_indices=[])
        if state.parity_chunk is None:
            return None
        missing = [idx for idx in range(state.total) if idx not in state.data_chunks]
        if len(missing) != 1:
            return None
        recovered = self._recover_missing_chunk(state, missing[0])
        state.data_chunks[missing[0]] = recovered
        return self._assemble_full(session_id, state, used_parity=True, recovered_indices=missing)

    def _recover_missing_chunk(self, state: SessionState, missing_idx: int) -> SplitChunk:
        if state.parity_chunk is None:
            raise PayloadError('Parity chunk is not available for recovery')
        parity_meta = XorParityBlock.decode(state.parity_chunk.chunk_bytes)
        if len(parity_meta.lengths) != state.total:
            raise PayloadError('Parity metadata does not match total chunk count')
        if missing_idx >= len(parity_meta.lengths):
            raise PayloadError('Missing chunk index is outside parity metadata')
        recovered_len = parity_meta.lengths[missing_idx]
        pieces = [chunk.chunk_bytes for idx, chunk in sorted(state.data_chunks.items()) if idx != missing_idx]
        parity_input = pieces + [parity_meta.parity_bytes]
        recovered = _xor_parity(parity_input)[:recovered_len]
        crc = hashlib.sha256(recovered).hexdigest()[:16]
        if len(recovered) != recovered_len:
            raise PayloadError(f'XOR recovery for chunk {missing_idx} returned an unexpected length')
        return SplitChunk(
            session_id=state.parity_chunk.session_id,
            index=missing_idx,
            total=state.total,
            digest=state.digest,
            crc32=f'{(zlib_crc32(recovered)):08x}',
            chunk_bytes=recovered,
        )

    def _assemble_full(self, session_id: str, state: SessionState, used_parity: bool, recovered_indices: list[int]) -> AssembledPayload:
        payload = b''.join(state.data_chunks[idx].chunk_bytes for idx in range(state.total))
        got = hashlib.sha256(payload).hexdigest()[:16]
        if got != state.digest:
            raise PayloadError('Assembled split payload digest mismatch')
        return AssembledPayload(
            session_id=session_id,
            total=state.total,
            digest=got,
            payload=payload,
            recovered_indices=list(recovered_indices),
            used_parity=used_parity,
        )

    def progress(self, session_id: str) -> SplitProgress | None:
        state = self.sessions.get(session_id)
        if state is None:
            return None
        missing = [idx for idx in range(state.total) if idx not in state.data_chunks]
        have_parity = state.parity_chunk is not None
        can_recover = have_parity and len(missing) <= 1
        done = len(missing) == 0 or (have_parity and len(missing) == 1)
        return SplitProgress(
            session_id=session_id,
            total=state.total,
            collected=len(state.data_chunks),
            missing_data=missing,
            have_parity=have_parity,
            can_recover=can_recover,
            done=done,
        )


def zlib_crc32(data: bytes) -> int:
    import zlib
    return zlib.crc32(data) & 0xFFFFFFFF


def split_payload(payload: bytes, session_id: str, max_chunk_bytes: int = 180, with_parity: bool = False) -> List[SplitChunk]:
    total = max(1, math.ceil(len(payload) / max_chunk_bytes))
    digest = hashlib.sha256(payload).hexdigest()[:16]
    out: list[SplitChunk] = []
    data_chunks: list[bytes] = []
    for i in range(total):
        chunk = payload[i * max_chunk_bytes:(i + 1) * max_chunk_bytes]
        data_chunks.append(chunk)
        out.append(SplitChunk(**parse_chunk_text(encode_chunk_text(chunk, session_id, i, total, payload_digest=digest))))
    if with_parity:
        parity_meta = XorParityBlock([len(chunk) for chunk in data_chunks], _xor_parity(data_chunks)).encode()
        out.append(SplitChunk(**parse_chunk_text(encode_chunk_text(parity_meta, session_id, total, total, payload_digest=digest))))
    return out


def chunk_texts(payload: bytes, session_id: str, max_chunk_bytes: int = 180, with_parity: bool = False) -> List[str]:
    return [chunk.to_text() for chunk in split_payload(payload, session_id, max_chunk_bytes=max_chunk_bytes, with_parity=with_parity)]
