from __future__ import annotations
import json, subprocess, time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol
import requests
class WifiBackend(Protocol):
    def connect(self, ssid: str, password: Optional[str], security: str='WPA-PSK', hidden: bool=False) -> dict[str, Any]: ...
    def current_connection(self) -> dict[str, Any]: ...
@dataclass
class CommandResult: command: list[str]; returncode: int; stdout: str; stderr: str
@dataclass
class RetryPolicy: attempts: int = 3; backoff_seconds: float = 1.0
@dataclass
class ProvisioningSession:
    registration_id: str; registration_token: str; bootstrap_uri: Optional[str]=None; session_token: Optional[str]=None; session_expires_at: Optional[float]=None; audit_log: list[str]=field(default_factory=list)
class SubprocessRunner:
    def run(self, command: list[str], timeout: int=30) -> CommandResult:
        p=subprocess.run(command, capture_output=True, text=True, timeout=timeout); return CommandResult(command,p.returncode,p.stdout,p.stderr)
class NmcliWifiAdapter:
    def __init__(self, runner: Optional[SubprocessRunner]=None): self.runner=runner or SubprocessRunner()
    def connect(self, ssid: str, password: Optional[str], security: str='WPA-PSK', hidden: bool=False) -> dict[str, Any]:
        cmd=['nmcli','device','wifi','connect',ssid]
        if password: cmd+=['password',password]
        if hidden: cmd+=['hidden','yes']
        r=self.runner.run(cmd); return {'ok':r.returncode==0,'stdout':r.stdout,'stderr':r.stderr,'backend':'nmcli'}
    def current_connection(self) -> dict[str, Any]:
        r=self.runner.run(['nmcli','-t','-f','ACTIVE,SSID,DEVICE','device','wifi']); return {'ok':r.returncode==0,'stdout':r.stdout,'stderr':r.stderr,'backend':'nmcli'}
class WpaCliWifiAdapter:
    def __init__(self, interface: str='wlan0', runner: Optional[SubprocessRunner]=None): self.interface=interface; self.runner=runner or SubprocessRunner()
    def connect(self, ssid: str, password: Optional[str], security: str='WPA-PSK', hidden: bool=False) -> dict[str, Any]:
        seq=[['wpa_cli','-i',self.interface,'add_network'],['wpa_cli','-i',self.interface,'set_network','0','ssid',f'"{ssid}"']]
        seq.append(['wpa_cli','-i',self.interface,'set_network','0','psk',f'"{password}"'] if password else ['wpa_cli','-i',self.interface,'set_network','0','key_mgmt','NONE'])
        if hidden: seq.append(['wpa_cli','-i',self.interface,'set_network','0','scan_ssid','1'])
        seq += [['wpa_cli','-i',self.interface,'enable_network','0'],['wpa_cli','-i',self.interface,'save_config'],['wpa_cli','-i',self.interface,'reconnect']]
        logs=[]; ok=True
        for cmd in seq:
            r=self.runner.run(cmd); logs.append({'cmd':cmd,'stdout':r.stdout,'stderr':r.stderr,'rc':r.returncode}); ok &= r.returncode==0
        return {'ok':ok,'logs':logs,'backend':'wpa_cli'}
    def current_connection(self) -> dict[str, Any]:
        r=self.runner.run(['wpa_cli','-i',self.interface,'status']); return {'ok':r.returncode==0,'stdout':r.stdout,'stderr':r.stderr,'backend':'wpa_cli'}
class CloudBootstrapClient:
    def __init__(self, base_url: str, timeout: float=5.0): self.base_url=base_url.rstrip('/'); self.timeout=timeout
    def create_session(self, registration_id: str, registration_token: str) -> dict[str, Any]:
        r=requests.post(f'{self.base_url}/bootstrap/session', json={'registration_id':registration_id,'registration_token':registration_token}, timeout=self.timeout); r.raise_for_status(); return r.json()
    def fetch_context(self, session_token: str) -> dict[str, Any]:
        r=requests.get(f'{self.base_url}/bootstrap/context', headers={'Authorization':f'Bearer {session_token}'}, timeout=self.timeout); r.raise_for_status(); return r.json()
    def acknowledge(self, session_token: str, device_id: str) -> dict[str, Any]:
        r=requests.post(f'{self.base_url}/bootstrap/ack', headers={'Authorization':f'Bearer {session_token}'}, json={'device_id':device_id}, timeout=self.timeout); r.raise_for_status(); return r.json()
class ProvisioningManager:
    def __init__(self, wifi_backend: WifiBackend, cloud_client: Optional[CloudBootstrapClient]=None, retry_policy: Optional[RetryPolicy]=None): self.wifi_backend=wifi_backend; self.cloud_client=cloud_client; self.retry_policy=retry_policy or RetryPolicy()
    def _retry(self, op: Callable[[],Any], label: str, session: ProvisioningSession) -> Any:
        last=None
        for attempt in range(1,self.retry_policy.attempts+1):
            try: session.audit_log.append(f'{label}: attempt {attempt}'); return op()
            except Exception as exc:
                last=exc; session.audit_log.append(f'{label}: failure {exc}')
                if attempt<self.retry_policy.attempts: time.sleep(self.retry_policy.backoff_seconds*attempt)
        raise RuntimeError(f'{label} failed after {self.retry_policy.attempts} attempts') from last
    def hydrate_context(self, session: ProvisioningSession) -> dict[str, Any]:
        if not self.cloud_client: raise RuntimeError('Cloud client is not configured')
        if not session.session_token or (session.session_expires_at and time.time()>=session.session_expires_at):
            created=self._retry(lambda: self.cloud_client.create_session(session.registration_id, session.registration_token), 'create-session', session); session.session_token=created['session_token']; session.session_expires_at=float(created['expires_at'])
        return self._retry(lambda: self.cloud_client.fetch_context(session.session_token), 'fetch-context', session)
    def apply_wifi_context(self, context: dict[str, Any], session: ProvisioningSession) -> dict[str, Any]:
        wifi=context.get('wifi') or context
        r=self._retry(lambda: self.wifi_backend.connect(wifi['ssid'], wifi.get('psk'), wifi.get('security','WPA-PSK'), bool(wifi.get('hidden',False))), 'wifi-connect', session)
        if not r.get('ok'): raise RuntimeError(f'Wi-Fi provisioning failed: {json.dumps(r)}')
        return r
    def provision(self, parsed_payload: dict[str, Any], device_id: str='embedded-linux-device') -> dict[str, Any]:
        s=ProvisioningSession(parsed_payload.get('registration-id',''), parsed_payload.get('registration-token',''), parsed_payload.get('uri')); out={'context':None,'wifi':None,'ack':None,'audit_log':s.audit_log}
        if 'ssid' in parsed_payload: out['wifi']=self.apply_wifi_context({'wifi':parsed_payload}, s); return out
        ctx=self.hydrate_context(s); out['context']=ctx; out['wifi']=self.apply_wifi_context(ctx,s)
        if self.cloud_client and s.session_token: out['ack']=self._retry(lambda: self.cloud_client.acknowledge(s.session_token, device_id), 'ack', s)
        return out
