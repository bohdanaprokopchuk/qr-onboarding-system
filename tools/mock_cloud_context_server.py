from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse
import json
DEMO_DB={'118f61fdf9f04ff75405a8dc': {'network': {'ssid':'my_shiny_ssid','security':'WPA2'}, 'cloud': {'endpoint':'mqtts://broker.example:8883','tenant':'demo-lab'}, 'expires_in':300}}
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed=urlparse(self.path)
        if parsed.path!='/setup': self.send_response(404); self.end_headers(); return
        rid=parse_qs(parsed.query).get('rid',[''])[0]; body=DEMO_DB.get(rid, {'error':'unknown rid'}); payload=json.dumps(body).encode('utf-8')
        self.send_response(200); self.send_header('Content-Type','application/json; charset=utf-8'); self.send_header('Content-Length', str(len(payload))); self.end_headers(); self.wfile.write(payload)
if __name__=='__main__':
    server=HTTPServer(('127.0.0.1',8080), Handler); print('Mock cloud context server running on http://127.0.0.1:8080'); server.serve_forever()
