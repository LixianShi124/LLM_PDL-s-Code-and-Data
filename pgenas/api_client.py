import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request

class APIClient:

    def __init__(self, api_key, base_url, model, output, max_calls=120, timeout=90, max_tokens=5000, temperature=1.0):
        url = urllib.parse.urlsplit(base_url)
        if url.scheme != 'https' or not url.netloc or url.username or url.password or url.query:
            raise ValueError('PGENAS_BASE_URL must be an HTTPS URL without credentials or query parameters')
        if not api_key or not model or max_calls < 1:
            raise ValueError('An API key, model and positive request budget are required')
        self.endpoint = base_url.rstrip('/')
        if not self.endpoint.endswith('/chat/completions'):
            self.endpoint += '/chat/completions'
        self.api_key, self.model = (api_key, model)
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.max_calls, self.timeout = (max_calls, timeout)
        self.max_tokens, self.temperature = (max_tokens, temperature)
        self.records = []

    @classmethod
    def from_environment(cls, output, config):
        return cls(os.environ.get('PGENAS_API_KEY', ''), os.environ.get('PGENAS_BASE_URL', ''), os.environ.get('PGENAS_MODEL', 'deepseek-v4-flash'), output, config['max_api_calls'], config['api_timeout'], config['api_max_tokens'], config['temperature'])

    def save(self, name, value):
        text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
        (self.output / name).write_text(text.replace(self.api_key, '[REDACTED]'), encoding='utf-8')

    def generate(self, prompt):
        if len(self.records) >= self.max_calls:
            return dict(success=False, error='API request budget exhausted')
        index = len(self.records) + 1
        body = dict(model=self.model, temperature=self.temperature, max_tokens=self.max_tokens, messages=[dict(role='system', content='Follow the six-module architecture design specification. Return only executable PyTorch architecture code.'), dict(role='user', content=prompt)])
        self.save(f'call_{index:06d}_request.json', body)
        request = urllib.request.Request(self.endpoint, data=json.dumps(body).encode('utf-8'), headers={'Authorization': 'Bearer ' + self.api_key, 'Content-Type': 'application/json'})
        start = time.monotonic()
        record = dict(call_index=index, success=False)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read(2000001)
            if len(raw) > 2000000:
                raise ValueError('API response exceeds 2 MB')
            result = json.loads(raw)
            code = result['choices'][0]['message']['content']
            if not isinstance(code, str) or not code.strip():
                raise ValueError('API response has no architecture code')
            self.save(f'call_{index:06d}_response.json', result)
            record.update(success=True, code=code.replace(self.api_key, '[REDACTED]'), usage=result.get('usage'), returned_model=result.get('model'))
        except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
            record['error'] = str(error).replace(self.api_key, '[REDACTED]')[:2000]
        record['latency_seconds'] = time.monotonic() - start
        self.records.append(record)
        self.save(f'call_{index:06d}_result.json', record)
        return record

    def summary(self):
        totals = {key: 0 for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')}
        missing = {key: 0 for key in totals}
        for row in self.records:
            usage = row.get('usage') or {}
            for key in totals:
                value = usage.get(key)
                if isinstance(value, int) and value >= 0:
                    totals[key] += value
                else:
                    missing[key] += 1
        return dict(requests=len(self.records), successes=sum((r['success'] for r in self.records)), failures=sum((not r['success'] for r in self.records)), **totals, missing_usage_records=missing, api_seconds=sum((r['latency_seconds'] for r in self.records)))
