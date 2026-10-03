"""Persistent budget guard for this paid hardening run, including image calls."""

import json
import math
from pathlib import Path

import httpx

from app.provider import ManualProvider, ProviderError
from app.structured import atomic_json, manual_lock


class RunBudget:
    def __init__(self, settings, limit=20, max_attempts=128):
        self.settings = settings
        self.max_attempts = max_attempts
        self.path = settings.runtime / 'evaluations' / 'adversarial_api_ledger.json'
        if not self.path.exists():
            atomic_json(self.path, {'limit_usd': limit, 'input_per_million': 5,
                                   'output_per_million': 25, 'safety_multiplier': 2,
                                   'pricing_source': 'https://api.mwapi.dev/api/v1/model-plaza',
                                   'baseline': self.usage(), 'attempts': []})

    def usage(self):
        with httpx.Client(timeout=20) as client:
            response = client.get(self.settings.base_url + '/usage',
                                  headers={'Authorization': 'Bearer ' + self.settings.api_key})
            response.raise_for_status()
            data = response.json()
        if data.get('unit') != 'USD' or not data.get('isValid'):
            raise ProviderError('Cannot verify USD key usage; paid tests are disabled')
        total = data['usage']['total']
        values = {'balance': data['balance'], 'actual_cost': total['actual_cost']}
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in values.values()):
            raise ProviderError('Invalid gateway accounting; paid tests are disabled')
        return values

    def reserve(self, payload):
        # Byte count bounds text tokens conservatively. Image reservation uses
        # the ingestion pipeline's deliberately high 32k input-token allowance.
        text_bytes, images = 0, 0
        for message in payload['messages']:
            content = message['content']
            if isinstance(content, str):
                text_bytes += len(content.encode())
            else:
                for item in content:
                    if item['type'] == 'text':
                        text_bytes += len(item['text'].encode())
                    elif item['type'] == 'image_url':
                        images += 1
        if images > 1:
            raise ProviderError('Hardening budget permits one page image per request')
        current = self.usage()
        with manual_lock(self.settings.runtime, 'adversarial-api-budget'):
            ledger = json.loads(self.path.read_text())
            reservation = ((max(text_bytes + 1024, 32000 if images else 0) * ledger['input_per_million'] +
                            payload['max_tokens'] * ledger['output_per_million']) / 1e6 * ledger['safety_multiplier'])
            accounted = sum(attempt.get('estimated_usd', attempt['reserved_usd']) for attempt in ledger['attempts'])
            speech_path = self.settings.runtime / 'evaluations/deepgram_speech_budget.json'
            if speech_path.exists():
                accounted += json.loads(speech_path.read_text())['reserved_usd']
            observed = max(0, ledger['baseline']['balance'] - current['balance'],
                           current['actual_cost'] - ledger['baseline']['actual_cost'])
            if max(accounted, observed) + reservation > ledger['limit_usd'] or len(ledger['attempts']) >= self.max_attempts:
                raise ProviderError('US$20 hardening allowance or request limit would be exceeded')
            index = len(ledger['attempts'])
            ledger['attempts'].append({'reserved_usd': reservation, 'image_count': images, 'status': 'pending'})
            ledger['latest_gateway_usage'] = current
            atomic_json(self.path, ledger)
        return index

    def complete(self, index, response):
        with manual_lock(self.settings.runtime, 'adversarial-api-budget'):
            ledger = json.loads(self.path.read_text())
            usage = response.get('usage') or {}
            if all(type(usage.get(key)) is int and usage[key] >= 0 for key in ('prompt_tokens', 'completion_tokens')):
                ledger['attempts'][index]['estimated_usd'] = (
                    usage['prompt_tokens'] * ledger['input_per_million'] +
                    usage['completion_tokens'] * ledger['output_per_million']) / 1e6 * ledger['safety_multiplier']
            ledger['attempts'][index].update(status='responded', usage=usage)
            atomic_json(self.path, ledger)


class BudgetedProvider(ManualProvider):
    def __init__(self, settings, budget):
        super().__init__(settings.base_url, settings.api_key, settings.model)
        self.budget = budget

    def _post_response(self, payload):
        index = self.budget.reserve(payload)
        response = super()._post_response(payload)
        self.budget.complete(index, response)
        return response
