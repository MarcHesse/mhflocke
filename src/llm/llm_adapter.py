#!/usr/bin/env python3
"""
MULTI-LLM ADAPTER v1.0
Orchestrates 5 free-tier LLM providers with automatic fallback and rate-limiting.

Providers (total ~16,750 requests/day):
  1. Gemini Flash-Lite      — 1,000 RPD  (Google)
  2. Groq Llama 3.3 70B     — 14,400 RPD (Groq)
  3. Mistral Codestral      — 1,000 RPD  (Mistral AI)
  4. OpenRouter Free Models — 50 RPD     (OpenRouter)
  5. HuggingFace StarCoder2 — 300 RPD    (HuggingFace)

Usage:
    adapter = MultiLLMAdapter(keys={'gemini': 'AIza...', 'groq': 'gsk_...'})
    response = adapter.generate("Write a function to calculate factorial")
    if response.success:
        print(response.text)

No external dependencies — uses urllib.request from stdlib.
"""

__version__ = "1.0"
__logbook__ = 148

from typing import Optional, Dict, List
from dataclasses import dataclass
from datetime import datetime
import json
import urllib.request
import urllib.parse
import time
from abc import ABC, abstractmethod


@dataclass
class LLMResponse:
    """Response from LLM API call."""
    success: bool
    text: str
    provider: str         # 'gemini', 'groq', 'mistral', etc.
    model: str            # actual model name
    tokens: int = 0       # tokens used (if available)
    latency_ms: float = 0.0
    error: str = ''


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, api_key: Optional[str], name: str, daily_limit: int):
        self.api_key = api_key
        self.name = name
        self.daily_limit = daily_limit
        self.enabled = api_key is not None and len(api_key) > 0
        
        # Rate limiting (simple counter, resets every 24h)
        self._request_count = 0
        self._count_reset_time = datetime.now()
    
    def _increment_count(self):
        """Increment request counter and reset if 24h passed."""
        now = datetime.now()
        if (now - self._count_reset_time).total_seconds() > 86400:
            self._request_count = 0
            self._count_reset_time = now
        self._request_count += 1
    
    def requests_remaining(self) -> int:
        """How many requests left today?"""
        return max(0, self.daily_limit - self._request_count)
    
    def can_request(self) -> bool:
        """Can we make another request?"""
        return self.enabled and self.requests_remaining() > 0
    
    def get_daily_limit(self) -> int:
        """Return the daily limit for this provider."""
        return self.daily_limit
    
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 1000) -> LLMResponse:
        """Generate text from prompt. Must be implemented by subclass."""
        pass


class GeminiProvider(LLMProvider):
    """Google Gemini Flash-Lite — 1,000 requests/day."""
    
    def __init__(self, api_key: Optional[str]):
        super().__init__(api_key, 'gemini', 1000)
        self.model = 'gemini-2.0-flash-lite'
        self.base_url = 'https://generativelanguage.googleapis.com/v1beta/models'
    
    def generate(self, prompt: str, max_tokens: int = 1000) -> LLMResponse:
        if not self.enabled:
            return LLMResponse(False, '', self.name, self.model, error='API key missing')
        
        if not self.can_request():
            return LLMResponse(False, '', self.name, self.model, 
                             error=f'Rate limit reached ({self.daily_limit}/day)')
        
        start = time.time()
        
        try:
            url = f'{self.base_url}/{self.model}:generateContent?key={self.api_key}'
            data = {
                'contents': [{'parts': [{'text': prompt}]}],
                'generationConfig': {'maxOutputTokens': max_tokens}
            }
            
            req = urllib.request.Request(
                url,
                data=json.dumps(data).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
            
            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read().decode('utf-8'))
            
            self._increment_count()
            
            text = result['candidates'][0]['content']['parts'][0]['text']
            tokens = result.get('usageMetadata', {}).get('totalTokenCount', 0)
            latency = (time.time() - start) * 1000
            
            return LLMResponse(True, text, self.name, self.model, tokens, latency)
        
        except Exception as e:
            return LLMResponse(False, '', self.name, self.model, 
                             error=f'Gemini error: {str(e)}')


class GroqProvider(LLMProvider):
    """Groq Llama 3.3 70B — 14,400 requests/day."""
    
    def __init__(self, api_key: Optional[str]):
        super().__init__(api_key, 'groq', 14400)
        self.model = 'llama-3.3-70b-versatile'
        self.base_url = 'https://api.groq.com/openai/v1/chat/completions'
    
    def generate(self, prompt: str, max_tokens: int = 1000) -> LLMResponse:
        if not self.enabled:
            return LLMResponse(False, '', self.name, self.model, error='API key missing')
        
        if not self.can_request():
            return LLMResponse(False, '', self.name, self.model,
                             error=f'Rate limit reached ({self.daily_limit}/day)')
        
        start = time.time()
        
        try:
            data = {
                'model': self.model,
                'messages': [{'role': 'user', 'content': prompt}],
                'max_tokens': max_tokens
            }
            
            req = urllib.request.Request(
                self.base_url,
                data=json.dumps(data).encode('utf-8'),
                headers={
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {self.api_key}'
                }
            )
            
            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read().decode('utf-8'))
            
            self._increment_count()
            
            text = result['choices'][0]['message']['content']
            tokens = result.get('usage', {}).get('total_tokens', 0)
            latency = (time.time() - start) * 1000
            
            return LLMResponse(True, text, self.name, self.model, tokens, latency)
        
        except Exception as e:
            return LLMResponse(False, '', self.name, self.model,
                             error=f'Groq error: {str(e)}')


class MistralProvider(LLMProvider):
    """Mistral Codestral — 1,000 requests/day."""
    
    def __init__(self, api_key: Optional[str]):
        super().__init__(api_key, 'mistral', 1000)
        self.model = 'codestral-latest'
        self.base_url = 'https://api.mistral.ai/v1/chat/completions'
    
    def generate(self, prompt: str, max_tokens: int = 1000) -> LLMResponse:
        if not self.enabled:
            return LLMResponse(False, '', self.name, self.model, error='API key missing')
        
        if not self.can_request():
            return LLMResponse(False, '', self.name, self.model,
                             error=f'Rate limit reached ({self.daily_limit}/day)')
        
        start = time.time()
        
        try:
            data = {
                'model': self.model,
                'messages': [{'role': 'user', 'content': prompt}],
                'max_tokens': max_tokens
            }
            
            req = urllib.request.Request(
                self.base_url,
                data=json.dumps(data).encode('utf-8'),
                headers={
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {self.api_key}'
                }
            )
            
            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read().decode('utf-8'))
            
            self._increment_count()
            
            text = result['choices'][0]['message']['content']
            tokens = result.get('usage', {}).get('total_tokens', 0)
            latency = (time.time() - start) * 1000
            
            return LLMResponse(True, text, self.name, self.model, tokens, latency)
        
        except Exception as e:
            return LLMResponse(False, '', self.name, self.model,
                             error=f'Mistral error: {str(e)}')


class OpenRouterProvider(LLMProvider):
    """OpenRouter Free Models — 50 requests/day."""
    
    def __init__(self, api_key: Optional[str]):
        super().__init__(api_key, 'openrouter', 50)
        self.model = 'meta-llama/llama-3.1-8b-instruct:free'
        self.base_url = 'https://openrouter.ai/api/v1/chat/completions'
    
    def generate(self, prompt: str, max_tokens: int = 1000) -> LLMResponse:
        if not self.enabled:
            return LLMResponse(False, '', self.name, self.model, error='API key missing')
        
        if not self.can_request():
            return LLMResponse(False, '', self.name, self.model,
                             error=f'Rate limit reached ({self.daily_limit}/day)')
        
        start = time.time()
        
        try:
            data = {
                'model': self.model,
                'messages': [{'role': 'user', 'content': prompt}],
                'max_tokens': max_tokens
            }
            
            req = urllib.request.Request(
                self.base_url,
                data=json.dumps(data).encode('utf-8'),
                headers={
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {self.api_key}'
                }
            )
            
            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read().decode('utf-8'))
            
            self._increment_count()
            
            text = result['choices'][0]['message']['content']
            tokens = result.get('usage', {}).get('total_tokens', 0)
            latency = (time.time() - start) * 1000
            
            return LLMResponse(True, text, self.name, self.model, tokens, latency)
        
        except Exception as e:
            return LLMResponse(False, '', self.name, self.model,
                             error=f'OpenRouter error: {str(e)}')


class HuggingFaceProvider(LLMProvider):
    """HuggingFace StarCoder2 — 300 requests/day."""
    
    def __init__(self, api_key: Optional[str]):
        super().__init__(api_key, 'huggingface', 300)
        self.model = 'bigcode/starcoder2-15b'
        self.base_url = f'https://api-inference.huggingface.co/models/{self.model}'
    
    def generate(self, prompt: str, max_tokens: int = 1000) -> LLMResponse:
        if not self.enabled:
            return LLMResponse(False, '', self.name, self.model, error='API key missing')
        
        if not self.can_request():
            return LLMResponse(False, '', self.name, self.model,
                             error=f'Rate limit reached ({self.daily_limit}/day)')
        
        start = time.time()
        
        try:
            data = {
                'inputs': prompt,
                'parameters': {'max_new_tokens': max_tokens}
            }
            
            req = urllib.request.Request(
                self.base_url,
                data=json.dumps(data).encode('utf-8'),
                headers={
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {self.api_key}'
                }
            )
            
            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read().decode('utf-8'))
            
            self._increment_count()
            
            # HuggingFace returns list of dicts
            if isinstance(result, list) and len(result) > 0:
                text = result[0].get('generated_text', '')
            else:
                text = result.get('generated_text', str(result))
            
            latency = (time.time() - start) * 1000
            
            return LLMResponse(True, text, self.name, self.model, 0, latency)
        
        except Exception as e:
            return LLMResponse(False, '', self.name, self.model,
                             error=f'HuggingFace error: {str(e)}')


class MultiLLMAdapter:
    """
    Orchestrates multiple LLM providers with automatic fallback.
    
    Strategy:
      1. Try provider with most remaining capacity
      2. If fails, try next provider
      3. If all fail, return error
    
    Usage:
        adapter = MultiLLMAdapter(keys={'gemini': 'AIza...', 'groq': 'gsk_...'})
        response = adapter.generate("Write Python code to sort a list")
    """
    
    def __init__(self, keys: Optional[Dict[str, str]] = None):
        """
        Args:
            keys: Dict mapping provider name to API key
                  e.g. {'gemini': 'AIza...', 'groq': 'gsk_...'}
                  Missing keys = provider disabled
        """
        if keys is None:
            keys = {}
        
        # Initialize all providers
        self.providers: List[LLMProvider] = [
            GeminiProvider(keys.get('gemini')),
            GroqProvider(keys.get('groq')),
            MistralProvider(keys.get('mistral')),
            OpenRouterProvider(keys.get('openrouter')),
            HuggingFaceProvider(keys.get('huggingface')),
        ]
        
        # Filter to enabled providers
        self.active_providers = [p for p in self.providers if p.enabled]
        self.enabled = len(self.active_providers) > 0
        
        # Statistics
        self.total_requests = 0
        self.total_successes = 0
        self.total_failures = 0
    
    def _get_best_provider(self) -> Optional[LLMProvider]:
        """Get provider with most remaining capacity."""
        available = [p for p in self.active_providers if p.can_request()]
        if not available:
            return None
        return max(available, key=lambda p: p.requests_remaining())
    
    def generate(self, prompt: str, max_tokens: int = 1000) -> LLMResponse:
        """
        Generate text using best available provider.
        Falls back to next provider on failure.
        """
        self.total_requests += 1
        
        # Try providers in order of remaining capacity
        tried = []
        
        while True:
            provider = self._get_best_provider()
            if not provider:
                self.total_failures += 1
                error = 'All providers exhausted or rate-limited'
                if tried:
                    error += f' (tried: {", ".join(tried)})'
                return LLMResponse(False, '', '', '', error=error)
            
            tried.append(provider.name)
            response = provider.generate(prompt, max_tokens)
            
            if response.success:
                self.total_successes += 1
                return response
            
            # Remove failed provider from active list for this request
            self.active_providers = [p for p in self.active_providers if p != provider]
            
            if not self.active_providers:
                self.total_failures += 1
                return LLMResponse(
                    False, '', '', '',
                    error=f'All providers failed. Tried: {", ".join(tried)}'
                )
    
    def generate_code(self, description: str, context_facts: List[str] = None) -> LLMResponse:
        """
        Generate Python code from natural language description.
        
        Args:
            description: What the code should do
            context_facts: Optional list of facts to include in prompt
        
        Returns:
            LLMResponse with generated code
        """
        prompt = f"""Generate Python code for the following task:

{description}

Requirements:
- Write clean, working Python code
- Include brief comments
- Use only Python standard library
- Code should be self-contained and executable

"""
        
        if context_facts:
            prompt += f"""
Context (relevant facts):
{chr(10).join('- ' + f for f in context_facts[:5])}

"""
        
        prompt += """
Respond with ONLY the Python code, no explanations before or after.
"""
        
        return self.generate(prompt, max_tokens=2000)
    
    def get_statistics(self) -> dict:
        """Get usage statistics."""
        total_capacity = sum(p.get_daily_limit() for p in self.providers)
        total_remaining = sum(p.requests_remaining() for p in self.active_providers)
        
        return {
            'total_requests': self.total_requests,
            'total_successes': self.total_successes,
            'total_failures': self.total_failures,
            'success_rate': self.total_successes / max(1, self.total_requests),
            'enabled': self.enabled,
            'active_providers': [p.name for p in self.active_providers],
            'total_capacity': total_capacity,
            'total_remaining': total_remaining,
            'providers': [
                {
                    'name': p.name,
                    'enabled': p.enabled,
                    'daily_limit': p.get_daily_limit(),
                    'remaining': p.requests_remaining() if p.enabled else 0,
                    'model': getattr(p, 'model', 'unknown')
                }
                for p in self.providers
            ]
        }
