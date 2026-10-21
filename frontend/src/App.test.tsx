import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import React from 'react';
import { SeverityBadge, CategoryBadge, VerificationBadge } from './components/Badge';
import { DiffViewer } from './components/DiffViewer';
import { SlothMascot } from './components/SlothMascot';

describe('Badge Components', () => {
  it('renders severity badge with correct label', () => {
    render(<SeverityBadge severity="high" />);
    expect(screen.getByText('high')).toBeDefined();
  });

  it('renders category badge with friendly text', () => {
    render(<CategoryBadge category="configuration" />);
    expect(screen.getByText('Config')).toBeDefined();
  });

  it('renders verification badge with checkmark on passed', () => {
    render(<VerificationBadge status="passed" />);
    expect(screen.getByText(/Passed/)).toBeDefined();
  });
});

describe('SlothMascot Component', () => {
  it('renders SVG with accessible label', () => {
    render(<SlothMascot size={32} />);
    const el = screen.getByLabelText('QuietPatch Sloth Mascot');
    expect(el).toBeDefined();
    expect(el.getAttribute('width')).toBe('32');
  });
});

describe('DiffViewer Component', () => {
  it('renders diff lines and file path', () => {
    const diff = '--- a/file.txt\n+++ b/file.txt\n@@ -1,1 +1,2 @@\n-old line\n+new line';
    render(
      <DiffViewer
        diff={diff}
        originalContent="old line"
        proposedContent="new line"
        filePath="src/file.txt"
      />
    );
    expect(screen.getByText('src/file.txt')).toBeDefined();
    expect(screen.getByText('+new line')).toBeDefined();
    expect(screen.getByText('-old line')).toBeDefined();
  });
});

describe('Header AI Provider Status Component', () => {
  const dummyConfig = {
    app_mode: 'local' as const,
    ai_provider: 'ollama' as const,
    ollama_base_url: 'http://localhost:11434',
    ollama_model: 'gemma4:e4b',
    gemma_api_model: 'gemma-2.5-9b-it',
    has_gemma_api_key: false,
    allowed_root: '/tmp',
    sentry_configured: false,
    sentry_environment: 'development',
    sentry_sample_rate: 1.0,
    max_repo_files: 500,
    max_repo_bytes: 10000000,
    max_scan_seconds: 60,
    sample_repo_available: true,
  };

  it('renders Connected when Ollama is reachable and model is downloaded', async () => {
    const { Header } = await import('./components/Header');
    render(
      <Header
        config={dummyConfig}
        providerStatus={{
          ai: {
            available: true,
            status: 'connected',
            model: 'gemma4:e4b',
            message: 'Ollama is connected',
            active_provider: 'ollama',
            app_mode: 'local',
          },
          observability: {
            status: 'not_configured',
            message: 'Not configured',
            environment: 'development',
            sample_rate: 1.0,
            configured: false,
          },
        }}
        activeRepoName="Sample Project"
        onOpenNewScan={() => {}}
        onRunDemoScan={() => {}}
        isScanning={false}
      />
    );
    expect(screen.getByText(/Connected \(gemma4:e4b\)/i)).toBeDefined();
  });

  it('renders Model Missing when Ollama is running but model is missing', async () => {
    const { Header } = await import('./components/Header');
    render(
      <Header
        config={dummyConfig}
        providerStatus={{
          ai: {
            available: false,
            status: 'model_missing',
            model: 'gemma4:e4b',
            message: 'Model missing',
            active_provider: 'ollama',
            app_mode: 'local',
          },
          observability: {
            status: 'not_configured',
            message: 'Not configured',
            environment: 'development',
            sample_rate: 1.0,
            configured: false,
          },
        }}
        activeRepoName="Sample Project"
        onOpenNewScan={() => {}}
        onRunDemoScan={() => {}}
        isScanning={false}
      />
    );
    expect(screen.getByText(/Model Missing \(gemma4:e4b\)/i)).toBeDefined();
  });

  it('renders Offline when Ollama daemon is unreachable', async () => {
    const { Header } = await import('./components/Header');
    render(
      <Header
        config={dummyConfig}
        providerStatus={{
          ai: {
            available: false,
            status: 'offline',
            model: 'gemma4:e4b',
            message: 'Daemon unreachable',
            active_provider: 'ollama',
            app_mode: 'local',
          },
          observability: {
            status: 'not_configured',
            message: 'Not configured',
            environment: 'development',
            sample_rate: 1.0,
            configured: false,
          },
        }}
        activeRepoName="Sample Project"
        onOpenNewScan={() => {}}
        onRunDemoScan={() => {}}
        isScanning={false}
      />
    );
    expect(screen.getByText(/Offline \(gemma4:e4b\)/i)).toBeDefined();
  });
});

describe('API Error Handling and Display', () => {
  it('extracts structured JSON error message on upstream 503/500 error', async () => {
    const { api } = await import('./api/client');
    const originalFetch = globalThis.fetch;
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 503,
      json: async () => ({
        error: 'Hosted Gemma model is temporarily unavailable (upstream HTTP 500). Please retry in a few moments.',
        error_type: 'upstream_transient_error',
      }),
      text: async () => '',
    }) as any;

    try {
      await expect(api.startScan('sample', 'sample-repo')).rejects.toThrow(
        /Hosted Gemma model is temporarily unavailable/
      );
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it('falls back to non-HTML text when response is not JSON', async () => {
    const { api } = await import('./api/client');
    const originalFetch = globalThis.fetch;
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 502,
      json: async () => {
        throw new Error('Unexpected token');
      },
      text: async () => 'Bad Gateway: Upstream connection failed',
    }) as any;

    try {
      await expect(api.startScan('sample', 'sample-repo')).rejects.toThrow(
        /Bad Gateway: Upstream connection failed/
      );
    } finally {
      globalThis.fetch = originalFetch;
    }
  });
});

