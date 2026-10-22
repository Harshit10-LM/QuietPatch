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

describe('MaintenanceRunPage and FindingsPage Alert State Mapping', () => {
  const mockRunningRun: any = {
    id: 'run-active-123',
    repository_path: '/repos/sample',
    repository_name: 'sample',
    status: 'analyzing',
    started_at: '2026-10-22T08:00:00Z',
    completed_at: null,
    total_files: 10,
    total_findings: 5,
    files_scanned: 10,
    stages: {
      initialization: { name: 'initialization', status: 'completed', duration_seconds: 0.1 },
      discovery: { name: 'discovery', status: 'completed', duration_seconds: 0.2 },
      scan: { name: 'scan', status: 'completed', duration_seconds: 0.5 },
      analysis: { name: 'analysis', status: 'running', duration_seconds: 5.0 },
      prioritization: { name: 'prioritization', status: 'pending', duration_seconds: 0 },
      proposal: { name: 'proposal', status: 'pending', duration_seconds: 0 },
      verification: { name: 'verification', status: 'pending', duration_seconds: 0 },
      report: { name: 'report', status: 'pending', duration_seconds: 0 },
    },
    findings: [
      {
        id: 'f-1',
        category: 'configuration',
        title: 'Finding 1',
        file_path: 'conf.py',
        line_number: 10,
        severity: 'high',
        evidence: 'key=""',
        explanation: 'Needs key',
        suggested_action: 'Add key',
        estimated_effort_mins: 5,
        verification_status: 'unverified',
        ai_assisted: false,
        is_demo: false,
      },
    ],
    patches: [],
    proposed_patches: [],
    ai_provider: 'gemma_api',
    model_name: 'gemma-4-26b-a4b-it',
  };

  const mockCompletedPartialRun: any = {
    ...mockRunningRun,
    id: 'run-partial-456',
    status: 'completed',
    completed_at: '2026-10-22T08:02:00Z',
    stages: {
      ...mockRunningRun.stages,
      analysis: { name: 'analysis', status: 'partial', duration_seconds: 15.0 },
      prioritization: { name: 'prioritization', status: 'completed', duration_seconds: 0.5 },
      proposal: { name: 'proposal', status: 'completed', duration_seconds: 2.0 },
      verification: { name: 'verification', status: 'completed', duration_seconds: 0.5 },
      report: { name: 'report', status: 'completed', duration_seconds: 0.1 },
    },
  };

  it('MaintenanceRunPage does NOT show partial/failed warning while actively analyzing', async () => {
    const { MaintenanceRunPage } = await import('./pages/MaintenanceRunPage');
    render(
      <MaintenanceRunPage
        currentRun={mockRunningRun}
        runs={[mockRunningRun]}
        onSelectRun={() => {}}
        onNavigate={() => {}}
        isScanning={true}
      />
    );

    // Banner should NOT appear while actively running
    expect(screen.queryByText(/Partial Gemma AI Analysis Completed/i)).toBeNull();
    expect(screen.queryByText(/Gemma AI Analysis Failed/i)).toBeNull();
    expect(screen.queryByText(/Findings Awaiting Gemma AI Evaluation/i)).toBeNull();
  });

  it('MaintenanceRunPage DOES show Partial Gemma AI Analysis Completed when run is completed with partial stage', async () => {
    const { MaintenanceRunPage } = await import('./pages/MaintenanceRunPage');
    render(
      <MaintenanceRunPage
        currentRun={mockCompletedPartialRun}
        runs={[mockCompletedPartialRun]}
        onSelectRun={() => {}}
        onNavigate={() => {}}
        isScanning={false}
      />
    );

    expect(screen.getByText(/Partial Gemma AI Analysis Completed/i)).toBeDefined();
  });

  it('FindingsPage does NOT show Retry AI banner while run is actively analyzing', async () => {
    const { FindingsPage } = await import('./pages/FindingsPage');
    render(
      <FindingsPage
        currentRun={mockRunningRun}
        onProposeFix={async () => {}}
        onNavigateToQueue={() => {}}
        isProposingFix={false}
      />
    );

    expect(screen.queryByText(/Some Findings Awaiting Gemma AI Evaluation/i)).toBeNull();
  });

  it('FindingsPage DOES show Retry AI banner when run is completed with unassisted findings', async () => {
    const { FindingsPage } = await import('./pages/FindingsPage');
    render(
      <FindingsPage
        currentRun={mockCompletedPartialRun}
        onProposeFix={async () => {}}
        onNavigateToQueue={() => {}}
        isProposingFix={false}
        onRetryAi={async () => {}}
      />
    );

    expect(screen.getByText(/Some Findings Awaiting Gemma AI Evaluation/i)).toBeDefined();
    expect(screen.getByRole('button', { name: /Retry AI Analysis/i })).toBeDefined();
  });
});


