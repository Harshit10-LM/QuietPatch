import {
  MaintenanceReport,
  MaintenanceRun,
  ProposedPatch,
  ProviderStatus,
  PublicConfig,
} from '../types';

const BASE_URL = '';

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${url}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  });

  if (!res.ok) {
    let errorMsg = `Request failed with status ${res.status}`;
    try {
      const errData = await res.json();
      if (errData.error) errorMsg = errData.error;
    } catch {
      // ignore json parse error
    }
    throw new Error(errorMsg);
  }

  return res.json();
}

export const api = {
  getHealth: () => fetchJson<{ status: string; app: string; mode: string; provider: string }>('/api/health'),
  
  getConfig: () => fetchJson<PublicConfig>('/api/config'),
  
  getProviderStatus: (provider?: string) =>
    fetchJson<ProviderStatus>(`/api/providers/status${provider ? `?provider=${provider}` : ''}`),
  
  getRepositories: () =>
    fetchJson<{ repositories: Array<{ name: string; path: string; type: string; description: string; languages: string[] }>; allowed_root: string }>('/api/repositories'),
  
  validateRepository: (type: 'local' | 'github', path: string) =>
    fetchJson<{ valid: boolean; resolved_path?: string; owner?: string; repo?: string; error?: string; message?: string }>('/api/repositories/validate', {
      method: 'POST',
      body: JSON.stringify({ type, path }),
    }),
  
  startScan: (type: 'local' | 'sample' | 'github', path: string, provider?: string) =>
    fetchJson<MaintenanceRun>('/api/scans', {
      method: 'POST',
      body: JSON.stringify({ type, path, provider }),
    }),
  
  startDemoScan: () =>
    fetchJson<MaintenanceRun>('/api/demo/scan', {
      method: 'POST',
    }),
  
  getScans: () => fetchJson<MaintenanceRun[]>('/api/scans'),
  
  getScan: (runId: string) => fetchJson<MaintenanceRun>(`/api/scans/${runId}`),
  
  getFindings: (runId: string, category?: string, severity?: string) => {
    const params = new URLSearchParams();
    if (category) params.set('category', category);
    if (severity) params.set('severity', severity);
    return fetchJson<MaintenanceRun['findings']>(`/api/scans/${runId}/findings?${params.toString()}`);
  },
  
  getReport: (runId: string) => fetchJson<MaintenanceReport>(`/api/scans/${runId}/report`),
  
  exportReportMarkdown: async (runId: string): Promise<string> => {
    const res = await fetch(`/api/scans/${runId}/report/markdown`);
    if (!res.ok) throw new Error('Failed to export markdown report');
    return res.text();
  },
  
  proposeFix: (findingId: string) =>
    fetchJson<ProposedPatch>(`/api/findings/${findingId}/propose-fix`, {
      method: 'POST',
    }),
  
  approvePatch: (patchId: string) =>
    fetchJson<ProposedPatch>(`/api/patches/${patchId}/approve`, {
      method: 'POST',
    }),
  
  rejectPatch: (patchId: string) =>
    fetchJson<ProposedPatch>(`/api/patches/${patchId}/reject`, {
      method: 'POST',
    }),
  
  applyPatch: (patchId: string) =>
    fetchJson<ProposedPatch>(`/api/patches/${patchId}/apply`, {
      method: 'POST',
    }),
  
  undoPatch: (patchId: string) =>
    fetchJson<ProposedPatch>(`/api/patches/${patchId}/undo`, {
      method: 'POST',
    }),
  
  verifyPatch: (patchId: string) =>
    fetchJson<{ status: string; passed: boolean; details: string }>(`/api/patches/${patchId}/verify`, {
      method: 'POST',
    }),
  
  runTestCommand: (runId: string, command: string) =>
    fetchJson<{ success: boolean; command: string; exit_code: number; stdout: string; stderr: string; message: string }>('/api/verify/test-command', {
      method: 'POST',
      body: JSON.stringify({ run_id: runId, command }),
    }),

  retryAiAnalysis: (runId: string) =>
    fetchJson<MaintenanceRun>(`/api/scans/${runId}/retry-ai`, {
      method: 'POST',
    }),
};

