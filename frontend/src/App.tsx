import React, { useEffect, useState } from 'react';
import { api } from './api/client';
import { Header } from './components/Header';
import { Modal } from './components/Modal';
import { Sidebar, TabKey } from './components/Sidebar';
import { FindingsPage } from './pages/FindingsPage';
import { FixQueuePage } from './pages/FixQueuePage';
import { MaintenanceRunPage } from './pages/MaintenanceRunPage';
import { OverviewPage } from './pages/OverviewPage';
import { ReportsPage } from './pages/ReportsPage';
import { RepositoriesPage } from './pages/RepositoriesPage';
import { SettingsPage } from './pages/SettingsPage';
import { MaintenanceRun, ProviderStatus, PublicConfig } from './types';
import { Loader2, Play, Sparkles } from 'lucide-react';

export function App() {
  const [activeTab, setActiveTab] = useState<TabKey>('overview');
  const [config, setConfig] = useState<PublicConfig | null>(null);
  const [providerStatus, setProviderStatus] = useState<ProviderStatus | null>(null);
  const [runs, setRuns] = useState<MaintenanceRun[]>([]);
  const [currentRun, setCurrentRun] = useState<MaintenanceRun | null>(null);
  const [isScanning, setIsScanning] = useState(false);
  const [isProposingFix, setIsProposingFix] = useState(false);
  const [isRetryingAi, setIsRetryingAi] = useState(false);
  const [isNewScanModalOpen, setIsNewScanModalOpen] = useState(false);

  // New Scan modal fields
  const [scanType, setScanType] = useState<'sample' | 'local' | 'github'>('sample');
  const [scanPath, setScanPath] = useState('');
  const [scanProvider, setScanProvider] = useState<string>('ollama');
  const [scanError, setScanError] = useState<string | null>(null);

  // Fetch initial system configuration and runs
  const loadSystemData = async () => {
    try {
      const [cfg, status, allRuns] = await Promise.all([
        api.getConfig(),
        api.getProviderStatus(),
        api.getScans(),
      ]);
      setConfig(cfg);
      setProviderStatus(status);
      setScanProvider(cfg.ai_provider);
      setRuns(allRuns);

      if (allRuns.length > 0 && !currentRun) {
        setCurrentRun(allRuns[0]);
      }
    } catch (err) {
      console.error('Failed to load initial application state:', err);
    }
  };

  useEffect(() => {
    loadSystemData();
  }, []);

  // Poll active run status until terminal status ('completed' or 'failed')
  useEffect(() => {
    if (!currentRun) return;
    const terminalStatuses = ['completed', 'failed', 'cancelled'];
    if (terminalStatuses.includes(currentRun.status)) {
      setIsScanning(false);
      setIsRetryingAi(false);
      return;
    }

    setIsScanning(true);

    const intervalId = setInterval(async () => {
      try {
        const refreshed = await api.getScan(currentRun.id);
        setCurrentRun((prev) => (prev && prev.id === refreshed.id ? refreshed : prev));
        setRuns((prev) => prev.map((r) => (r.id === refreshed.id ? refreshed : r)));

        if (terminalStatuses.includes(refreshed.status)) {
          setIsScanning(false);
          setIsRetryingAi(false);
          clearInterval(intervalId);
        }
      } catch (pollErr) {
        console.error('Error polling run status:', pollErr);
      }
    }, 1500);

    return () => clearInterval(intervalId);
  }, [currentRun?.id, currentRun?.status]);

  // 1-Click Fast Demo Scan
  const handleStartDemoScan = async () => {
    setIsScanning(true);
    try {
      const run = await api.startDemoScan();
      setCurrentRun(run);
      setRuns((prev) => [run, ...prev.filter((r) => r.id !== run.id)]);
      setActiveTab('runs');
    } catch (e: any) {
      alert(`Demo scan failed: ${e.message}`);
    } finally {
      setIsScanning(false);
    }
  };

  // Launch scan from modal or Repositories page
  const handleLaunchScan = async (
    type: 'local' | 'sample' | 'github',
    path: string,
    provider?: string
  ) => {
    setIsScanning(true);
    setScanError(null);
    try {
      const run = await api.startScan(type, path, provider || config?.ai_provider);
      setCurrentRun(run);
      setRuns((prev) => [run, ...prev.filter((r) => r.id !== run.id)]);
      setIsNewScanModalOpen(false);
      setActiveTab('runs');
      if (['completed', 'failed', 'cancelled'].includes(run.status)) {
        setIsScanning(false);
      }
    } catch (e: any) {
      setScanError(e.message);
      setIsScanning(false);
    }
  };

  // Propose fix for a specific finding
  const handleProposeFix = async (findingId: string) => {
    setIsProposingFix(true);
    try {
      const patch = await api.proposeFix(findingId);
      if (currentRun) {
        const updatedPatches = [...currentRun.patches];
        const existingIdx = updatedPatches.findIndex((p) => p.id === patch.id);
        if (existingIdx >= 0) {
          updatedPatches[existingIdx] = patch;
        } else {
          updatedPatches.unshift(patch);
        }
        const updatedRun = { ...currentRun, patches: updatedPatches };
        setCurrentRun(updatedRun);
        setRuns((prev) => prev.map((r) => (r.id === updatedRun.id ? updatedRun : r)));
      }
    } catch (e: any) {
      alert(`Failed to propose fix: ${e.message}`);
    } finally {
      setIsProposingFix(false);
    }
  };

  const handleRetryAi = async () => {
    if (!currentRun) return;
    setIsRetryingAi(true);
    try {
      const updated = await api.retryAiAnalysis(currentRun.id);
      setCurrentRun(updated);
      setRuns((prev) => prev.map((r) => (r.id === updated.id ? updated : r)));
      if (['completed', 'failed', 'cancelled'].includes(updated.status)) {
        setIsRetryingAi(false);
      }
    } catch (e: any) {
      alert(`Failed to retry AI analysis: ${e.message}`);
      setIsRetryingAi(false);
    }
  };

  // Patch actions
  const handleApprovePatch = async (patchId: string) => {
    const updated = await api.approvePatch(patchId);
    updatePatchInState(updated);
  };

  const handleRejectPatch = async (patchId: string) => {
    const updated = await api.rejectPatch(patchId);
    updatePatchInState(updated);
  };

  const handleApplyPatch = async (patchId: string) => {
    const updated = await api.applyPatch(patchId);
    updatePatchInState(updated);
  };

  const handleUndoPatch = async (patchId: string) => {
    const updated = await api.undoPatch(patchId);
    updatePatchInState(updated);
  };

  const handleVerifyPatch = async (patchId: string) => {
    const res = await api.verifyPatch(patchId);
    // Reload active run to update verification status on finding
    if (currentRun) {
      const refreshed = await api.getScan(currentRun.id);
      setCurrentRun(refreshed);
      setRuns((prev) => prev.map((r) => (r.id === refreshed.id ? refreshed : r)));
    }
    return res;
  };

  const handleRunTestCommand = async (command: string) => {
    if (!currentRun) throw new Error('No active run');
    return api.runTestCommand(currentRun.id, command);
  };

  const updatePatchInState = (updatedPatch: any) => {
    if (!currentRun) return;
    const updatedPatches = currentRun.patches.map((p) =>
      p.id === updatedPatch.id ? updatedPatch : p
    );
    const updatedRun = { ...currentRun, patches: updatedPatches };
    setCurrentRun(updatedRun);
    setRuns((prev) => prev.map((r) => (r.id === updatedRun.id ? updatedRun : r)));
  };

  const pendingPatchesCount =
    currentRun?.patches.filter((p) => p.status === 'pending').length || 0;
  const totalFindingsCount = currentRun?.findings.length || 0;

  return (
    <div className="min-h-screen bg-[#0b0d11] text-slate-100 flex flex-col font-sans">
      {/* Top Application Header */}
      <Header
        config={config}
        providerStatus={providerStatus}
        activeRepoName={currentRun?.repository_name || 'No Repository Selected'}
        onOpenNewScan={() => setIsNewScanModalOpen(true)}
        onRunDemoScan={handleStartDemoScan}
        isScanning={isScanning}
      />

      {/* Main Layout Body */}
      <div className="flex-1 flex overflow-hidden">
        {/* Navigation Sidebar */}
        <Sidebar
          activeTab={activeTab}
          onSelectTab={setActiveTab}
          pendingPatchesCount={pendingPatchesCount}
          totalFindingsCount={totalFindingsCount}
        />

        {/* Dynamic Page Content */}
        <main className="flex-1 overflow-y-auto p-8">
          {activeTab === 'overview' && (
            <OverviewPage
              currentRun={currentRun}
              config={config}
              providerStatus={providerStatus}
              onNavigate={setActiveTab}
              onStartDemoScan={handleStartDemoScan}
              onOpenNewScan={() => setIsNewScanModalOpen(true)}
              isScanning={isScanning}
            />
          )}

          {activeTab === 'repositories' && (
            <RepositoriesPage
              config={config}
              onSelectRepoAndScan={handleLaunchScan}
              isScanning={isScanning}
            />
          )}

          {activeTab === 'runs' && (
            <MaintenanceRunPage
              currentRun={currentRun}
              runs={runs}
              onSelectRun={(id) => {
                const found = runs.find((r) => r.id === id);
                if (found) setCurrentRun(found);
              }}
              onNavigate={setActiveTab}
              isScanning={isScanning}
              onRetryAi={handleRetryAi}
              isRetryingAi={isRetryingAi}
            />
          )}

          {activeTab === 'findings' && (
            <FindingsPage
              currentRun={currentRun}
              onProposeFix={handleProposeFix}
              onNavigateToQueue={() => setActiveTab('queue')}
              isProposingFix={isProposingFix}
              onRetryAi={handleRetryAi}
              isRetryingAi={isRetryingAi}
            />
          )}

          {activeTab === 'queue' && (
            <FixQueuePage
              currentRun={currentRun}
              config={config}
              onApprovePatch={handleApprovePatch}
              onRejectPatch={handleRejectPatch}
              onApplyPatch={handleApplyPatch}
              onUndoPatch={handleUndoPatch}
              onVerifyPatch={handleVerifyPatch}
              onRunTestCommand={handleRunTestCommand}
            />
          )}

          {activeTab === 'reports' && <ReportsPage currentRun={currentRun} />}

          {activeTab === 'settings' && (
            <SettingsPage
              config={config}
              providerStatus={providerStatus}
              onRefreshStatus={loadSystemData}
            />
          )}
        </main>
      </div>

      {/* New Maintenance Scan Modal */}
      <Modal
        isOpen={isNewScanModalOpen}
        onClose={() => setIsNewScanModalOpen(false)}
        title="Start New Maintenance Run"
      >
        <div className="space-y-4 text-xs">
          <div>
            <label className="block text-slate-300 font-semibold mb-1.5">Repository Source</label>
            <div className="grid grid-cols-3 gap-2">
              <button
                type="button"
                onClick={() => {
                  setScanType('sample');
                  setScanPath('sample-repo');
                }}
                className={`p-2.5 rounded-lg border text-center font-medium transition ${
                  scanType === 'sample'
                    ? 'bg-emerald-950/70 border-emerald-600 text-emerald-300'
                    : 'bg-[#151a26] border-[#222b3b] text-slate-400 hover:text-slate-200'
                }`}
              >
                Sample Project
              </button>
              <button
                type="button"
                onClick={() => {
                  setScanType('local');
                  setScanPath('.');
                }}
                className={`p-2.5 rounded-lg border text-center font-medium transition ${
                  scanType === 'local'
                    ? 'bg-blue-950/70 border-blue-600 text-blue-300'
                    : 'bg-[#151a26] border-[#222b3b] text-slate-400 hover:text-slate-200'
                }`}
              >
                Local Path
              </button>
              <button
                type="button"
                onClick={() => {
                  setScanType('github');
                  setScanPath('');
                }}
                className={`p-2.5 rounded-lg border text-center font-medium transition ${
                  scanType === 'github'
                    ? 'bg-purple-950/70 border-purple-600 text-purple-300'
                    : 'bg-[#151a26] border-[#222b3b] text-slate-400 hover:text-slate-200'
                }`}
              >
                GitHub URL
              </button>
            </div>
          </div>

          <div>
            <label className="block text-slate-300 font-semibold mb-1">
              {scanType === 'github'
                ? 'Public GitHub URL'
                : scanType === 'local'
                ? 'Local Relative Path'
                : 'Bundled Sample Project'}
            </label>
            <input
              type="text"
              value={scanPath}
              onChange={(e) => setScanPath(e.target.value)}
              placeholder={
                scanType === 'github'
                  ? 'https://github.com/owner/repository'
                  : scanType === 'local'
                  ? 'e.g. . or services/api'
                  : 'sample-repo'
              }
              disabled={scanType === 'sample'}
              className="w-full px-3 py-2 rounded-lg bg-[#161b27] border border-[#263145] text-slate-100 placeholder-slate-500 focus:outline-none focus:border-emerald-500 font-mono disabled:opacity-60"
            />
          </div>

          <div>
            <label className="block text-slate-300 font-semibold mb-1">AI Caretaker Provider</label>
            <select
              value={scanProvider}
              onChange={(e) => setScanProvider(e.target.value)}
              className="w-full px-3 py-2 rounded-lg bg-[#161b27] border border-[#263145] text-slate-200 focus:outline-none focus:border-emerald-500 font-medium"
            >
              <option value="ollama">Local Gemma via Ollama (Privacy-first)</option>
              <option value="demo">Demonstration Mode (Curated demo fixtures)</option>
              <option value="gemma_api">Hosted Gemma API (Google Gen AI SDK)</option>
            </select>
          </div>

          {scanError && (
            <div className="p-3 rounded-lg bg-red-950/50 border border-red-800 text-red-300 text-xs">
              {scanError}
            </div>
          )}

          <div className="pt-3 flex justify-end space-x-2">
            <button
              type="button"
              onClick={() => setIsNewScanModalOpen(false)}
              className="px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 font-medium transition"
            >
              Cancel
            </button>
            <button
              type="button"
              onClick={() => handleLaunchScan(scanType, scanPath || 'sample-repo', scanProvider)}
              disabled={isScanning}
              className="flex items-center space-x-1.5 px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white font-semibold shadow transition disabled:opacity-50"
            >
              {isScanning ? (
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
              ) : (
                <Play className="w-3.5 h-3.5 fill-current" />
              )}
              <span>Launch Caretaker Run</span>
            </button>
          </div>
        </div>
      </Modal>
    </div>
  );
}

export default App;
