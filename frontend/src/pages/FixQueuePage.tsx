import React, { useState } from 'react';
import {
  AlertTriangle,
  Check,
  CheckCircle2,
  FileCode,
  Flame,
  GitPullRequest,
  History,
  Info,
  Loader2,
  Play,
  RotateCcw,
  ShieldAlert,
  ThumbsDown,
  ThumbsUp,
  X,
  Zap,
} from 'lucide-react';
import { MaintenanceRun, PatchStatus, ProposedPatch, PublicConfig } from '../types';
import { PatchStatusBadge } from '../components/Badge';
import { DiffViewer } from '../components/DiffViewer';

interface FixQueuePageProps {
  currentRun: MaintenanceRun | null;
  config: PublicConfig | null;
  onApprovePatch: (patchId: string) => Promise<void>;
  onRejectPatch: (patchId: string) => Promise<void>;
  onApplyPatch: (patchId: string) => Promise<void>;
  onUndoPatch: (patchId: string) => Promise<void>;
  onVerifyPatch: (patchId: string) => Promise<{ status: string; passed: boolean; details: string }>;
  onRunTestCommand: (command: string) => Promise<any>;
}

export const FixQueuePage: React.FC<FixQueuePageProps> = ({
  currentRun,
  config,
  onApprovePatch,
  onRejectPatch,
  onApplyPatch,
  onUndoPatch,
  onVerifyPatch,
  onRunTestCommand,
}) => {
  const patches = currentRun?.patches || [];
  const [selectedPatchId, setSelectedPatchId] = useState<string>(patches[0]?.id || '');
  const [actionLoading, setActionLoading] = useState<string | null>(null);
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);
  const [testResult, setTestResult] = useState<any | null>(null);

  const selectedPatch = patches.find((p) => p.id === selectedPatchId) || patches[0];
  const isHosted = config?.app_mode === 'hosted';

  const handleAction = async (action: 'approve' | 'reject' | 'apply' | 'undo' | 'verify') => {
    if (!selectedPatch) return;
    setActionLoading(action);
    setMessage(null);
    setTestResult(null);

    try {
      if (action === 'approve') {
        await onApprovePatch(selectedPatch.id);
        setMessage({ type: 'success', text: `Patch approved. You may now apply it to disk.` });
      } else if (action === 'reject') {
        await onRejectPatch(selectedPatch.id);
        setMessage({ type: 'success', text: `Patch rejected.` });
      } else if (action === 'apply') {
        await onApplyPatch(selectedPatch.id);
        setMessage({
          type: 'success',
          text: `Patch applied successfully! Original contents backed up safely.`,
        });
      } else if (action === 'undo') {
        await onUndoPatch(selectedPatch.id);
        setMessage({
          type: 'success',
          text: `Patch reverted! Original file restored from backup.`,
        });
      } else if (action === 'verify') {
        const res = await onVerifyPatch(selectedPatch.id);
        setMessage({
          type: res.passed ? 'success' : 'error',
          text: res.details,
        });
      }
    } catch (e: any) {
      setMessage({ type: 'error', text: e.message });
    } finally {
      setActionLoading(null);
    }
  };

  const handleRunTest = async (cmd: string) => {
    setActionLoading('test');
    setMessage(null);
    try {
      const res = await onRunTestCommand(cmd);
      setTestResult(res);
    } catch (e: any) {
      setMessage({ type: 'error', text: e.message });
    } finally {
      setActionLoading(null);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-xl font-bold text-slate-100">Fix Queue &amp; Reviewable Patches</h2>
        <p className="text-sm text-slate-400 mt-1">
          Review, approve, apply, and safely undo AI-proposed maintenance fixes before anything touches disk.
        </p>
      </div>

      {patches.length === 0 ? (
        <div className="p-12 text-center border border-dashed border-[#232b3b] rounded-xl text-slate-500 space-y-2">
          <GitPullRequest className="w-8 h-8 text-slate-600 mx-auto" />
          <p className="text-sm font-medium">No proposed patches in the queue.</p>
          <p className="text-xs text-slate-400">
            Start a maintenance run or click &quot;Propose Fix&quot; on any finding to generate a reviewable patch.
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left Column: Patch List (4 Cols) */}
          <div className="lg:col-span-4 space-y-2.5">
            <div className="text-xs font-semibold uppercase tracking-wider text-slate-400 px-1">
              Proposed Patches ({patches.length})
            </div>
            <div className="space-y-2">
              {patches.map((p) => {
                const isSelected = p.id === (selectedPatch?.id || '');
                return (
                  <button
                    key={p.id}
                    onClick={() => {
                      setSelectedPatchId(p.id);
                      setMessage(null);
                      setTestResult(null);
                    }}
                    className={`w-full text-left p-3.5 rounded-xl border transition flex flex-col space-y-1.5 ${
                      isSelected
                        ? 'bg-[#151c28] border-emerald-500/80 shadow-md ring-1 ring-emerald-500/20'
                        : 'bg-[#121622] border-[#21293a] hover:bg-[#161c2b] hover:border-[#2f3b52]'
                    }`}
                  >
                    <div className="flex items-center justify-between w-full">
                      <span className="font-mono text-xs font-semibold text-slate-200 truncate max-w-[190px]">
                        {p.file_path}
                      </span>
                      <PatchStatusBadge status={p.status} />
                    </div>
                    <p className="text-xs text-slate-400 line-clamp-2 leading-relaxed">
                      {p.change_summary}
                    </p>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Right Column: Diff and Review Controls (8 Cols) */}
          {selectedPatch && (
            <div className="lg:col-span-8 space-y-5">
              {/* Notification Banner */}
              {message && (
                <div
                  className={`p-3.5 rounded-lg border text-xs flex items-center space-x-2.5 ${
                    message.type === 'success'
                      ? 'bg-emerald-950/60 border-emerald-800 text-emerald-300'
                      : 'bg-red-950/60 border-red-800 text-red-300'
                  }`}
                >
                  {message.type === 'success' ? (
                    <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
                  ) : (
                    <AlertTriangle className="w-4 h-4 text-red-400 shrink-0" />
                  )}
                  <span>{message.text}</span>
                </div>
              )}

              {/* Patch Summary Header */}
              <div className="p-5 rounded-xl bg-[#121622] border border-[#21293a] space-y-3">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div className="space-y-1">
                    <div className="flex items-center space-x-2">
                      <span className="font-mono text-xs font-bold text-emerald-400">
                        {selectedPatch.file_path}
                      </span>
                      <PatchStatusBadge status={selectedPatch.status} />
                      {selectedPatch.is_demo ? (
                        <span className="px-2 py-0.5 rounded text-[11px] font-mono bg-amber-950/70 text-amber-300 border border-amber-800/60 font-semibold">
                          Demo Fixture
                        </span>
                      ) : selectedPatch.ai_model ? (
                        <span className="px-2 py-0.5 rounded text-[11px] font-mono bg-purple-950/70 text-purple-300 border border-purple-800/60 font-semibold">
                          Gemma ({selectedPatch.ai_model})
                        </span>
                      ) : null}
                    </div>
                    <h3 className="text-base font-bold text-slate-100">
                      {selectedPatch.change_summary}
                    </h3>
                  </div>

                  {/* Patch Actions Bar */}
                  <div className="flex flex-wrap items-center gap-2">
                    {/* Approve / Reject */}
                    {selectedPatch.status === 'pending' && (
                      <>
                        <button
                          onClick={() => handleAction('approve')}
                          disabled={!!actionLoading}
                          className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold shadow transition disabled:opacity-50"
                        >
                          {actionLoading === 'approve' ? (
                            <Loader2 className="w-3.5 h-3.5 animate-spin" />
                          ) : (
                            <ThumbsUp className="w-3.5 h-3.5" />
                          )}
                          <span>Approve</span>
                        </button>
                        <button
                          onClick={() => handleAction('reject')}
                          disabled={!!actionLoading}
                          className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-medium transition disabled:opacity-50"
                        >
                          {actionLoading === 'reject' ? (
                            <Loader2 className="w-3.5 h-3.5 animate-spin" />
                          ) : (
                            <ThumbsDown className="w-3.5 h-3.5" />
                          )}
                          <span>Reject</span>
                        </button>
                      </>
                    )}

                    {/* Apply Button */}
                    {selectedPatch.status === 'approved' && (
                      <button
                        onClick={() => handleAction('apply')}
                        disabled={!!actionLoading || isHosted}
                        className={`flex items-center space-x-1.5 px-3.5 py-1.5 rounded-lg text-xs font-semibold shadow transition ${
                          isHosted
                            ? 'bg-slate-800 text-slate-500 cursor-not-allowed'
                            : 'bg-emerald-600 hover:bg-emerald-500 text-white'
                        }`}
                        title={
                          isHosted
                            ? 'Patch application is disabled on Render hosted demo for security.'
                            : 'Apply patch to local disk with automatic backup'
                        }
                      >
                        {actionLoading === 'apply' ? (
                          <Loader2 className="w-3.5 h-3.5 animate-spin" />
                        ) : (
                          <Check className="w-3.5 h-3.5" />
                        )}
                        <span>{isHosted ? 'Apply Disabled in Demo' : 'Apply Patch to Disk'}</span>
                      </button>
                    )}

                    {/* Verify Button */}
                    {selectedPatch.status === 'applied' && (
                      <button
                        onClick={() => handleAction('verify')}
                        disabled={!!actionLoading}
                        className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold shadow transition disabled:opacity-50"
                      >
                        {actionLoading === 'verify' ? (
                          <Loader2 className="w-3.5 h-3.5 animate-spin" />
                        ) : (
                          <CheckCircle2 className="w-3.5 h-3.5" />
                        )}
                        <span>Verify Resolution</span>
                      </button>
                    )}

                    {/* Undo Button */}
                    {selectedPatch.status === 'applied' && (
                      <button
                        onClick={() => handleAction('undo')}
                        disabled={!!actionLoading || isHosted}
                        className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-[#222a3a] hover:bg-red-950/80 hover:text-red-300 border border-[#313c52] text-slate-300 text-xs font-medium transition disabled:opacity-50"
                      >
                        {actionLoading === 'undo' ? (
                          <Loader2 className="w-3.5 h-3.5 animate-spin" />
                        ) : (
                          <RotateCcw className="w-3.5 h-3.5" />
                        )}
                        <span>Undo Patch</span>
                      </button>
                    )}
                  </div>
                </div>

                {/* Explanation and Risk Notes */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3 pt-2 text-xs">
                  <div className="p-3 rounded-lg bg-[#151a26] border border-[#21293a] space-y-1">
                    <span className="font-semibold text-slate-300">Why this change:</span>
                    <p className="text-slate-400 leading-relaxed">{selectedPatch.explanation}</p>
                  </div>
                  <div className="p-3 rounded-lg bg-[#151a26] border border-[#21293a] space-y-1">
                    <span className="font-semibold text-slate-300">Risks &amp; Assumptions:</span>
                    <p className="text-slate-400 leading-relaxed">
                      {selectedPatch.risks_assumptions}
                    </p>
                  </div>
                </div>
              </div>

              {/* Interactive Diff Viewer */}
              <div className="space-y-2">
                <div className="text-xs font-semibold uppercase tracking-wider text-slate-400 px-1">
                  Unified Code Diff
                </div>
                <DiffViewer
                  diff={selectedPatch.diff}
                  originalContent={selectedPatch.original_content}
                  proposedContent={selectedPatch.proposed_content}
                  filePath={selectedPatch.file_path}
                />
              </div>

              {/* Allowlisted Test Execution Box */}
              <div className="p-4 rounded-xl bg-[#121622] border border-[#21293a] space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <Play className="w-4 h-4 text-emerald-400" />
                    <h4 className="text-xs font-semibold text-slate-200">
                      Approved Verification Commands
                    </h4>
                  </div>
                  <span className="text-[11px] text-slate-500 font-mono">
                    Explicit Allowlist • No Arbitrary Shell
                  </span>
                </div>

                <div className="flex flex-wrap gap-2 text-xs font-mono">
                  <button
                    onClick={() => handleRunTest('pytest')}
                    disabled={!!actionLoading || isHosted}
                    className="px-3 py-1.5 rounded-lg bg-[#181e2b] hover:bg-[#222a3d] border border-[#273247] text-slate-200 transition disabled:opacity-50"
                  >
                    Run pytest -v
                  </button>
                  <button
                    onClick={() => handleRunTest('npm-test')}
                    disabled={!!actionLoading || isHosted}
                    className="px-3 py-1.5 rounded-lg bg-[#181e2b] hover:bg-[#222a3d] border border-[#273247] text-slate-200 transition disabled:opacity-50"
                  >
                    Run npm test
                  </button>
                  <button
                    onClick={() => handleRunTest('python-check')}
                    disabled={!!actionLoading || isHosted}
                    className="px-3 py-1.5 rounded-lg bg-[#181e2b] hover:bg-[#222a3d] border border-[#273247] text-slate-200 transition disabled:opacity-50"
                  >
                    Run python py_compile
                  </button>
                </div>

                {testResult && (
                  <div className="p-3 rounded-lg bg-[#0b0e14] border border-[#1d2535] space-y-1.5 font-mono text-xs">
                    <div className="flex items-center justify-between">
                      <span className="text-slate-400">Command: {testResult.command}</span>
                      <span
                        className={`text-[10px] font-bold uppercase px-1.5 py-0.5 rounded ${
                          testResult.success
                            ? 'bg-emerald-950 text-emerald-400'
                            : 'bg-red-950 text-red-400'
                        }`}
                      >
                        Exit Code {testResult.exit_code}
                      </span>
                    </div>
                    {testResult.stdout && (
                      <pre className="text-slate-300 text-[11px] overflow-x-auto whitespace-pre-wrap">
                        {testResult.stdout}
                      </pre>
                    )}
                    {testResult.stderr && (
                      <pre className="text-red-400 text-[11px] overflow-x-auto whitespace-pre-wrap">
                        {testResult.stderr}
                      </pre>
                    )}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
