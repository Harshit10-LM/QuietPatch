import React from 'react';
import {
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  Clock,
  Cpu,
  FileSearch,
  Filter,
  Flame,
  ListTodo,
  Loader2,
  Sparkles,
  Zap,
} from 'lucide-react';
import { MaintenanceRun, StageRecord } from '../types';

interface MaintenanceRunPageProps {
  currentRun: MaintenanceRun | null;
  runs: MaintenanceRun[];
  onSelectRun: (runId: string) => void;
  onNavigate: (tab: any) => void;
  isScanning: boolean;
  onRetryAi?: () => Promise<void>;
  isRetryingAi?: boolean;
}

export const MaintenanceRunPage: React.FC<MaintenanceRunPageProps> = ({
  currentRun,
  runs,
  onSelectRun,
  onNavigate,
  isScanning,
  onRetryAi,
  isRetryingAi,
}) => {
  const workflowStages = [
    { key: 'initialization', label: '1. Initialize', desc: 'Validate environment & security limits' },
    { key: 'discovery', label: '2. Discover', desc: 'Walk repository files & detect manifests' },
    { key: 'scan', label: '3. Scan', desc: 'Run deterministic scanners (TODO, docs, config, etc.)' },
    { key: 'analysis', label: '4. Analyze', desc: 'Gemma model reviews and grounds findings' },
    { key: 'prioritization', label: '5. Prioritize', desc: 'Organize into Critical to Informational' },
    { key: 'proposal', label: '6. Propose', desc: 'Generate targeted reviewable unified diffs' },
    { key: 'verification', label: '7. Verify', desc: 'Deterministic re-checks on approved fixes' },
    { key: 'report', label: '8. Report', desc: 'Compile time saved & completed maintenance summary' },
  ];

  const getStageStatus = (key: string): 'completed' | 'partial' | 'running' | 'pending' | 'failed' | 'skipped' => {
    if (!currentRun) return 'pending';
    const stage = currentRun.stages[key];
    if (stage) {
      if (stage.status === 'completed') return 'completed';
      if (stage.status === 'partial') return 'partial';
      if (stage.status === 'failed') return 'failed';
      if (stage.status === 'skipped') return 'skipped';
      if (stage.status === 'running') return 'running';
      if (stage.status === 'pending') return 'pending';
    }
    // In our sequential run: if run completed, earlier stages are completed
    if (currentRun.status === 'completed') {
      if (key === 'verification') {
        const hasVerified = currentRun.findings.some((f) => f.verification_status === 'passed');
        return hasVerified ? 'completed' : 'pending';
      }
      return 'completed';
    }
    return isScanning ? 'running' : 'pending';
  };

  const analysisStage = currentRun?.stages?.['analysis'];
  const hasAnalysisIssue =
    analysisStage &&
    (analysisStage.status === 'failed' || analysisStage.status === 'partial') ||
    (currentRun && currentRun.findings.some((f) => !f.ai_assisted && !f.is_demo));

  return (
    <div className="space-y-6">
      {/* Header and Run Switcher */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold text-slate-100">Maintenance Run Progression</h2>
          <p className="text-sm text-slate-400 mt-1">
            Real backend job workflow executing the QuietPatch agent stages with Sentry telemetry.
          </p>
        </div>

        {/* Run Selector Dropdown */}
        {runs.length > 0 && (
          <div className="flex items-center space-x-2">
            <span className="text-xs text-slate-400 font-medium">Active Run:</span>
            <select
              value={currentRun?.id || ''}
              onChange={(e) => onSelectRun(e.target.value)}
              className="px-3 py-1.5 rounded-lg bg-[#141924] border border-[#232b3b] text-xs text-slate-200 focus:outline-none focus:border-emerald-500 font-mono"
            >
              {runs.map((r) => (
                <option key={r.id} value={r.id}>
                  {r.repository_name} ({r.id}) — {r.status}
                </option>
              ))}
            </select>
          </div>
        )}
      </div>

      {/* Active Run Status Banner */}
      {currentRun && (
        <div className="p-5 rounded-xl bg-[#121622] border border-[#21293a] flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center space-x-3.5">
            <div
              className={`p-3 rounded-lg flex items-center justify-center ${
                currentRun.status === 'completed'
                  ? 'bg-emerald-950/70 border border-emerald-800 text-emerald-400'
                  : currentRun.status === 'failed'
                  ? 'bg-red-950/70 border border-red-800 text-red-400'
                  : 'bg-blue-950/70 border border-blue-800 text-blue-400'
              }`}
            >
              {currentRun.status === 'completed' ? (
                <CheckCircle2 className="w-5 h-5" />
              ) : currentRun.status === 'failed' ? (
                <AlertTriangle className="w-5 h-5" />
              ) : (
                <Loader2 className="w-5 h-5 animate-spin" />
              )}
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <span className="text-sm font-bold text-slate-100 font-mono">
                  {currentRun.repository_name}
                </span>
                <span
                  className={`text-[10px] font-bold uppercase px-2 py-0.5 rounded ${
                    currentRun.status === 'completed'
                      ? 'bg-emerald-950 text-emerald-400 border border-emerald-800'
                      : currentRun.status === 'failed'
                      ? 'bg-red-950 text-red-400 border border-red-800'
                      : 'bg-blue-950 text-blue-300 border border-blue-800'
                  }`}
                >
                  {currentRun.status}
                </span>
              </div>
              <p className="text-xs text-slate-400 mt-0.5">
                Run ID: <code className="text-slate-300">{currentRun.id}</code> • Model:{' '}
                <code className="text-slate-300">{currentRun.ai_model}</code>
              </p>
            </div>
          </div>

          <div className="flex items-center space-x-4 text-xs font-mono">
            <div>
              <span className="text-slate-500 block">Duration</span>
              <span className="text-slate-200 font-semibold">{currentRun.duration_seconds}s</span>
            </div>
            <div>
              <span className="text-slate-500 block">Files Scanned</span>
              <span className="text-slate-200 font-semibold">{currentRun.files_scanned}</span>
            </div>
            <div>
              <span className="text-slate-500 block">Findings</span>
              <span className="text-amber-400 font-semibold">{currentRun.findings.length}</span>
            </div>
            <div>
              <span className="text-slate-500 block">Patches</span>
              <span className="text-emerald-400 font-semibold">{currentRun.patches.length}</span>
            </div>
          </div>
        </div>
      )}

      {/* Actionable Timeout / Partial AI Banner */}
      {hasAnalysisIssue && (
        <div className="p-4 rounded-xl bg-amber-950/40 border border-amber-800/80 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs">
          <div className="flex items-start space-x-3 text-amber-300">
            <AlertTriangle className="w-5 h-5 shrink-0 mt-0.5 text-amber-400" />
            <div>
              <h4 className="font-semibold text-amber-200">
                {analysisStage?.status === 'failed'
                  ? (currentRun?.ai_provider === 'gemma_api'
                      ? 'Hosted Gemma AI Analysis Failed'
                      : 'Local Gemma AI Inference Timed Out or Failed')
                  : 'Partial Gemma AI Analysis Completed'}
              </h4>
              <p className="text-amber-300/80 mt-0.5 text-[11px]">
                {analysisStage?.details ||
                  (currentRun?.ai_provider === 'gemma_api'
                    ? 'Hosted Gemma response was empty or blocked. Deterministic findings were preserved.'
                    : 'Deterministic findings were preserved. AI analysis can be retried without re-scanning files.')}
              </p>
            </div>
          </div>
          {onRetryAi && (
            <button
              onClick={onRetryAi}
              disabled={isRetryingAi}
              className="flex items-center space-x-1.5 px-3.5 py-2 rounded-lg bg-amber-600 hover:bg-amber-500 text-white font-semibold transition disabled:opacity-50 shrink-0 self-start sm:self-auto"
            >
              {isRetryingAi ? (
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
              ) : (
                <Sparkles className="w-3.5 h-3.5" />
              )}
              <span>Retry AI Analysis</span>
            </button>
          )}
        </div>
      )}

      {/* 8-Stage Sequential Workflow Stepper */}
      <div className="p-6 rounded-xl bg-[#121622] border border-[#21293a] space-y-5">
        <h3 className="text-sm font-semibold text-slate-200">Agent Workflow Stages</h3>

        <div className="space-y-3">
          {workflowStages.map((stage, idx) => {
            const status = getStageStatus(stage.key);
            const recordedStage = currentRun?.stages[stage.key];
            const duration = recordedStage?.duration_seconds;

            return (
              <div
                key={stage.key}
                className={`p-3.5 rounded-lg border transition flex items-center justify-between ${
                  status === 'completed'
                    ? 'bg-[#151b27] border-emerald-900/40 text-slate-200'
                    : status === 'partial'
                    ? 'bg-amber-950/20 border-amber-900/40 text-amber-200'
                    : status === 'running'
                    ? 'bg-[#161d2d] border-blue-800/60 text-blue-200 animate-pulse'
                    : status === 'failed'
                    ? 'bg-red-950/20 border-red-900/40 text-red-300'
                    : 'bg-[#0f131a] border-[#1d2534] text-slate-500'
                }`}
              >
                <div className="flex items-center space-x-3.5">
                  <div
                    className={`w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold ${
                      status === 'completed'
                        ? 'bg-emerald-950 text-emerald-400 border border-emerald-700'
                        : status === 'partial'
                        ? 'bg-amber-950 text-amber-400 border border-amber-700'
                        : status === 'running'
                        ? 'bg-blue-950 text-blue-300 border border-blue-700'
                        : status === 'failed'
                        ? 'bg-red-950 text-red-400 border border-red-700'
                        : 'bg-slate-900 text-slate-600 border border-slate-800'
                    }`}
                  >
                    {status === 'completed' ? (
                      '✓'
                    ) : status === 'partial' ? (
                      '!'
                    ) : status === 'running' ? (
                      <Loader2 className="w-3.5 h-3.5 animate-spin" />
                    ) : (
                      idx + 1
                    )}
                  </div>
                  <div>
                    <h4 className="text-xs font-semibold">{stage.label}</h4>
                    <p className="text-[11px] text-slate-400">{stage.desc}</p>
                  </div>
                </div>

                <div className="flex items-center space-x-3 text-xs font-mono">
                  {recordedStage?.details && (
                    <span className="hidden md:inline text-slate-400 text-[11px]">
                      {recordedStage.details}
                    </span>
                  )}
                  {duration !== undefined && duration > 0 && (
                    <span className="px-2 py-0.5 rounded bg-[#1b2230] text-slate-300 text-[11px]">
                      {duration.toFixed(3)}s
                    </span>
                  )}
                  <span
                    className={`text-[10px] font-bold uppercase px-1.5 py-0.5 rounded ${
                      status === 'completed'
                        ? 'text-emerald-400 bg-emerald-950/80'
                        : status === 'partial'
                        ? 'text-amber-400 bg-amber-950/80'
                        : status === 'running'
                        ? 'text-blue-300 bg-blue-950/80'
                        : 'text-slate-500 bg-slate-900'
                    }`}
                  >
                    {status}
                  </span>
                </div>
              </div>
            );
          })}
        </div>

        {/* Action button to proceed */}
        {currentRun?.status === 'completed' && (
          <div className="pt-2 flex justify-end gap-3">
            <button
              onClick={() => onNavigate('findings')}
              className="flex items-center space-x-1.5 px-4 py-2 rounded-lg bg-[#1a2232] hover:bg-[#222d42] border border-[#2b374d] text-xs font-medium text-slate-200 transition"
            >
              <span>Explore {currentRun.findings.length} Findings</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={() => onNavigate('queue')}
              className="flex items-center space-x-1.5 px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold shadow transition"
            >
              <span>Review {currentRun.patches.length} Fixes in Fix Queue</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>
        )}
      </div>
    </div>
  );
};
