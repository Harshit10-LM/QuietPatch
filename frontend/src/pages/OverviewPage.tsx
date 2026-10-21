import React from 'react';
import {
  AlertTriangle,
  ArrowRight,
  CheckCircle,
  Clock,
  Cpu,
  FileCode,
  Flame,
  Info,
  Play,
  Shield,
  Sparkles,
  Zap,
} from 'lucide-react';
import { MaintenanceRun, ProposedPatch, ProviderStatus, PublicConfig, Severity } from '../types';
import { CategoryBadge, SeverityBadge } from '../components/Badge';

interface OverviewPageProps {
  currentRun: MaintenanceRun | null;
  config: PublicConfig | null;
  providerStatus: ProviderStatus | null;
  onNavigate: (tab: any) => void;
  onStartDemoScan: () => void;
  onOpenNewScan: () => void;
  isScanning: boolean;
}

export const OverviewPage: React.FC<OverviewPageProps> = ({
  currentRun,
  config,
  providerStatus,
  onNavigate,
  onStartDemoScan,
  onOpenNewScan,
  isScanning,
}) => {
  const findings = currentRun?.findings || [];
  const patches = currentRun?.patches || [];

  const severityCounts: Record<Severity, number> = {
    critical: 0,
    high: 0,
    medium: 0,
    low: 0,
    informational: 0,
  };

  findings.forEach((f) => {
    if (f.severity in severityCounts) {
      severityCounts[f.severity]++;
    }
  });

  const pendingPatches = patches.filter((p) => p.status === 'pending');
  const appliedPatches = patches.filter((p) => p.status === 'applied');

  const totalTimeSavedMins = currentRun?.time_saved_minutes || 0;
  const totalTimeSavedHours = (totalTimeSavedMins / 60).toFixed(1);

  return (
    <div className="space-y-6">
      {/* Hero Section */}
      <div className="relative overflow-hidden rounded-2xl bg-gradient-to-r from-[#111723] via-[#141b29] to-[#101925] border border-[#222c3e] p-7 shadow-xl">
        <div className="relative z-10 max-w-3xl space-y-3">
          <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-emerald-950/70 border border-emerald-800/60 text-emerald-400 text-xs font-semibold">
            <Sparkles className="w-3.5 h-3.5" />
            <span>Slow and Steady AI Maintenance</span>
          </div>
          <h2 className="text-3xl font-extrabold text-white tracking-tight sm:text-4xl">
            Let maintenance wait on you.
          </h2>
          <p className="text-slate-300 text-sm sm:text-base leading-relaxed">
            QuietPatch scans your codebase for technical debt, documentation gaps, and configuration drift.
            Gemma prioritizes confirmed evidence, proposes minimal reviewable fixes, and verifies approved changes.
          </p>

          <div className="flex flex-wrap items-center gap-3 pt-2">
            <button
              onClick={onOpenNewScan}
              disabled={isScanning}
              className="flex items-center space-x-2 px-4 py-2.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-sm font-semibold shadow-md transition disabled:opacity-50"
            >
              <Play className="w-4 h-4 fill-current" />
              <span>Start Maintenance Run</span>
            </button>
            <button
              onClick={onStartDemoScan}
              disabled={isScanning}
              className="flex items-center space-x-2 px-4 py-2.5 rounded-lg bg-[#1a2232] hover:bg-[#232e44] border border-[#2c3954] text-slate-200 text-sm font-medium transition disabled:opacity-50"
            >
              <Sparkles className="w-4 h-4 text-amber-400" />
              <span>Run Sample Demo Scan</span>
            </button>
            <button
              onClick={() => onNavigate('settings')}
              className="px-3.5 py-2.5 text-xs text-slate-400 hover:text-slate-200 transition"
            >
              Configure Mode &amp; Models →
            </button>
          </div>
        </div>

        {/* Decorative corner glow */}
        <div className="absolute right-0 top-0 w-96 h-96 bg-emerald-500/5 rounded-full blur-3xl pointer-events-none" />
      </div>

      {/* Primary Metrics Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Files Scanned */}
        <div className="p-5 rounded-xl bg-[#121622] border border-[#21293a] flex flex-col justify-between">
          <div className="flex items-center justify-between text-slate-400">
            <span className="text-xs font-semibold uppercase tracking-wider">Files Scanned</span>
            <FileCode className="w-4 h-4 text-slate-400" />
          </div>
          <div className="mt-4">
            <div className="text-3xl font-extrabold text-white">
              {currentRun ? currentRun.files_scanned : 0}
            </div>
            <p className="text-xs text-slate-400 mt-1">Excluding caches and build artifacts</p>
          </div>
        </div>

        {/* Confirmed Findings */}
        <div className="p-5 rounded-xl bg-[#121622] border border-[#21293a] flex flex-col justify-between">
          <div className="flex items-center justify-between text-slate-400">
            <span className="text-xs font-semibold uppercase tracking-wider">Confirmed Findings</span>
            <AlertTriangle className="w-4 h-4 text-amber-400" />
          </div>
          <div className="mt-4">
            <div className="text-3xl font-extrabold text-amber-400">{findings.length}</div>
            <p className="text-xs text-slate-400 mt-1">
              {severityCounts.critical + severityCounts.high} critical / high priority
            </p>
          </div>
        </div>

        {/* Fixes Ready / Pending */}
        <div className="p-5 rounded-xl bg-[#121622] border border-[#21293a] flex flex-col justify-between">
          <div className="flex items-center justify-between text-slate-400">
            <span className="text-xs font-semibold uppercase tracking-wider">Proposed Patches</span>
            <Zap className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="mt-4">
            <div className="text-3xl font-extrabold text-emerald-400">{patches.length}</div>
            <p className="text-xs text-slate-400 mt-1">
              {pendingPatches.length} pending review • {appliedPatches.length} applied
            </p>
          </div>
        </div>

        {/* Estimated Time Saved */}
        <div className="p-5 rounded-xl bg-[#121622] border border-[#21293a] flex flex-col justify-between">
          <div className="flex items-center justify-between text-slate-400">
            <span className="text-xs font-semibold uppercase tracking-wider">Time Saved (Est.)</span>
            <Clock className="w-4 h-4 text-blue-400" />
          </div>
          <div className="mt-4">
            <div className="text-3xl font-extrabold text-blue-400">~{totalTimeSavedHours}h</div>
            <p className="text-xs text-slate-400 mt-1">
              {totalTimeSavedMins} manual minutes avoided
            </p>
          </div>
        </div>
      </div>

      {/* Severity Breakdown Bar */}
      <div className="p-5 rounded-xl bg-[#121622] border border-[#21293a] space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-slate-200">Findings by Severity</h3>
          <span className="text-xs text-slate-400">Grounded in deterministic scanner evidence</span>
        </div>

        {/* Segmented bar */}
        <div className="w-full h-3 rounded-full bg-[#1b2230] flex overflow-hidden">
          {findings.length === 0 ? (
            <div className="w-full h-full bg-slate-700/40" />
          ) : (
            <>
              <div
                style={{ width: `${(severityCounts.critical / findings.length) * 100}%` }}
                className="bg-red-500 h-full"
                title={`Critical: ${severityCounts.critical}`}
              />
              <div
                style={{ width: `${(severityCounts.high / findings.length) * 100}%` }}
                className="bg-orange-500 h-full"
                title={`High: ${severityCounts.high}`}
              />
              <div
                style={{ width: `${(severityCounts.medium / findings.length) * 100}%` }}
                className="bg-amber-500 h-full"
                title={`Medium: ${severityCounts.medium}`}
              />
              <div
                style={{ width: `${(severityCounts.low / findings.length) * 100}%` }}
                className="bg-blue-500 h-full"
                title={`Low: ${severityCounts.low}`}
              />
              <div
                style={{ width: `${(severityCounts.informational / findings.length) * 100}%` }}
                className="bg-slate-500 h-full"
                title={`Informational: ${severityCounts.informational}`}
              />
            </>
          )}
        </div>

        {/* Severity Count Pills */}
        <div className="flex flex-wrap gap-2.5 pt-1">
          <div className="flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-red-950/40 border border-red-900/40 text-xs">
            <span className="w-2 h-2 rounded-full bg-red-500" />
            <span className="text-slate-300">Critical:</span>
            <span className="font-bold text-red-400">{severityCounts.critical}</span>
          </div>
          <div className="flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-orange-950/40 border border-orange-900/40 text-xs">
            <span className="w-2 h-2 rounded-full bg-orange-500" />
            <span className="text-slate-300">High:</span>
            <span className="font-bold text-orange-400">{severityCounts.high}</span>
          </div>
          <div className="flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-amber-950/40 border border-amber-900/40 text-xs">
            <span className="w-2 h-2 rounded-full bg-amber-500" />
            <span className="text-slate-300">Medium:</span>
            <span className="font-bold text-amber-300">{severityCounts.medium}</span>
          </div>
          <div className="flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-blue-950/40 border border-blue-900/40 text-xs">
            <span className="w-2 h-2 rounded-full bg-blue-500" />
            <span className="text-slate-300">Low:</span>
            <span className="font-bold text-blue-300">{severityCounts.low}</span>
          </div>
          <div className="flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-xs">
            <span className="w-2 h-2 rounded-full bg-slate-500" />
            <span className="text-slate-400">Info:</span>
            <span className="font-bold text-slate-300">{severityCounts.informational}</span>
          </div>
        </div>
      </div>

      {/* Two Column Layout: Suggested Low-Effort Fixes & Caretaker Status */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: Quick Fix Queue Teaser */}
        <div className="lg:col-span-2 p-5 rounded-xl bg-[#121622] border border-[#21293a] space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Zap className="w-4 h-4 text-emerald-400" />
              <h3 className="text-sm font-semibold text-slate-100">Suggested Minimal Fixes</h3>
            </div>
            <button
              onClick={() => onNavigate('queue')}
              className="text-xs text-emerald-400 hover:text-emerald-300 flex items-center space-x-1 font-medium transition"
            >
              <span>View All in Fix Queue</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>

          {patches.length === 0 ? (
            <div className="p-8 text-center border border-dashed border-[#232b3b] rounded-lg text-slate-500 text-sm">
              No patches proposed yet. Start a maintenance run to generate reviewable fixes.
            </div>
          ) : (
            <div className="space-y-2.5">
              {patches.slice(0, 3).map((patch) => (
                <div
                  key={patch.id}
                  className="p-3.5 rounded-lg bg-[#161b27] border border-[#242e42] hover:border-emerald-800/60 transition flex items-center justify-between"
                >
                  <div className="space-y-1 pr-4">
                    <div className="flex items-center space-x-2">
                      <span className="font-mono text-xs text-slate-300 font-semibold truncate max-w-[220px]">
                        {patch.file_path}
                      </span>
                      <span
                        className={`text-[10px] uppercase font-bold px-1.5 py-0.5 rounded ${
                          patch.status === 'applied'
                            ? 'bg-emerald-950 text-emerald-400 border border-emerald-800'
                            : 'bg-amber-950 text-amber-300 border border-amber-800'
                        }`}
                      >
                        {patch.status}
                      </span>
                    </div>
                    <p className="text-xs text-slate-300 leading-snug">{patch.change_summary}</p>
                  </div>
                  <button
                    onClick={() => onNavigate('queue')}
                    className="shrink-0 px-3 py-1.5 rounded bg-[#1e2738] hover:bg-emerald-600 hover:text-white text-xs text-slate-300 transition"
                  >
                    Review Diff
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Right: Architecture & Security Badges */}
        <div className="p-5 rounded-xl bg-[#121622] border border-[#21293a] space-y-4">
          <div className="flex items-center space-x-2">
            <Shield className="w-4 h-4 text-emerald-400" />
            <h3 className="text-sm font-semibold text-slate-100">Caretaker Status</h3>
          </div>

          <div className="space-y-3 text-xs">
            <div className="p-3 rounded-lg bg-[#151a26] border border-[#222b3b] space-y-1">
              <div className="flex items-center justify-between text-slate-300 font-medium">
                <span>Execution Mode</span>
                <span className="uppercase text-[11px] font-mono text-emerald-400 font-semibold">
                  {config?.app_mode || 'local'}
                </span>
              </div>
              <p className="text-[11px] text-slate-400">
                {config?.app_mode === 'hosted'
                  ? 'Render Web Service: Read-only analysis and patch preview enabled.'
                  : 'Local machine: Private code stays on your filesystem.'}
              </p>
            </div>

            <div className="p-3 rounded-lg bg-[#151a26] border border-[#222b3b] space-y-1">
              <div className="flex items-center justify-between text-slate-300 font-medium">
                <span>Gemma Provider</span>
                <span className="font-mono text-[11px] text-amber-300">
                  {config?.ai_provider || 'demo'}
                </span>
              </div>
              <p className="text-[11px] text-slate-400 truncate">
                {providerStatus?.ai.message || 'Ready for maintenance runs.'}
              </p>
            </div>

            <div className="p-3 rounded-lg bg-[#151a26] border border-[#222b3b] space-y-1">
              <div className="flex items-center justify-between text-slate-300 font-medium">
                <span>Sentry Tracing</span>
                <span
                  className={`text-[11px] font-mono ${
                    providerStatus?.observability?.status === 'connected'
                      ? 'text-emerald-400'
                      : 'text-slate-500'
                  }`}
                >
                  {providerStatus?.observability?.status === 'connected' ? 'Active' : 'Not Configured'}
                </span>
              </div>
              <p className="text-[11px] text-slate-400">
                {providerStatus?.observability?.message || 'DSN omitted in local development.'}
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
