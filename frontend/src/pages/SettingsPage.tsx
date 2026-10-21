import React from 'react';
import {
  Activity,
  AlertCircle,
  CheckCircle2,
  Cpu,
  ExternalLink,
  Flame,
  HardDrive,
  Lock,
  RefreshCw,
  Shield,
  Sparkles,
  Terminal,
} from 'lucide-react';
import { ProviderStatus, PublicConfig } from '../types';

interface SettingsPageProps {
  config: PublicConfig | null;
  providerStatus: ProviderStatus | null;
  onRefreshStatus: () => void;
}

export const SettingsPage: React.FC<SettingsPageProps> = ({
  config,
  providerStatus,
  onRefreshStatus,
}) => {
  const isSentryActive = providerStatus?.observability?.status === 'connected';

  return (
    <div className="space-y-6 max-w-4xl">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-slate-100">Settings &amp; Environment Inspection</h2>
          <p className="text-sm text-slate-400 mt-1">
            Review your AI execution mode, Ollama reachability, Sentry observability, and security boundaries.
          </p>
        </div>
        <button
          onClick={onRefreshStatus}
          className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-[#18202d] hover:bg-[#222c3e] border border-[#29364d] text-xs font-medium text-slate-200 transition"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          <span>Refresh Status</span>
        </button>
      </div>

      {/* 1. Execution Mode & Privacy Details */}
      <div className="p-6 rounded-xl bg-[#121622] border border-[#21293a] space-y-4">
        <div className="flex items-center space-x-2 text-slate-100">
          <Shield className="w-5 h-5 text-emerald-400" />
          <h3 className="text-base font-semibold">Execution Mode &amp; Privacy Architecture</h3>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          <div
            className={`p-3.5 rounded-lg border text-xs space-y-1.5 ${
              config?.app_mode === 'local' && config?.ai_provider === 'ollama'
                ? 'bg-emerald-950/40 border-emerald-700 text-emerald-200 ring-1 ring-emerald-500/30'
                : 'bg-[#151b27] border-[#222b3b] text-slate-400'
            }`}
          >
            <div className="flex items-center justify-between font-bold">
              <span>Mode A: Local Gemma</span>
              {config?.app_mode === 'local' && config?.ai_provider === 'ollama' && (
                <span className="text-[10px] uppercase font-mono text-emerald-400">Active</span>
              )}
            </div>
            <p className="text-[11px] leading-relaxed">
              Privacy-first local execution via Ollama. Code never leaves your computer. Fully offline once downloaded.
            </p>
          </div>

          <div
            className={`p-3.5 rounded-lg border text-xs space-y-1.5 ${
              config?.ai_provider === 'gemma_api'
                ? 'bg-purple-950/40 border-purple-700 text-purple-200 ring-1 ring-purple-500/30'
                : 'bg-[#151b27] border-[#222b3b] text-slate-400'
            }`}
          >
            <div className="flex items-center justify-between font-bold">
              <span>Mode B: Hosted Gemma</span>
              {config?.ai_provider === 'gemma_api' && (
                <span className="text-[10px] uppercase font-mono text-purple-400">Active</span>
              )}
            </div>
            <p className="text-[11px] leading-relaxed">
              Powers genuine Gemma AI on deployed Render Web Service via official Google Gen AI SDK.
            </p>
          </div>

          <div
            className={`p-3.5 rounded-lg border text-xs space-y-1.5 ${
              config?.ai_provider === 'demo'
                ? 'bg-amber-950/40 border-amber-700 text-amber-200 ring-1 ring-amber-500/30'
                : 'bg-[#151b27] border-[#222b3b] text-slate-400'
            }`}
          >
            <div className="flex items-center justify-between font-bold">
              <span>Mode C: Demo Mode</span>
              {config?.ai_provider === 'demo' && (
                <span className="text-[10px] uppercase font-mono text-amber-400">Active</span>
              )}
            </div>
            <p className="text-[11px] leading-relaxed">
              Zero external credentials needed. Runs deterministic scanners and curated demonstration explanations.
            </p>
          </div>
        </div>
      </div>

      {/* 2. AI Provider Status & Diagnostics */}
      <div className="p-6 rounded-xl bg-[#121622] border border-[#21293a] space-y-4">
        <div className="flex items-center space-x-2 text-slate-100">
          <Cpu className="w-5 h-5 text-purple-400" />
          <h3 className="text-base font-semibold">Active AI Provider Diagnostics</h3>
        </div>

        <div className="p-4 rounded-lg bg-[#0c0f16] border border-[#1b2230] space-y-2 text-xs font-mono">
          <div className="flex items-center justify-between">
            <span className="text-slate-500">Configured Provider:</span>
            <span className="text-slate-200 font-bold uppercase">{config?.ai_provider}</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-slate-500">Target Model:</span>
            <span className="text-slate-200">
              {config?.ai_provider === 'ollama'
                ? config?.ollama_model
                : config?.ai_provider === 'gemma_api'
                ? config?.gemma_api_model
                : 'Demonstration Model'}
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-slate-500">Connection State:</span>
            {providerStatus?.ai.status === 'connected' ? (
              <span className="px-2 py-0.5 rounded text-[11px] font-bold bg-emerald-950 text-emerald-400 border border-emerald-800">
                CONNECTED
              </span>
            ) : providerStatus?.ai.status === 'model_missing' ? (
              <span className="px-2 py-0.5 rounded text-[11px] font-bold bg-amber-950 text-amber-400 border border-amber-800">
                MODEL MISSING
              </span>
            ) : providerStatus?.ai.status === 'offline' ? (
              <span className="px-2 py-0.5 rounded text-[11px] font-bold bg-red-950 text-red-400 border border-red-800">
                OFFLINE
              </span>
            ) : (
              <span className="px-2 py-0.5 rounded text-[11px] font-bold bg-slate-800 text-slate-300 border border-slate-700">
                {providerStatus?.ai.status?.toUpperCase() || 'READY'}
              </span>
            )}
          </div>
          <div className="flex items-center justify-between">
            <span className="text-slate-500">Provider Health:</span>
            <span
              className={
                providerStatus?.ai.available
                  ? 'text-emerald-400 font-semibold'
                  : providerStatus?.ai.status === 'model_missing'
                  ? 'text-amber-400 font-semibold'
                  : 'text-red-400 font-semibold'
              }
            >
              {providerStatus?.ai.message || 'Ready'}
            </span>
          </div>
          {providerStatus?.ai.installed_models && providerStatus.ai.installed_models.length > 0 && (
            <div className="flex items-center justify-between pt-1">
              <span className="text-slate-500">Ollama Models:</span>
              <span className="text-slate-300 font-mono text-[11px]">
                {providerStatus.ai.installed_models.join(', ')}
              </span>
            </div>
          )}
          {config?.ai_provider === 'ollama' && (
            <>
              <div className="flex items-center justify-between">
                <span className="text-slate-500">Inference Request Timeout:</span>
                <span className="text-emerald-400 font-semibold">{config.ollama_request_timeout_seconds ?? 240}s</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-slate-500">Model Keep-Alive:</span>
                <span className="text-slate-300">{config.ollama_keep_alive ?? '15m'}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-slate-500">Inference Batch Size:</span>
                <span className="text-slate-300">{config.ollama_batch_size ?? 3} findings/request</span>
              </div>
            </>
          )}
        </div>

        {/* Actionable Instruction when Ollama model not installed */}
        {providerStatus?.ai.instruction && (
          <div className="p-4 rounded-lg bg-amber-950/20 border border-amber-800/40 text-xs text-amber-300 space-y-2">
            <div className="flex items-center space-x-2 font-semibold">
              <Terminal className="w-4 h-4 text-amber-400" />
              <span>Actionable Setup Instruction:</span>
            </div>
            <p className="leading-relaxed">{providerStatus.ai.instruction}</p>
          </div>
        )}

        {/* Privacy Notice for Hosted Provider */}
        {providerStatus?.ai.privacy_notice && (
          <div className="p-3 rounded-lg bg-blue-950/20 border border-blue-800/40 text-xs text-blue-300">
            {providerStatus.ai.privacy_notice}
          </div>
        )}
      </div>

      {/* 3. Sentry Agent Tracing & Observability */}
      <div className="p-6 rounded-xl bg-[#121622] border border-[#21293a] space-y-4">
        <div className="flex items-center space-x-2 text-slate-100">
          <Activity className="w-5 h-5 text-emerald-400" />
          <h3 className="text-base font-semibold">Sentry Agent Tracing Integration</h3>
        </div>

        <div className="p-4 rounded-lg bg-[#0c0f16] border border-[#1b2230] space-y-2 text-xs font-mono">
          <div className="flex items-center justify-between">
            <span className="text-slate-500">Observability Status:</span>
            <span
              className={`font-semibold ${
                isSentryActive ? 'text-emerald-400' : 'text-slate-500'
              }`}
            >
              {isSentryActive ? `Connected (${config?.sentry_environment})` : 'Observability not configured'}
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-slate-500">Traces Sample Rate:</span>
            <span className="text-slate-300">{config?.sentry_sample_rate ?? 0.1}</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-slate-500">Telemetry Sanitization:</span>
            <span className="text-emerald-400">Active (Zero code snippets or secrets transmitted)</span>
          </div>
        </div>

        <div className="text-xs text-slate-400 space-y-2 leading-relaxed">
          <p className="font-semibold text-slate-300">How to observe agent stages in Sentry:</p>
          <ol className="list-decimal list-inside space-y-1 text-slate-400 pl-1">
            <li>Configure <code className="text-slate-200">SENTRY_DSN</code> in your environment variables.</li>
            <li>Run any maintenance scan from QuietPatch.</li>
            <li>Open Sentry &gt; Performance &gt; Transactions &gt; Find <code className="text-slate-200">QuietPatch Stage: ...</code> spans.</li>
            <li>Inspect stage durations (discovery, scanners, Gemma model invocation, patch validation).</li>
          </ol>
        </div>
      </div>

      {/* 4. Security Boundaries & Path Containment */}
      <div className="p-6 rounded-xl bg-[#121622] border border-[#21293a] space-y-4">
        <div className="flex items-center space-x-2 text-slate-100">
          <Lock className="w-5 h-5 text-blue-400" />
          <h3 className="text-base font-semibold">Security Boundaries &amp; Resource Limits</h3>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs font-mono">
          <div className="p-3 rounded-lg bg-[#0c0f16] border border-[#1b2230]">
            <span className="text-slate-500 block">Allowed Repository Root:</span>
            <span className="text-slate-200 font-semibold break-all">
              {config?.allowed_root || 'Current Workspace'}
            </span>
          </div>
          <div className="p-3 rounded-lg bg-[#0c0f16] border border-[#1b2230]">
            <span className="text-slate-500 block">Max Files Per Scan:</span>
            <span className="text-slate-200 font-semibold">{config?.max_repo_files || 500} files</span>
          </div>
          <div className="p-3 rounded-lg bg-[#0c0f16] border border-[#1b2230]">
            <span className="text-slate-500 block">Max Scan Size Limit:</span>
            <span className="text-slate-200 font-semibold">
              {((config?.max_repo_bytes || 10485760) / (1024 * 1024)).toFixed(0)} MB
            </span>
          </div>
          <div className="p-3 rounded-lg bg-[#0c0f16] border border-[#1b2230]">
            <span className="text-slate-500 block">Max Scan Timeout:</span>
            <span className="text-slate-200 font-semibold">{config?.max_scan_seconds || 60} seconds</span>
          </div>
        </div>

        <p className="text-[11px] text-slate-500">
          In Hosted mode on Render, patch application and arbitrary command execution are strictly disabled on the backend.
        </p>
      </div>
    </div>
  );
};
