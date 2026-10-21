import React from 'react';
import { Activity, Cpu, Play, ShieldAlert, Sparkles } from 'lucide-react';
import { ProviderStatus, PublicConfig } from '../types';
import { SlothMascot } from './SlothMascot';

interface HeaderProps {
  config: PublicConfig | null;
  providerStatus: ProviderStatus | null;
  activeRepoName: string;
  onOpenNewScan: () => void;
  onRunDemoScan: () => void;
  isScanning: boolean;
}

export const Header: React.FC<HeaderProps> = ({
  config,
  providerStatus,
  activeRepoName,
  onOpenNewScan,
  onRunDemoScan,
  isScanning,
}) => {
  const isSentryActive = providerStatus?.observability?.status === 'connected';
  const providerType = config?.ai_provider || 'demo';

  return (
    <header className="h-16 border-b border-[#1f2635] bg-[#0d1017] px-6 flex items-center justify-between sticky top-0 z-40">
      {/* Brand */}
      <div className="flex items-center space-x-3.5">
        <div className="p-1.5 rounded-lg bg-emerald-950/60 border border-emerald-800/50 flex items-center justify-center">
          <SlothMascot size={26} />
        </div>
        <div>
          <div className="flex items-center space-x-2">
            <h1 className="text-base font-bold text-slate-100 tracking-tight">QuietPatch</h1>
            <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-emerald-950 text-emerald-400 border border-emerald-800/60 font-semibold uppercase">
              Gemma Caretaker
            </span>
          </div>
          <p className="text-xs text-slate-400 hidden sm:block">Slow and steady repository care</p>
        </div>
      </div>

      {/* Active Repository & Status Pills */}
      <div className="flex items-center space-x-3">
        {/* Active Repo Pill */}
        <div className="hidden md:flex items-center space-x-2 px-3 py-1 rounded-full bg-[#141924] border border-[#232b3b] text-xs">
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
          <span className="text-slate-400">Target:</span>
          <span className="font-medium text-slate-200 truncate max-w-[180px]">{activeRepoName}</span>
        </div>

        {/* AI Provider Status */}
        <div className="hidden lg:flex items-center space-x-1.5 px-3 py-1 rounded-full bg-[#141924] border border-[#232b3b] text-xs">
          <Cpu className={`w-3.5 h-3.5 ${
            providerType === 'ollama'
              ? providerStatus?.ai.status === 'connected'
                ? 'text-emerald-400'
                : providerStatus?.ai.status === 'model_missing'
                ? 'text-amber-400'
                : 'text-red-400'
              : providerType === 'gemma_api'
              ? providerStatus?.ai.available
                ? 'text-purple-400'
                : 'text-amber-400'
              : 'text-amber-400'
          }`} />
          <span className="text-slate-400">AI:</span>
          {providerType === 'ollama' ? (
            providerStatus?.ai.status === 'connected' ? (
              <span className="font-semibold text-emerald-300 uppercase text-[11px]">
                Connected ({config?.ollama_model || 'Gemma'})
              </span>
            ) : providerStatus?.ai.status === 'model_missing' ? (
              <span
                className="font-semibold text-amber-400 uppercase text-[11px]"
                title={providerStatus?.ai.message}
              >
                Model Missing ({config?.ollama_model || 'Gemma'})
              </span>
            ) : (
              <span
                className="font-semibold text-red-400 uppercase text-[11px]"
                title={providerStatus?.ai.message}
              >
                Offline ({config?.ollama_model || 'Gemma'})
              </span>
            )
          ) : providerType === 'gemma_api' ? (
            <span className="font-semibold text-purple-300 uppercase text-[11px]">
              {providerStatus?.ai.available ? 'Hosted Gemma' : 'Gemma API (Unconfigured)'}
            </span>
          ) : (
            <span className="font-semibold text-amber-300 uppercase text-[11px]">
              Demo Fixtures
            </span>
          )}
        </div>

        {/* Sentry Observability Status */}
        <div className="hidden xl:flex items-center space-x-1.5 px-3 py-1 rounded-full bg-[#141924] border border-[#232b3b] text-xs">
          <Activity className={`w-3.5 h-3.5 ${isSentryActive ? 'text-emerald-400' : 'text-slate-500'}`} />
          <span className="text-slate-400">Sentry:</span>
          <span className={`text-[11px] font-medium ${isSentryActive ? 'text-emerald-300' : 'text-slate-500'}`}>
            {isSentryActive ? 'Active' : 'Offline'}
          </span>
        </div>

        {/* Action Buttons */}
        <div className="flex items-center space-x-2">
          <button
            onClick={onRunDemoScan}
            disabled={isScanning}
            className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-[#192230] hover:bg-[#232e42] border border-[#2c374d] text-xs font-medium text-slate-200 transition disabled:opacity-50"
            title="Scan bundled sample repository with deterministic findings"
          >
            <Sparkles className="w-3.5 h-3.5 text-amber-400" />
            <span className="hidden sm:inline">1-Click Demo Scan</span>
            <span className="sm:hidden">Demo</span>
          </button>

          <button
            onClick={onOpenNewScan}
            disabled={isScanning}
            className="flex items-center space-x-1.5 px-3.5 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold shadow-sm transition disabled:opacity-50"
          >
            <Play className="w-3.5 h-3.5 fill-current" />
            <span>Start Maintenance Run</span>
          </button>
        </div>
      </div>
    </header>
  );
};
