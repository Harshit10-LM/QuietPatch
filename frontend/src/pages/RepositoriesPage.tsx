import React, { useState } from 'react';
import {
  AlertCircle,
  CheckCircle2,
  FolderGit2,
  GitBranch,
  Globe,
  HardDrive,
  Play,
  ShieldAlert,
  Sparkles,
} from 'lucide-react';
import { api } from '../api/client';
import { PublicConfig } from '../types';

interface RepositoriesPageProps {
  config: PublicConfig | null;
  onSelectRepoAndScan: (type: 'local' | 'sample' | 'github', path: string) => void;
  isScanning: boolean;
}

export const RepositoriesPage: React.FC<RepositoriesPageProps> = ({
  config,
  onSelectRepoAndScan,
  isScanning,
}) => {
  const [localPath, setLocalPath] = useState('');
  const [githubUrl, setGithubUrl] = useState('');
  const [valStatus, setValStatus] = useState<{
    valid?: boolean;
    message?: string;
    error?: string;
  } | null>(null);
  const [isValidating, setIsValidating] = useState(false);

  const handleValidateLocal = async () => {
    if (!localPath.trim()) return;
    setIsValidating(true);
    setValStatus(null);
    try {
      const res = await api.validateRepository('local', localPath.trim());
      setValStatus({ valid: true, message: res.message || 'Valid repository path' });
    } catch (e: any) {
      setValStatus({ valid: false, error: e.message });
    } finally {
      setIsValidating(false);
    }
  };

  const handleValidateGithub = async () => {
    if (!githubUrl.trim()) return;
    setIsValidating(true);
    setValStatus(null);
    try {
      const res = await api.validateRepository('github', githubUrl.trim());
      setValStatus({ valid: true, message: res.message || 'Valid public GitHub repository' });
    } catch (e: any) {
      setValStatus({ valid: false, error: e.message });
    } finally {
      setIsValidating(false);
    }
  };

  return (
    <div className="space-y-6 max-w-4xl">
      <div>
        <h2 className="text-xl font-bold text-slate-100">Repository Selection</h2>
        <p className="text-sm text-slate-400 mt-1">
          Choose a repository source to scan. QuietPatch enforces strict local containment boundaries and safe GitHub archive limits.
        </p>
      </div>

      {/* 1. Bundled Sample Repository Card */}
      <div className="p-6 rounded-xl bg-gradient-to-r from-[#121824] to-[#161f30] border border-emerald-800/50 shadow-md flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div className="space-y-1.5">
          <div className="flex items-center space-x-2">
            <Sparkles className="w-4 h-4 text-emerald-400" />
            <h3 className="text-base font-semibold text-slate-100">
              NovaTask (Bundled Demonstration Project)
            </h3>
            <span className="text-[10px] font-bold px-2 py-0.5 rounded bg-emerald-950 text-emerald-300 border border-emerald-800 uppercase tracking-wider">
              Zero Credentials
            </span>
          </div>
          <p className="text-xs text-slate-300 leading-relaxed max-w-xl">
            A realistic asynchronous job orchestrator project containing authentic broken documentation links, undocumented environment variables, missing test scripts, and maintenance TODOs.
          </p>
          <div className="flex items-center space-x-2 text-[11px] text-slate-400 font-mono">
            <span>Stack: Python 3 • TypeScript • Markdown</span>
          </div>
        </div>

        <button
          onClick={() => onSelectRepoAndScan('sample', 'sample-repo')}
          disabled={isScanning}
          className="shrink-0 flex items-center space-x-2 px-4 py-2.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold shadow transition disabled:opacity-50"
        >
          <Play className="w-3.5 h-3.5 fill-current" />
          <span>Scan Sample Project</span>
        </button>
      </div>

      {/* 2. Local File System Explorer Card */}
      <div className="p-6 rounded-xl bg-[#121622] border border-[#21293a] space-y-4">
        <div className="flex items-center space-x-2.5 text-slate-100">
          <HardDrive className="w-5 h-5 text-blue-400" />
          <h3 className="text-base font-semibold">Local Repository (On Your Machine)</h3>
        </div>
        <p className="text-xs text-slate-400 leading-relaxed">
          Select a project inside your configured allowed root directory. Private code is never transmitted outside your permitted directory.
        </p>

        <div className="p-3 rounded-lg bg-[#0c0f16] border border-[#1b2230] text-xs font-mono space-y-1">
          <span className="text-slate-500">Configured Allowed Root:</span>
          <p className="text-slate-300 font-semibold break-all">
            {config?.allowed_root || 'Current Workspace'}
          </p>
        </div>

        <div className="space-y-2">
          <label className="text-xs font-medium text-slate-300">
            Path relative to allowed root (or subfolder name):
          </label>
          <div className="flex gap-2">
            <input
              type="text"
              value={localPath}
              onChange={(e) => setLocalPath(e.target.value)}
              placeholder="e.g. . or my-backend-service"
              className="flex-1 px-3 py-2 rounded-lg bg-[#161b27] border border-[#263145] text-xs text-slate-100 placeholder-slate-500 focus:outline-none focus:border-emerald-500"
            />
            <button
              onClick={handleValidateLocal}
              disabled={isValidating || !localPath.trim()}
              className="px-3.5 py-2 rounded-lg bg-[#1e2738] hover:bg-[#28354c] text-xs font-medium text-slate-200 transition disabled:opacity-50"
            >
              Validate
            </button>
            <button
              onClick={() => onSelectRepoAndScan('local', localPath.trim() || '.')}
              disabled={isScanning}
              className="flex items-center space-x-1.5 px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold shadow transition disabled:opacity-50"
            >
              <Play className="w-3.5 h-3.5 fill-current" />
              <span>Scan Local Repo</span>
            </button>
          </div>
        </div>
      </div>

      {/* 3. Public GitHub Repository Card */}
      <div className="p-6 rounded-xl bg-[#121622] border border-[#21293a] space-y-4">
        <div className="flex items-center space-x-2.5 text-slate-100">
          <GitBranch className="w-5 h-5 text-purple-400" />
          <h3 className="text-base font-semibold">Public GitHub Repository</h3>
        </div>
        <p className="text-xs text-slate-400 leading-relaxed">
          Inspect an open-source GitHub repository. QuietPatch downloads the archive safely with strict file count, decompression bomb, and SSRF defenses.
        </p>

        <div className="space-y-2">
          <label className="text-xs font-medium text-slate-300">
            Public GitHub Repository URL:
          </label>
          <div className="flex gap-2">
            <input
              type="text"
              value={githubUrl}
              onChange={(e) => setGithubUrl(e.target.value)}
              placeholder="https://github.com/owner/repository"
              className="flex-1 px-3 py-2 rounded-lg bg-[#161b27] border border-[#263145] text-xs text-slate-100 placeholder-slate-500 focus:outline-none focus:border-emerald-500"
            />
            <button
              onClick={handleValidateGithub}
              disabled={isValidating || !githubUrl.trim()}
              className="px-3.5 py-2 rounded-lg bg-[#1e2738] hover:bg-[#28354c] text-xs font-medium text-slate-200 transition disabled:opacity-50"
            >
              Validate
            </button>
            <button
              onClick={() => onSelectRepoAndScan('github', githubUrl.trim())}
              disabled={isScanning || !githubUrl.trim()}
              className="flex items-center space-x-1.5 px-4 py-2 rounded-lg bg-purple-600 hover:bg-purple-500 text-white text-xs font-semibold shadow transition disabled:opacity-50"
            >
              <Play className="w-3.5 h-3.5 fill-current" />
              <span>Analyze Public Repo</span>
            </button>
          </div>
        </div>

        <div className="p-3 rounded-lg bg-amber-950/20 border border-amber-800/40 text-xs text-amber-300/90 flex items-start space-x-2">
          <AlertCircle className="w-4 h-4 shrink-0 text-amber-400 mt-0.5" />
          <span>
            Safety Guarantee: Hosted and remote scans execute static deterministic analysis only. Build scripts and test suites are never executed on untrusted remote code.
          </span>
        </div>
      </div>

      {/* Live Validation Alert */}
      {valStatus && (
        <div
          className={`p-4 rounded-xl border flex items-center space-x-3 text-xs ${
            valStatus.valid
              ? 'bg-emerald-950/40 border-emerald-800 text-emerald-300'
              : 'bg-red-950/40 border-red-800 text-red-300'
          }`}
        >
          {valStatus.valid ? (
            <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
          ) : (
            <ShieldAlert className="w-4 h-4 text-red-400 shrink-0" />
          )}
          <span>{valStatus.message || valStatus.error}</span>
        </div>
      )}
    </div>
  );
};
