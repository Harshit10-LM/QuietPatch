import React, { useMemo, useState } from 'react';
import {
  AlertCircle,
  ArrowRight,
  Code2,
  FileCode,
  Filter,
  Lightbulb,
  Loader2,
  Search,
  Sparkles,
  Wrench,
  Zap,
} from 'lucide-react';
import { Finding, FindingCategory, MaintenanceRun, Severity } from '../types';
import { CategoryBadge, SeverityBadge, VerificationBadge } from '../components/Badge';

interface FindingsPageProps {
  currentRun: MaintenanceRun | null;
  onProposeFix: (findingId: string) => Promise<void>;
  onNavigateToQueue: () => void;
  isProposingFix: boolean;
  onRetryAi?: () => Promise<void>;
  isRetryingAi?: boolean;
}

export const FindingsPage: React.FC<FindingsPageProps> = ({
  currentRun,
  onProposeFix,
  onNavigateToQueue,
  isProposingFix,
  onRetryAi,
  isRetryingAi,
}) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedSeverity, setSelectedSeverity] = useState<string>('all');
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [sortBy, setSortBy] = useState<'severity' | 'file' | 'category' | 'effort'>('severity');
  const [activeFixId, setActiveFixId] = useState<string | null>(null);

  const findings = currentRun?.findings || [];

  const severityWeight: Record<Severity, number> = {
    critical: 5,
    high: 4,
    medium: 3,
    low: 2,
    informational: 1,
  };

  const filteredFindings = useMemo(() => {
    return findings
      .filter((f) => {
        if (selectedSeverity !== 'all' && f.severity !== selectedSeverity) return false;
        if (selectedCategory !== 'all' && f.category !== selectedCategory) return false;
        if (searchQuery.trim()) {
          const q = searchQuery.toLowerCase();
          const matchTitle = f.title.toLowerCase().includes(q);
          const matchFile = f.file_path.toLowerCase().includes(q);
          const matchEvidence = f.evidence.toLowerCase().includes(q);
          const matchExplanation = f.explanation.toLowerCase().includes(q);
          if (!matchTitle && !matchFile && !matchEvidence && !matchExplanation) return false;
        }
        return true;
      })
      .sort((a, b) => {
        if (sortBy === 'severity') {
          return severityWeight[b.severity] - severityWeight[a.severity];
        }
        if (sortBy === 'file') {
          return a.file_path.localeCompare(b.file_path);
        }
        if (sortBy === 'category') {
          return a.category.localeCompare(b.category);
        }
        if (sortBy === 'effort') {
          return b.estimated_effort_mins - a.estimated_effort_mins;
        }
        return 0;
      });
  }, [findings, selectedSeverity, selectedCategory, searchQuery, sortBy]);

  const handlePropose = async (findingId: string) => {
    setActiveFixId(findingId);
    try {
      await onProposeFix(findingId);
      onNavigateToQueue();
    } finally {
      setActiveFixId(null);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold text-slate-100">Confirmed Repository Findings</h2>
          <p className="text-sm text-slate-400 mt-1">
            Deterministic scanner detections prioritized by Gemma open-weight models.
          </p>
        </div>
        <div className="text-xs text-slate-400 bg-[#121622] px-3 py-1.5 rounded-lg border border-[#21293a] self-start sm:self-auto font-mono">
          Showing <span className="font-bold text-emerald-400">{filteredFindings.length}</span> of{' '}
          {findings.length} findings
        </div>
      </div>

      {/* Unassisted Findings / Retry AI Banner */}
      {currentRun && currentRun.findings.some((f) => !f.ai_assisted && !f.is_demo) && (
        <div className="p-4 rounded-xl bg-amber-950/40 border border-amber-800/80 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs">
          <div className="flex items-start space-x-3 text-amber-300">
            <AlertCircle className="w-5 h-5 shrink-0 mt-0.5 text-amber-400" />
            <div>
              <h4 className="font-semibold text-amber-200">
                Some Findings Awaiting Gemma AI Evaluation
              </h4>
              <p className="text-amber-300/80 mt-0.5 text-[11px]">
                Deterministic scanner findings were confirmed, but local Gemma inference timed out or was partially completed.
                You can retry AI analysis without re-scanning repository files.
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

      {/* Search and Filter Controls */}
      <div className="p-4 rounded-xl bg-[#121622] border border-[#21293a] space-y-3">
        <div className="flex flex-col md:flex-row gap-3">
          {/* Search Bar */}
          <div className="relative flex-1">
            <Search className="w-4 h-4 text-slate-400 absolute left-3 top-2.5" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search by keyword, filename, or evidence..."
              className="w-full pl-9 pr-3 py-2 rounded-lg bg-[#161b27] border border-[#263145] text-xs text-slate-100 placeholder-slate-500 focus:outline-none focus:border-emerald-500"
            />
          </div>

          {/* Sort By Dropdown */}
          <div className="flex items-center space-x-2 shrink-0">
            <span className="text-xs text-slate-400">Sort:</span>
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as any)}
              className="px-3 py-2 rounded-lg bg-[#161b27] border border-[#263145] text-xs text-slate-200 focus:outline-none focus:border-emerald-500 font-medium"
            >
              <option value="severity">Severity (High to Low)</option>
              <option value="file">File Path</option>
              <option value="category">Category</option>
              <option value="effort">Estimated Effort</option>
            </select>
          </div>
        </div>

        {/* Filter Badges Row */}
        <div className="flex flex-wrap items-center gap-2 pt-1 text-xs">
          <span className="text-slate-500 text-[11px] font-semibold uppercase tracking-wider mr-1">
            Severity:
          </span>
          {['all', 'critical', 'high', 'medium', 'low', 'informational'].map((sev) => (
            <button
              key={sev}
              onClick={() => setSelectedSeverity(sev)}
              className={`px-2.5 py-1 rounded-md capitalize font-medium transition ${
                selectedSeverity === sev
                  ? 'bg-emerald-600 text-white shadow'
                  : 'bg-[#181e2b] text-slate-400 hover:text-slate-200 hover:bg-[#20283a]'
              }`}
            >
              {sev}
            </button>
          ))}

          <span className="text-slate-500 text-[11px] font-semibold uppercase tracking-wider ml-3 mr-1">
            Category:
          </span>
          {['all', 'todo', 'documentation', 'configuration', 'python', 'javascript', 'general'].map(
            (cat) => (
              <button
                key={cat}
                onClick={() => setSelectedCategory(cat)}
                className={`px-2.5 py-1 rounded-md capitalize font-medium transition ${
                  selectedCategory === cat
                    ? 'bg-emerald-600 text-white shadow'
                    : 'bg-[#181e2b] text-slate-400 hover:text-slate-200 hover:bg-[#20283a]'
                }`}
              >
                {cat}
              </button>
            )
          )}
        </div>
      </div>

      {/* Findings List */}
      {filteredFindings.length === 0 ? (
        <div className="p-12 text-center border border-dashed border-[#232b3b] rounded-xl text-slate-500 space-y-2">
          <p className="text-sm font-medium">No findings match your current filters.</p>
          <button
            onClick={() => {
              setSelectedSeverity('all');
              setSelectedCategory('all');
              setSearchQuery('');
            }}
            className="text-xs text-emerald-400 hover:underline"
          >
            Clear all filters
          </button>
        </div>
      ) : (
        <div className="space-y-3.5">
          {filteredFindings.map((finding) => (
            <div
              key={finding.id}
              className="p-5 rounded-xl bg-[#121622] border border-[#21293a] hover:border-[#2f3b52] transition space-y-3.5"
            >
              {/* Finding Top Bar */}
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div className="space-y-1">
                  <div className="flex flex-wrap items-center gap-2">
                    <SeverityBadge severity={finding.severity} />
                    <CategoryBadge category={finding.category} />
                    <VerificationBadge status={finding.verification_status} />
                    {finding.is_demo ? (
                      <span className="inline-flex items-center space-x-1 px-2 py-0.5 rounded text-[11px] font-mono bg-amber-950/70 text-amber-300 border border-amber-800/60 font-semibold">
                        <Sparkles className="w-3 h-3" />
                        <span>Demo Fixture</span>
                      </span>
                    ) : finding.ai_assisted ? (
                      <span className="inline-flex items-center space-x-1 px-2 py-0.5 rounded text-[11px] font-mono bg-purple-950/70 text-purple-300 border border-purple-800/60 font-semibold">
                        <Sparkles className="w-3 h-3" />
                        <span>Gemma Evaluated ({finding.ai_model || 'Local'})</span>
                      </span>
                    ) : (
                      <span className="inline-flex items-center space-x-1 px-2 py-0.5 rounded text-[11px] font-mono bg-slate-900 text-slate-400 border border-slate-700/60 font-medium">
                        <span>Deterministic Scan</span>
                      </span>
                    )}
                  </div>
                  <h3 className="text-base font-semibold text-slate-100">{finding.title}</h3>
                </div>

                <button
                  onClick={() => handlePropose(finding.id)}
                  disabled={isProposingFix && activeFixId === finding.id}
                  className="flex items-center space-x-1.5 px-3.5 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold shadow transition disabled:opacity-50 shrink-0"
                >
                  {isProposingFix && activeFixId === finding.id ? (
                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  ) : (
                    <Zap className="w-3.5 h-3.5 fill-current" />
                  )}
                  <span>Propose Fix</span>
                </button>
              </div>

              {/* File and Line Location */}
              <div className="flex items-center space-x-2 text-xs font-mono text-slate-400 bg-[#0c0f16] px-3 py-1.5 rounded border border-[#1b2230]">
                <FileCode className="w-3.5 h-3.5 text-slate-400" />
                <span className="text-slate-300 font-semibold">{finding.file_path}</span>
                {finding.line_number && (
                  <span className="text-slate-500">line {finding.line_number}</span>
                )}
                <span className="text-slate-600 ml-auto">
                  Effort: ~{finding.estimated_effort_mins} mins
                </span>
              </div>

              {/* Code Evidence Snippet */}
              {finding.evidence && (
                <div className="space-y-1">
                  <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
                    Grounded Evidence:
                  </span>
                  <pre className="p-2.5 rounded bg-[#0a0d13] border border-[#1d2535] text-xs font-mono text-emerald-300 overflow-x-auto whitespace-pre-wrap">
                    {finding.evidence}
                  </pre>
                </div>
              )}

              {/* Why This Matters (Explanation) */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs pt-1">
                <div className="p-3 rounded-lg bg-[#161c28] border border-[#232c3f] space-y-1">
                  <div className="flex items-center space-x-1.5 font-semibold text-slate-300">
                    <Lightbulb className="w-3.5 h-3.5 text-amber-400" />
                    <span>Why This Matters</span>
                  </div>
                  <p className="text-slate-300 leading-relaxed">{finding.explanation}</p>
                </div>

                <div className="p-3 rounded-lg bg-[#161c28] border border-[#232c3f] space-y-1">
                  <div className="flex items-center space-x-1.5 font-semibold text-slate-300">
                    <Wrench className="w-3.5 h-3.5 text-blue-400" />
                    <span>Recommended Action</span>
                  </div>
                  <p className="text-slate-300 leading-relaxed">{finding.suggested_action}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
