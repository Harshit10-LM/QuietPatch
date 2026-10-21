import React, { useEffect, useState } from 'react';
import {
  AlertCircle,
  CheckCircle2,
  Clock,
  Copy,
  Download,
  FileCode,
  FileDown,
  FileText,
  Loader2,
  Sparkles,
} from 'lucide-react';
import { api } from '../api/client';
import { MaintenanceReport, MaintenanceRun } from '../types';

interface ReportsPageProps {
  currentRun: MaintenanceRun | null;
}

export const ReportsPage: React.FC<ReportsPageProps> = ({ currentRun }) => {
  const [report, setReport] = useState<MaintenanceReport | null>(null);
  const [loading, setLoading] = useState(false);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (!currentRun?.id) return;
    setLoading(true);
    api
      .getReport(currentRun.id)
      .then(setReport)
      .catch((err) => console.error('Failed to load report:', err))
      .finally(() => setLoading(false));
  }, [currentRun?.id]);

  const handleDownloadMarkdown = async () => {
    if (!currentRun?.id) return;
    try {
      const md = await api.exportReportMarkdown(currentRun.id);
      const blob = new Blob([md], { type: 'text/markdown;charset=utf-8' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `quietpatch-report-${currentRun.id}.md`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e: any) {
      alert(`Export failed: ${e.message}`);
    }
  };

  const handleCopyMarkdown = async () => {
    if (!currentRun?.id) return;
    try {
      const md = await api.exportReportMarkdown(currentRun.id);
      await navigator.clipboard.writeText(md);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (e: any) {
      alert(`Copy failed: ${e.message}`);
    }
  };

  const handleDownloadJson = () => {
    if (!report) return;
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `quietpatch-report-${report.run_id}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  if (!currentRun) {
    return (
      <div className="p-12 text-center border border-dashed border-[#232b3b] rounded-xl text-slate-500">
        No maintenance run selected. Run a scan to generate a report.
      </div>
    );
  }

  if (loading || !report) {
    return (
      <div className="p-12 text-center text-slate-400 flex items-center justify-center space-x-2">
        <Loader2 className="w-5 h-5 animate-spin text-emerald-400" />
        <span>Compiling maintenance report...</span>
      </div>
    );
  }

  return (
    <div className="space-y-6 max-w-5xl">
      {/* Report Header Bar */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold text-slate-100">Caretaker Maintenance Report</h2>
          <p className="text-sm text-slate-400 mt-0.5">
            Run ID: <code className="text-slate-300 font-mono">{report.run_id}</code> • Generated at{' '}
            {new Date(report.generated_at).toLocaleTimeString()}
          </p>
        </div>

        {/* Export Buttons */}
        <div className="flex items-center space-x-2">
          <button
            onClick={handleCopyMarkdown}
            className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-[#18202d] hover:bg-[#222c3e] border border-[#29364d] text-xs font-medium text-slate-200 transition"
          >
            <Copy className="w-3.5 h-3.5" />
            <span>{copied ? 'Copied!' : 'Copy Markdown'}</span>
          </button>
          <button
            onClick={handleDownloadMarkdown}
            className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-[#18202d] hover:bg-[#222c3e] border border-[#29364d] text-xs font-medium text-slate-200 transition"
          >
            <FileDown className="w-3.5 h-3.5 text-emerald-400" />
            <span>Export .md</span>
          </button>
          <button
            onClick={handleDownloadJson}
            className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-[#18202d] hover:bg-[#222c3e] border border-[#29364d] text-xs font-medium text-slate-200 transition"
          >
            <Download className="w-3.5 h-3.5 text-blue-400" />
            <span>Export JSON</span>
          </button>
        </div>
      </div>

      {/* Summary Highlight Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="p-4 rounded-xl bg-[#121622] border border-[#21293a]">
          <span className="text-[11px] font-semibold uppercase text-slate-500">Files Scanned</span>
          <div className="text-2xl font-bold text-slate-100 mt-1">{report.files_scanned}</div>
        </div>
        <div className="p-4 rounded-xl bg-[#121622] border border-[#21293a]">
          <span className="text-[11px] font-semibold uppercase text-slate-500">Total Findings</span>
          <div className="text-2xl font-bold text-amber-400 mt-1">{report.total_findings}</div>
        </div>
        <div className="p-4 rounded-xl bg-[#121622] border border-[#21293a]">
          <span className="text-[11px] font-semibold uppercase text-slate-500">Fixes Applied</span>
          <div className="text-2xl font-bold text-emerald-400 mt-1">
            {report.fixes_applied} / {report.fixes_proposed}
          </div>
        </div>
        <div className="p-4 rounded-xl bg-[#121622] border border-[#21293a]">
          <span className="text-[11px] font-semibold uppercase text-slate-500">Time Saved</span>
          <div className="text-2xl font-bold text-blue-400 mt-1">
            ~{report.time_saved.total_hours} hrs
          </div>
        </div>
      </div>

      {/* Verification Outcomes & Time Saved Section */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Verification Checks */}
        <div className="p-5 rounded-xl bg-[#121622] border border-[#21293a] space-y-3">
          <h3 className="text-sm font-semibold text-slate-200">Real Verification Results</h3>
          <div className="space-y-2 text-xs">
            <div className="flex items-center justify-between p-2.5 rounded-lg bg-[#161c28] border border-[#222b3b]">
              <span className="text-slate-300">Passed Checks</span>
              <span className="font-bold text-emerald-400 font-mono">{report.checks_passed}</span>
            </div>
            <div className="flex items-center justify-between p-2.5 rounded-lg bg-[#161c28] border border-[#222b3b]">
              <span className="text-slate-300">Failed Checks</span>
              <span className="font-bold text-red-400 font-mono">{report.checks_failed}</span>
            </div>
            <div className="flex items-center justify-between p-2.5 rounded-lg bg-[#161c28] border border-[#222b3b]">
              <span className="text-slate-300">Skipped / Pending</span>
              <span className="font-bold text-slate-400 font-mono">{report.checks_skipped}</span>
            </div>
          </div>
        </div>

        {/* Time Saved Breakdown */}
        <div className="p-5 rounded-xl bg-[#121622] border border-[#21293a] space-y-3">
          <h3 className="text-sm font-semibold text-slate-200">Lazy Payoff Accounting</h3>
          <p className="text-xs text-slate-400 leading-relaxed">
            {report.time_saved.assumptions_note}
          </p>
          <div className="pt-1 flex items-center justify-between text-xs font-mono border-t border-[#222b3b]">
            <span className="text-slate-400">Total Minutes Avoided:</span>
            <span className="font-bold text-blue-400 text-sm">
              {report.time_saved.total_minutes} mins
            </span>
          </div>
        </div>
      </div>

      {/* Stage Durations Table */}
      <div className="p-5 rounded-xl bg-[#121622] border border-[#21293a] space-y-3">
        <h3 className="text-sm font-semibold text-slate-200">Stage Performance Traces</h3>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 text-xs font-mono">
          {Object.entries(report.stage_durations).map(([stage, dur]) => (
            <div
              key={stage}
              className="p-2.5 rounded-lg bg-[#151a26] border border-[#21293a] flex items-center justify-between"
            >
              <span className="capitalize text-slate-400">{stage}:</span>
              <span className="font-bold text-slate-200">{dur.toFixed(3)}s</span>
            </div>
          ))}
        </div>
      </div>

      {/* Remaining Work Requiring Human Review */}
      <div className="p-5 rounded-xl bg-[#121622] border border-[#21293a] space-y-3">
        <h3 className="text-sm font-semibold text-slate-200">
          Remaining Work Requiring Developer Attention
        </h3>
        {report.remaining_work.length === 0 ? (
          <p className="text-xs text-emerald-400 italic">
            All identified findings have been addressed and verified!
          </p>
        ) : (
          <div className="space-y-2 text-xs">
            {report.remaining_work.map((item) => (
              <div
                key={item.id}
                className="p-3 rounded-lg bg-[#151a26] border border-[#222938] flex flex-col sm:flex-row sm:items-center justify-between gap-2"
              >
                <div>
                  <div className="flex items-center space-x-2">
                    <span className="font-mono text-emerald-400 font-semibold truncate max-w-[220px]">
                      {item.file_path}
                    </span>
                    <span className="uppercase text-[10px] font-bold px-1.5 py-0.5 rounded bg-slate-800 text-slate-300">
                      {item.severity}
                    </span>
                  </div>
                  <p className="text-slate-300 mt-0.5">{item.title}</p>
                </div>
                <span className="text-slate-500 text-[11px] shrink-0">{item.action}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
