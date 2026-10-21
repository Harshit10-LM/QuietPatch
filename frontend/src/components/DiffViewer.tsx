import React, { useState } from 'react';
import { Columns, FileCode, Split } from 'lucide-react';

interface DiffViewerProps {
  diff: string;
  originalContent: string;
  proposedContent: string;
  filePath: string;
}

export const DiffViewer: React.FC<DiffViewerProps> = ({
  diff,
  originalContent,
  proposedContent,
  filePath,
}) => {
  const [viewMode, setViewMode] = useState<'diff' | 'side-by-side'>('diff');

  const diffLines = diff.split('\n');

  return (
    <div className="bg-[#0f131a] rounded-lg border border-[#232b3b] overflow-hidden text-sm font-mono">
      {/* Diff Toolbar */}
      <div className="flex items-center justify-between px-4 py-2.5 bg-[#141923] border-b border-[#232b3b]">
        <div className="flex items-center space-x-2 text-slate-300">
          <FileCode className="w-4 h-4 text-emerald-400" />
          <span className="font-semibold text-slate-200">{filePath}</span>
        </div>
        <div className="flex items-center space-x-1 bg-[#0c0f14] p-0.5 rounded border border-[#222a38]">
          <button
            onClick={() => setViewMode('diff')}
            className={`flex items-center space-x-1.5 px-2.5 py-1 rounded text-xs transition-colors ${
              viewMode === 'diff'
                ? 'bg-emerald-600 text-white font-medium shadow'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Split className="w-3.5 h-3.5" />
            <span>Unified Diff</span>
          </button>
          <button
            onClick={() => setViewMode('side-by-side')}
            className={`flex items-center space-x-1.5 px-2.5 py-1 rounded text-xs transition-colors ${
              viewMode === 'side-by-side'
                ? 'bg-emerald-600 text-white font-medium shadow'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Columns className="w-3.5 h-3.5" />
            <span>Before / After</span>
          </button>
        </div>
      </div>

      {/* Diff Content View */}
      {viewMode === 'diff' ? (
        <div className="overflow-x-auto max-h-[420px] p-2 leading-relaxed">
          {diffLines.length === 0 || (diffLines.length === 1 && !diffLines[0]) ? (
            <div className="p-4 text-slate-500 italic">No textual difference detected.</div>
          ) : (
            diffLines.map((line, idx) => {
              let lineStyle = 'text-slate-300';
              let bgStyle = 'bg-transparent';
              let prefix = ' ';

              if (line.startsWith('+') && !line.startsWith('+++')) {
                lineStyle = 'text-emerald-300';
                bgStyle = 'bg-emerald-950/40 border-l-2 border-emerald-500';
                prefix = '+';
              } else if (line.startsWith('-') && !line.startsWith('---')) {
                lineStyle = 'text-red-300';
                bgStyle = 'bg-red-950/40 border-l-2 border-red-500';
                prefix = '-';
              } else if (line.startsWith('@@')) {
                lineStyle = 'text-blue-400 font-semibold';
                bgStyle = 'bg-blue-950/20';
                prefix = '@';
              } else if (line.startsWith('---') || line.startsWith('+++')) {
                lineStyle = 'text-slate-500 italic';
              }

              return (
                <div
                  key={idx}
                  className={`flex px-2 py-0.5 rounded-sm whitespace-pre font-mono text-xs ${bgStyle}`}
                >
                  <span className="w-8 select-none text-slate-600 text-right pr-3">{idx + 1}</span>
                  <span className={`${lineStyle} flex-1`}>{line}</span>
                </div>
              );
            })
          )}
        </div>
      ) : (
        /* Side by Side (Before vs After) View */
        <div className="grid grid-cols-2 divide-x divide-[#232b3b] max-h-[420px] overflow-hidden">
          <div className="flex flex-col">
            <div className="px-3 py-1.5 bg-[#121620] text-xs font-semibold text-slate-400 border-b border-[#232b3b]">
              Original Version
            </div>
            <pre className="p-3 text-xs text-slate-300 overflow-auto flex-1 leading-relaxed whitespace-pre-wrap">
              {originalContent || '<New file will be created>'}
            </pre>
          </div>
          <div className="flex flex-col">
            <div className="px-3 py-1.5 bg-[#121620] text-xs font-semibold text-emerald-400 border-b border-[#232b3b]">
              Proposed Version (Patched)
            </div>
            <pre className="p-3 text-xs text-emerald-200 overflow-auto flex-1 leading-relaxed whitespace-pre-wrap">
              {proposedContent}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
};
