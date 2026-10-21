import React from 'react';
import { FindingCategory, PatchStatus, Severity, VerificationStatus } from '../types';

interface SeverityBadgeProps {
  severity: Severity;
}

export const SeverityBadge: React.FC<SeverityBadgeProps> = ({ severity }) => {
  const styles: Record<Severity, string> = {
    critical: 'bg-red-950/80 text-red-400 border-red-800/60',
    high: 'bg-orange-950/80 text-orange-400 border-orange-800/60',
    medium: 'bg-amber-950/80 text-amber-300 border-amber-800/60',
    low: 'bg-blue-950/80 text-blue-300 border-blue-800/60',
    informational: 'bg-slate-900 text-slate-400 border-slate-800',
  };

  return (
    <span
      className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium uppercase tracking-wider border ${
        styles[severity] || styles.informational
      }`}
    >
      {severity}
    </span>
  );
};

interface CategoryBadgeProps {
  category: FindingCategory;
}

export const CategoryBadge: React.FC<CategoryBadgeProps> = ({ category }) => {
  const labels: Record<FindingCategory, string> = {
    todo: 'TODO / Marker',
    documentation: 'Docs',
    configuration: 'Config',
    python: 'Python',
    javascript: 'JS/TS',
    general: 'Hygiene',
  };

  return (
    <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-mono bg-slate-800/80 text-slate-300 border border-slate-700/60">
      {labels[category] || category}
    </span>
  );
};

interface VerificationBadgeProps {
  status: VerificationStatus;
}

export const VerificationBadge: React.FC<VerificationBadgeProps> = ({ status }) => {
  const styles: Record<VerificationStatus, string> = {
    passed: 'bg-emerald-950/80 text-emerald-400 border-emerald-800/60',
    failed: 'bg-red-950/80 text-red-400 border-red-800/60',
    unverified: 'bg-slate-900 text-slate-400 border-slate-800',
    skipped: 'bg-slate-900 text-slate-500 border-slate-800',
  };

  return (
    <span
      className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium border ${
        styles[status] || styles.unverified
      }`}
    >
      {status === 'passed' && '✓ '}
      {status === 'failed' && '✕ '}
      {status.charAt(0).toUpperCase() + status.slice(1)}
    </span>
  );
};

interface PatchStatusBadgeProps {
  status: PatchStatus;
}

export const PatchStatusBadge: React.FC<PatchStatusBadgeProps> = ({ status }) => {
  const styles: Record<PatchStatus, string> = {
    pending: 'bg-amber-950/80 text-amber-300 border-amber-800/60',
    approved: 'bg-emerald-950/80 text-emerald-300 border-emerald-800/60',
    rejected: 'bg-slate-900 text-slate-400 border-slate-800',
    applied: 'bg-emerald-900/90 text-emerald-200 border-emerald-700',
    undone: 'bg-blue-950/80 text-blue-300 border-blue-800/60',
  };

  return (
    <span
      className={`inline-flex items-center px-2.5 py-0.5 rounded text-xs font-semibold uppercase tracking-wider border ${
        styles[status] || styles.pending
      }`}
    >
      {status}
    </span>
  );
};
