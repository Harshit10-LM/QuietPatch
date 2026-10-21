import React from 'react';
import {
  CheckCircle2,
  Clock,
  FileSearch,
  FolderGit2,
  GitPullRequest,
  LayoutDashboard,
  Settings,
  ShieldCheck,
} from 'lucide-react';

export type TabKey =
  | 'overview'
  | 'repositories'
  | 'runs'
  | 'findings'
  | 'queue'
  | 'reports'
  | 'settings';

interface SidebarProps {
  activeTab: TabKey;
  onSelectTab: (tab: TabKey) => void;
  pendingPatchesCount: number;
  totalFindingsCount: number;
}

export const Sidebar: React.FC<SidebarProps> = ({
  activeTab,
  onSelectTab,
  pendingPatchesCount,
  totalFindingsCount,
}) => {
  const navItems: Array<{
    key: TabKey;
    label: string;
    icon: React.ComponentType<{ className?: string }>;
    badge?: number;
  }> = [
    { key: 'overview', label: 'Overview', icon: LayoutDashboard },
    { key: 'repositories', label: 'Repositories', icon: FolderGit2 },
    { key: 'runs', label: 'Maintenance Runs', icon: Clock },
    { key: 'findings', label: 'Findings', icon: FileSearch, badge: totalFindingsCount },
    { key: 'queue', label: 'Fix Queue', icon: GitPullRequest, badge: pendingPatchesCount },
    { key: 'reports', label: 'Reports', icon: CheckCircle2 },
    { key: 'settings', label: 'Settings', icon: Settings },
  ];

  return (
    <aside className="w-64 border-r border-[#1f2635] bg-[#0b0e14] flex flex-col justify-between p-4 shrink-0">
      <div className="space-y-1.5">
        <div className="px-3 py-2 text-[11px] font-semibold tracking-wider text-slate-500 uppercase">
          Navigation
        </div>
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeTab === item.key;
          return (
            <button
              key={item.key}
              onClick={() => onSelectTab(item.key)}
              className={`w-full flex items-center justify-between px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                isActive
                  ? 'bg-emerald-950/60 text-emerald-300 border border-emerald-800/50 shadow-sm'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-[#131722]'
              }`}
            >
              <div className="flex items-center space-x-3">
                <Icon className={`w-4 h-4 ${isActive ? 'text-emerald-400' : 'text-slate-500'}`} />
                <span>{item.label}</span>
              </div>
              {item.badge !== undefined && item.badge > 0 && (
                <span
                  className={`text-xs px-2 py-0.5 rounded-full font-mono font-medium ${
                    item.key === 'queue'
                      ? 'bg-amber-900/60 text-amber-300 border border-amber-700/50'
                      : 'bg-[#1b2230] text-slate-300 border border-[#2a354b]'
                  }`}
                >
                  {item.badge}
                </span>
              )}
            </button>
          );
        })}
      </div>

      {/* Safety Boundary Footer Box */}
      <div className="p-3.5 rounded-xl bg-[#111622] border border-[#20293a] space-y-2">
        <div className="flex items-center space-x-2 text-xs font-semibold text-slate-300">
          <ShieldCheck className="w-4 h-4 text-emerald-400" />
          <span>Local-First Safety</span>
        </div>
        <p className="text-[11px] text-slate-400 leading-relaxed">
          Patches require explicit user review. No automatic git commits or unverified writes.
        </p>
      </div>
    </aside>
  );
};
