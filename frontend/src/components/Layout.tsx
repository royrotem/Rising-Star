import { Outlet, Link, useLocation } from 'react-router-dom';
import {
  LayoutDashboard,
  Cpu,
  Settings,
  Activity,
  Zap,
  ChevronRight,
} from 'lucide-react';
import clsx from 'clsx';

const navigation = [
  { name: 'Dashboard', href: '/', icon: LayoutDashboard },
  { name: 'Systems', href: '/systems', icon: Cpu },
  { name: 'Settings', href: '/settings', icon: Settings },
];

export default function Layout() {
  const location = useLocation();

  return (
    <div className="min-h-screen bg-surface-950">
      {/* Sidebar */}
      <aside className="fixed inset-y-0 left-0 w-64 bg-surface-900/95 backdrop-blur-sm border-r border-stone-800/80 z-30 flex flex-col">
        {/* Logo */}
        <div className="flex items-center gap-3 px-5 py-5 border-b border-stone-800/60">
          <div className="relative p-2.5 bg-gradient-to-br from-primary-500 to-primary-700 rounded-xl shadow-lg shadow-primary-500/20">
            <Zap className="w-5 h-5 text-white" />
            <div className="absolute -top-0.5 -right-0.5 w-2.5 h-2.5 bg-accent-400 rounded-full border-2 border-surface-900" />
          </div>
          <div>
            <h1 className="text-base font-bold text-white tracking-tight">UAIE</h1>
            <p className="text-[11px] text-stone-500 font-medium">Insight Engine</p>
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 px-3 py-4 space-y-1">
          {navigation.map((item) => {
            const isActive = location.pathname === item.href ||
              (item.href !== '/' && location.pathname.startsWith(item.href));

            return (
              <Link
                key={item.name}
                to={item.href}
                className={clsx(
                  'group flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all duration-200',
                  isActive
                    ? 'bg-primary-500/10 text-primary-400 shadow-sm shadow-primary-500/5'
                    : 'text-stone-400 hover:bg-stone-800/60 hover:text-stone-200'
                )}
              >
                <item.icon className={clsx(
                  'w-[18px] h-[18px] transition-colors',
                  isActive ? 'text-primary-400' : 'text-stone-500 group-hover:text-stone-300'
                )} />
                <span className="flex-1">{item.name}</span>
                {isActive && (
                  <ChevronRight className="w-3.5 h-3.5 text-primary-500/60" />
                )}
              </Link>
            );
          })}
        </nav>

        {/* Agent Status */}
        <div className="px-4 py-4 border-t border-stone-800/60">
          <div className="flex items-center gap-2.5">
            <div className="relative">
              <Activity className="w-4 h-4 text-accent-500" />
              <div className="absolute -top-0.5 -right-0.5 w-1.5 h-1.5 bg-accent-400 rounded-full animate-pulse" />
            </div>
            <span className="text-xs text-stone-500 font-medium">Agents Online</span>
          </div>
        </div>
      </aside>

      {/* Main content */}
      <main className="pl-64">
        <div className="min-h-screen">
          <Outlet />
        </div>
      </main>
    </div>
  );
}
