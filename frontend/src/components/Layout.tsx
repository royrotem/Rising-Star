import { Outlet, Link, useLocation } from 'react-router-dom';
import {
  LayoutDashboard,
  Cpu,
  Settings,
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
    <div className="min-h-screen">
      {/* Sidebar */}
      <aside className="fixed inset-y-0 left-0 w-56 bg-stone-900 border-r border-stone-600/50 z-30 flex flex-col">
        {/* Logo */}
        <div className="px-5 py-6">
          <h1 className="text-sm font-semibold text-white tracking-wide">UAIE</h1>
          <p className="text-[11px] text-stone-400 mt-0.5">Insight Engine</p>
        </div>

        {/* Navigation */}
        <nav className="flex-1 px-3 space-y-0.5">
          {navigation.map((item) => {
            const isActive = location.pathname === item.href ||
              (item.href !== '/' && location.pathname.startsWith(item.href));

            return (
              <Link
                key={item.name}
                to={item.href}
                className={clsx(
                  'flex items-center gap-2.5 px-3 py-2 rounded-lg text-[13px] font-medium transition-colors duration-150',
                  isActive
                    ? 'bg-stone-700 text-white'
                    : 'text-stone-300 hover:text-stone-100 hover:bg-stone-700/60'
                )}
              >
                <item.icon className={clsx(
                  'w-4 h-4',
                  isActive ? 'text-primary-400' : 'text-stone-400'
                )} />
                {item.name}
              </Link>
            );
          })}
        </nav>

        {/* Footer */}
        <div className="px-5 py-4 border-t border-stone-600/40">
          <p className="text-[10px] text-stone-500">v1.0</p>
        </div>
      </aside>

      {/* Main content */}
      <main className="pl-56">
        <div className="min-h-screen">
          <Outlet />
        </div>
      </main>
    </div>
  );
}
