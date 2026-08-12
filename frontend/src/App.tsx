import { useEffect, useState } from 'react'
import Dashboard from '@/pages/Dashboard'
import AnalyticsPage from '@/pages/AnalyticsPage'
import EvaluationPage from '@/pages/EvaluationPage'
import LandingPage from '@/pages/LandingPage'
import LoginPage from '@/pages/LoginPage'
import SignupPage from '@/pages/SignupPage'
import { logout } from '@/services/api'
import { useCurrentUser } from '@/hooks/useCurrentUser'
import { Brain, BarChart2, LogOut, LayoutDashboard } from 'lucide-react'

type Tab = 'landing' | 'login' | 'signup' | 'dashboard' | 'analytics' | 'evaluation'

// Evaluation is a researcher/developer tool (model benchmarks, ablation).
// It is intentionally excluded from the user-facing nav but remains accessible
// as a route for research use. To open it: click the version text in the nav.
const NAV_TABS: { id: Tab; label: string; Icon: React.FC<{ className?: string }> }[] = [
  { id: 'dashboard', label: 'Dashboard', Icon: LayoutDashboard },
  { id: 'analytics', label: 'Analytics', Icon: BarChart2 },
]

export default function App() {
  const { user, loading } = useCurrentUser()
  const [tab, setTab] = useState<Tab>('landing')

  useEffect(() => {
    if (!loading) {
      if (user && (tab === 'landing' || tab === 'login' || tab === 'signup')) {
        setTab('dashboard')
      } else if (!user && (tab === 'dashboard' || tab === 'analytics' || tab === 'evaluation')) {
        setTab('login')
      }
    }
  }, [user, loading, tab])

  if (loading) {
    return (
      <div className="min-h-screen bg-ivory flex items-center justify-center">
        <div className="flex flex-col items-center gap-4">
          <div className="w-10 h-10 border-4 border-plum/20 border-t-plum rounded-full animate-spin" />
          <p className="text-sm text-gray-500 font-medium">Loading…</p>
        </div>
      </div>
    )
  }

  if (tab === 'landing') return <LandingPage onGetStarted={() => setTab('signup')} onLogin={() => setTab('login')} />
  if (tab === 'login')   return <LoginPage onBack={() => setTab('landing')} onLoginSuccess={() => window.location.reload()} onNavigateSignup={() => setTab('signup')} />
  if (tab === 'signup')  return <SignupPage onBack={() => setTab('landing')} onSignupSuccess={() => window.location.reload()} onNavigateLogin={() => setTab('login')} />

  return (
    <div className="min-h-screen bg-ivory text-gray-900 flex flex-col">
      {/* ── Top navigation ── */}
      <nav className="sticky top-0 z-50 bg-white/80 backdrop-blur-md border-b border-gray-200 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 h-14 flex items-center gap-1">
          {/* Logo — double-click to open Evaluation (researcher tool) */}
          <div
            className="flex items-center gap-2 mr-5 cursor-default select-none"
            onDoubleClick={() => setTab('evaluation')}
            title="Double-click for research tools"
          >
            <Brain className="h-5 w-5 text-plum" />
            <span className="text-sm font-bold tracking-tight text-gray-900">MHBAP</span>
          </div>

          {/* Page tabs */}
          <div className="flex items-center gap-0.5 flex-1">
            {NAV_TABS.map(({ id, label, Icon }) => (
              <button
                key={id}
                onClick={() => setTab(id)}
                aria-current={tab === id ? 'page' : undefined}
                className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                  tab === id
                    ? 'bg-plum text-white shadow-sm'
                    : 'text-gray-500 hover:text-gray-900 hover:bg-gray-100'
                }`}
              >
                <Icon className="h-3.5 w-3.5" />
                <span className="hidden sm:inline">{label}</span>
              </button>
            ))}
          </div>

          {/* User + logout */}
          <div className="flex items-center gap-3 ml-auto">
            {user && (
              <span className="hidden md:block text-xs text-gray-400 truncate max-w-[180px]">
                {user.email}
              </span>
            )}
            <button
              onClick={async () => { try { await logout() } finally { window.location.reload() } }}
              title="Log out"
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium text-gray-500 hover:text-red-600 hover:bg-red-50 transition-all"
            >
              <LogOut className="h-3.5 w-3.5" />
              <span className="hidden sm:inline">Logout</span>
            </button>
          </div>
        </div>
      </nav>

      {/* ── Page content ── */}
      <div className="flex-1">
        {tab === 'dashboard'  && <Dashboard user={user} />}
        {tab === 'analytics'  && <AnalyticsPage />}
        {tab === 'evaluation' && <EvaluationPage />}
      </div>
    </div>
  )
}
