import { useState, useEffect } from 'react'
import Dashboard from '@/pages/Dashboard'
import AnalyticsPage from '@/pages/AnalyticsPage'
import EvaluationPage from '@/pages/EvaluationPage'
import LandingPage from '@/pages/LandingPage'
import LoginPage from '@/pages/LoginPage'
import SignupPage from '@/pages/SignupPage'
import { logout } from '@/services/api'
import { useCurrentUser } from '@/hooks/useCurrentUser'

type Tab = 'landing' | 'login' | 'signup' | 'dashboard' | 'analytics' | 'evaluation'

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
      <div className="min-h-screen bg-gray-900 flex items-center justify-center text-white">
        <div className="w-8 h-8 border-4 border-plum/30 border-t-plum rounded-full animate-spin" />
      </div>
    )
  }

  if (tab === 'landing') {
    return (
      <LandingPage 
        onGetStarted={() => setTab('signup')} 
        onLogin={() => setTab('login')}
      />
    )
  }

  if (tab === 'login') {
    return (
      <LoginPage 
        onBack={() => setTab('landing')} 
        onLoginSuccess={() => window.location.reload()} 
        onNavigateSignup={() => setTab('signup')}
      />
    )
  }

  if (tab === 'signup') {
    return (
      <SignupPage 
        onBack={() => setTab('landing')} 
        onSignupSuccess={() => window.location.reload()} 
        onNavigateLogin={() => setTab('login')}
      />
    )
  }

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Top nav */}
      <nav className="border-b border-gray-800 px-6 py-3 flex items-center gap-1">
        <span className="text-sm font-bold text-indigo-400 mr-6">MHBAP</span>
        {(['dashboard', 'analytics', 'evaluation'] as Tab[]).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`text-sm px-4 py-1.5 rounded capitalize transition-colors ${
              tab === t
                ? 'bg-indigo-600 text-white'
                : 'text-gray-400 hover:text-white hover:bg-gray-800'
            }`}
          >{t}</button>
        ))}
        
        {/* Simple logout button in nav */}
        <button
          onClick={async () => {
            await logout();
            window.location.reload();
          }}
          className="ml-auto text-sm px-4 py-1.5 rounded text-gray-400 hover:text-white hover:bg-gray-800 transition-colors"
        >
          Logout
        </button>
      </nav>

      {/* Page content */}
      {tab === 'dashboard' ? <Dashboard user={user} /> : tab === 'analytics' ? <AnalyticsPage /> : <EvaluationPage />}
    </div>
  )
}
