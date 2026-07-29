import { useState } from 'react'
import Dashboard from '@/pages/Dashboard'
import AnalyticsPage from '@/pages/AnalyticsPage'

type Tab = 'dashboard' | 'analytics'

export default function App() {
  const [tab, setTab] = useState<Tab>('dashboard')

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Top nav */}
      <nav className="border-b border-gray-800 px-6 py-3 flex items-center gap-1">
        <span className="text-sm font-bold text-indigo-400 mr-6">MHBAP</span>
        {(['dashboard', 'analytics'] as Tab[]).map((t) => (
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
      </nav>

      {/* Page content */}
      {tab === 'dashboard' ? <Dashboard /> : <AnalyticsPage />}
    </div>
  )
}
