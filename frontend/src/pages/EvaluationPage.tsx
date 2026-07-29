import { useState } from 'react'
import type { BenchmarkResponse, AblationStudyResponse, EvaluationReport } from '@/types/evaluation'

const API = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

function pct(v: number) { return `${(v * 100).toFixed(1)}%` }
function fmt(v: number) { return v.toFixed(4) }

function Badge({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-gray-700 rounded-lg p-3 text-center">
      <div className="text-xs text-gray-400 mb-1">{label}</div>
      <div className="text-lg font-bold text-indigo-300">{value}</div>
    </div>
  )
}

function ReportCard({ report }: { report: EvaluationReport }) {
  const [open, setOpen] = useState(false)
  const isFusion = report.name === 'fusion'
  return (
    <div className={`rounded-xl border p-4 ${isFusion ? 'border-indigo-500 bg-indigo-900/20' : 'border-gray-700 bg-gray-800'}`}>
      <div className="flex items-center justify-between mb-3">
        <h3 className="font-semibold capitalize text-white">{report.name}</h3>
        {isFusion && <span className="text-xs bg-indigo-600 text-white px-2 py-0.5 rounded">FUSION</span>}
      </div>
      <div className="grid grid-cols-3 gap-2 mb-3">
        <Badge label="Accuracy" value={pct(report.accuracy)} />
        <Badge label="Macro-F1" value={pct(report.macro_f1)} />
        <Badge label="κ (Kappa)" value={fmt(report.cohen_kappa)} />
      </div>
      <button onClick={() => setOpen(!open)} className="text-xs text-indigo-400 hover:text-indigo-300">
        {open ? '▲ Hide per-class' : '▼ Show per-class'}
      </button>
      {open && (
        <table className="w-full mt-2 text-xs">
          <thead>
            <tr className="text-gray-400 border-b border-gray-700">
              <th className="text-left py-1">Label</th>
              <th>Prec</th><th>Rec</th><th>F1</th><th>Sup</th>
            </tr>
          </thead>
          <tbody>
            {report.per_class.map(c => (
              <tr key={c.label} className="border-b border-gray-700/50 text-gray-300">
                <td className="py-1 capitalize">{c.label}</td>
                <td className="text-center">{pct(c.precision)}</td>
                <td className="text-center">{pct(c.recall)}</td>
                <td className="text-center">{pct(c.f1)}</td>
                <td className="text-center">{c.support}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}

export default function EvaluationPage() {
  const [benchmark, setBenchmark] = useState<BenchmarkResponse | null>(null)
  const [ablation, setAblation] = useState<AblationStudyResponse | null>(null)
  const [loading, setLoading] = useState<'bench' | 'ablation' | null>(null)
  const [error, setError] = useState<string | null>(null)

  async function fetchBenchmark() {
    setLoading('bench'); setError(null)
    try {
      const res = await fetch(`${API}/api/v1/evaluation/benchmark?n_samples=1000`)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      setBenchmark(await res.json())
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed')
    } finally { setLoading(null) }
  }

  async function fetchAblation() {
    setLoading('ablation'); setError(null)
    try {
      const res = await fetch(`${API}/api/v1/evaluation/ablation?n_samples=500`)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      setAblation(await res.json())
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed')
    } finally { setLoading(null) }
  }

  return (
    <div className="max-w-5xl mx-auto px-6 py-8 space-y-10">
      <div>
        <h1 className="text-xl font-bold text-white mb-1">Evaluation &amp; Benchmarks</h1>
        <p className="text-sm text-gray-400">Per-modality metrics, fusion comparison, and ablation study.</p>
      </div>

      {error && <div className="text-red-400 text-sm bg-red-900/20 rounded p-3">{error}</div>}

      {/* Benchmark section */}
      <section>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-base font-semibold text-gray-200">Benchmark (n=1 000)</h2>
          <button
            onClick={fetchBenchmark}
            disabled={loading === 'bench'}
            className="text-sm bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white px-4 py-1.5 rounded transition-colors"
          >
            {loading === 'bench' ? 'Running…' : 'Run benchmark'}
          </button>
        </div>
        {benchmark && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {benchmark.reports.map(r => <ReportCard key={r.name} report={r} />)}
          </div>
        )}
      </section>

      {/* Ablation section */}
      <section>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-base font-semibold text-gray-200">Ablation Study (n=500)</h2>
          <button
            onClick={fetchAblation}
            disabled={loading === 'ablation'}
            className="text-sm bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white px-4 py-1.5 rounded transition-colors"
          >
            {loading === 'ablation' ? 'Running…' : 'Run ablation'}
          </button>
        </div>
        {ablation && (
          <div className="space-y-2">
            <p className="text-xs text-gray-400 mb-3">
              Baseline (all modalities) Macro-F1: <span className="text-indigo-300 font-bold">{pct(ablation.baseline_f1)}</span>
            </p>
            <table className="w-full text-xs">
              <thead>
                <tr className="text-gray-400 border-b border-gray-700">
                  <th className="text-left py-2">Active modalities</th>
                  <th className="text-left">Dropped</th>
                  <th>Accuracy</th>
                  <th>Macro-F1</th>
                  <th>Δ F1</th>
                  <th>κ</th>
                </tr>
              </thead>
              <tbody>
                {ablation.results
                  .sort((a, b) => b.macro_f1 - a.macro_f1)
                  .map(r => {
                    const delta = r.macro_f1 - ablation.baseline_f1
                    return (
                      <tr key={r.active_modalities.join('+')} className="border-b border-gray-700/50 text-gray-300">
                        <td className="py-1.5">{r.active_modalities.join(', ')}</td>
                        <td className="text-gray-500">{r.dropped_modalities.join(', ') || '—'}</td>
                        <td className="text-center">{pct(r.accuracy)}</td>
                        <td className="text-center">{pct(r.macro_f1)}</td>
                        <td className={`text-center ${delta < 0 ? 'text-red-400' : 'text-green-400'}`}>
                          {delta >= 0 ? '+' : ''}{pct(delta)}
                        </td>
                        <td className="text-center">{fmt(r.cohen_kappa)}</td>
                      </tr>
                    )
                  })}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  )
}
