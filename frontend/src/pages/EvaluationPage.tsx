import { useState } from 'react'
import type { BenchmarkResponse, AblationStudyResponse, EvaluationReport } from '@/types/evaluation'
import { getBenchmark, getAblation } from '@/services/api'

function pct(v: number) { return `${(v * 100).toFixed(1)}%` }
function fmt(v: number) { return v.toFixed(4) }

function Badge({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-gray-50 border border-gray-100 rounded-xl p-3 text-center shadow-sm">
      <div className="text-xs text-gray-500 mb-1 font-medium">{label}</div>
      <div className="text-lg font-bold text-plum">{value}</div>
    </div>
  )
}

function ReportCard({ report }: { report: EvaluationReport }) {
  const [open, setOpen] = useState(false)
  const isFusion = report.name === 'fusion'
  return (
    <div className={`rounded-2xl border p-5 shadow-sm transition-all ${isFusion ? 'border-plum bg-plum/5' : 'border-gray-200 bg-white'}`}>
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-semibold capitalize text-gray-900 text-lg">{report.name}</h3>
        {isFusion && <span className="text-xs bg-plum text-white px-2 py-1 rounded-md font-medium shadow-sm">FUSION</span>}
      </div>
      <div className="grid grid-cols-3 gap-2 mb-3">
        <Badge label="Accuracy" value={pct(report.accuracy)} />
        <Badge label="Macro-F1" value={pct(report.macro_f1)} />
        <Badge label="κ (Kappa)" value={fmt(report.cohen_kappa)} />
      </div>
      <button onClick={() => setOpen(!open)} className="text-xs font-medium text-plum hover:text-plum-dark mt-2 transition-colors">
        {open ? '▲ Hide per-class' : '▼ Show per-class'}
      </button>
      {open && (
        <table className="w-full mt-4 text-xs">
          <thead>
            <tr className="text-gray-500 border-b border-gray-200">
              <th className="text-left py-1">Label</th>
              <th>Prec</th><th>Rec</th><th>F1</th><th>Sup</th>
            </tr>
          </thead>
          <tbody>
            {report.per_class.map(c => (
              <tr key={c.label} className="border-b border-gray-100 text-gray-700">
                <td className="py-2 capitalize">{c.label}</td>
                <td className="py-2 text-center">{pct(c.precision)}</td>
                <td className="py-2 text-center">{pct(c.recall)}</td>
                <td className="py-2 text-center">{pct(c.f1)}</td>
                <td className="py-2 text-center">{c.support}</td>
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
      const data = await getBenchmark(1000)
      setBenchmark(data)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed')
    } finally { setLoading(null) }
  }

  async function fetchAblation() {
    setLoading('ablation'); setError(null)
    try {
      const data = await getAblation(500)
      setAblation(data)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed')
    } finally { setLoading(null) }
  }

  return (
    <div className="min-h-screen bg-ivory text-gray-900 font-sans pb-12">
      <div className="max-w-5xl mx-auto px-6 py-8 space-y-10">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 mb-1">Evaluation &amp; Benchmarks</h1>
          <p className="text-sm text-gray-500">Per-modality metrics, fusion comparison, and ablation study.</p>
        </div>

        {error && <div className="text-red-700 text-sm bg-red-50 border border-red-100 rounded-xl p-4 shadow-sm">{error}</div>}

        {/* Benchmark section */}
        <section>
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-lg font-semibold text-gray-900">Benchmark (n=1 000)</h2>
            <button
              onClick={fetchBenchmark}
              disabled={loading === 'bench'}
              className="text-sm bg-plum hover:bg-plum-dark disabled:opacity-50 text-white px-4 py-2 rounded-xl transition-all shadow-sm font-medium flex items-center gap-2"
            >
              {loading === 'bench' && <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />}
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
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-lg font-semibold text-gray-900">Ablation Study (n=500)</h2>
          <button
            onClick={fetchAblation}
            disabled={loading === 'ablation'}
            className="text-sm bg-plum hover:bg-plum-dark disabled:opacity-50 text-white px-4 py-2 rounded-xl transition-all shadow-sm font-medium flex items-center gap-2"
          >
            {loading === 'ablation' && <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />}
            {loading === 'ablation' ? 'Running…' : 'Run ablation'}
          </button>
        </div>
        {ablation && (
          <div className="bg-white/80 backdrop-blur-sm border border-gray-100 shadow-sm rounded-2xl p-6">
            <p className="text-sm text-gray-500 mb-6">
              Baseline (all modalities) Macro-F1: <span className="text-plum font-bold ml-1">{pct(ablation.baseline_f1)}</span>
            </p>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-gray-500 border-b border-gray-200">
                    <th className="text-left py-3 pr-4">Active modalities</th>
                    <th className="text-left py-3 pr-4">Dropped</th>
                    <th className="py-3 px-2">Accuracy</th>
                    <th className="py-3 px-2">Macro-F1</th>
                    <th className="py-3 px-2">Δ F1</th>
                    <th className="py-3 px-2">κ</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100">
                  {ablation.results
                    .sort((a, b) => b.macro_f1 - a.macro_f1)
                    .map(r => {
                      const delta = r.macro_f1 - ablation.baseline_f1
                      return (
                        <tr key={r.active_modalities.join('+')} className="hover:bg-gray-50/80 transition-colors text-gray-700">
                          <td className="py-3 pr-4">{r.active_modalities.join(', ')}</td>
                          <td className="py-3 pr-4 text-gray-400">{r.dropped_modalities.join(', ') || '—'}</td>
                          <td className="py-3 px-2 text-center">{pct(r.accuracy)}</td>
                          <td className="py-3 px-2 text-center">{pct(r.macro_f1)}</td>
                          <td className={`py-3 px-2 text-center font-medium ${delta < 0 ? 'text-red-500' : 'text-green-500'}`}>
                            {delta >= 0 ? '+' : ''}{pct(delta)}
                          </td>
                          <td className="py-3 px-2 text-center text-gray-500">{fmt(r.cohen_kappa)}</td>
                        </tr>
                      )
                    })}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </section>
      </div>
    </div>
  )
}
