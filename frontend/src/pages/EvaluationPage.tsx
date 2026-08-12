import { useState } from 'react'
import { FlaskConical, ChevronDown, ChevronUp } from 'lucide-react'
import type { BenchmarkResponse, AblationStudyResponse, EvaluationReport } from '@/types/evaluation'
import { getBenchmark, getAblation } from '@/services/api'

function pct(v: number) { return `${(v * 100).toFixed(1)}%` }
function fmt(v: number) { return v.toFixed(4) }

function MetricBadge({ label, value, accent = false }: { label: string; value: string; accent?: boolean }) {
  return (
    <div className={`rounded-xl border p-3 text-center ${accent ? 'border-plum/20 bg-plum/5' : 'border-gray-100 bg-gray-50'}`}>
      <p className="text-[10px] font-semibold text-gray-400 uppercase tracking-wider mb-1">{label}</p>
      <p className={`text-xl font-bold ${accent ? 'text-plum' : 'text-gray-900'}`}>{value}</p>
    </div>
  )
}

function ReportCard({ report }: { report: EvaluationReport }) {
  const [open, setOpen] = useState(false)
  const isFusion = report.name === 'fusion'

  return (
    <div className={`rounded-2xl border p-5 shadow-sm transition-all ${isFusion ? 'border-plum/25 bg-plum/5 ring-1 ring-plum/10' : 'border-gray-100 bg-white'}`}>
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-bold capitalize text-gray-900">{report.name}</h3>
        {isFusion && (
          <span className="inline-flex items-center gap-1 text-xs bg-plum text-white px-2.5 py-1 rounded-lg font-semibold shadow-sm">
            <FlaskConical className="h-3 w-3" /> FUSION
          </span>
        )}
      </div>
      <div className="grid grid-cols-3 gap-2 mb-3">
        <MetricBadge label="Accuracy" value={pct(report.accuracy)} accent={isFusion} />
        <MetricBadge label="Macro-F1" value={pct(report.macro_f1)} accent={isFusion} />
        <MetricBadge label="κ (Kappa)" value={fmt(report.cohen_kappa)} />
      </div>
      <button
        onClick={() => setOpen(!open)}
        className="inline-flex items-center gap-1 text-xs font-semibold text-plum hover:text-plum-dark mt-1 transition-colors"
      >
        {open ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
        {open ? 'Hide per-class' : 'Show per-class'}
      </button>
      {open && (
        <div className="mt-4 overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-gray-400 border-b border-gray-100 font-semibold uppercase tracking-wider">
                <th className="text-left py-2">Label</th>
                <th className="py-2 text-center">Prec</th>
                <th className="py-2 text-center">Recall</th>
                <th className="py-2 text-center">F1</th>
                <th className="py-2 text-center">Support</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-50">
              {report.per_class.map((c) => (
                <tr key={c.label} className="text-gray-700 hover:bg-gray-50/60 transition-colors">
                  <td className="py-2 capitalize font-medium">{c.label}</td>
                  <td className="py-2 text-center tabular-nums">{pct(c.precision)}</td>
                  <td className="py-2 text-center tabular-nums">{pct(c.recall)}</td>
                  <td className="py-2 text-center tabular-nums font-semibold">{pct(c.f1)}</td>
                  <td className="py-2 text-center tabular-nums text-gray-400">{c.support}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

function RunButton({
  label,
  loading,
  onClick,
}: {
  label: string
  loading: boolean
  onClick: () => void
}) {
  return (
    <button
      onClick={onClick}
      disabled={loading}
      className="inline-flex items-center gap-2 text-sm bg-plum hover:bg-plum-dark disabled:opacity-50 text-white px-5 py-2.5 rounded-xl font-semibold shadow-sm transition-all"
    >
      {loading && (
        <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
      )}
      <FlaskConical className="h-4 w-4" />
      {loading ? 'Running…' : label}
    </button>
  )
}

export default function EvaluationPage() {
  const [benchmark, setBenchmark] = useState<BenchmarkResponse | null>(null)
  const [ablation, setAblation] = useState<AblationStudyResponse | null>(null)
  const [loading, setLoading] = useState<'bench' | 'ablation' | null>(null)
  const [error, setError] = useState<string | null>(null)

  async function fetchBenchmark() {
    setLoading('bench'); setError(null)
    try { const data = await getBenchmark(1000); setBenchmark(data) }
    catch (e: unknown) { setError(e instanceof Error ? e.message : 'Failed') }
    finally { setLoading(null) }
  }

  async function fetchAblation() {
    setLoading('ablation'); setError(null)
    try { const data = await getAblation(500); setAblation(data) }
    catch (e: unknown) { setError(e instanceof Error ? e.message : 'Failed') }
    finally { setLoading(null) }
  }

  return (
    <div className="min-h-screen bg-ivory text-gray-900 font-sans pb-12">
      <div className="max-w-5xl mx-auto px-4 sm:px-6 py-8 space-y-10">
        {/* Header */}
        <div>
          <h1 className="text-2xl font-bold text-gray-900 mb-1">Evaluation &amp; Benchmarks</h1>
          <p className="text-sm text-gray-400">Per-modality metrics, fusion comparison, and ablation study.</p>
        </div>

        {/* Error */}
        {error && (
          <div className="flex items-center gap-3 text-sm text-red-600 bg-red-50 border border-red-100 rounded-xl px-4 py-3 shadow-sm">
            <FlaskConical className="h-4 w-4 shrink-0" />
            {error}
          </div>
        )}

        {/* ── Benchmark ── */}
        <section>
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-6">
            <div>
              <h2 className="text-base font-semibold text-gray-900">Benchmark</h2>
              <p className="text-xs text-gray-400 mt-0.5">n = 1,000 samples — full modality comparison</p>
            </div>
            <RunButton label="Run Benchmark" loading={loading === 'bench'} onClick={fetchBenchmark} />
          </div>
          {!benchmark && loading !== 'bench' && (
            <div className="rounded-2xl border-2 border-dashed border-gray-200 bg-white/40 p-12 text-center text-gray-300 text-sm">
              Click <strong className="text-gray-400">Run Benchmark</strong> to evaluate all modalities.
            </div>
          )}
          {loading === 'bench' && (
            <div className="rounded-2xl border border-gray-100 bg-white p-12 flex flex-col items-center gap-3 text-gray-400 text-sm shadow-sm">
              <div className="w-6 h-6 border-4 border-plum/20 border-t-plum rounded-full animate-spin" />
              Running benchmark on 1,000 samples…
            </div>
          )}
          {benchmark && (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {benchmark.reports.map((r) => <ReportCard key={r.name} report={r} />)}
            </div>
          )}
        </section>

        {/* ── Ablation ── */}
        <section>
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-6">
            <div>
              <h2 className="text-base font-semibold text-gray-900">Ablation Study</h2>
              <p className="text-xs text-gray-400 mt-0.5">n = 500 samples — modality contribution analysis</p>
            </div>
            <RunButton label="Run Ablation" loading={loading === 'ablation'} onClick={fetchAblation} />
          </div>
          {!ablation && loading !== 'ablation' && (
            <div className="rounded-2xl border-2 border-dashed border-gray-200 bg-white/40 p-12 text-center text-gray-300 text-sm">
              Click <strong className="text-gray-400">Run Ablation</strong> to measure each modality's impact.
            </div>
          )}
          {loading === 'ablation' && (
            <div className="rounded-2xl border border-gray-100 bg-white p-12 flex flex-col items-center gap-3 text-gray-400 text-sm shadow-sm">
              <div className="w-6 h-6 border-4 border-plum/20 border-t-plum rounded-full animate-spin" />
              Running ablation on 500 samples…
            </div>
          )}
          {ablation && (
            <div className="bg-white border border-gray-100 shadow-sm rounded-2xl p-6">
              <p className="text-sm text-gray-500 mb-5">
                Baseline Macro-F1 (all modalities):{' '}
                <span className="text-plum font-bold ml-1">{pct(ablation.baseline_f1)}</span>
              </p>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-xs text-gray-400 font-semibold uppercase tracking-wider border-b border-gray-100">
                      <th className="text-left py-3 pr-4">Active Modalities</th>
                      <th className="text-left py-3 pr-4">Dropped</th>
                      <th className="py-3 px-2 text-center">Accuracy</th>
                      <th className="py-3 px-2 text-center">Macro-F1</th>
                      <th className="py-3 px-2 text-center">Δ F1</th>
                      <th className="py-3 px-2 text-center">κ</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-50">
                    {ablation.results
                      .sort((a, b) => b.macro_f1 - a.macro_f1)
                      .map((r) => {
                        const delta = r.macro_f1 - ablation.baseline_f1
                        const isBaseline = r.dropped_modalities.length === 0
                        return (
                          <tr
                            key={r.active_modalities.join('+')}
                            className={`transition-colors text-gray-700 ${isBaseline ? 'bg-plum/3' : 'hover:bg-gray-50/60'}`}
                          >
                            <td className="py-3 pr-4 font-medium">{r.active_modalities.join(', ')}</td>
                            <td className="py-3 pr-4 text-gray-400">{r.dropped_modalities.join(', ') || '—'}</td>
                            <td className="py-3 px-2 text-center tabular-nums">{pct(r.accuracy)}</td>
                            <td className="py-3 px-2 text-center tabular-nums font-semibold">{pct(r.macro_f1)}</td>
                            <td className={`py-3 px-2 text-center font-bold tabular-nums ${delta < -0.005 ? 'text-red-500' : delta > 0.005 ? 'text-green-500' : 'text-gray-400'}`}>
                              {delta >= 0 ? '+' : ''}{pct(delta)}
                            </td>
                            <td className="py-3 px-2 text-center tabular-nums text-gray-400">{fmt(r.cohen_kappa)}</td>
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
