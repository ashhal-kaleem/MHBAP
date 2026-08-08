import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer,
} from 'recharts'
import type { MetricSeries } from '@/types'

interface Props { data: MetricSeries[] }

export function TimeSeriesChart({ data }: Props) {
  return (
    <div className="rounded-2xl bg-white/80 backdrop-blur-sm border border-gray-100 shadow-sm p-5">
      <p className="mb-3 text-sm font-semibold text-gray-900">Signal History</p>
      <ResponsiveContainer width="100%" height={200}>
        <LineChart data={data} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis dataKey="time" tick={{ fill: '#6b7280', fontSize: 10 }}
            interval="preserveStartEnd" />
          <YAxis domain={[0, 1]} tick={{ fill: '#6b7280', fontSize: 10 }} />
          <Tooltip
            contentStyle={{ backgroundColor: '#ffffff', border: '1px solid #f3f4f6', borderRadius: '0.75rem', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
            labelStyle={{ color: '#374151', fontWeight: 600, marginBottom: '0.25rem' }}
            itemStyle={{ color: '#4b5563', fontSize: '0.875rem' }}
            formatter={(v: number) => `${Math.round(v * 100)}%`}
          />
          <Legend wrapperStyle={{ fontSize: 11, color: '#4b5563', paddingTop: '10px' }} />
          <Line type="monotone" dataKey="stress"     stroke="#f87171" dot={false} strokeWidth={2} />
          <Line type="monotone" dataKey="engagement" stroke="#4ade80" dot={false} strokeWidth={2} />
          <Line type="monotone" dataKey="attention"  stroke="#60a5fa" dot={false} strokeWidth={2} />
          <Line type="monotone" dataKey="fatigue"    stroke="#fb923c" dot={false} strokeWidth={2} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

