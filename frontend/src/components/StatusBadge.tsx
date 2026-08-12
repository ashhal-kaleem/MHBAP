import { clsx } from 'clsx'
import type { ConnectionStatus } from '@/types'

const MAP: Record<ConnectionStatus, { label: string; dot: string; ring: string; text: string }> = {
  connecting: { label: 'Connecting',  dot: 'bg-amber-400 animate-pulse', ring: 'border-amber-200 bg-amber-50',  text: 'text-amber-700' },
  open:       { label: 'Live',        dot: 'bg-green-400 animate-pulse', ring: 'border-green-200 bg-green-50',  text: 'text-green-700' },
  closed:     { label: 'Offline',     dot: 'bg-gray-300',                ring: 'border-gray-200 bg-gray-50',    text: 'text-gray-500' },
  error:      { label: 'Stream Error',dot: 'bg-red-500 animate-pulse',   ring: 'border-red-200 bg-red-50',      text: 'text-red-600' },
}

export function StatusBadge({ status }: { status: ConnectionStatus }) {
  const { label, dot, ring, text } = MAP[status]
  return (
    <span
      role="status"
      aria-label={`WebSocket status: ${label}`}
      className={clsx(
        'inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-semibold transition-colors',
        ring, text,
      )}
    >
      <span className={clsx('h-2 w-2 rounded-full', dot)} />
      {label}
    </span>
  )
}
