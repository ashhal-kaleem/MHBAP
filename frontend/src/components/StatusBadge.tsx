import { clsx } from 'clsx'
import type { ConnectionStatus } from '@/types'

const MAP: Record<ConnectionStatus, { label: string; dot: string }> = {
  connecting: { label: 'Connecting…', dot: 'bg-yellow-400 animate-pulse' },
  open:       { label: 'Live',        dot: 'bg-green-400 animate-pulse' },
  closed:     { label: 'Offline',     dot: 'bg-gray-400' },
  error:      { label: 'Error',       dot: 'bg-red-500' },
}

export function StatusBadge({ status }: { status: ConnectionStatus }) {
  const { label, dot } = MAP[status]
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full bg-gray-800 px-3 py-1 text-xs font-medium text-gray-200">
      <span className={clsx('h-2 w-2 rounded-full', dot)} />
      {label}
    </span>
  )
}
