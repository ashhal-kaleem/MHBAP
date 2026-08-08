import { Activity, Camera, CameraOff, Mic, MicOff, Server } from 'lucide-react'

interface CapturePreviewProps {
  active: boolean
  onPermissionDenied?: (err: string) => void // Kept for compatibility with parent components
}

export function CapturePreview({ active }: CapturePreviewProps) {
  // Since the backend handles hardware capture natively (OpenCV/sounddevice),
  // we do not call navigator.mediaDevices.getUserMedia here to avoid hardware locking conflicts
  // and 'Camera access denied' errors on the host OS.

  return (
    <div className="rounded-2xl bg-white/80 backdrop-blur-sm border border-gray-100 shadow-sm overflow-hidden flex flex-col h-full">
      {/* Viewport placeholder */}
      <div className="relative bg-gray-900 aspect-video w-full overflow-hidden flex flex-col items-center justify-center">
        
        {/* Offline overlay */}
        {!active && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-gray-500">
            <Server className="h-8 w-8 opacity-50" />
            <span className="text-xs">No active session</span>
          </div>
        )}

        {/* Active backend processing animation */}
        {active && (
          <div className="absolute inset-0 flex flex-col items-center justify-center text-white/80 z-10">
            <div className="relative flex items-center justify-center h-16 w-16 mb-3">
              <div className="absolute inset-0 rounded-full border border-blue-500/30 animate-[ping_3s_cubic-bezier(0,0,0.2,1)_infinite]" />
              <div className="absolute inset-2 rounded-full border border-blue-400/40 animate-[ping_2s_cubic-bezier(0,0,0.2,1)_infinite_200ms]" />
              <div className="absolute inset-4 rounded-full border border-blue-300/50 animate-[ping_1s_cubic-bezier(0,0,0.2,1)_infinite_400ms]" />
              <Activity className="h-6 w-6 text-blue-400" />
            </div>
            <span className="text-sm font-medium tracking-wide">Processing Pipeline</span>
            <span className="text-[10px] text-white/50 mt-1 uppercase tracking-wider">Native Hardware Capture</span>
          </div>
        )}

        {/* Live badge */}
        {active && (
          <span className="absolute top-2 left-2 z-20 inline-flex items-center gap-1 rounded-full bg-blue-600/90 px-2 py-0.5 text-[10px] font-semibold text-white backdrop-blur-sm shadow-sm border border-blue-500/50">
            <span className="h-1.5 w-1.5 rounded-full bg-white animate-pulse" />
            LIVE
          </span>
        )}
      </div>

      {/* Status row */}
      <div className="flex items-center gap-4 px-4 py-3 bg-white mt-auto">
        <div className="flex items-center gap-1.5 text-xs">
          {active 
            ? <Camera className="h-3.5 w-3.5 text-blue-500" />
            : <CameraOff className="h-3.5 w-3.5 text-gray-400" />}
          <span className={active ? 'text-blue-600 font-medium' : 'text-gray-400'}>
            {active ? 'Backend Video' : 'Video Idle'}
          </span>
        </div>

        <div className="flex items-center gap-1.5 text-xs">
          {active 
            ? <Mic className="h-3.5 w-3.5 text-blue-500" />
            : <MicOff className="h-3.5 w-3.5 text-gray-400" />}
          <span className={active ? 'text-blue-600 font-medium' : 'text-gray-400'}>
            {active ? 'Backend Audio' : 'Audio Idle'}
          </span>
        </div>
      </div>
    </div>
  )
}
