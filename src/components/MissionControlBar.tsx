import React from 'react';
import { Gauge, Navigation, Compass, Radio, Volume2, VolumeX, BarChart3, Wifi, Layers } from 'lucide-react';
import { RobotTelemetry } from '../types/simulation';
import { soundEffects } from '../services/soundEffects';

interface MissionControlBarProps {
  telemetry: RobotTelemetry;
  isConnected: boolean;
  isMuted: boolean;
  onToggleMute: () => void;
  onOpenBenchmark: () => void;
  isBaselineMode: boolean;
  onToggleBaseline: () => void;
}

export const MissionControlBar: React.FC<MissionControlBarProps> = ({
  telemetry,
  isConnected,
  isMuted,
  onToggleMute,
  onOpenBenchmark,
  isBaselineMode,
  onToggleBaseline
}) => {
  return (
    <header className="h-14 bg-[#0a0f16]/95 border-b border-gray-800 px-4 flex items-center justify-between text-xs backdrop-blur-md select-none z-30">
      {/* Brand & Project Identity */}
      <div className="flex items-center space-x-3">
        <div className="flex items-center space-x-2">
          <div className="w-7 h-7 rounded bg-gradient-to-tr from-cyan-600 to-cyan-400 flex items-center justify-center font-black text-black font-mono shadow-[0_0_12px_rgba(0,240,255,0.5)]">
            A
          </div>
          <div>
            <div className="flex items-center space-x-1.5">
              <span className="font-bold text-white tracking-widest text-sm font-mono">
                KRONOS-1M // ARGON
              </span>
              <span className="px-1.5 py-0.5 rounded text-[9px] font-mono bg-cyan-950 text-cyan-400 border border-cyan-800">
                MANAGED AGENTS + BREV GPU
              </span>
            </div>
            <p className="text-[10px] text-gray-400 tracking-tight font-mono">
              1M-CONTEXT ROBOTICS DATA ENGINE &amp; MULTIVERSE DPO ARENA
            </p>
          </div>
        </div>

        <div className="h-6 w-px bg-gray-800 mx-2" />

        <div className="flex items-center space-x-1.5 font-mono text-[11px] text-gray-300">
          <Wifi className={`w-3.5 h-3.5 ${isConnected ? 'text-emerald-400 animate-pulse' : 'text-amber-400'}`} />
          <span>{isConnected ? 'LIVE WS ORCHESTRATOR' : 'CLIENT KRONOS ENGINE'}</span>
        </div>
      </div>

      {/* Real-Time Rover Telemetry */}
      <div className="hidden lg:flex items-center space-x-6 font-mono text-[11px] bg-black/40 px-4 py-1.5 rounded-lg border border-gray-800/80">
        <div className="flex items-center space-x-1.5">
          <Gauge className="w-3.5 h-3.5 text-cyan-400" />
          <span className="text-gray-400">VELOCITY:</span>
          <span className="text-white font-bold">{telemetry.speedKmh.toFixed(1)} km/h</span>
        </div>

        <div className="flex items-center space-x-1.5">
          <Navigation className="w-3.5 h-3.5 text-cyan-400" />
          <span className="text-gray-400">TRAVERSED:</span>
          <span className="text-white font-bold">{telemetry.traversedDistanceMeters.toFixed(1)} m</span>
        </div>

        <div className="flex items-center space-x-1.5">
          <Compass className="w-3.5 h-3.5 text-cyan-400" />
          <span className="text-gray-400">CLEARANCE:</span>
          <span className={`font-bold ${telemetry.obstacleClearanceMeters < 1.5 ? 'text-red-400' : 'text-emerald-400'}`}>
            {telemetry.obstacleClearanceMeters.toFixed(2)} m
          </span>
        </div>

        <div className="flex items-center space-x-1.5">
          <Radio className="w-3.5 h-3.5 text-cyan-400" />
          <span className="text-gray-400">LiDAR HITS:</span>
          <span className="text-white font-bold">{telemetry.lidarRayHits} / 32 RAYS</span>
        </div>

        <div className="flex items-center space-x-1.5">
          <span className="text-gray-400">STATUS:</span>
          <span className="px-1.5 py-0.5 rounded text-[10px] uppercase font-bold bg-cyan-950 text-cyan-300 border border-cyan-800">
            {telemetry.autonomousState.replace('_', ' ')}
          </span>
        </div>
      </div>

      {/* Action Controls & Benchmark Mode */}
      <div className="flex items-center space-x-3">
        {/* Baseline Comparison Toggle */}
        <button
          onClick={onToggleBaseline}
          className={`px-3 py-1.5 rounded text-xs font-mono font-bold border transition flex items-center space-x-1.5 ${
            isBaselineMode
              ? 'bg-red-950/80 border-red-500 text-red-300 shadow-[0_0_12px_rgba(255,51,68,0.4)]'
              : 'bg-gray-900 border-gray-700 text-gray-300 hover:border-cyan-500 hover:text-cyan-400'
          }`}
          title="Toggle 64K Chunked Baseline comparison to demonstrate mesh tears"
        >
          <Layers className="w-3.5 h-3.5" />
          <span>{isBaselineMode ? 'MODE: 64K BASELINE (FRACTURED)' : 'MODE: ARGON 1M (SEAMLESS)'}</span>
        </button>

        {/* Benchmark Metric Modal Button */}
        <button
          onClick={onOpenBenchmark}
          className="px-3 py-1.5 bg-cyan-950/70 hover:bg-cyan-900 border border-cyan-700/80 text-cyan-300 rounded text-xs font-mono font-bold flex items-center space-x-1.5 transition shadow-[0_0_8px_rgba(0,240,255,0.25)]"
        >
          <BarChart3 className="w-3.5 h-3.5 text-cyan-400" />
          <span>METRICS</span>
        </button>

        {/* Audio Mute Toggle */}
        <button
          onClick={onToggleMute}
          className="p-2 bg-gray-900 border border-gray-700 hover:border-gray-500 text-gray-300 rounded transition"
          title={isMuted ? "Unmute Tactical SFX" : "Mute Audio"}
        >
          {isMuted ? <VolumeX className="w-4 h-4 text-gray-500" /> : <Volume2 className="w-4 h-4 text-cyan-400" />}
        </button>
      </div>
    </header>
  );
};

