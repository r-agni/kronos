import React from 'react';
import { GitBranch, Cpu, Activity, Flame, Skull, Sparkles } from 'lucide-react';
import {
  AdversaryCritique,
  DenseProcessRewardTick,
  MultiverseForkEvent,
  RubricScoreCard,
  TemporalSegment,
} from '../types/simulation';

interface AdversaryHUDProps {
  lastCritique: AdversaryCritique | null;
  activeHazardsCount: number;
  threatLevel: string;
  onManualTriggerHazard?: (type: string) => void;
  onTriggerMultiverseFork?: () => void;
  recentTicks?: DenseProcessRewardTick[];
  currentSegment?: TemporalSegment | null;
  scoreCard?: RubricScoreCard;
  lastFork?: MultiverseForkEvent | null;
}

export const AdversaryHUD: React.FC<AdversaryHUDProps> = ({
  lastCritique,
  onManualTriggerHazard,
  onTriggerMultiverseFork,
  recentTicks = [],
  currentSegment,
  scoreCard,
  lastFork,
}) => {
  const outcomeScore = scoreCard?.overallOutcomeScore1to5 ?? 4.7;
  const compliance = scoreCard?.compliancePct ?? 96.4;

  return (
    <div className="bg-[#0b1017]/95 backdrop-blur-md border border-cyan-900/70 rounded-lg p-3 shadow-xl font-mono">
      <div className="flex items-center justify-between pb-2 mb-2 border-b border-gray-800 text-xs">
        <div className="flex items-center space-x-2">
          <Cpu className="w-4 h-4 text-cyan-400 animate-pulse" />
          <span className="font-bold tracking-wider text-cyan-300 uppercase">
            KRONOS-1M // TRAJECTORY REWARD &amp; DPO ENGINE
          </span>
        </div>
        <div className="flex items-center space-x-1.5">
          <span className="px-1.5 py-0.5 rounded text-[9px] font-bold bg-violet-950/80 text-violet-300 border border-violet-700/70">
            antigravity-preview-09-2026
          </span>
          <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-950/70 text-emerald-300 border border-emerald-700/60">
            OUTCOME: {outcomeScore.toFixed(1)}/5 ({compliance}%)
          </span>
        </div>
      </div>

      <div className="bg-black/50 border border-gray-800/90 rounded p-2 mb-2 text-[11px]">
        <div className="flex items-center justify-between text-[10px] text-gray-400 mb-0.5">
          <span className="text-cyan-400 font-bold">
            {currentSegment
              ? `${currentSegment.label} [${currentSegment.startSec.toFixed(1)}s–${currentSegment.endSec.toFixed(1)}s]`
              : 'SEG-01: CONTINUOUS CRUISE [0.0s–4.5s]'}
          </span>
          <span
            className={`px-1.5 py-0.2 rounded text-[9px] font-bold ${
              currentSegment?.stitchStatus === 'window_fractured'
                ? 'bg-red-950 text-red-400 border border-red-800'
                : 'bg-cyan-950 text-cyan-300 border border-cyan-800'
            }`}
          >
            {currentSegment?.stitchStatus === 'window_fractured'
              ? '64K STITCH FRACTURE'
              : '1M UNIFIED CONTEXT'}
          </span>
        </div>
        <p className="text-gray-200 truncate text-[11px]">
          &gt; {currentSegment?.instructionCaption || 'Maintain center-corridor alignment and monitor 32-beam LiDAR frontier.'}
        </p>
      </div>

      <div className="mb-2">
        <div className="flex items-center justify-between text-[10px] text-gray-400 mb-1">
          <span className="flex items-center space-x-1">
            <Activity className="w-3 h-3 text-emerald-400" />
            <span>0.5s DENSE PROCESS REWARDS (3D CHRONO-RIBBON):</span>
          </span>
          <span className="text-[9px] text-gray-400">
            <span className="text-emerald-400">■ NORMAL</span>{' '}
            <span className="text-amber-400">■ SUBOPTIMAL</span>{' '}
            <span className="text-red-400">■ FAILURE</span>
          </span>
        </div>
        <div className="flex items-center space-x-0.5 h-3 bg-gray-950 p-0.5 rounded border border-gray-800 overflow-hidden">
          {(recentTicks.length > 0 ? recentTicks.slice(-28) : Array.from({ length: 18 })).map(
            (tick: any, idx) => {
              const rating = tick?.rating || 'normal';
              const color =
                rating === 'normal'
                  ? 'bg-emerald-500'
                  : rating === 'suboptimal'
                    ? 'bg-amber-400'
                    : 'bg-red-500 animate-pulse';
              return (
                <div
                  key={tick?.id || idx}
                  className={`flex-1 h-full rounded-[1px] ${color}`}
                  title={tick ? `${tick.timestampSec}s: ${tick.reason}` : 'Nominal 0.5s step'}
                />
              );
            }
          )}
        </div>
      </div>

      {lastFork ? (
        <div className="bg-cyan-950/25 border border-cyan-700/50 rounded p-2 mb-2 text-[10px]">
          <div className="flex items-center justify-between text-cyan-300 font-bold mb-1">
            <span className="flex items-center space-x-1">
              <Sparkles className="w-3 h-3" />
              <span>4-WAY MULTIVERSE DPO FORK @ {lastFork.simTimeSec}s [thinking: HIGH]</span>
            </span>
            <span className="text-emerald-300">CHOSEN: GHOST Ω (4.9/5)</span>
          </div>
          <div className="grid grid-cols-4 gap-1 text-[9px]">
            {lastFork.branches.map(b => (
              <div
                key={b.branchId}
                className={`px-1.5 py-0.5 rounded border truncate ${
                  b.dpoRole === 'chosen'
                    ? 'bg-cyan-950/70 border-cyan-400 text-cyan-200 font-bold'
                    : 'bg-black/50 border-gray-800 text-gray-400'
                }`}
              >
                <div>{b.branchId.split('_')[0].toUpperCase()}: {b.outcomeScore1to5}/5</div>
                <div className="text-[8px] uppercase">{b.dpoRole}</div>
              </div>
            ))}
          </div>
        </div>
      ) : lastCritique ? (
        <div className="bg-red-950/20 border border-red-900/40 rounded p-1.5 mb-2 text-[10px] text-red-200 truncate">
          <span className="font-bold text-red-400 uppercase mr-1">
            {lastCritique.eavesdropIntercepted ? '[EAVESDROP TRAP]:' : '[STRESS EVENT]:'}
          </span>
          {lastCritique.actionPlan}
        </div>
      ) : null}

      <div className="pt-2 border-t border-gray-800/80 flex items-center justify-between text-[10px]">
        <button
          onClick={onTriggerMultiverseFork}
          className="px-2.5 py-1 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-black font-bold rounded flex items-center space-x-1.5 shadow-[0_0_12px_rgba(0,240,255,0.35)] transition"
        >
          <GitBranch className="w-3.5 h-3.5" />
          <span>⚡ FORK 4X MULTIVERSE DPO</span>
        </button>

        <div className="flex space-x-1.5">
          <button
            onClick={() => onManualTriggerHazard?.('chasm_rupture')}
            className="px-2 py-1 bg-red-900/40 hover:bg-red-800/60 border border-red-700/50 text-red-200 rounded flex items-center space-x-1 transition"
          >
            <Skull className="w-2.5 h-2.5" />
            <span>Inject Chasm</span>
          </button>
          <button
            onClick={() => onManualTriggerHazard?.('seismic_collapse')}
            className="px-2 py-1 bg-amber-900/40 hover:bg-amber-800/60 border border-amber-700/50 text-amber-200 rounded flex items-center space-x-1 transition"
          >
            <Flame className="w-2.5 h-2.5" />
            <span>Rockfall</span>
          </button>
        </div>
      </div>
    </div>
  );
};

