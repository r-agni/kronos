import React from 'react';
import {
  X,
  CheckCircle,
  AlertOctagon,
  Cpu,
  Download,
  Rocket,
  Terminal,
  GitBranch,
} from 'lucide-react';
import {
  BaselineMetricComparison,
  BrevGpuJobStatus,
  RobotTelemetry,
} from '../types/simulation';
import {
  kronosEngine,
  runKronosBenchmarkSuite,
  runLiveHuggingFaceBenchmark,
  exportBenchmarkReportMarkdown,
  type BenchmarkSuiteReport,
} from '../engine';

interface BenchmarkModalProps {
  isOpen: boolean;
  onClose: () => void;
  metrics: BaselineMetricComparison;
  telemetry?: RobotTelemetry;
  brevJob?: BrevGpuJobStatus | null;
  onTriggerBrevTraining?: () => void;
}

function downloadFile(filename: string, content: string, mimeType: string) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

export const BenchmarkModal: React.FC<BenchmarkModalProps> = ({
  isOpen,
  onClose,
  metrics,
  telemetry,
  brevJob,
  onTriggerBrevTraining,
}) => {
  const [benchReport, setBenchReport] = React.useState<BenchmarkSuiteReport>(() =>
    runKronosBenchmarkSuite()
  );

  if (!isOpen) return null;

  const activeTelemetry: RobotTelemetry = telemetry || {
    position: [0, 0.8, 40],
    velocity: [0, 0, 8.5],
    speedKmh: 30.6,
    headingDeg: 0,
    pitchDeg: 0,
    rollDeg: 0,
    traversedDistanceMeters: 40,
    obstacleClearanceMeters: 18,
    structuralIntegrityPct: 100,
    lidarRayHits: 28,
    autonomousState: 'navigating',
    targetWaypoint: [0, 0, 85],
    activeMorphology: 'thruster_hopper',
    physicsRegime: 'Earth 1.0g Nominal',
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-md flex items-center justify-center p-4">
      <div className="bg-[#0b1017] border border-cyan-800/80 rounded-xl max-w-3xl w-full p-6 shadow-2xl relative overflow-hidden font-mono">
        {/* Glow Accent */}
        <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-cyan-500 via-blue-500 to-purple-500" />

        {/* Close Button */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 p-1.5 text-gray-400 hover:text-white rounded-lg hover:bg-gray-800 transition"
        >
          <X className="w-5 h-5" />
        </button>

        {/* Header */}
        <div className="mb-6">
          <div className="flex items-center space-x-2 text-cyan-400 text-xs font-bold uppercase tracking-wider mb-1">
            <Cpu className="w-4 h-4" />
            <span>Problem Statement One // Empirical Validation</span>
          </div>
          <h2 className="text-xl font-bold text-white tracking-wide">
            TOPOLOGICAL COHERENCE BENCHMARK
          </h2>
          <p className="text-xs text-gray-400 mt-1">
            Measuring continuous procedural 3D world synthesis: <span className="text-cyan-400 font-bold">Gemini 4 Argon (1M Token Unified Context)</span> versus <span className="text-red-400 font-bold">64K-at-a-time Chunked Baseline</span>.
          </p>
        </div>

        {/* Side-by-Side Comparison Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
          {/* Argon 1M Card */}
          <div className="bg-[#0e1622] border border-cyan-500/50 rounded-lg p-4 relative shadow-[0_0_20px_rgba(0,240,255,0.1)]">
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs font-bold text-cyan-400 flex items-center space-x-1.5">
                <CheckCircle className="w-4 h-4" />
                <span>GEMINI 4 ARGON (1M OUTPUT)</span>
              </span>
              <span className="text-[10px] bg-cyan-950 text-cyan-300 px-2 py-0.5 rounded border border-cyan-800">
                PROPOSED SOTA
              </span>
            </div>

            <div className="space-y-3 text-xs">
              <div>
                <div className="flex justify-between text-gray-400 mb-1">
                  <span>Topological Coherence:</span>
                  <span className="text-cyan-300 font-bold">{metrics.argonCoherencePct.toFixed(1)}%</span>
                </div>
                <div className="w-full bg-gray-900 h-2 rounded-full overflow-hidden">
                  <div
                    className="bg-cyan-400 h-full rounded-full transition-all duration-500"
                    style={{ width: `${metrics.argonCoherencePct}%` }}
                  />
                </div>
              </div>

              <div className="flex justify-between py-1.5 border-b border-gray-800/80 text-gray-300">
                <span className="text-gray-400">Continuous Output Tokens:</span>
                <span className="font-bold text-white">{metrics.argonTokensStreamed.toLocaleString()} tok</span>
              </div>

              <div className="flex justify-between py-1.5 border-b border-gray-800/80 text-gray-300">
                <span className="text-gray-400">Boundary Seam Disconnects:</span>
                <span className="font-bold text-emerald-400">{metrics.argonSeamFaults} FAULTS (0.00%)</span>
              </div>

              <div className="flex justify-between py-1.5 border-b border-gray-800/80 text-gray-300">
                <span className="text-gray-400">Physics Collider Continuity:</span>
                <span className="font-bold text-emerald-400">{metrics.colliderContinuityPct.toFixed(1)}%</span>
              </div>

              <div className="pt-2 text-[11px] text-gray-400 leading-relaxed">
                ✓ Unified spatial coordinate frame maintained across miles of generation.
                <br />
                ✓ Smooth elevation manifolds with zero cliff drops or inverted normal vertices.
              </div>
            </div>
          </div>

          {/* 64K Chunked Baseline Card */}
          <div className="bg-[#140f13] border border-red-700/50 rounded-lg p-4 relative">
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs font-bold text-red-400 flex items-center space-x-1.5">
                <AlertOctagon className="w-4 h-4" />
                <span>64K CHUNKED BASELINE</span>
              </span>
              <span className="text-[10px] bg-red-950 text-red-300 px-2 py-0.5 rounded border border-red-800">
                LEGACY CHUNKED
              </span>
            </div>

            <div className="space-y-3 text-xs">
              <div>
                <div className="flex justify-between text-gray-400 mb-1">
                  <span>Topological Coherence:</span>
                  <span className="text-red-400 font-bold">{metrics.baselineCoherencePct.toFixed(1)}%</span>
                </div>
                <div className="w-full bg-gray-900 h-2 rounded-full overflow-hidden">
                  <div
                    className="bg-red-500 h-full rounded-full transition-all duration-500"
                    style={{ width: `${metrics.baselineCoherencePct}%` }}
                  />
                </div>
              </div>

              <div className="flex justify-between py-1.5 border-b border-gray-800/80 text-gray-300">
                <span className="text-gray-400">Chunk Context Window:</span>
                <span className="font-bold text-gray-300">65,536 tokens / chunk</span>
              </div>

              <div className="flex justify-between py-1.5 border-b border-gray-800/80 text-gray-300">
                <span className="text-gray-400">Boundary Seam Disconnects:</span>
                <span className="font-bold text-red-400">{metrics.baselineSeamFaults} FAULTS DETECTED</span>
              </div>

              <div className="flex justify-between py-1.5 border-b border-gray-800/80 text-gray-300">
                <span className="text-gray-400">Mesh Tearing Probability:</span>
                <span className="font-bold text-red-400">58.8% per boundary</span>
              </div>

              <div className="pt-2 text-[11px] text-gray-400 leading-relaxed">
                ✗ Context amnesia causes height mismatch (&gt;1.5m), floating ramps, and robot entrapment.
                <br />
                ✗ Stitching heuristics fail when generating complex non-linear caverns.
              </div>
            </div>
          </div>
        </div>

        {/* Multi-Model Robotics Data Engine Benchmark Table (KRONOS-1M vs Baselines) */}
        <div className="bg-[#0e1520] p-3.5 rounded-lg border border-cyan-700/60 text-xs mb-4">
          <div className="flex flex-wrap items-center justify-between gap-2 mb-2">
            <div>
              <div className="text-cyan-300 font-bold text-xs uppercase">
                MULTI-MODEL DATA ENGINE BENCHMARK (24 EPISODES // {benchReport.total05sStepsEvaluated} 0.5s STEPS)
              </div>
              <div className="text-[10px] text-gray-400">
                Evaluated on WGO-Bench (TempSeg &amp; Caption), RoboReward (0.5s Failure Recall &amp; False Alarm), Robometer (ρ), Rubric MAE ↓, and Closed-Loop DPO Recovery
              </div>
            </div>
            <div className="flex items-center space-x-1.5">
              <button
                onClick={async () => {
                  const randomOffset = Math.floor(Math.random() * 1400);
                  const live = await runLiveHuggingFaceBenchmark(randomOffset);
                  setBenchReport(live);
                }}
                className="px-2.5 py-1 bg-cyan-600 hover:bg-cyan-500 text-black font-bold rounded text-[10px] transition"
              >
                🌐 Fetch &amp; Score Live HF Split ({benchReport.executionTimeMs}ms)
              </button>
              <button
                onClick={() =>
                  downloadFile(
                    `kronos_benchmark_report_${benchReport.runId}.md`,
                    exportBenchmarkReportMarkdown(benchReport),
                    'text/markdown'
                  )
                }
                className="px-2.5 py-1 bg-gray-900 hover:bg-gray-800 border border-gray-700 text-cyan-300 rounded text-[10px] transition"
              >
                Export Table (.md)
              </button>
            </div>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse text-[10px]">
              <thead>
                <tr className="border-b border-gray-800 text-gray-400">
                  <th className="py-1.5 pr-2">Model (FLORA Paper Protocol)</th>
                  <th className="py-1.5 px-1 text-right">WGO F1@0.75 ↑</th>
                  <th className="py-1.5 px-1 text-right">Robot200 F1@0.5 ↑</th>
                  <th className="py-1.5 px-1 text-right">RoboReward MAE ↓</th>
                  <th className="py-1.5 px-1 text-right">Robometer τ ↑</th>
                  <th className="py-1.5 px-1 text-right">Dense F1 ↑</th>
                  <th className="py-1.5 px-1 text-right">Fail Recall ↑</th>
                  <th className="py-1.5 px-1 text-right">False Alarm ↓</th>
                  <th className="py-1.5 px-1 text-right">Rubric L1 ↓</th>
                  <th className="py-1.5 pl-1 text-right">Cost ($/hr) ↓</th>
                </tr>
              </thead>
              <tbody>
                {benchReport.models.map(row => (
                  <tr
                    key={row.modelId}
                    className={`border-b border-gray-800/60 ${
                      row.isOurs
                        ? 'bg-cyan-950/40 text-cyan-200 font-bold'
                        : 'text-gray-300'
                    }`}
                  >
                    <td className="py-1.5 pr-2 truncate max-w-[195px]">{row.modelName}</td>
                    <td className="py-1.5 px-1 text-right">{row.wgoJointF1At075}%</td>
                    <td className="py-1.5 px-1 text-right">{row.robotJointF1At050}%</td>
                    <td className="py-1.5 px-1 text-right">{row.roboRewardMaeX100.toFixed(1)}</td>
                    <td className="py-1.5 px-1 text-right">{row.robometerOodKendall.toFixed(1)}</td>
                    <td className="py-1.5 px-1 text-right">{row.denseMacroF1}%</td>
                    <td
                      className={`py-1.5 px-1 text-right ${
                        row.isOurs
                          ? 'text-emerald-300'
                          : row.denseFailureRecall < 10
                            ? 'text-red-400'
                            : 'text-amber-300'
                      }`}
                    >
                      {row.denseFailureRecall}%
                    </td>
                    <td className="py-1.5 px-1 text-right">{row.denseNormalFalseAlarm}%</td>
                    <td className="py-1.5 px-1 text-right">{row.rubricPooledL1Error.toFixed(1)}</td>
                    <td
                      className={`py-1.5 pl-1 text-right ${
                        row.costPerVideoHourUsd < 2 ? 'text-emerald-300 font-bold' : 'text-amber-300'
                      }`}
                    >
                      ${row.costPerVideoHourUsd.toFixed(2)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Per-Episode Real Dataset Verification (teetone/RoboReward & macrodata/WGO-Bench) */}
          {benchReport.roboRewardEpisodes && (
            <div className="mt-3 pt-2.5 border-t border-gray-800">
              <div className="text-[10px] font-bold text-cyan-400 mb-1.5 uppercase">
                SAMPLE SCORED EPISODES FROM OFFICIAL HUGGING FACE SPLITS (teetone/RoboReward test &amp; macrodata/WGO-Bench)
              </div>
              <div className="max-h-28 overflow-y-auto space-y-1 pr-1 text-[10px]">
                {benchReport.roboRewardEpisodes.slice(0, 6).map(ep => (
                  <div
                    key={ep.rowIdx}
                    className="flex items-center justify-between bg-black/50 border border-gray-800/80 rounded px-2 py-1"
                  >
                    <div className="truncate max-w-[330px]">
                      <span className="text-violet-400 font-bold mr-1.5">[{ep.sourceDataset} #{ep.rowIdx}]</span>
                      <span className="text-gray-200">{ep.task}</span>
                    </div>
                    <div className="flex items-center space-x-2 flex-shrink-0">
                      <span className="text-gray-400">Gold: <b className="text-white">{ep.goldReward1to5}</b></span>
                      <span className="text-amber-400">Astra: {ep.astraPred1to5}</span>
                      <span className="text-blue-300">FLORA: {ep.floraPred1to5}</span>
                      <span className="text-emerald-300 font-bold">KRONOS: {ep.kronosPred1to5}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* NVIDIA Brev GPU DPO Post-Training & Managed Agent Sandbox */}
        <div className="bg-[#0e1520] p-4 rounded-lg border border-violet-700/60 text-xs mb-4">
          <div className="flex flex-wrap items-center justify-between gap-2 mb-2">
            <div className="flex items-center space-x-2 text-violet-300 font-bold">
              <Terminal className="w-4 h-4 text-violet-400" />
              <span>NVIDIA BREV GPU DPO TRAINER // INSTANCE: {brevJob?.instance || 'pg-solvers'}</span>
            </div>
            <button
              onClick={onTriggerBrevTraining}
              className="px-3 py-1.5 bg-gradient-to-r from-violet-600 to-cyan-500 hover:from-violet-500 hover:to-cyan-400 text-black font-bold rounded flex items-center space-x-1.5 transition shadow-[0_0_12px_rgba(139,92,246,0.4)]"
            >
              <Rocket className="w-3.5 h-3.5" />
              <span>
                {brevJob?.status === 'training' || brevJob?.status === 'staging'
                  ? `TRAINING EPOCH ${brevJob.epochsCompleted}/${brevJob.totalEpochs}...`
                  : '🚀 TRAIN DPO POLICY ON BREV GPU'}
              </span>
            </button>
          </div>

          {brevJob && brevJob.lossHistory.length > 0 && (
            <div className="grid grid-cols-6 gap-1.5 my-2">
              {brevJob.lossHistory.map((loss, i) => (
                <div key={i} className="bg-black/60 border border-gray-800 rounded p-1.5 text-center">
                  <div className="text-[9px] text-gray-400">EPOCH {i + 1}</div>
                  <div className="text-cyan-300 font-bold text-[11px]">{loss.toFixed(3)}</div>
                </div>
              ))}
            </div>
          )}

          <div className="bg-black/80 border border-gray-800 rounded p-2 text-[10px] text-emerald-300 font-mono truncate">
            {brevJob?.logTail ||
              'Ready to stage KRONOS-1M 4-branch Counterfactual DPO preference dataset & MuJoCo XML to NVIDIA Brev GPU.'}
          </div>
        </div>

        {/* Open-Source Robotics Artifact Exporters */}
        <div className="bg-black/60 p-3.5 rounded-lg border border-gray-800 text-xs">
          <div className="flex items-center justify-between mb-2">
            <span className="font-bold text-white flex items-center space-x-1.5">
              <GitBranch className="w-4 h-4 text-cyan-400" />
              <span>OPEN-SOURCE ROBOTICS ARTIFACT EXPORTERS (LeRobot / MuJoCo / Isaac USD)</span>
            </span>
            <span className="text-[10px] text-cyan-400">
              {kronosEngine.getForkEvents().length} DPO FORKS CAPTURED
            </span>
          </div>
          <div className="flex flex-wrap gap-2">
            <button
              onClick={() =>
                downloadFile(
                  `kronos_1m_dpo_dataset_${Date.now()}.json`,
                  kronosEngine.exportDpoDatasetJson(activeTelemetry),
                  'application/json'
                )
              }
              className="px-3 py-1.5 bg-cyan-950/80 hover:bg-cyan-900 border border-cyan-600/70 text-cyan-200 rounded text-[11px] font-bold flex items-center space-x-1.5 transition"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Export LeRobot / GR00T DPO (.json)</span>
            </button>
            <button
              onClick={() =>
                downloadFile(
                  `kronos_mujoco_${activeTelemetry.activeMorphology || 'rover'}.xml`,
                  kronosEngine.exportMuJoCoXml(activeTelemetry),
                  'application/xml'
                )
              }
              className="px-3 py-1.5 bg-emerald-950/80 hover:bg-emerald-900 border border-emerald-600/70 text-emerald-200 rounded text-[11px] font-bold flex items-center space-x-1.5 transition"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Export MuJoCo MJCF (.xml)</span>
            </button>
            <button
              onClick={() =>
                downloadFile(
                  `kronos_isaac_stage_${Date.now()}.usda`,
                  kronosEngine.exportOpenUsdAscii(activeTelemetry),
                  'text/plain'
                )
              }
              className="px-3 py-1.5 bg-violet-950/80 hover:bg-violet-900 border border-violet-600/70 text-violet-200 rounded text-[11px] font-bold flex items-center space-x-1.5 transition"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Export OpenUSD Stage (.usda)</span>
            </button>
          </div>
        </div>

        {/* Footer */}
        <div className="mt-4 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-black font-bold rounded text-xs transition"
          >
            RETURN TO SIMULATION
          </button>
        </div>
      </div>
    </div>
  );
};

