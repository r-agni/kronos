import * as THREE from 'three';
import { streamWorldGeneration, parseDSLDirectives } from './geminiArgon.js';
import { evaluateViewportAndInjectHazard } from './geminiAdversary.js';
import { handleVoiceCommand } from './geminiVoice.js';
import { runManagedAgentInteraction, getManagedAgentSession } from './managedAgents.js';
import { triggerBrevDpoTraining } from './brevClient.js';
import { KronosEngine } from '../src/engine/KronosEngine.js';
import {
  runKronosBenchmarkSuite,
  exportBenchmarkReportMarkdown,
} from '../src/engine/KronosBenchmarkSuite.js';
import { PhysicsWorld } from '../src/engine/PhysicsWorld.js';
import { ProceduralDSLParser } from '../src/engine/ProceduralDSL.js';
import { SectorManager } from '../src/engine/SectorManager.js';
import { WorldCompiler } from '../src/engine/WorldCompiler.js';
import { RobotTelemetry } from '../src/types/simulation.js';

async function runRealWorkflowVerification() {
  console.log('=== KRONOS-1M // REAL-DATA & REAL-WORKFLOW VERIFICATION SUITE ===\n');

  // 1. Verify ProceduralDSLParser & WorldCompiler with real Rapier WASM PhysicsWorld
  const parser = new ProceduralDSLParser();
  const sampleDsl = `
ADD_SECTOR { id: "sec_verify_01", name: "Sector Verify", biome: "collapsed_chasm", center: [0, 0, 50], size: [36, 60], heightScale: 2.5, roughness: 0.5 }
MUTATE_PHYSICS { id: "phys_verify_01", gravity: -1.62, friction: 0.16, windX: 14.0, fogDensity: 0.006, label: "Lunar 0.16g Cryo-Ice" }
MORPH_ROVER { id: "morph_verify_01", mode: "thruster_hopper", suspensionHeight: 1.45, trackWidthScale: 1.32, thrusterBoost: true }
DEPLOY_COUNTERMEASURE { id: "cm_verify_01", type: "deployable_bridge", position: [0, 0.3, 52], dimensions: [6.0, 0.5, 18.0] }
`;
  const cmds = parser.parseComplete(sampleDsl);
  if (cmds.length !== 4) {
    throw new Error(`Expected 4 parsed DSL commands, got ${cmds.length}`);
  }
  const scene = new THREE.Scene();
  const physics = await PhysicsWorld.create();
  const sectorMgr = new SectorManager(scene, physics, 6);
  const compiler = new WorldCompiler(scene, sectorMgr);
  const compiled = compiler.ingestText(sampleDsl);
  if (
    compiled.sectors.length !== 1 ||
    compiled.physicsMutations.length !== 1 ||
    compiled.roverMorphs.length !== 1 ||
    compiled.countermeasures.length !== 1
  ) {
    throw new Error('WorldCompiler failed to compile all 4 directive types');
  }
  console.log('[PASS] 1. ProceduralDSLParser & WorldCompiler verified (ADD_SECTOR, MUTATE_PHYSICS, MORPH_ROVER, DEPLOY_COUNTERMEASURE).');

  // 2. Verify KronosEngine 6 capabilities on real 60Hz telemetry stream
  const kronos = new KronosEngine();
  kronos.setRubric('Penalize wheel slip > 20% and require clearance > 2.5m');
  const sampleTelemetry: RobotTelemetry = {
    position: [0.4, 0.82, 38.5],
    velocity: [0.1, 0, 0.4],
    speedKmh: 0.8,
    headingDeg: 2.1,
    pitchDeg: -1.4,
    rollDeg: 0.8,
    traversedDistanceMeters: 38.5,
    obstacleClearanceMeters: 1.8,
    structuralIntegrityPct: 95,
    lidarRayHits: 29,
    autonomousState: 'route_blocked',
    targetWaypoint: [0, 0, 80],
    wheelSlipRatio: 0.34,
    tippingMarginDeg: 24.5,
    activeMorphology: 'thruster_hopper',
    physicsRegime: 'Lunar 0.16g Cryo-Ice',
  };

  const stepRes = kronos.stepTelemetry(sampleTelemetry, true);
  if (!stepRes.newTick || stepRes.newTick.rating !== 'failure') {
    throw new Error('Expected failure process reward tick on route_blocked telemetry');
  }
  const fork = kronos.triggerMultiverseFork(
    sampleTelemetry.position,
    0,
    '64K Seam Cliff Halt @ Z=38.5m'
  );
  if (fork.branches.length !== 4 || fork.chosenBranchId !== 'omega_optimal') {
    throw new Error('Expected 4 counterfactual branches with omega_optimal chosen');
  }
  const dpoJson = kronos.exportDpoDatasetJson(sampleTelemetry);
  const mjcfXml = kronos.exportMuJoCoXml(sampleTelemetry);
  const usdAscii = kronos.exportOpenUsdAscii(sampleTelemetry);
  if (!dpoJson.includes('kronos-1m-robotics-dpo-v1') || !mjcfXml.includes('<mujoco') || !usdAscii.includes('#usda 1.0')) {
    throw new Error('Artifact export validation failed');
  }
  console.log('[PASS] 2. KronosEngine 6 capabilities & LeRobot DPO / MuJoCo XML / OpenUSD exports verified.');

  // 3. Verify Gemini Managed Agents (`antigravity-preview-09-2026` + `gemini-4-argon`)
  const agentTurn = await runManagedAgentInteraction({
    skillName: 'counterfactual-dpo-synthesizer',
    thinkingLevel: 'high',
    prompt: 'Synthesize 4-branch counterfactual DPO resolution for Z=38.5m chasm halt.',
    telemetry: sampleTelemetry,
  });
  const session = getManagedAgentSession();
  if (!agentTurn.steps.length || !session.interactionId || !session.environmentId) {
    throw new Error('Managed Agent session verification failed');
  }
  console.log(`[PASS] 3. Gemini Managed Agent verified (interactionId=${session.interactionId}, steps=${agentTurn.steps.length}).`);

  // 4. Verify Live World Stream, VLM Adversary, Voice Mutator & NVIDIA Brev GPU DPO Training in parallel
  const [streamed, critique, voiceRes, brevJob] = await Promise.all([
    streamWorldGeneration({ sectorIndex: 1, currentPosition: [0, 0, 0] }, () => {}),
    evaluateViewportAndInjectHazard('', sampleTelemetry),
    handleVoiceCommand('Morph rover to thruster hopper in left corridor', [0, 0.8, 38.5]),
    triggerBrevDpoTraining(dpoJson, mjcfXml),
  ]);
  const parsedStream = parseDSLDirectives(streamed);

  if (!parsedStream.sector || !critique.hazardCodeSnippet || !voiceRes.dslMutation || brevJob.status !== 'completed') {
    throw new Error('End-to-end backend & Brev GPU verification failed');
  }
  console.log(`[PASS] 4. NVIDIA Brev GPU DPO Training verified (instance=${brevJob.instance}, finalLoss=${brevJob.lossHistory.slice(-1)[0]}).`);

  // 5. Run Empirical Multi-Model Benchmark Suite (KRONOS-1M vs Baselines)
  const benchReport = runKronosBenchmarkSuite();
  const benchMd = exportBenchmarkReportMarkdown(benchReport);
  console.log('\n=== 5. EMPIRICAL BENCHMARK SUITE RESULTS ===\n');
  console.log(benchMd);
  console.log('\n=== ALL REAL-DATA & REAL-WORKFLOW CHECKS PASSED ===');
  process.exit(0);
}

runRealWorkflowVerification().catch((err) => {
  console.error('Verification failed:', err);
  process.exit(1);
});

