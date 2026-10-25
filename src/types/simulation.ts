// Shared simulation and data contracts across 3D Engine, VLM Red Team, and Gemini 4 Argon

export type Vec3 = [number, number, number];

export interface TerrainSector {
  id: string;
  name: string;
  center: Vec3;
  size: [number, number]; // width, length
  heightScale: number;
  roughness: number;
  biome: 'industrial_hazmat' | 'subterranean_tunnel' | 'collapsed_chasm' | 'reinforced_highway';
  hasBridge?: boolean;
  hasTunnel?: boolean;
  clearanceHeight?: number;
  timestamp: number;
}

export type HazardType = 
  | 'seismic_collapse'
  | 'radioactive_leak'
  | 'chasm_rupture'
  | 'steam_pipe_burst'
  | 'structural_overpass_fall'
  | 'emp_interference';

export interface HazardEvent {
  id: string;
  type: HazardType;
  title: string;
  position: Vec3;
  radius: number;
  severity: 'warning' | 'critical' | 'catastrophic';
  timeRemainingSec: number;
  triggerSource: 'vlm_adversary' | 'environmental' | 'benchmark_stress';
  active: boolean;
}

export interface TacticalCountermeasure {
  id: string;
  name: string;
  type: 'carbon_ramp' | 'deployable_bridge' | 'sensor_beacon' | 'air_scrubber' | 'blast_barrier';
  position: Vec3;
  rotation?: Vec3;
  dimensions?: Vec3;
  deployedAt: number;
  operatorVoicePrompt?: string;
}

export interface LiDARPoint {
  origin: Vec3;
  direction: Vec3;
  hitPoint: Vec3;
  distance: number;
  hitNormal: Vec3;
  entityType: 'terrain' | 'hazard' | 'structure' | 'countermeasure';
}

export type RoverMorphologyMode =
  | 'standard_cruiser'
  | 'high_clearance_crawler'
  | 'wide_track_heavy'
  | 'thruster_hopper';

export interface RoverMorphConfig {
  mode: RoverMorphologyMode;
  suspensionHeight: number;
  trackWidthScale: number;
  thrusterBoost: boolean;
}

export interface PhysicsMutationState {
  gravityY: number;
  surfaceFriction: number;
  lateralWindX: number;
  fogDensity: number;
  environmentLabel: string;
}

export interface Sim2RealStressSnapshot {
  id: string;
  timestamp: number;
  sectorId: string;
  position: Vec3;
  eventType:
    | 'seam_fault_halt'
    | 'eavesdrop_trap'
    | 'physics_mutation'
    | 'hazard_evasion'
    | 'chassis_morph'
    | 'multiverse_dpo_fork';
  wheelSlipRatio: number;
  tippingMarginDeg: number;
  deltaElevationMeters: number;
  summary: string;
  dslSnapshot: string;
}

export interface TemporalSegment {
  id: string;
  index: number;
  label: string;
  startSec: number;
  endSec: number;
  instructionCaption: string;
  stitchStatus: 'continuous_1m' | 'window_fractured';
  dominantState: RobotTelemetry['autonomousState'];
  meanReward: number;
}

export interface DenseProcessRewardTick {
  id: string;
  timestampSec: number;
  rating: 'normal' | 'suboptimal' | 'failure';
  score: number; // 0.0 to 1.0
  reason: string;
  position: Vec3;
  wheelSlipRatio: number;
  clearanceMeters: number;
}

export interface CounterfactualBranch {
  branchId: 'alpha_fail' | 'beta_swerve' | 'gamma_crawl' | 'omega_optimal';
  label: string;
  policyName: string;
  colorHex: number;
  cssColor: string;
  dpoRole: 'chosen' | 'rejected';
  outcomeScore1to5: number;
  processRewardMean: number;
  trajectory: Vec3[];
  actionDsl: string;
  rationale: string;
}

export interface MultiverseForkEvent {
  id: string;
  timestamp: number;
  simTimeSec: number;
  triggerReason: string;
  originPosition: Vec3;
  originHeading: number;
  thinkingLevel: 'low' | 'high';
  branches: CounterfactualBranch[];
  chosenBranchId: CounterfactualBranch['branchId'];
}

export interface RubricScoreCard {
  rubricText: string;
  overallOutcomeScore1to5: number;
  compliancePct: number;
  normalStepsCount: number;
  suboptimalStepsCount: number;
  failureStepsCount: number;
  activeViolations: string[];
}

export interface ManagedAgentStep {
  id: string;
  timestamp: number;
  agent: 'antigravity-preview-09-2026';
  model: 'gemini-4-argon';
  stepType: 'reasoning' | 'code_execution' | 'filesystem' | 'model_output';
  thinkingLevel: 'low' | 'high';
  interactionId: string;
  environmentId: string;
  skillMounted: string;
  summary: string;
  codeOrOutput?: string;
}

export interface BrevGpuJobStatus {
  jobId: string;
  instance: string;
  orgId: string;
  status: 'idle' | 'staging' | 'training' | 'completed' | 'error';
  mode: 'brev_cloud_gpu' | 'local_gpu_verified';
  dpoPairsCount: number;
  epochsCompleted: number;
  totalEpochs: number;
  lossHistory: number[];
  policyRewardGainPct: number;
  logTail: string;
  updatedAt: number;
}

export interface RobotTelemetry {
  position: Vec3;
  velocity: Vec3;
  speedKmh: number;
  headingDeg: number;
  pitchDeg: number;
  rollDeg: number;
  traversedDistanceMeters: number;
  obstacleClearanceMeters: number;
  structuralIntegrityPct: number;
  lidarRayHits: number;
  autonomousState: 'navigating' | 'avoiding_hazard' | 'scaling_ramp' | 'voice_override' | 'route_blocked';
  targetWaypoint: Vec3;
  wheelSlipRatio?: number;
  tippingMarginDeg?: number;
  activeMorphology?: RoverMorphologyMode;
  physicsRegime?: string;
}

export interface AdversaryCritique {
  id: string;
  timestamp: number;
  threatLevel: 'low' | 'moderate' | 'elevated' | 'severe' | 'lethal';
  vulnerabilityTargeted: string;
  actionPlan: string;
  hazardCodeSnippet: string;
  mitigationAdviseForBlueTeam: string;
  eavesdropIntercepted?: string;
}

export interface StreamToken {
  token: string;
  kind: 'keyword' | 'identifier' | 'geometry' | 'physics' | 'meta';
  timestamp: number;
}

export interface BaselineMetricComparison {
  argonTokensStreamed: number;
  argonCoherencePct: number;
  argonSeamFaults: number;
  baselineTokensGenerated: number;
  baselineCoherencePct: number;
  baselineSeamFaults: number;
  colliderContinuityPct: number;
}

