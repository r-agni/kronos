import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import {
  SceneManager,
  RoverModel,
  LiDARVisualizer,
  PhysicsWorld,
  SectorManager,
  BaselineSimulator,
  PostProcessingManager,
  WorldCompiler,
  kronosEngine,
  type CameraViewMode,
  type VisionMode,
  type WorldCompileResult,
  type WorldDeltaPayload,
  type HazardSpawnDescriptor,
  type CountermeasureSpawnDescriptor,
} from '../engine';
import {
  AdversaryCritique,
  BaselineMetricComparison,
  HazardType,
  MultiverseForkEvent,
  PhysicsMutationState,
  RobotTelemetry,
  RoverMorphConfig,
  Sim2RealStressSnapshot,
  Vec3,
} from '../types/simulation';
import { geminiService, type WorldDelta } from '../services/geminiClient';
import { soundEffects } from '../services/soundEffects';

interface Viewport3DProps {
  onTelemetryUpdate: (telemetry: RobotTelemetry) => void;
  onAdversaryCritique: (critique: AdversaryCritique) => void;
  onMetrics: (metrics: BaselineMetricComparison) => void;
  onStressSnapshot?: (snapshot: Sim2RealStressSnapshot) => void;
  onMultiverseFork?: (fork: MultiverseForkEvent) => void;
  forkRequestCounter?: number;
  isBaselineMode: boolean;
  tokensReceived: number;
  hazardInjection?: AdversaryCritique | null;
}

type CameraUiMode = 'chase' | 'isometric' | 'cockpit';
type VisionUiMode = 'standard' | 'nvg' | 'flir';

const ROVER_HALF: Vec3 = [0.9, 0.45, 1.6];
const ROVER_HALF_Y = ROVER_HALF[1];
const HAZARD_TYPES: readonly HazardType[] = [
  'seismic_collapse',
  'radioactive_leak',
  'chasm_rupture',
  'steam_pipe_burst',
  'structural_overpass_fall',
  'emp_interference',
];

const STARTING_SECTORS = [
  {
    id: 'sec_00_origin',
    name: 'Containment Zone Alpha',
    center: [0, 0, 0] as Vec3,
    size: [28, 40] as [number, number],
    heightScale: 1.0,
    roughness: 0.25,
    biome: 'reinforced_highway' as const,
    timestamp: Date.now(),
  },
  {
    id: 'sec_01_hazmat',
    name: 'Reactor Corridor Beta',
    center: [0, 0, 40] as Vec3,
    size: [28, 40] as [number, number],
    heightScale: 1.8,
    roughness: 0.45,
    biome: 'industrial_hazmat' as const,
    timestamp: Date.now(),
  },
  {
    id: 'sec_02_chasm',
    name: 'Seismic Rupture Gamma',
    center: [0, 0, 80] as Vec3,
    size: [28, 40] as [number, number],
    heightScale: 2.2,
    roughness: 0.65,
    biome: 'collapsed_chasm' as const,
    timestamp: Date.now(),
  },
];

let voiceSerial = 0;

function mapCamera(mode: CameraUiMode): CameraViewMode {
  if (mode === 'cockpit') return 'first_person';
  if (mode === 'isometric') return 'orbital';
  return 'chase';
}

function mapVision(mode: VisionUiMode): VisionMode {
  if (mode === 'nvg') return 'night_vision';
  if (mode === 'flir') return 'thermal';
  return 'standard';
}

function fmt(value: number): string {
  return Number.isFinite(value) ? value.toFixed(2) : '0.00';
}

function safeId(raw: string, prefix: string): string {
  const cleaned = (raw || '').replace(/[^A-Za-z0-9_-]/g, '_').slice(0, 48);
  return cleaned ? `${prefix}_${cleaned}` : `${prefix}_${Date.now()}`;
}

function loosenHazard(raw: string): HazardType | null {
  const value = raw.toLowerCase();
  if ((HAZARD_TYPES as readonly string[]).includes(value)) return value as HazardType;
  if (value.includes('overpass')) return 'structural_overpass_fall';
  if (value.includes('chasm') || value.includes('rupture') || value.includes('fissure')) return 'chasm_rupture';
  if (value.includes('steam')) return 'steam_pipe_burst';
  if (value.includes('radio') || value.includes('leak') || value.includes('hazmat')) return 'radioactive_leak';
  if (value.includes('emp')) return 'emp_interference';
  if (value.includes('collapse') || value.includes('seismic') || value.includes('rock') || value.includes('boulder')) {
    return 'seismic_collapse';
  }
  return null;
}

function hazardTypeFromCritique(critique: AdversaryCritique): HazardType {
  const snippet = critique.hazardCodeSnippet ?? '';
  const explicit = snippet.match(/type\s*[:=]\s*["']([a-zA-Z0-9_]+)["']/i)?.[1]
    ?? snippet.match(/type\s*[:=]\s*([a-zA-Z0-9_]+)/i)?.[1];
  const fromExplicit = explicit ? loosenHazard(explicit) : null;
  if (fromExplicit) return fromExplicit;

  const blob = `${snippet} ${critique.actionPlan} ${critique.vulnerabilityTargeted}`.toLowerCase();
  return loosenHazard(blob) ?? 'seismic_collapse';
}

function severityFromThreat(level: AdversaryCritique['threatLevel']): 'warning' | 'critical' | 'catastrophic' {
  if (level === 'lethal' || level === 'severe') return 'catastrophic';
  if (level === 'elevated' || level === 'moderate') return 'critical';
  return 'warning';
}

function parseVec3(snippet: string, key: string): Vec3 | null {
  const matcher = new RegExp(
    `${key}\\s*[:=]\\s*\\[\\s*(-?\\d+(?:\\.\\d+)?)\\s*,\\s*(-?\\d+(?:\\.\\d+)?)\\s*,\\s*(-?\\d+(?:\\.\\d+)?)\\s*\\]`,
    'i'
  );
  const match = matcher.exec(snippet);
  if (!match) return null;
  return [Number(match[1]), Number(match[2]), Number(match[3])];
}

function synthesizedCritiqueDirectives(critique: AdversaryCritique, roverPos: Vec3, heading: number): string {
  const snippet = critique.hazardCodeSnippet ?? '';
  const hazardType = hazardTypeFromCritique(critique);
  const position = parseVec3(snippet, 'position') ?? [
    roverPos[0] + Math.sin(heading) * 22,
    roverPos[1],
    roverPos[2] + Math.cos(heading) * 22,
  ];
  const radiusMatch = snippet.match(/radius\s*[:=]\s*(\d+(?:\.\d+)?)/i);
  const radius = radiusMatch ? Number(radiusMatch[1]) : 14;
  const severityMatch = snippet.match(/severity\s*[:=]\s*["']?(warning|critical|catastrophic|\d+(?:\.\d+)?)["']?/i);
  const severityLiteral = severityMatch
    ? (/^\d/.test(severityMatch[1]) ? severityMatch[1] : `"${severityMatch[1].toLowerCase()}"`)
    : `"${severityFromThreat(critique.threatLevel)}"`;

  return `TRIGGER_HAZARD { id: "${safeId(critique.id, 'hz')}", type: "${hazardType}", position: [${fmt(position[0])}, ${fmt(position[1])}, ${fmt(position[2])}], radius: ${radius}, severity: ${severityLiteral} }`;
}

function voiceCountermeasureDsl(countermeasureType: string, roverPos: Vec3, heading: number): string {
  const id = `cm_voice_${Date.now()}_${voiceSerial++}`;
  const type = (countermeasureType || '').toLowerCase();
  const fx = Math.sin(heading);
  const fz = Math.cos(heading);
  const rx = Math.cos(heading);
  const rz = -Math.sin(heading);

  if (type.includes('beacon') || type.includes('sensor')) {
    const x = roverPos[0] + rx * 3.5 + fx * 8;
    const z = roverPos[2] + rz * 3.5 + fz * 8;
    return `DEPLOY_COUNTERMEASURE { id: "${id}", type: "sensor_beacon", position: [${fmt(x)}, ${fmt(roverPos[1] + 1.25)}, ${fmt(z)}], dimensions: [0.8, 2.5, 0.8] }`;
  }

  if (type.includes('scrub') || type.includes('air') || type.includes('smoke') || type.includes('fog')) {
    const x = roverPos[0] - rx * 3.5 + fx * 8;
    const z = roverPos[2] - rz * 3.5 + fz * 8;
    return `DEPLOY_COUNTERMEASURE { id: "${id}", type: "air_scrubber", position: [${fmt(x)}, ${fmt(roverPos[1] + 1.2)}, ${fmt(z)}], dimensions: [1.8, 2.4, 1.8] }`;
  }

  if (type.includes('barrier') || type.includes('blast') || type.includes('wall')) {
    const x = roverPos[0] + fx * 10;
    const z = roverPos[2] + fz * 10;
    return `DEPLOY_COUNTERMEASURE { id: "${id}", type: "blast_barrier", position: [${fmt(x)}, ${fmt(roverPos[1] + 1.25)}, ${fmt(z)}], rotation: [0, ${fmt(heading)}, 0], dimensions: [8, 2.5, 0.6] }`;
  }

  const bridge = type.includes('bridge');
  const length = bridge ? 18 : 14;
  const rise = bridge ? 1.15 : 2.5;
  const lead = 3.5;
  const startX = roverPos[0] + fx * lead;
  const startZ = roverPos[2] + fz * lead;
  const endX = startX + fx * length;
  const endZ = startZ + fz * length;
  return `EXTEND_RAMP { id: "${id}", start: [${fmt(startX)}, ${fmt(roverPos[1])}, ${fmt(startZ)}], end: [${fmt(endX)}, ${fmt(roverPos[1] + rise)}, ${fmt(endZ)}], width: ${bridge ? 6 : 6.5}, material: "carbon_fiber" }`;
}

function asCompilerDelta(delta: WorldDelta): WorldDeltaPayload {
  return {
    sector: delta.sector ?? null,
    structures: delta.structures ?? [],
    hazards: delta.hazards ?? [],
    rawDirectives: delta.rawDirectives ?? '',
  };
}

function applyDirectives(compiler: WorldCompiler, rawDirectives: string): WorldCompileResult {
  return compiler.applyDelta({
    sector: null,
    structures: [],
    hazards: [],
    rawDirectives,
  });
}

function debrisMesh(hazardType: HazardType, radius: number): THREE.Mesh {
  if (hazardType === 'structural_overpass_fall') {
    return new THREE.Mesh(
      new THREE.BoxGeometry(radius * 1.5, radius * 0.9, radius * 1.7),
      new THREE.MeshStandardMaterial({ color: 0x9ca3af, roughness: 0.82, metalness: 0.12 })
    );
  }
  if (hazardType === 'chasm_rupture') {
    return new THREE.Mesh(
      new THREE.DodecahedronGeometry(radius, 0),
      new THREE.MeshStandardMaterial({ color: 0x292524, roughness: 0.96, metalness: 0.04 })
    );
  }
  return new THREE.Mesh(
    new THREE.DodecahedronGeometry(radius, 1),
    new THREE.MeshStandardMaterial({ color: 0x78716c, roughness: 0.9, metalness: 0.08 })
  );
}

function hazardMarker(hazard: HazardSpawnDescriptor): THREE.Mesh {
  const color = hazard.hazardType === 'radioactive_leak'
    ? 0x84cc16
    : hazard.hazardType === 'steam_pipe_burst'
      ? 0xe2e8f0
      : hazard.hazardType === 'emp_interference'
        ? 0xa855f7
        : 0xf59e0b;
  const mesh = new THREE.Mesh(
    new THREE.SphereGeometry(Math.min(8, Math.max(2, hazard.radius * 0.35)), 18, 12),
    new THREE.MeshStandardMaterial({
      color,
      transparent: true,
      opacity: 0.28,
      emissive: color,
      emissiveIntensity: 0.7,
      depthWrite: false,
    })
  );
  mesh.position.set(hazard.position[0], hazard.position[1] + 1.5, hazard.position[2]);
  mesh.name = hazard.id;
  return mesh;
}

function countermeasureMesh(countermeasure: CountermeasureSpawnDescriptor): THREE.Mesh {
  const { visual, staticStructure } = countermeasure;
  const material = new THREE.MeshStandardMaterial({
    color: visual.color,
    emissive: visual.emissive,
    emissiveIntensity: visual.emissiveIntensity,
    metalness: visual.shape === 'box' ? 0.85 : 0.4,
    roughness: visual.shape === 'box' ? 0.25 : 0.45,
  });
  const mesh = visual.shape === 'cylinder'
    ? new THREE.Mesh(
      new THREE.CylinderGeometry(visual.dimensions[0] / 2, visual.dimensions[2] / 2, visual.dimensions[1], 16),
      material
    )
    : new THREE.Mesh(
      new THREE.BoxGeometry(visual.dimensions[0], visual.dimensions[1], visual.dimensions[2]),
      material
    );
  mesh.position.set(staticStructure.position[0], staticStructure.position[1], staticStructure.position[2]);
  mesh.quaternion.set(
    staticStructure.rotationQuat[0],
    staticStructure.rotationQuat[1],
    staticStructure.rotationQuat[2],
    staticStructure.rotationQuat[3]
  );
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  mesh.name = countermeasure.id;
  return mesh;
}

function realizeSpawns(
  scene: THREE.Scene,
  physics: PhysicsWorld,
  result: WorldCompileResult,
  onEmp: () => void
): void {
  for (const hazard of result.hazards) {
    if (hazard.dynamicSpheres.length === 0) {
      scene.add(hazardMarker(hazard));
      if (hazard.hazardType === 'emp_interference') onEmp();
      continue;
    }
    for (const sphere of hazard.dynamicSpheres) {
      const mesh = debrisMesh(hazard.hazardType, sphere.radius);
      mesh.position.set(sphere.position[0], sphere.position[1], sphere.position[2]);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      mesh.name = sphere.id;
      scene.add(mesh);
      physics.addDynamicSphere(sphere.id, sphere.position, sphere.radius, sphere.options);
      physics.registerDynamicMesh(sphere.id, mesh);
    }
  }

  for (const countermeasure of result.countermeasures) {
    scene.add(countermeasureMesh(countermeasure));
    const body = countermeasure.staticStructure;
    physics.addStaticStructure(body.id, body.position, body.halfExtents, body.entityType, body.rotationQuat);
  }
}

export const Viewport3D: React.FC<Viewport3DProps> = ({
  onTelemetryUpdate,
  onAdversaryCritique,
  onMetrics,
  onStressSnapshot,
  onMultiverseFork,
  forkRequestCounter = 0,
  isBaselineMode,
  tokensReceived,
  hazardInjection = null,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [cameraMode, setCameraMode] = useState<CameraUiMode>('chase');
  const [visionMode, setVisionMode] = useState<VisionUiMode>('standard');
  const [isAutonomous, setIsAutonomous] = useState<boolean>(true);
  const [judgeMode, setJudgeMode] = useState<boolean>(() => geminiService.isJudgeMode());
  const [hudBanner, setHudBanner] = useState<string | null>(null);

  const sceneMgrRef = useRef<SceneManager | null>(null);
  const physicsWorldRef = useRef<PhysicsWorld | null>(null);
  const roverModelRef = useRef<RoverModel | null>(null);
  const sectorMgrRef = useRef<SectorManager | null>(null);
  const postPipelineRef = useRef<PostProcessingManager | null>(null);
  const seamRibbonsRef = useRef<THREE.Mesh[]>([]);
  const forkHandlerRef = useRef<((reason: string) => void) | null>(null);

  const cameraModeRef = useRef(cameraMode);
  const visionModeRef = useRef(visionMode);
  const isAutonomousRef = useRef(isAutonomous);
  const baselineModeRef = useRef(isBaselineMode);
  const tokenCountRef = useRef(tokensReceived);
  const onTelemetryRef = useRef(onTelemetryUpdate);
  const onAdversaryRef = useRef(onAdversaryCritique);
  const onMetricsRef = useRef(onMetrics);
  const onStressSnapshotRef = useRef(onStressSnapshot);
  const onMultiverseForkRef = useRef(onMultiverseFork);
  const hazardHandlerRef = useRef<((critique: AdversaryCritique) => void) | null>(null);
  const pendingHazardRef = useRef<AdversaryCritique | null>(hazardInjection);

  const roverStateRef = useRef({
    pos: [0, 0.8, 0] as Vec3,
    velocity: [0, 0, 8.5] as Vec3,
    speed: 8.5,
    heading: 0,
    steerAngle: 0,
    traversedDistance: 0,
    activeHazards: 0,
    hazardZones: [] as Array<{ pos: Vec3; radius: number; type: HazardType }>,
    bridgedZones: [] as Array<{ pos: Vec3; span: number }>,
    lidarMaxRange: 45,
    integrityBonus: 0,
    verticalJumpOffset: 0,
    lastHaltSnapshotZ: -999,
    autonomousState: 'navigating' as RobotTelemetry['autonomousState'],
    physicsMutation: {
      gravityY: -9.81,
      surfaceFriction: 0.78,
      lateralWindX: 0,
      fogDensity: 0.006,
      environmentLabel: 'Earth 1.0g Nominal',
    } as PhysicsMutationState,
    roverMorph: {
      mode: 'standard_cruiser',
      suspensionHeight: 1.0,
      trackWidthScale: 1.0,
      thrusterBoost: false,
    } as RoverMorphConfig,
  });
  const keysPressed = useRef<{ [key: string]: boolean }>({});
  const sectorFrontierCounter = useRef<number>(2);

  onTelemetryRef.current = onTelemetryUpdate;
  onAdversaryRef.current = onAdversaryCritique;
  onMetricsRef.current = onMetrics;
  onStressSnapshotRef.current = onStressSnapshot;
  onMultiverseForkRef.current = onMultiverseFork;

  useEffect(() => {
    if (forkRequestCounter > 0) {
      forkHandlerRef.current?.('Operator triggered 4x Counterfactual Multiverse DPO Fork');
    }
  }, [forkRequestCounter]);

  useEffect(() => {
    cameraModeRef.current = cameraMode;
    sceneMgrRef.current?.setViewMode(mapCamera(cameraMode));
  }, [cameraMode]);

  useEffect(() => {
    visionModeRef.current = visionMode;
    postPipelineRef.current?.setVisionMode(mapVision(visionMode));
  }, [visionMode]);

  useEffect(() => {
    isAutonomousRef.current = isAutonomous;
  }, [isAutonomous]);

  useEffect(() => {
    baselineModeRef.current = isBaselineMode;
    const sectorMgr = sectorMgrRef.current;
    if (!sectorMgr) return;
    sectorMgr.setFractureMode(isBaselineMode);
    for (const ribbon of seamRibbonsRef.current) {
      const mat = ribbon.material as THREE.MeshBasicMaterial;
      mat.color.setHex(isBaselineMode ? 0xff2244 : 0x00f0ff);
      mat.opacity = isBaselineMode ? 0.85 : 0.35;
      ribbon.scale.y = isBaselineMode ? 3.5 : 1.0;
    }
    if (isBaselineMode) {
      setHudBanner('64K CHUNKED BASELINE ACTIVE — SOUTH SECTOR SEAMS FRACTURED (ΔY -2.4m)');
    } else {
      setHudBanner('ARGON 1M CONTINUOUS CONTEXT — TOPOLOGICAL SEAMS RESTORED (100% COHERENCE)');
    }
    onMetricsRef.current(sectorMgr.getComparisonMetrics(tokenCountRef.current));
  }, [isBaselineMode]);

  useEffect(() => {
    tokenCountRef.current = tokensReceived;
    const sectorMgr = sectorMgrRef.current;
    if (!sectorMgr) return;
    onMetricsRef.current(sectorMgr.getComparisonMetrics(tokensReceived));
  }, [tokensReceived]);

  useEffect(() => {
    pendingHazardRef.current = hazardInjection;
    if (!hazardInjection || !hazardHandlerRef.current) return;
    hazardHandlerRef.current(hazardInjection);
  }, [hazardInjection]);

  useEffect(() => {
    geminiService.setJudgeMode(judgeMode);
  }, [judgeMode]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    let disposed = false;
    let teardown: (() => void) | null = null;

    const boot = async () => {
      const sceneMgr = new SceneManager(container);
      const physics = await PhysicsWorld.create([0, -9.81, 0]);
      if (disposed) {
        physics.dispose();
        sceneMgr.dispose();
        return;
      }

      const width = container.clientWidth || window.innerWidth;
      const height = container.clientHeight || window.innerHeight;
      const postPipeline = new PostProcessingManager(width, height);
      postPipeline.setVisionMode(mapVision(visionModeRef.current));

      const sectorMgr = new SectorManager(sceneMgr.getScene(), physics, 8);
      const baselineSim = new BaselineSimulator();
      sectorMgr.bindBaselineSimulator(baselineSim);
      const compiler = new WorldCompiler(sceneMgr.getScene(), sectorMgr);

      for (const sector of STARTING_SECTORS) {
        sectorMgr.addSector({ ...sector, timestamp: Date.now() });
      }
      sectorMgr.setFractureMode(baselineModeRef.current);

      const ribbons: THREE.Mesh[] = [];
      for (const seamZ of [20, 60, 100, 140, 180]) {
        const ribGeo = new THREE.BoxGeometry(28, 0.22, 0.28);
        const ribMat = new THREE.MeshBasicMaterial({
          color: baselineModeRef.current ? 0xff2244 : 0x00f0ff,
          transparent: true,
          opacity: baselineModeRef.current ? 0.85 : 0.35,
        });
        const ribbon = new THREE.Mesh(ribGeo, ribMat);
        ribbon.position.set(0, 0.45, seamZ);
        sceneMgr.add(ribbon);
        ribbons.push(ribbon);
      }
      seamRibbonsRef.current = ribbons;

      const chronoTrailGroup = new THREE.Group();
      sceneMgr.add(chronoTrailGroup);
      const multiverseGhostGroup = new THREE.Group();
      sceneMgr.add(multiverseGhostGroup);

      const activeGhostAnimations: Array<{
        mesh: THREE.Group;
        trajectory: Vec3[];
        elapsed: number;
        duration: number;
      }> = [];

      const rover = new RoverModel({
        scale: 1,
        enableSpotlights: true,
        spotlightIntensity: 3.5,
        initialPosition: new THREE.Vector3(0, 0.8, 0),
      });
      sceneMgr.add(rover.getGroup());
      physics.ensureRoverKinematic('argon_rover', ROVER_HALF, [0, 0.8 + ROVER_HALF_Y, 0]);

      const lidarVis = new LiDARVisualizer({
        beamCount: 32,
        maxRange: 45,
        beamColor: 0x00f0ff,
      });
      sceneMgr.add(lidarVis.getGroup());

      sceneMgr.setViewMode(mapCamera(cameraModeRef.current));
      sceneMgrRef.current = sceneMgr;
      physicsWorldRef.current = physics;
      roverModelRef.current = rover;
      sectorMgrRef.current = sectorMgr;
      postPipelineRef.current = postPipeline;
      soundEffects.startRoverEngine();

      const publishMetrics = () => {
        onMetricsRef.current(sectorMgr.getComparisonMetrics(tokenCountRef.current));
      };

      const present = (result: WorldCompileResult) => {
        if (disposed) return;
        const st = roverStateRef.current;
        const hasAny =
          result.hazards.length > 0 ||
          result.countermeasures.length > 0 ||
          result.sectors.length > 0 ||
          result.physicsMutations.length > 0 ||
          result.roverMorphs.length > 0;
        if (!hasAny) return;

        realizeSpawns(sceneMgr.getScene(), physics, result, () => postPipeline.triggerGlitch(0.85));

        for (const hz of result.hazards) {
          st.hazardZones.push({ pos: hz.position, radius: hz.radius, type: hz.hazardType });
          if (hz.hazardType === 'steam_pipe_burst' || hz.hazardType === 'radioactive_leak') {
            st.lidarMaxRange = 22;
            const fog = sceneMgr.getScene().fog;
            if (fog instanceof THREE.FogExp2) fog.density = 0.018;
          }
        }

        for (const cm of result.countermeasures) {
          if (cm.countermeasureType === 'deployable_bridge' || cm.countermeasureType === 'carbon_ramp') {
            st.bridgedZones.push({ pos: cm.position, span: Math.max(14, cm.dimensions[2]) });
            setHudBanner(`DEPLOYED ${cm.countermeasureType.toUpperCase().replace('_', ' ')} — CHASM / SEAM BRIDGED`);
          } else if (cm.countermeasureType === 'air_scrubber') {
            st.lidarMaxRange = 55;
            const fog = sceneMgr.getScene().fog;
            if (fog instanceof THREE.FogExp2) fog.density = 0.004;
            setHudBanner('ELECTROSTATIC AIR SCRUBBER ACTIVE — PARTICULATE FOG PURGED');
          } else if (cm.countermeasureType === 'sensor_beacon') {
            st.lidarMaxRange = 70;
            st.integrityBonus = Math.min(25, st.integrityBonus + 12);
            setHudBanner('SENSOR BEACON ACTIVE — LIDAR RANGE BOOSTED TO 70m');
          }
        }

        for (const physMut of result.physicsMutations) {
          st.physicsMutation = {
            gravityY: physMut.gravityY,
            surfaceFriction: physMut.surfaceFriction,
            lateralWindX: physMut.lateralWindX,
            fogDensity: physMut.fogDensity,
            environmentLabel: physMut.environmentLabel,
          };
          physics.world.gravity.y = physMut.gravityY;
          const fog = sceneMgr.getScene().fog;
          if (fog instanceof THREE.FogExp2) fog.density = physMut.fogDensity;
          setHudBanner(
            `KRONOS PHYSICS MUTATION: ${physMut.environmentLabel.toUpperCase()} (G=${physMut.gravityY.toFixed(2)}, μ=${physMut.surfaceFriction.toFixed(2)}, WIND=${physMut.lateralWindX.toFixed(1)}m/s)`
          );
        }

        for (const morph of result.roverMorphs) {
          st.roverMorph = {
            mode: morph.mode,
            suspensionHeight: morph.suspensionHeight,
            trackWidthScale: morph.trackWidthScale,
            thrusterBoost: morph.thrusterBoost,
          };
          rover.applyMorphology(st.roverMorph);
          setHudBanner(
            `KRONOS CO-DESIGN MORPH: ${morph.mode.toUpperCase().replace(/_/g, ' ')} (LIFT ${morph.suspensionHeight.toFixed(2)}x, TRACK ${morph.trackWidthScale.toFixed(2)}x)`
          );
        }

        const solid = result.hazards.some(hazard => hazard.dynamicSpheres.length > 0);
        if (solid) soundEffects.playStructuralCollapse();
        else if (result.hazards.length > 0) soundEffects.playAdversaryAlarm();
        if (result.countermeasures.length > 0 || result.roverMorphs.length > 0) {
          soundEffects.playCountermeasureDeploy();
        }
        if (result.hazards.length > 0) roverStateRef.current.activeHazards += result.hazards.length;
        if (result.sectors.length > 0) publishMetrics();
      };

      const appliedCritiques = new Set<string>();
      const applyCritique = (critique: AdversaryCritique) => {
        if (disposed || appliedCritiques.has(critique.id)) return;
        appliedCritiques.add(critique.id);
        const st = roverStateRef.current;
        const snippet = critique.hazardCodeSnippet ?? '';
        const parsed = /TRIGGER_HAZARD\s*\{[\s\S]*\}/.test(snippet)
          ? applyDirectives(compiler, snippet)
          : null;
        if (parsed && parsed.hazards.length > 0) {
          present(parsed);
          return;
        }
        present(applyDirectives(compiler, synthesizedCritiqueDirectives(critique, st.pos, st.heading)));
      };

      hazardHandlerRef.current = applyCritique;
      if (pendingHazardRef.current) applyCritique(pendingHazardRef.current);
      publishMetrics();

      const handleKeyDown = (event: KeyboardEvent) => {
        keysPressed.current[event.key.toLowerCase()] = true;
        if (['w', 'a', 's', 'd', 'arrowup', 'arrowdown', 'arrowleft', 'arrowright'].includes(event.key.toLowerCase())) {
          isAutonomousRef.current = false;
          setIsAutonomous(false);
        }
      };
      const handleKeyUp = (event: KeyboardEvent) => {
        keysPressed.current[event.key.toLowerCase()] = false;
      };
      window.addEventListener('keydown', handleKeyDown);
      window.addEventListener('keyup', handleKeyUp);

      const unsubVoice = geminiService.onVoice((toolCall) => {
        const st = roverStateRef.current;
        present(applyDirectives(compiler, voiceCountermeasureDsl(toolCall.countermeasureType, st.pos, st.heading)));
      });
      const unsubAdversary = geminiService.onAdversary((critique) => {
        onAdversaryRef.current(critique);
        applyCritique(critique);
      });
      const unsubDelta = geminiService.onWorldDelta((delta) => {
        present(compiler.applyDelta(asCompilerDelta(delta)));
      });
      const unsubToken = geminiService.onToken((chunk) => {
        if (!chunk) return;
        present(compiler.ingestText(chunk));
      });

      const handleResize = () => {
        const host = containerRef.current;
        const liveScene = sceneMgrRef.current;
        if (!host || !liveScene) return;
        liveScene.handleResize(host.clientWidth, host.clientHeight);
        postPipelineRef.current?.handleResize(host.clientWidth, host.clientHeight);
      };
      window.addEventListener('resize', handleResize);

      const yaw = new THREE.Quaternion();
      const yawAxis = new THREE.Vector3(0, 1, 0);
      const cameraTarget = new THREE.Vector3();
      let lastTime = performance.now();
      let frameCounter = 0;
      const executeMultiverseFork = (reason: string) => {
        if (disposed) return;
        const st = roverStateRef.current;
        const fork = kronosEngine.triggerMultiverseFork(st.pos, st.heading, reason, {
          sampleSupportHeight: (x, z) =>
            physics.sampleSupportHeight(x, z, st.pos[1] + 25) ?? sectorMgr.getHeightAt(x, z),
          surfaceFriction: st.physicsMutation.surfaceFriction,
          lateralWindX: st.physicsMutation.lateralWindX,
          isFractured64k: baselineModeRef.current,
          hazardZones: st.hazardZones,
        });
        onMultiverseForkRef.current?.(fork);
        soundEffects.playCountermeasureDeploy();
        setHudBanner(
          `⚡ 4-WAY COUNTERFACTUAL MULTIVERSE FORK [thinking_level: HIGH] — CHOSEN: GHOST Ω (SCORE 4.9/5)`
        );

        while (multiverseGhostGroup.children.length > 0) {
          multiverseGhostGroup.remove(multiverseGhostGroup.children[0]);
        }
        activeGhostAnimations.length = 0;

        for (const branch of fork.branches) {
          const pts = branch.trajectory.map(p => new THREE.Vector3(p[0], p[1] + 0.35, p[2]));
          const curveGeo = new THREE.BufferGeometry().setFromPoints(pts);
          const lineMat = new THREE.LineBasicMaterial({
            color: branch.colorHex,
            transparent: true,
            opacity: branch.dpoRole === 'chosen' ? 0.95 : 0.65,
          });
          multiverseGhostGroup.add(new THREE.Line(curveGeo, lineMat));

          const ghostGroup = new THREE.Group();
          const bodyGeo = new THREE.BoxGeometry(1.1, 0.5, 2.1);
          const bodyMat = new THREE.MeshBasicMaterial({
            color: branch.colorHex,
            wireframe: branch.dpoRole === 'rejected',
            transparent: true,
            opacity: branch.dpoRole === 'chosen' ? 0.72 : 0.48,
          });
          ghostGroup.add(new THREE.Mesh(bodyGeo, bodyMat));
          ghostGroup.position.set(st.pos[0], st.pos[1], st.pos[2]);
          multiverseGhostGroup.add(ghostGroup);

          activeGhostAnimations.push({
            mesh: ghostGroup,
            trajectory: branch.trajectory,
            elapsed: 0,
            duration: 3.2,
          });
        }

        const chosen = fork.branches.find(b => b.dpoRole === 'chosen');
        if (chosen) {
          geminiService.emitDslSnippet(`\n// [KRONOS-1M CHOSEN DPO BRANCH Ω]\n${chosen.actionDsl}\n`);
        }
      };

      forkHandlerRef.current = executeMultiverseFork;

      let lastRayHits: ReturnType<PhysicsWorld['sweepLiDARScan']> = [];

      let animationFrameId = 0;

      const updateRoverMovement = (dt: number) => {
        const st = roverStateRef.current;
        const fx = Math.sin(st.heading);
        const fz = Math.cos(st.heading);

        const currentSupport =
          physics.sampleSupportHeight(st.pos[0], st.pos[2], st.pos[1] + 25) ??
          sectorMgr.getHeightAt(st.pos[0], st.pos[2]);
        let maxDropAhead = 0;
        let hasRampOrBridgeAhead = false;

        for (const dist of [3.0, 6.0, 9.0]) {
          const px = st.pos[0] + fx * dist;
          const pz = st.pos[2] + fz * dist;
          const aheadSupport =
            physics.sampleSupportHeight(px, pz, st.pos[1] + 25) ?? sectorMgr.getHeightAt(px, pz);
          const drop = currentSupport - aheadSupport;
          if (drop > maxDropAhead) maxDropAhead = drop;
          if (aheadSupport - currentSupport > 0.35) hasRampOrBridgeAhead = true;
        }

        const isNearBridgedZone = st.bridgedZones.some(
          b => Math.hypot(b.pos[0] - st.pos[0], b.pos[2] - (st.pos[2] + 5)) < b.span * 0.85
        );
        const isNearUnbridgedChasm =
          !isNearBridgedZone &&
          st.hazardZones.some(
            h =>
              h.type === 'chasm_rupture' &&
              h.pos[2] > st.pos[2] - 2 &&
              h.pos[2] - st.pos[2] < 13 &&
              Math.abs(h.pos[0] - st.pos[0]) < h.radius * 0.75
          );

        const seamBlockAhead =
          (baselineModeRef.current && maxDropAhead > 1.05 && !isNearBridgedZone) || isNearUnbridgedChasm;

        if (isAutonomousRef.current) {
          if (seamBlockAhead && !st.roverMorph.thrusterBoost) {
            st.speed = THREE.MathUtils.lerp(st.speed, 0, 0.35);
            st.steerAngle = 0;
            st.autonomousState = 'route_blocked';
            st.verticalJumpOffset = THREE.MathUtils.lerp(st.verticalJumpOffset, 0, 0.25);
          } else if (seamBlockAhead && st.roverMorph.thrusterBoost) {
            st.speed = THREE.MathUtils.lerp(st.speed, 11.5, 0.15);
            st.steerAngle = THREE.MathUtils.lerp(st.steerAngle, 0, 0.2);
            st.autonomousState = 'scaling_ramp';
            st.verticalJumpOffset = THREE.MathUtils.lerp(st.verticalJumpOffset, 2.4, 0.18);
          } else if (hasRampOrBridgeAhead || isNearBridgedZone) {
            st.speed = THREE.MathUtils.lerp(st.speed, 7.5, 0.12);
            st.steerAngle = THREE.MathUtils.lerp(st.steerAngle, -st.pos[0] * 0.04, 0.15);
            st.autonomousState = 'scaling_ramp';
            st.verticalJumpOffset = THREE.MathUtils.lerp(st.verticalJumpOffset, 0, 0.2);
          } else {
            let leftClear = 40;
            let centerClear = 40;
            let rightClear = 40;
            const n = lastRayHits.length;
            for (let i = 0; i < n; i++) {
              const hit = lastRayHits[i];
              if (hit.entityType === 'terrain' && hit.hitNormal[1] > 0.72) continue;
              if (i < n * 0.35) leftClear = Math.min(leftClear, hit.distance);
              else if (i > n * 0.65) rightClear = Math.min(rightClear, hit.distance);
              else centerClear = Math.min(centerClear, hit.distance);
            }

            if (centerClear < 13.5) {
              const turnDir = leftClear >= rightClear ? 1 : -1;
              st.steerAngle = THREE.MathUtils.lerp(st.steerAngle, turnDir * 0.38, 0.22);
              st.speed = THREE.MathUtils.lerp(st.speed, 5.8, 0.15);
              st.autonomousState = 'avoiding_hazard';
            } else {
              const recenter = THREE.MathUtils.clamp(-st.pos[0] * 0.035 - st.heading * 0.45, -0.22, 0.22);
              st.steerAngle = THREE.MathUtils.lerp(st.steerAngle, recenter, 0.12);
              st.speed = THREE.MathUtils.lerp(st.speed, 8.8, 0.1);
              st.autonomousState = 'navigating';
            }
            st.verticalJumpOffset = THREE.MathUtils.lerp(st.verticalJumpOffset, 0, 0.2);
          }
          st.heading += st.steerAngle * dt;
        } else {
          if (keysPressed.current['w'] || keysPressed.current['arrowup']) {
            st.speed = Math.min(st.speed + 12 * dt, 14);
          } else if (keysPressed.current['s'] || keysPressed.current['arrowdown']) {
            st.speed = Math.max(st.speed - 15 * dt, -5);
          } else {
            st.speed = THREE.MathUtils.lerp(st.speed, 0, 0.05);
          }

          if (keysPressed.current['a'] || keysPressed.current['arrowleft']) {
            st.steerAngle = Math.min(st.steerAngle + 2 * dt, 0.45);
          } else if (keysPressed.current['d'] || keysPressed.current['arrowright']) {
            st.steerAngle = Math.max(st.steerAngle - 2 * dt, -0.45);
          } else {
            st.steerAngle = THREE.MathUtils.lerp(st.steerAngle, 0, 0.15);
          }

          st.heading += st.steerAngle * (st.speed / 5) * dt;
          st.autonomousState = 'voice_override';
        }

        const windDriftX =
          st.speed > 0.2
            ? st.physicsMutation.lateralWindX * 0.045 * (1.1 - st.physicsMutation.surfaceFriction) * dt
            : 0;
        const dx = Math.sin(st.heading) * st.speed * dt + windDriftX;
        const dz = Math.cos(st.heading) * st.speed * dt;
        st.pos[0] = THREE.MathUtils.clamp(st.pos[0] + dx, -11.5, 11.5);
        st.pos[2] += dz;
        st.traversedDistance += Math.abs(dz);
        st.velocity = [dt > 0 ? dx / dt : 0, 0, dt > 0 ? dz / dt : 0];
      };

      const frame = (currentTime: number) => {
        if (disposed) return;
        const dt = Math.min((currentTime - lastTime) / 1000, 0.05);
        lastTime = currentTime;
        frameCounter += 1;

        updateRoverMovement(dt);

        const st = roverStateRef.current;
        const support = physics.sampleSupportHeight(st.pos[0], st.pos[2], st.pos[1] + 25);
        const ground = support ?? sectorMgr.getHeightAt(st.pos[0], st.pos[2]);
        const previousY = st.pos[1];
        st.pos[1] = THREE.MathUtils.lerp(st.pos[1], ground + 0.02 + st.verticalJumpOffset, 0.45);
        rover.setPosition(st.pos[0], st.pos[1], st.pos[2]);
        rover.setRotation(0, st.heading, 0);
        rover.update(dt, st.speed, st.steerAngle);

        // Animate 4 Multiverse Ghost Rovers along their counterfactual trajectories
        for (const anim of activeGhostAnimations) {
          anim.elapsed += dt;
          const u = Math.min(1, anim.elapsed / anim.duration);
          const idxFloat = u * (anim.trajectory.length - 1);
          const i0 = Math.floor(idxFloat);
          const i1 = Math.min(anim.trajectory.length - 1, i0 + 1);
          const frac = idxFloat - i0;
          const p0 = anim.trajectory[i0];
          const p1 = anim.trajectory[i1];
          anim.mesh.position.set(
            THREE.MathUtils.lerp(p0[0], p1[0], frac),
            THREE.MathUtils.lerp(p0[1], p1[1], frac) + 0.4,
            THREE.MathUtils.lerp(p0[2], p1[2], frac)
          );
        }

        yaw.setFromAxisAngle(yawAxis, st.heading);
        physics.updateRoverKinematic(
          [st.pos[0], st.pos[1] + ROVER_HALF_Y, st.pos[2]],
          [yaw.x, yaw.y, yaw.z, yaw.w]
        );
        physics.step(dt);
        physics.syncDynamicMeshes();

        const mast: Vec3 = [st.pos[0], st.pos[1] + 1.15, st.pos[2]];
        const rayHits = physics.sweepLiDARScan(mast, st.heading, 32, 110, -0.08, st.lidarMaxRange);
        lastRayHits = rayHits;
        lidarVis.update(mast, rayHits);

        sceneMgr.setViewMode(mapCamera(cameraModeRef.current));
        cameraTarget.set(st.pos[0], st.pos[1], st.pos[2]);
        sceneMgr.updateCamera(cameraTarget, dt, st.heading, 0);

        const nextSectorFrontier = Math.floor(st.pos[2] / 40) + 1;
        if (nextSectorFrontier > sectorFrontierCounter.current) {
          sectorFrontierCounter.current = nextSectorFrontier;
          geminiService.requestSectorStream(nextSectorFrontier, [st.pos[0], st.pos[1], st.pos[2]]);
        }
        sectorMgr.update(st.pos);

        let minClearance = 35;
        for (const hit of rayHits) {
          if (hit.distance < minClearance) minClearance = hit.distance;
        }
        const verticalSpeed = dt > 0 ? (st.pos[1] - previousY) / dt : 0;
        const pitchDeg = THREE.MathUtils.radToDeg(Math.atan2(verticalSpeed, Math.max(1, Math.abs(st.speed))));
        const rollDeg = THREE.MathUtils.radToDeg(st.steerAngle) * 0.35;
        const wheelSlipRatio = +(
          Math.min(
            0.95,
            Math.max(0.04, (1 - st.physicsMutation.surfaceFriction) * 0.65 + Math.abs(st.physicsMutation.lateralWindX) * 0.012)
          )
        ).toFixed(2);
        const tippingMarginDeg = +(Math.max(4, 34 - Math.hypot(pitchDeg, rollDeg) * 1.4)).toFixed(1);

        const telemetry: RobotTelemetry = {
          position: [st.pos[0], st.pos[1], st.pos[2]],
          velocity: [st.velocity[0], verticalSpeed, st.velocity[2]],
          speedKmh: st.speed * 3.6,
          headingDeg: (st.heading * 180) / Math.PI,
          pitchDeg,
          rollDeg,
          traversedDistanceMeters: st.traversedDistance,
          obstacleClearanceMeters: minClearance,
          structuralIntegrityPct: Math.min(100, Math.max(62, 100 - st.activeHazards * 5 + st.integrityBonus)),
          lidarRayHits: rayHits.filter(hit => hit.distance < st.lidarMaxRange * 0.9).length,
          autonomousState: st.autonomousState,
          targetWaypoint: [0, 0, st.pos[2] + 45],
          wheelSlipRatio,
          tippingMarginDeg,
          activeMorphology: st.roverMorph.mode,
          physicsRegime: st.physicsMutation.environmentLabel,
        };
        onTelemetryRef.current(telemetry);
        soundEffects.updateEnginePitch(telemetry.speedKmh);

        // Step KRONOS-1M 0.5s Dense Process Rewards & 3D Volumetric Chrono-Ribbon
        const { newTick, shouldTriggerAutoFork } = kronosEngine.stepTelemetry(
          telemetry,
          baselineModeRef.current
        );
        if (newTick) {
          const tickColor =
            newTick.rating === 'normal'
              ? 0x10b981
              : newTick.rating === 'suboptimal'
                ? 0xf59e0b
                : 0xff2244;
          const nodeGeo = new THREE.BoxGeometry(0.65, 0.14, 0.9);
          const nodeMat = new THREE.MeshBasicMaterial({
            color: tickColor,
            transparent: true,
            opacity: 0.72,
          });
          const nodeMesh = new THREE.Mesh(nodeGeo, nodeMat);
          nodeMesh.position.set(newTick.position[0], newTick.position[1] + 0.12, newTick.position[2]);
          chronoTrailGroup.add(nodeMesh);
          if (chronoTrailGroup.children.length > 70) {
            chronoTrailGroup.remove(chronoTrailGroup.children[0]);
          }
        }
        if (shouldTriggerAutoFork) {
          executeMultiverseFork(
            baselineModeRef.current
              ? 'Auto-Fork: 64K Seam Cliff Halt detected by 0.5s Process Reward'
              : 'Auto-Fork: Corridor Block detected by 0.5s Process Reward'
          );
        }

        if (frameCounter % 300 === 0) {
          try {
            const frameBase64 = sceneMgr.getRenderer().domElement.toDataURL('image/jpeg', 0.55);
            geminiService.submitViewportFrame(frameBase64, telemetry);
          } catch {
            // The canvas can be unavailable for a single frame during resize.
          }
        }

        postPipeline.render(sceneMgr.getRenderer(), sceneMgr.getScene(), sceneMgr.getCamera(), dt);
        animationFrameId = requestAnimationFrame(frame);
      };

      animationFrameId = requestAnimationFrame(frame);

      teardown = () => {
        disposed = true;
        hazardHandlerRef.current = null;
        cancelAnimationFrame(animationFrameId);
        window.removeEventListener('keydown', handleKeyDown);
        window.removeEventListener('keyup', handleKeyUp);
        window.removeEventListener('resize', handleResize);
        unsubVoice();
        unsubAdversary();
        unsubDelta();
        unsubToken();
        sectorMgr.dispose();
        lidarVis.dispose();
        physics.dispose();
        sceneMgr.dispose();
        sceneMgrRef.current = null;
        physicsWorldRef.current = null;
        roverModelRef.current = null;
        sectorMgrRef.current = null;
        postPipelineRef.current = null;
      };
    };

    boot();

    return () => {
      disposed = true;
      teardown?.();
      teardown = null;
    };
  }, []);

  return (
    <div className="relative w-full h-full overflow-hidden bg-black select-none">
      <div ref={containerRef} className="w-full h-full" />

      <div className="absolute top-4 left-4 z-20 flex flex-wrap items-center gap-2">
        <div className="bg-[#0b1017]/90 backdrop-blur-md border border-gray-800 p-1 rounded-lg flex space-x-1 text-[11px] font-mono text-gray-300">
          <button
            onClick={() => setCameraMode('chase')}
            className={`px-2.5 py-1 rounded transition ${
              cameraMode === 'chase' ? 'bg-cyan-500 text-black font-bold' : 'hover:bg-gray-800'
            }`}
          >
            CHASE CAM
          </button>
          <button
            onClick={() => setCameraMode('isometric')}
            className={`px-2.5 py-1 rounded transition ${
              cameraMode === 'isometric' ? 'bg-cyan-500 text-black font-bold' : 'hover:bg-gray-800'
            }`}
          >
            TACTICAL ORBIT
          </button>
          <button
            onClick={() => setCameraMode('cockpit')}
            className={`px-2.5 py-1 rounded transition ${
              cameraMode === 'cockpit' ? 'bg-cyan-500 text-black font-bold' : 'hover:bg-gray-800'
            }`}
          >
            ROVER MAST POV
          </button>
        </div>

        <div className="bg-[#0b1017]/90 backdrop-blur-md border border-gray-800 p-1 rounded-lg flex space-x-1 text-[11px] font-mono text-gray-300">
          <button
            onClick={() => setVisionMode('standard')}
            className={`px-2.5 py-1 rounded transition ${
              visionMode === 'standard' ? 'bg-cyan-500 text-black font-bold' : 'hover:bg-gray-800'
            }`}
          >
            STD
          </button>
          <button
            onClick={() => setVisionMode('nvg')}
            className={`px-2.5 py-1 rounded transition ${
              visionMode === 'nvg' ? 'bg-cyan-500 text-black font-bold' : 'hover:bg-gray-800'
            }`}
          >
            NVG
          </button>
          <button
            onClick={() => setVisionMode('flir')}
            className={`px-2.5 py-1 rounded transition ${
              visionMode === 'flir' ? 'bg-cyan-500 text-black font-bold' : 'hover:bg-gray-800'
            }`}
          >
            FLIR
          </button>
        </div>

        <button
          onClick={() => {
            const next = !isAutonomousRef.current;
            isAutonomousRef.current = next;
            setIsAutonomous(next);
          }}
          className={`px-3 py-1.5 rounded-lg border text-xs font-mono font-bold transition flex items-center space-x-1.5 ${
            isAutonomous
              ? 'bg-emerald-950/80 border-emerald-500/80 text-emerald-300 shadow-[0_0_10px_rgba(16,185,129,0.3)]'
              : 'bg-amber-950/80 border-amber-500/80 text-amber-300'
          }`}
          title="Toggle Autonomous Navigation AI vs Manual WASD Controls"
        >
          <span className={`w-2 h-2 rounded-full ${isAutonomous ? 'bg-emerald-400 animate-pulse' : 'bg-amber-400'}`} />
          <span>{isAutonomous ? 'AI AUTOPILOT: ON' : 'MANUAL DRIVE: WASD'}</span>
        </button>

        <button
          onClick={() => setJudgeMode(prev => !prev)}
          className={`px-3 py-1.5 rounded-lg border text-xs font-mono font-bold transition ${
            judgeMode
              ? 'bg-violet-950/80 border-violet-400 text-violet-200 shadow-[0_0_10px_rgba(167,139,250,0.35)]'
              : 'bg-gray-900 border-gray-700 text-gray-300 hover:border-violet-400 hover:text-violet-200'
          }`}
          title="Replay the checked-in Argon transcript for sector requests, including while the socket is connected"
        >
          {judgeMode ? 'JUDGE MODE: REPLAY' : 'JUDGE MODE: OFF'}
        </button>
      </div>

      <div className="absolute inset-0 pointer-events-none flex items-center justify-center">
        <div className="w-8 h-8 border border-cyan-400/20 rounded-full flex items-center justify-center">
          <div className="w-1.5 h-1.5 bg-cyan-400/40 rounded-full" />
        </div>
      </div>

      {hudBanner && (
        <div className="absolute top-14 left-4 z-20 pointer-events-none">
          <div
            className={`px-3 py-1.5 rounded-lg border text-[11px] font-mono font-bold tracking-wide backdrop-blur-md shadow-lg ${
              hudBanner.includes('BLOCKED') || hudBanner.includes('FRACTURED')
                ? 'bg-red-950/90 border-red-500 text-red-200 animate-pulse'
                : 'bg-cyan-950/85 border-cyan-500/80 text-cyan-200'
            }`}
          >
            {hudBanner}
          </div>
        </div>
      )}
    </div>
  );
};
