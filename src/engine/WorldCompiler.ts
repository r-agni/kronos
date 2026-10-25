import * as THREE from 'three';
import {
  ProceduralDSLParser,
  type DSLCommand,
  type AddSectorCommand,
  type ExtendRampCommand,
  type SpawnStructureCommand,
  type TriggerHazardCommand,
  type DeployCountermeasureCommand,
  type MutatePhysicsCommand,
  type MorphRoverCommand,
} from './ProceduralDSL';
import { SectorManager, type SectorInstance } from './SectorManager';
import type {
  HazardEvent,
  HazardType,
  PhysicsMutationState,
  RoverMorphConfig,
  TacticalCountermeasure,
  TerrainSector,
  Vec3,
} from '../types/simulation';

/**
 * Payload broadcast by the server as `{ type: 'WORLD_DELTA', payload }` when a
 * sector stream completes. `sector` / `structures` / `hazards` come from
 * `parseDSLDirectives`. `rawDirectives` is the full DSL text, which the client
 * parser treats as authoritative for incremental compilation.
 */
export interface WorldDeltaPayload {
  sector: Partial<TerrainSector> | null;
  structures: WorldDeltaStructure[];
  hazards: WorldDeltaHazard[];
  rawDirectives: string;
}

export interface WorldDeltaMessage {
  type: 'WORLD_DELTA';
  payload: WorldDeltaPayload;
}

export interface WorldDeltaStructure {
  type: string;
  position: Vec3;
  scale?: Vec3;
  material?: string;
}

export type WorldDeltaHazard = Partial<HazardEvent> & {
  severity?: HazardEvent['severity'] | number;
};

/** Arguments for `PhysicsWorld.addStaticStructure(id, position, halfExtents, entityType, rotationQuat)`. */
export interface StaticStructureSpawn {
  id: string;
  position: Vec3;
  halfExtents: Vec3;
  entityType: 'structure' | 'countermeasure';
  rotationQuat: [number, number, number, number];
}

/** Arguments for `PhysicsWorld.addDynamicSphere(id, position, radius, options)`. */
export interface DynamicSphereSpawn {
  id: string;
  position: Vec3;
  radius: number;
  options: {
    mass: number;
    friction: number;
    restitution: number;
    entityType: 'hazard';
  };
}

export interface HazardSpawnDescriptor {
  kind: 'hazard';
  id: string;
  hazardType: HazardType;
  title: string;
  /** Area center from the directive. Debris bodies may spawn above this point. */
  position: Vec3;
  radius: number;
  severity: 'warning' | 'critical' | 'catastrophic' | number;
  /**
   * Solid debris to register with the physics world. Empty for non-solid hazards
   * (leaks, steam, EMP). The compiler does not create bodies or hazard meshes.
   */
  dynamicSpheres: DynamicSphereSpawn[];
}

export interface CountermeasureVisualHint {
  shape: 'box' | 'cylinder';
  dimensions: Vec3;
  color: number;
  emissive: number;
  emissiveIntensity: number;
}

export interface CountermeasureSpawnDescriptor {
  kind: 'countermeasure';
  id: string;
  countermeasureType: TacticalCountermeasure['type'];
  origin: 'EXTEND_RAMP' | 'DEPLOY_COUNTERMEASURE';
  /** Collider center. For ramps this sits half a deck-thickness below the start/end centerline. */
  position: Vec3;
  rotationEuler: Vec3;
  dimensions: Vec3;
  material: string;
  visual: CountermeasureVisualHint;
  /** Pass to `PhysicsWorld.addStaticStructure`. The compiler does not create the body or the mesh. */
  staticStructure: StaticStructureSpawn;
}

export interface SpawnedStructure {
  id: string;
  structureType: string;
  position: Vec3;
  scale: Vec3;
  material: string;
  /** Already added to the scene. */
  mesh: THREE.Object3D;
}

export interface PhysicsMutationDescriptor extends PhysicsMutationState {
  kind: 'physics_mutation';
  id: string;
}

export interface RoverMorphDescriptor extends RoverMorphConfig {
  kind: 'rover_morph';
  id: string;
}

export interface WorldCompileResult {
  sectors: SectorInstance[];
  structures: SpawnedStructure[];
  hazards: HazardSpawnDescriptor[];
  countermeasures: CountermeasureSpawnDescriptor[];
  physicsMutations: PhysicsMutationDescriptor[];
  roverMorphs: RoverMorphDescriptor[];
}

const HAZARD_TYPES: readonly HazardType[] = [
  'seismic_collapse',
  'radioactive_leak',
  'chasm_rupture',
  'steam_pipe_burst',
  'structural_overpass_fall',
  'emp_interference',
];

const FALLING_HAZARDS = new Set<HazardType>([
  'seismic_collapse',
  'chasm_rupture',
  'structural_overpass_fall',
]);

const DEBRIS_PATTERN: ReadonlyArray<{ offset: Vec3; radius: number }> = [
  { offset: [-1.6, 6.0, -0.8], radius: 0.75 },
  { offset: [1.4, 7.2, 0.6], radius: 0.95 },
  { offset: [-0.4, 8.5, 1.5], radius: 0.7 },
  { offset: [0.8, 6.6, -1.6], radius: 1.05 },
];

const DEFAULT_STRUCTURE_SCALE: Record<string, Vec3> = {
  pipe_rack: [2, 7.5, 2],
  monolithic_arch: [14, 6, 3],
  retaining_wall: [8, 4, 0.8],
  catwalk: [1.6, 0.25, 12],
  cooling_tower: [8, 16, 8],
  tower: [3, 14, 3],
  overpass: [8, 1.2, 18],
  tunnel: [12, 6, 14],
  bridge: [8, 1, 16],
};

/**
 * Compiles Argon DSL into the live world.
 *
 * Sectors go through `SectorManager.addSector`. Structure meshes are added to the
 * provided scene. Hazards and countermeasures are returned as plain descriptors
 * so the integrator can create Rapier bodies; this class never touches physics.
 *
 * Ids are idempotent. `ingestText` accepts each new stream chunk (the parser
 * buffers partial tokens). `applyDelta` accepts a completed `WORLD_DELTA` payload.
 * Calling both for the same directives does not duplicate sectors, meshes, or spawns.
 */
export class WorldCompiler {
  private readonly scene: THREE.Scene;
  private readonly sectorManager: SectorManager;
  private readonly parser: ProceduralDSLParser;
  private readonly appliedKeys = new Set<string>();

  constructor(scene: THREE.Scene, sectorManager: SectorManager) {
    if (!scene) {
      throw new Error('WorldCompiler requires a THREE.Scene');
    }
    if (!sectorManager) {
      throw new Error('WorldCompiler requires a SectorManager');
    }
    this.scene = scene;
    this.sectorManager = sectorManager;
    this.parser = new ProceduralDSLParser();
  }

  /**
   * Apply one completed world delta. Parses `rawDirectives` with a fresh
   * `ProceduralDSLParser` (client parser is authoritative), then fills any
   * sector, structure, or hazard the structured fields carried and the text
   * parser missed.
   */
  public applyDelta(delta: WorldDeltaPayload): WorldCompileResult {
    const result = emptyResult();
    if (!delta) return result;

    const raw = typeof delta.rawDirectives === 'string' ? delta.rawDirectives : '';
    if (raw.length > 0) {
      const commands = new ProceduralDSLParser().parseComplete(raw);
      appendResult(result, this.compileCommands(commands));
    }

    appendResult(result, this.compileStructured(delta));
    return result;
  }

  /**
   * Ingest the next chunk of an Argon token stream. Partial commands stay
   * buffered until their braces close. Pass each new chunk, not the accumulated
   * document.
   */
  public ingestText(chunk: string): WorldCompileResult {
    if (!chunk) return emptyResult();
    return this.compileCommands(this.parser.pushChunk(chunk));
  }

  private compileStructured(delta: WorldDeltaPayload): WorldCompileResult {
    const result = emptyResult();
    const commands: DSLCommand[] = [];

    const sector = delta.sector;
    if (sector && typeof sector === 'object') {
      if (sector.id) this.patchSector(sector.id, sector);
      const sectorCommand = partialSectorCommand(sector);
      if (sectorCommand) commands.push(sectorCommand);
    }

    const structures = Array.isArray(delta.structures) ? delta.structures : [];
    for (const structure of structures) {
      const command = partialStructureCommand(structure);
      if (command) commands.push(command);
    }

    const hazards = Array.isArray(delta.hazards) ? delta.hazards : [];
    for (const hazard of hazards) {
      const command = partialHazardCommand(hazard);
      if (command) commands.push(command);
    }

    appendResult(result, this.compileCommands(commands));

    if (sector && sector.id) this.patchSector(sector.id, sector);
    return result;
  }

  private compileCommands(commands: DSLCommand[]): WorldCompileResult {
    const result = emptyResult();
    for (const command of commands) {
      if (!command) continue;
      const stable = canonicalize(command);
      switch (stable.type) {
        case 'ADD_SECTOR':
          this.applyAddSector(stable, result);
          break;
        case 'SPAWN_STRUCTURE':
          this.applyStructure(stable, result);
          break;
        case 'TRIGGER_HAZARD':
          this.applyHazard(stable, result);
          break;
        case 'EXTEND_RAMP':
          this.applyRamp(stable, result);
          break;
        case 'DEPLOY_COUNTERMEASURE':
          this.applyCountermeasure(stable, result);
          break;
        case 'MUTATE_PHYSICS':
          this.applyPhysicsMutation(stable, result);
          break;
        case 'MORPH_ROVER':
          this.applyRoverMorph(stable, result);
          break;
        default:
          break;
      }
    }
    return result;
  }

  private applyAddSector(cmd: AddSectorCommand, result: WorldCompileResult): void {
    if (!isFiniteVec(cmd.center)) return;
    if (!(cmd.size[0] > 0) || !(cmd.size[1] > 0)) return;

    const keys = [`sector-id:${cmd.id}`, `sector-at:${fmtVec(cmd.center)}`];
    if (!this.claim(keys)) return;
    if (this.sectorManager.getSector(cmd.id)) return;

    const instance = this.sectorManager.addSector(sectorFromAdd(cmd));
    result.sectors.push(instance);
  }

  private applyStructure(cmd: SpawnStructureCommand, result: WorldCompileResult): void {
    if (!isFiniteVec(cmd.position)) return;

    const structureType = structureTypeOf(cmd);
    const keys = [`struct-id:${cmd.id}`, `struct-at:${fmtVec(cmd.position)}`];
    if (!this.claim(keys)) return;

    const scale = resolveStructureScale(cmd, structureType);
    const material = readStringField(cmd.rawSnippet, 'material') ?? 'steel';
    const mesh = buildStructureMesh(structureType, material);
    mesh.name = cmd.id;
    mesh.position.set(cmd.position[0], cmd.position[1], cmd.position[2]);
    mesh.scale.set(scale[0], scale[1], scale[2]);
    mesh.userData = {
      argonId: cmd.id,
      role: 'structure',
      structureType,
      material,
    };
    this.scene.add(mesh);

    result.structures.push({
      id: cmd.id,
      structureType,
      position: [cmd.position[0], cmd.position[1], cmd.position[2]],
      scale,
      material,
      mesh,
    });
  }

  private applyHazard(cmd: TriggerHazardCommand, result: WorldCompileResult): void {
    if (!isFiniteVec(cmd.position)) return;

    const keys = [`hazard-id:${cmd.id}`, `hazard-at:${fmtVec(cmd.position)}`];
    if (!this.claim(keys)) return;

    result.hazards.push(describeHazard(cmd));
  }

  private applyRamp(cmd: ExtendRampCommand, result: WorldCompileResult): void {
    if (!isFiniteVec(cmd.start) || !isFiniteVec(cmd.end)) return;

    const midpoint: Vec3 = [
      (cmd.start[0] + cmd.end[0]) / 2,
      (cmd.start[1] + cmd.end[1]) / 2,
      (cmd.start[2] + cmd.end[2]) / 2,
    ];
    const keys = [
      `cm-id:${cmd.id}`,
      `ramp-span:${fmtVec(cmd.start)}>${fmtVec(cmd.end)}`,
      `cm-at:carbon_ramp@${fmtVec(midpoint)}`,
    ];
    if (!this.claim(keys)) return;

    result.countermeasures.push(describeRamp(cmd));
  }

  private applyCountermeasure(cmd: DeployCountermeasureCommand, result: WorldCompileResult): void {
    if (!isFiniteVec(cmd.position)) return;

    const keys = [
      `cm-id:${cmd.id}`,
      `cm-at:${cmd.countermeasureType}@${fmtVec(cmd.position)}`,
    ];
    if (!this.claim(keys)) return;

    result.countermeasures.push(describeCountermeasure(cmd));
  }

  private applyPhysicsMutation(cmd: MutatePhysicsCommand, result: WorldCompileResult): void {
    const keys = [`phys-id:${cmd.id}`];
    if (!this.claim(keys)) return;

    result.physicsMutations.push({
      kind: 'physics_mutation',
      id: cmd.id,
      gravityY: cmd.gravityY,
      surfaceFriction: cmd.surfaceFriction,
      lateralWindX: cmd.lateralWindX,
      fogDensity: cmd.fogDensity,
      environmentLabel: cmd.environmentLabel,
    });
  }

  private applyRoverMorph(cmd: MorphRoverCommand, result: WorldCompileResult): void {
    const keys = [`morph-id:${cmd.id}`];
    if (!this.claim(keys)) return;

    result.roverMorphs.push({
      kind: 'rover_morph',
      id: cmd.id,
      mode: cmd.mode,
      suspensionHeight: cmd.suspensionHeight,
      trackWidthScale: cmd.trackWidthScale,
      thrusterBoost: cmd.thrusterBoost,
    });
  }

  /** Metadata that does not change the already built terrain mesh. */
  private patchSector(id: string, partial: Partial<TerrainSector>): void {
    const instance = this.sectorManager.getSector(id);
    if (!instance) return;
    const sector = instance.sector;
    if (partial.name) sector.name = partial.name;
    if (typeof partial.roughness === 'number' && Number.isFinite(partial.roughness)) {
      sector.roughness = partial.roughness;
    }
    if (typeof partial.heightScale === 'number' && Number.isFinite(partial.heightScale)) {
      sector.heightScale = partial.heightScale;
    }
    if (partial.hasBridge !== undefined) sector.hasBridge = partial.hasBridge;
    if (partial.hasTunnel !== undefined) sector.hasTunnel = partial.hasTunnel;
    if (typeof partial.clearanceHeight === 'number' && Number.isFinite(partial.clearanceHeight)) {
      sector.clearanceHeight = partial.clearanceHeight;
    }
  }

  /** @returns true when none of the keys were seen before. Always records every key. */
  private claim(keys: string[]): boolean {
    const seen = keys.some(key => this.appliedKeys.has(key));
    for (const key of keys) this.appliedKeys.add(key);
    return !seen;
  }
}

function emptyResult(): WorldCompileResult {
  return {
    sectors: [],
    structures: [],
    hazards: [],
    countermeasures: [],
    physicsMutations: [],
    roverMorphs: [],
  };
}

function appendResult(target: WorldCompileResult, extra: WorldCompileResult): void {
  target.sectors.push(...extra.sectors);
  target.structures.push(...extra.structures);
  target.hazards.push(...extra.hazards);
  target.countermeasures.push(...extra.countermeasures);
  target.physicsMutations.push(...extra.physicsMutations);
  target.roverMorphs.push(...extra.roverMorphs);
}

function canonicalize(cmd: DSLCommand): DSLCommand {
  if (cmd.id && !isUnstableId(cmd.id)) return cmd;
  switch (cmd.type) {
    case 'ADD_SECTOR':
      return { ...cmd, id: `sec_${fmtVec(cmd.center)}` };
    case 'SPAWN_STRUCTURE':
      return { ...cmd, id: `struct_${structureTypeOf(cmd)}_${fmtVec(cmd.position)}` };
    case 'TRIGGER_HAZARD':
      return { ...cmd, id: `hz_${cmd.hazardType}_${fmtVec(cmd.position)}` };
    case 'EXTEND_RAMP':
      return { ...cmd, id: `ramp_${fmtVec(cmd.start)}_${fmtVec(cmd.end)}` };
    case 'DEPLOY_COUNTERMEASURE':
      return { ...cmd, id: `cm_${cmd.countermeasureType}_${fmtVec(cmd.position)}` };
    case 'MUTATE_PHYSICS':
      return { ...cmd, id: `phys_${fmt(cmd.gravityY)}_${fmt(cmd.surfaceFriction)}_${fmt(cmd.lateralWindX)}` };
    case 'MORPH_ROVER':
      return { ...cmd, id: `morph_${cmd.mode}_${fmt(cmd.suspensionHeight)}_${cmd.thrusterBoost ? '1' : '0'}` };
    default:
      return cmd;
  }
}

function partialSectorCommand(partial: Partial<TerrainSector>): AddSectorCommand | null {
  if (!Array.isArray(partial.center) || partial.center.length < 3) return null;
  const center = asVec3(partial.center, [0, 0, 0]);
  if (!isFiniteVec(center)) return null;

  const id = partial.id && !isUnstableId(partial.id) ? partial.id : `sec_${fmtVec(center)}`;
  const biome = asBiome(partial.biome);
  const size = asSize(partial.size);
  const roughness = typeof partial.roughness === 'number' && Number.isFinite(partial.roughness)
    ? partial.roughness
    : 0.35;
  const heightScale = typeof partial.heightScale === 'number' && Number.isFinite(partial.heightScale)
    ? partial.heightScale
    : 4;
  const name = partial.name ?? `Argon Sector ${id}`;
  const clearance = typeof partial.clearanceHeight === 'number' ? partial.clearanceHeight : undefined;
  const rawSnippet = [
    'ADD_SECTOR {',
    `id: "${id}",`,
    `name: "${String(name).replace(/"/g, '')}",`,
    `biome: "${biome}",`,
    `hasBridge: ${partial.hasBridge ? 'true' : 'false'},`,
    `hasTunnel: ${partial.hasTunnel ? 'true' : 'false'},`,
    clearance !== undefined ? `clearanceHeight: ${clearance}` : '',
    '}',
  ].filter(part => part.length > 0).join(' ');

  return {
    type: 'ADD_SECTOR',
    id,
    center,
    size,
    roughness,
    heightScale,
    biome,
    rawSnippet,
  };
}

function partialStructureCommand(structure: WorldDeltaStructure): SpawnStructureCommand | null {
  if (!structure || !isFiniteVec(asVec3(structure.position, [Number.NaN, Number.NaN, Number.NaN]))) {
    return null;
  }
  const position = asVec3(structure.position, [0, 0, 0]);
  const typeName = normalizeStructureType(structure.type || 'bridge');
  const scale = isFiniteVec(structure.scale) ? positiveVec(structure.scale, [1, 1, 1]) : undefined;
  const material = structure.material || 'steel';
  const rawParts = [
    'SPAWN_STRUCTURE {',
    `type: "${typeName}",`,
    `material: "${String(material).replace(/"/g, '')}",`,
  ];
  if (scale) {
    rawParts.push(`scale: [${scale[0]}, ${scale[1]}, ${scale[2]}],`);
  }
  rawParts.push('}');

  return {
    type: 'SPAWN_STRUCTURE',
    id: '',
    structureType: toParserStructureType(typeName),
    position,
    scale: scale ?? [1, 1, 1],
    rawSnippet: rawParts.join(' '),
  };
}

function partialHazardCommand(hazard: WorldDeltaHazard): TriggerHazardCommand | null {
  if (!hazard || !Array.isArray(hazard.position) || hazard.position.length < 3) return null;
  const position = asVec3(hazard.position, [0, 0, 0]);
  if (!isFiniteVec(position)) return null;

  const hazardType = asHazardType(hazard.type);
  const id = hazard.id && !isUnstableId(hazard.id) ? hazard.id : '';
  const radius = typeof hazard.radius === 'number' && hazard.radius > 0 ? hazard.radius : 15;
  const severity = asSeverity(hazard.severity);
  const title = hazard.title ?? `Hazard: ${hazardType.replace(/_/g, ' ').toUpperCase()}`;
  const severityText = typeof severity === 'number' ? String(severity) : `"${severity}"`;

  return {
    type: 'TRIGGER_HAZARD',
    id,
    hazardType,
    position,
    radius,
    severity,
    rawSnippet: `TRIGGER_HAZARD { id: "${id}", type: "${hazardType}", title: "${title.replace(/"/g, '')}", severity: ${severityText} }`,
  };
}

function sectorFromAdd(cmd: AddSectorCommand): TerrainSector {
  const raw = cmd.rawSnippet ?? '';
  const name = readStringField(raw, 'name') ?? `Argon Sector ${cmd.id}`;
  const clearance = readNumberField(raw, 'clearanceHeight');
  const sector: TerrainSector = {
    id: cmd.id,
    name,
    center: [cmd.center[0], cmd.center[1], cmd.center[2]],
    size: [cmd.size[0], cmd.size[1]],
    heightScale: cmd.heightScale,
    roughness: cmd.roughness,
    biome: asBiome(cmd.biome),
    hasBridge: /hasBridge\s*:\s*true/.test(raw),
    hasTunnel: /hasTunnel\s*:\s*true/.test(raw),
    timestamp: Date.now(),
  };
  if (clearance !== undefined) sector.clearanceHeight = clearance;
  return sector;
}

function describeHazard(cmd: TriggerHazardCommand): HazardSpawnDescriptor {
  const title = readStringField(cmd.rawSnippet, 'title')
    ?? `Hazard: ${cmd.hazardType.replace(/_/g, ' ').toUpperCase()}`;
  const dynamicSpheres = FALLING_HAZARDS.has(cmd.hazardType)
    ? debrisSpheres(cmd)
    : [];

  return {
    kind: 'hazard',
    id: cmd.id,
    hazardType: cmd.hazardType,
    title,
    position: [cmd.position[0], cmd.position[1], cmd.position[2]],
    radius: cmd.radius,
    severity: cmd.severity,
    dynamicSpheres,
  };
}

function debrisSpheres(cmd: TriggerHazardCommand): DynamicSphereSpawn[] {
  const spread = Math.min(1, Math.max(0.35, cmd.radius / 12));
  const mass = debrisMass(cmd.severity);
  return DEBRIS_PATTERN.map((piece, index) => ({
    id: `${cmd.id}:debris:${index}`,
    position: [
      cmd.position[0] + piece.offset[0] * spread,
      cmd.position[1] + piece.offset[1],
      cmd.position[2] + piece.offset[2] * spread,
    ],
    radius: piece.radius,
    options: {
      mass,
      friction: 0.8,
      restitution: 0.2,
      entityType: 'hazard' as const,
    },
  }));
}

function describeRamp(cmd: ExtendRampCommand): CountermeasureSpawnDescriptor {
  const start = new THREE.Vector3(cmd.start[0], cmd.start[1], cmd.start[2]);
  const end = new THREE.Vector3(cmd.end[0], cmd.end[1], cmd.end[2]);
  const delta = new THREE.Vector3().subVectors(end, start);
  const length = Math.max(1, delta.length());
  const width = Math.max(0.5, cmd.width || 6);
  const thickness = 0.4;
  const direction = delta.lengthSq() > 1e-8
    ? delta.clone().normalize()
    : new THREE.Vector3(0, 0, 1);
  const quat = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 0, 1), direction);
  const rotationQuat: [number, number, number, number] = [quat.x, quat.y, quat.z, quat.w];
  const up = new THREE.Vector3(0, 1, 0).applyQuaternion(quat);
  const center = new THREE.Vector3()
    .addVectors(start, end)
    .multiplyScalar(0.5)
    .addScaledVector(up, -thickness / 2);
  const position: Vec3 = [center.x, center.y, center.z];
  const dimensions: Vec3 = [width, thickness, length];
  const countermeasureType: TacticalCountermeasure['type'] = 'carbon_ramp';

  return {
    kind: 'countermeasure',
    id: cmd.id,
    countermeasureType,
    origin: 'EXTEND_RAMP',
    position,
    rotationEuler: eulerFromQuat(rotationQuat),
    dimensions,
    material: cmd.material || 'carbon_fiber',
    visual: countermeasureVisual(countermeasureType, dimensions),
    staticStructure: {
      id: cmd.id,
      position,
      halfExtents: halfOf(dimensions),
      entityType: 'countermeasure',
      rotationQuat,
    },
  };
}

function describeCountermeasure(cmd: DeployCountermeasureCommand): CountermeasureSpawnDescriptor {
  const fallback = defaultCountermeasureDimensions(cmd.countermeasureType);
  const dimensions = cmd.dimensions ? positiveVec(cmd.dimensions, fallback) : fallback;
  const rotationEuler: Vec3 = cmd.rotation
    ? [cmd.rotation[0], cmd.rotation[1], cmd.rotation[2]]
    : cmd.countermeasureType === 'carbon_ramp'
      ? [-0.15, 0, 0]
      : [0, 0, 0];
  const position: Vec3 = [cmd.position[0], cmd.position[1], cmd.position[2]];

  return {
    kind: 'countermeasure',
    id: cmd.id,
    countermeasureType: cmd.countermeasureType,
    origin: 'DEPLOY_COUNTERMEASURE',
    position,
    rotationEuler,
    dimensions,
    material: defaultCountermeasureMaterial(cmd.countermeasureType),
    visual: countermeasureVisual(cmd.countermeasureType, dimensions),
    staticStructure: {
      id: cmd.id,
      position,
      halfExtents: halfOf(dimensions),
      entityType: 'countermeasure',
      rotationQuat: quatFromEuler(rotationEuler),
    },
  };
}

function debrisMass(severity: TriggerHazardCommand['severity']): number {
  if (typeof severity === 'number' && Number.isFinite(severity)) {
    if (severity >= 10) return Math.min(2000, Math.max(50, severity));
    return 150 + severity * 700;
  }
  if (severity === 'warning') return 180;
  if (severity === 'catastrophic') return 900;
  return 350;
}

function defaultCountermeasureDimensions(type: TacticalCountermeasure['type']): Vec3 {
  switch (type) {
    case 'carbon_ramp':
      return [6.5, 0.4, 14];
    case 'deployable_bridge':
      return [6, 0.5, 18];
    case 'sensor_beacon':
      return [0.8, 2.5, 0.8];
    case 'air_scrubber':
      return [1.8, 2.4, 1.8];
    case 'blast_barrier':
      return [8, 2.5, 0.6];
    default:
      return [6.5, 0.4, 14];
  }
}

function defaultCountermeasureMaterial(type: TacticalCountermeasure['type']): string {
  switch (type) {
    case 'carbon_ramp':
    case 'deployable_bridge':
      return 'carbon_fiber';
    case 'sensor_beacon':
      return 'emissive_beacon';
    case 'air_scrubber':
      return 'scrubber_housing';
    case 'blast_barrier':
      return 'reinforced_plate';
    default:
      return 'carbon_fiber';
  }
}

function countermeasureVisual(
  type: TacticalCountermeasure['type'],
  dimensions: Vec3
): CountermeasureVisualHint {
  switch (type) {
    case 'sensor_beacon':
      return {
        shape: 'cylinder',
        dimensions,
        color: 0x00f0ff,
        emissive: 0x00f0ff,
        emissiveIntensity: 2.5,
      };
    case 'air_scrubber':
      return {
        shape: 'cylinder',
        dimensions,
        color: 0x14532d,
        emissive: 0x22c55e,
        emissiveIntensity: 0.6,
      };
    case 'blast_barrier':
      return {
        shape: 'box',
        dimensions,
        color: 0x334155,
        emissive: 0xf59e0b,
        emissiveIntensity: 0.2,
      };
    case 'carbon_ramp':
    case 'deployable_bridge':
    default:
      return {
        shape: 'box',
        dimensions,
        color: 0x1e293b,
        emissive: 0x00f0ff,
        emissiveIntensity: 0.35,
      };
  }
}

function structureTypeOf(cmd: SpawnStructureCommand): string {
  const fromRaw = readStringField(cmd.rawSnippet, 'type')
    ?? readStringField(cmd.rawSnippet, 'structureType');
  return normalizeStructureType(fromRaw ?? cmd.structureType);
}

function resolveStructureScale(cmd: SpawnStructureCommand, typeName: string): Vec3 {
  const fallback = DEFAULT_STRUCTURE_SCALE[typeName] ?? [4, 3, 4];
  const authored = /(?:^|[^A-Za-z0-9_])scale\s*[:=]/.test(cmd.rawSnippet ?? '');
  if (!authored) {
    const untouched = cmd.scale[0] === 1 && cmd.scale[1] === 1 && cmd.scale[2] === 1;
    if (untouched) return fallback;
  }
  return positiveVec(cmd.scale, fallback);
}

function normalizeStructureType(typeName: string): string {
  const value = (typeName || '').toLowerCase();
  if (value.includes('pipe')) return 'pipe_rack';
  if (value.includes('arch')) return 'monolithic_arch';
  if (value.includes('retain') || value.includes('wall')) return 'retaining_wall';
  if (value.includes('catwalk') || value.includes('walkway')) return 'catwalk';
  if (value.includes('cooling')) return 'cooling_tower';
  if (value.includes('tower')) return 'tower';
  if (value.includes('overpass')) return 'overpass';
  if (value.includes('tunnel')) return 'tunnel';
  if (value.includes('bridge')) return 'bridge';
  return value || 'bridge';
}

function toParserStructureType(typeName: string): SpawnStructureCommand['structureType'] {
  const normalized = normalizeStructureType(typeName);
  if (normalized === 'tunnel') return 'tunnel';
  if (normalized === 'overpass') return 'overpass';
  if (normalized === 'tower' || normalized === 'cooling_tower') return 'tower';
  return 'bridge';
}

function asBiome(value: unknown): TerrainSector['biome'] {
  if (
    value === 'industrial_hazmat' ||
    value === 'subterranean_tunnel' ||
    value === 'collapsed_chasm' ||
    value === 'reinforced_highway'
  ) {
    return value;
  }
  const text = String(value ?? '').toLowerCase();
  if (text.includes('hazmat') || text.includes('industrial')) return 'industrial_hazmat';
  if (text.includes('tunnel') || text.includes('subterranean')) return 'subterranean_tunnel';
  if (text.includes('chasm') || text.includes('collapse')) return 'collapsed_chasm';
  return 'reinforced_highway';
}

function asHazardType(value: unknown): HazardType {
  if (typeof value === 'string') {
    if ((HAZARD_TYPES as readonly string[]).includes(value)) return value as HazardType;
    const text = value.toLowerCase();
    if (text.includes('chasm')) return 'chasm_rupture';
    if (text.includes('radio')) return 'radioactive_leak';
    if (text.includes('steam')) return 'steam_pipe_burst';
    if (text.includes('overpass')) return 'structural_overpass_fall';
    if (text.includes('emp')) return 'emp_interference';
  }
  return 'seismic_collapse';
}

function asSeverity(value: unknown): 'warning' | 'critical' | 'catastrophic' | number {
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (value === 'warning' || value === 'critical' || value === 'catastrophic') return value;
  return 'critical';
}

function asSize(value: unknown): [number, number] {
  if (Array.isArray(value) && value.length >= 2) {
    const width = Number(value[0]);
    const length = Number(value[1]);
    if (Number.isFinite(width) && Number.isFinite(length) && width > 0 && length > 0) {
      return [width, length];
    }
  }
  return [64, 64];
}

function asVec3(value: unknown, fallback: Vec3): Vec3 {
  if (Array.isArray(value) && value.length >= 3) {
    const x = Number(value[0]);
    const y = Number(value[1]);
    const z = Number(value[2]);
    if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z)) return [x, y, z];
  }
  return [fallback[0], fallback[1], fallback[2]];
}

function positiveVec(value: Vec3, fallback: Vec3): Vec3 {
  const next: Vec3 = [fallback[0], fallback[1], fallback[2]];
  for (let i = 0; i < 3; i++) {
    const magnitude = Math.abs(Number(value[i]));
    if (Number.isFinite(magnitude) && magnitude >= 0.05) next[i] = magnitude;
  }
  return next;
}

function halfOf(dimensions: Vec3): Vec3 {
  return [dimensions[0] / 2, dimensions[1] / 2, dimensions[2] / 2];
}

function isFiniteVec(value: Vec3 | undefined | null): value is Vec3 {
  return !!value
    && value.length >= 3
    && Number.isFinite(value[0])
    && Number.isFinite(value[1])
    && Number.isFinite(value[2]);
}

function isUnstableId(id: string | undefined | null): boolean {
  if (!id) return true;
  return /^cmd_\d+_\d+$/.test(id);
}

function fmt(value: number): string {
  if (!Number.isFinite(value)) return '0.00';
  return value.toFixed(2);
}

function fmtVec(value: Vec3): string {
  return `${fmt(value[0])},${fmt(value[1])},${fmt(value[2])}`;
}

function readStringField(raw: string | undefined, field: string): string | null {
  if (!raw) return null;
  const quoted = new RegExp(`(?:^|[^A-Za-z0-9_])${field}\\s*[:=]\\s*['"]([^'"]+)['"]`, 'i');
  const quotedMatch = quoted.exec(raw);
  if (quotedMatch) return quotedMatch[1];
  const bare = new RegExp(`(?:^|[^A-Za-z0-9_])${field}\\s*[:=]\\s*([A-Za-z_][A-Za-z0-9_]*)`, 'i');
  const bareMatch = bare.exec(raw);
  return bareMatch ? bareMatch[1] : null;
}

function readNumberField(raw: string | undefined, field: string): number | undefined {
  if (!raw) return undefined;
  const matcher = new RegExp(`(?:^|[^A-Za-z0-9_])${field}\\s*[:=]\\s*(-?\\d+(?:\\.\\d+)?)`, 'i');
  const match = matcher.exec(raw);
  if (!match) return undefined;
  const value = Number(match[1]);
  return Number.isFinite(value) ? value : undefined;
}

function quatFromEuler(euler: Vec3): [number, number, number, number] {
  const quat = new THREE.Quaternion().setFromEuler(
    new THREE.Euler(euler[0], euler[1], euler[2], 'XYZ')
  );
  return [quat.x, quat.y, quat.z, quat.w];
}

function eulerFromQuat(quat: [number, number, number, number]): Vec3 {
  const euler = new THREE.Euler().setFromQuaternion(
    new THREE.Quaternion(quat[0], quat[1], quat[2], quat[3]),
    'XYZ'
  );
  return [euler.x, euler.y, euler.z];
}

function buildStructureMesh(typeName: string, materialName: string): THREE.Object3D {
  const material = structureMaterial(materialName);
  const group = new THREE.Group();
  switch (normalizeStructureType(typeName)) {
    case 'pipe_rack':
      addPipeRack(group, material);
      break;
    case 'monolithic_arch':
      addArch(group, material);
      break;
    case 'tunnel':
      addTunnel(group, material);
      break;
    case 'retaining_wall':
    case 'blast_barrier':
      group.add(new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), material));
      break;
    case 'catwalk':
      addCatwalk(group, material);
      break;
    case 'cooling_tower':
      group.add(new THREE.Mesh(new THREE.CylinderGeometry(0.42, 0.28, 1, 16), material));
      break;
    case 'tower':
      group.add(new THREE.Mesh(new THREE.CylinderGeometry(0.18, 0.28, 1, 12), material));
      break;
    case 'overpass':
      addDeck(group, material, 0.15);
      break;
    case 'bridge':
      addDeck(group, material, 0.2);
      break;
    default:
      group.add(new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), material));
      break;
  }
  enableShadows(group);
  return group;
}

function addPipeRack(group: THREE.Group, material: THREE.Material): void {
  const postGeom = new THREE.CylinderGeometry(0.04, 0.04, 1, 8);
  const posts: Array<[number, number]> = [[-0.4, -0.4], [0.4, -0.4], [-0.4, 0.4], [0.4, 0.4]];
  for (const [x, z] of posts) {
    const post = new THREE.Mesh(postGeom, material);
    post.position.set(x, 0, z);
    group.add(post);
  }
  const beamGeom = new THREE.CylinderGeometry(0.03, 0.03, 0.8, 8);
  for (const y of [-0.25, 0.05, 0.32]) {
    const beam = new THREE.Mesh(beamGeom, material);
    beam.rotation.z = Math.PI / 2;
    beam.position.set(0, y, 0);
    group.add(beam);
  }
}

function addArch(group: THREE.Group, material: THREE.Material): void {
  const arch = new THREE.Mesh(new THREE.TorusGeometry(0.5, 0.07, 10, 20, Math.PI), material);
  arch.scale.y = 2;
  arch.position.y = -0.5;
  group.add(arch);
}

function addTunnel(group: THREE.Group, material: THREE.Material): void {
  addArch(group, material);
  const wallGeom = new THREE.BoxGeometry(0.08, 0.55, 1);
  const left = new THREE.Mesh(wallGeom, material);
  left.position.set(-0.46, -0.2, 0);
  const right = new THREE.Mesh(wallGeom, material);
  right.position.set(0.46, -0.2, 0);
  group.add(left, right);
}

function addDeck(group: THREE.Group, material: THREE.Material, thickness: number): void {
  const deck = new THREE.Mesh(new THREE.BoxGeometry(1, thickness, 1), material);
  group.add(deck);
  const railGeom = new THREE.BoxGeometry(0.04, thickness * 2.2, 1);
  const left = new THREE.Mesh(railGeom, material);
  left.position.set(-0.48, thickness, 0);
  const right = new THREE.Mesh(railGeom, material);
  right.position.set(0.48, thickness, 0);
  group.add(left, right);
}

function addCatwalk(group: THREE.Group, material: THREE.Material): void {
  const deck = new THREE.Mesh(new THREE.BoxGeometry(1, 0.35, 1), material);
  deck.position.y = -0.15;
  group.add(deck);
  const railGeom = new THREE.BoxGeometry(0.06, 1, 1);
  const left = new THREE.Mesh(railGeom, material);
  left.position.set(-0.47, 0.15, 0);
  const right = new THREE.Mesh(railGeom, material);
  right.position.set(0.47, 0.15, 0);
  group.add(left, right);
}

function structureMaterial(materialName: string): THREE.MeshStandardMaterial {
  const name = materialName.toLowerCase();
  let color = 0x64748b;
  let metalness = 0.55;
  let roughness = 0.45;
  let emissive = 0x000000;
  let emissiveIntensity = 0;

  if (name.includes('carbon') || name.includes('nano')) {
    color = 0x1e293b;
    metalness = 0.85;
    roughness = 0.25;
    emissive = 0x00f0ff;
    emissiveIntensity = 0.25;
  } else if (name.includes('steel') || name.includes('galvan')) {
    color = 0x9aa4b2;
    metalness = 0.9;
    roughness = 0.32;
  } else if (name.includes('concrete') || name.includes('stone')) {
    color = 0x6b7280;
    metalness = 0.05;
    roughness = 0.9;
  }

  const material = new THREE.MeshStandardMaterial({
    color,
    metalness,
    roughness,
    emissive,
    emissiveIntensity,
  });
  material.name = materialName;
  return material;
}

function enableShadows(root: THREE.Object3D): void {
  root.traverse(child => {
    if (child instanceof THREE.Mesh) {
      child.castShadow = true;
      child.receiveShadow = true;
    }
  });
}
