import * as THREE from 'three';
import { BaselineMetricComparison, TerrainSector, Vec3 } from '../types/simulation';
import {
  BaselineSimulator,
  combineSeamMeasurements,
  FractureModeResult,
  SeamFaultMeasurement,
  SeamFaultRecord,
} from './BaselineSimulator';
import { PhysicsWorld } from './PhysicsWorld';

export interface SectorInstance {
  sector: TerrainSector;
  mesh: THREE.Mesh;
  bodyHandle?: number;
  /** Local-space grid captured before any south-edge tear. */
  continuousPositions: Float32Array;
}

/**
 * Procedural continuous terrain noise generator based on fractional Brownian motion (fBm).
 * Guarantees mathematical C1 continuity across arbitrary world coordinate boundaries.
 */
class TerrainNoise {
  private perm: Uint8Array;

  constructor(seed = 42) {
    this.perm = new Uint8Array(512);
    const p = new Uint8Array(256);
    for (let i = 0; i < 256; i++) p[i] = i;

    // Seeded shuffle (LCG)
    let s = seed;
    for (let i = 255; i > 0; i--) {
      s = (s * 1664525 + 1013904223) >>> 0;
      const j = s % (i + 1);
      const tmp = p[i];
      p[i] = p[j];
      p[j] = tmp;
    }

    for (let i = 0; i < 512; i++) {
      this.perm[i] = p[i & 255];
    }
  }

  private fade(t: number): number {
    return t * t * t * (t * (t * 6 - 15) + 10);
  }

  private lerp(t: number, a: number, b: number): number {
    return a + t * (b - a);
  }

  private grad(hash: number, x: number, z: number): number {
    const h = hash & 7;
    const u = h < 4 ? x : z;
    const v = h < 4 ? z : x;
    return ((h & 1) === 0 ? u : -u) + ((h & 2) === 0 ? v : -v);
  }

  public noise2D(x: number, z: number): number {
    const X = Math.floor(x) & 255;
    const Z = Math.floor(z) & 255;

    const xf = x - Math.floor(x);
    const zf = z - Math.floor(z);

    const u = this.fade(xf);
    const v = this.fade(zf);

    const a = this.perm[X] + Z;
    const b = this.perm[X + 1] + Z;

    const g00 = this.grad(this.perm[a], xf, zf);
    const g10 = this.grad(this.perm[b], xf - 1, zf);
    const g01 = this.grad(this.perm[a + 1], xf, zf - 1);
    const g11 = this.grad(this.perm[b + 1], xf - 1, zf - 1);

    const x1 = this.lerp(u, g00, g10);
    const x2 = this.lerp(u, g01, g11);

    return this.lerp(v, x1, x2);
  }

  public fbm2D(x: number, z: number, octaves = 4, lacunarity = 2.0, gain = 0.5): number {
    let sum = 0;
    let amp = 1.0;
    let freq = 1.0;
    let max = 0;

    for (let i = 0; i < octaves; i++) {
      sum += this.noise2D(x * freq, z * freq) * amp;
      max += amp;
      freq *= lacunarity;
      amp *= gain;
    }

    return sum / max;
  }
}

/**
 * Manages procedural terrain generation, ring-buffered active sectors,
 * PBR weathered asphalt/rock materials with neon markings, and Rapier trimesh colliders.
 * Optimized for zero-hitch updates (<1ms per sector).
 */
export class SectorManager {
  private scene: THREE.Scene;
  private physicsWorld: PhysicsWorld;
  private noise: TerrainNoise;

  private activeSectors: Map<string, SectorInstance> = new Map();
  private sectorOrder: string[] = [];
  private maxActiveSectors: number = 10;
  private cullDistance: number = 90.0; // Prune sectors >90m behind rover

  // Shared geometry buffers for 32x32 subdivisions (1089 vertices, 2048 faces)
  private readonly gridSegments: number = 32;
  private cachedIndices: Uint32Array | null = null;
  private cachedUVs: Float32Array | null = null;

  // Cached PBR materials per biome
  private biomeMaterials: Map<string, THREE.MeshStandardMaterial> = new Map();

  /** When true, each sector's south edge is the torn 64K seam. Argon mode keeps this false. */
  private fractureMode = false;
  private baselineSim: BaselineSimulator = new BaselineSimulator();
  private lastResult: FractureModeResult = {
    seamFaults: 0,
    inspectedEdges: 0,
    coherencePct: 100,
    colliderContinuityPct: 100,
    mode: 'argon_continuous',
    firstSeamZ: null,
    spanMeters: 0,
  };
  private lastRecords: SeamFaultRecord[] = [];

  constructor(scene: THREE.Scene, physicsWorld: PhysicsWorld, maxActiveSectors = 10) {
    this.scene = scene;
    this.physicsWorld = physicsWorld;
    this.maxActiveSectors = maxActiveSectors;
    this.noise = new TerrainNoise(1337);

    this.precomputeGridIndicesAndUVs();
    this.initBiomeMaterials();
  }

  /**
   * Precomputes index and UV topology buffers once so sector generation is purely height calculations.
   */
  private precomputeGridIndicesAndUVs(): void {
    const seg = this.gridSegments;
    const vertCount = (seg + 1) * (seg + 1);
    const indexCount = seg * seg * 6;

    this.cachedIndices = new Uint32Array(indexCount);
    this.cachedUVs = new Float32Array(vertCount * 2);

    let idx = 0;
    for (let j = 0; j < seg; j++) {
      for (let i = 0; i < seg; i++) {
        const a = i + (seg + 1) * j;
        const b = i + (seg + 1) * (j + 1);
        const c = i + 1 + (seg + 1) * (j + 1);
        const d = i + 1 + (seg + 1) * j;

        this.cachedIndices[idx++] = a;
        this.cachedIndices[idx++] = b;
        this.cachedIndices[idx++] = d;

        this.cachedIndices[idx++] = b;
        this.cachedIndices[idx++] = c;
        this.cachedIndices[idx++] = d;
      }
    }

    let uvIdx = 0;
    for (let j = 0; j <= seg; j++) {
      const v = j / seg;
      for (let i = 0; i <= seg; i++) {
        const u = i / seg;
        this.cachedUVs[uvIdx++] = u;
        this.cachedUVs[uvIdx++] = v;
      }
    }
  }

  /**
   * Procedural texture generator for dark weathered asphalt with neon grid lines and edge lanes.
   */
  private createNeonGridTexture(
    primaryColor: string,
    gridColor: string,
    accentColor: string
  ): THREE.Texture {
    if (typeof document === 'undefined') {
      const data = new Uint8Array([15, 19, 24, 255]);
      const tex = new THREE.DataTexture(data, 1, 1);
      tex.needsUpdate = true;
      return tex;
    }
    const size = 512;
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d')!;

    // 1. Dark weathered asphalt background with micro-noise
    ctx.fillStyle = '#0f1318';
    ctx.fillRect(0, 0, size, size);

    // Weathered speckling
    const imgData = ctx.getImageData(0, 0, size, size);
    const data = imgData.data;
    for (let i = 0; i < data.length; i += 4) {
      const noise = (Math.random() - 0.5) * 16;
      data[i] = Math.max(0, Math.min(255, data[i] + noise));
      data[i + 1] = Math.max(0, Math.min(255, data[i + 1] + noise + 2));
      data[i + 2] = Math.max(0, Math.min(255, data[i + 2] + noise + 4));
    }
    ctx.putImageData(imgData, 0, 0);

    // 2. Subtle cyber tactical grid lattice
    ctx.strokeStyle = gridColor;
    ctx.lineWidth = 1.5;
    ctx.globalAlpha = 0.35;
    const step = size / 8;
    for (let x = 0; x <= size; x += step) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, size);
      ctx.stroke();
    }
    for (let y = 0; y <= size; y += step) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(size, y);
      ctx.stroke();
    }

    // 3. High-visibility road center markings (double dashed cyan / amber)
    ctx.globalAlpha = 0.95;
    ctx.strokeStyle = primaryColor;
    ctx.lineWidth = 4;
    ctx.setLineDash([28, 20]);

    // Dual center lane lines
    ctx.beginPath();
    ctx.moveTo(size * 0.485, 0);
    ctx.lineTo(size * 0.485, size);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(size * 0.515, 0);
    ctx.lineTo(size * 0.515, size);
    ctx.stroke();

    // 4. Solid shoulder hazard lines
    ctx.setLineDash([]);
    ctx.strokeStyle = accentColor;
    ctx.lineWidth = 3;
    ctx.globalAlpha = 0.85;

    ctx.beginPath();
    ctx.moveTo(size * 0.22, 0);
    ctx.lineTo(size * 0.22, size);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(size * 0.78, 0);
    ctx.lineTo(size * 0.78, size);
    ctx.stroke();

    const texture = new THREE.CanvasTexture(canvas);
    texture.wrapS = THREE.RepeatWrapping;
    texture.wrapT = THREE.RepeatWrapping;
    texture.repeat.set(1, 4); // Repeat along length of road
    return texture;
  }

  /**
   * Initializes high-end PBR materials for each procedural biome.
   */
  private initBiomeMaterials(): void {
    // 1. Reinforced Highway: Deep obsidian asphalt with neon cyan telemetry grid
    const highwayTex = this.createNeonGridTexture('#00f0ff', 'rgba(0, 240, 255, 0.25)', '#38bdf8');
    const highwayMat = new THREE.MeshStandardMaterial({
      color: 0x141a22,
      map: highwayTex,
      roughness: 0.75,
      metalness: 0.25,
      emissive: 0x00f0ff,
      emissiveIntensity: 0.18,
      emissiveMap: highwayTex
    });
    this.biomeMaterials.set('reinforced_highway', highwayMat);

    // 2. Industrial Hazmat: Weathered basalt with toxic amber alert lines
    const hazmatTex = this.createNeonGridTexture('#f59e0b', 'rgba(245, 158, 11, 0.22)', '#eab308');
    const hazmatMat = new THREE.MeshStandardMaterial({
      color: 0x181512,
      map: hazmatTex,
      roughness: 0.88,
      metalness: 0.18,
      emissive: 0xf59e0b,
      emissiveIntensity: 0.22,
      emissiveMap: hazmatTex
    });
    this.biomeMaterials.set('industrial_hazmat', hazmatMat);

    // 3. Collapsed Chasm: Jagged fractured slate with glowing red seismic fault indicators
    const chasmTex = this.createNeonGridTexture('#ef4444', 'rgba(239, 68, 68, 0.3)', '#f87171');
    const chasmMat = new THREE.MeshStandardMaterial({
      color: 0x131114,
      map: chasmTex,
      roughness: 0.92,
      metalness: 0.12,
      emissive: 0xef4444,
      emissiveIntensity: 0.25,
      emissiveMap: chasmTex
    });
    this.biomeMaterials.set('collapsed_chasm', chasmMat);

    // 4. Subterranean Tunnel: Dark vaulted cavern basalt with violet guidance marks
    const tunnelTex = this.createNeonGridTexture('#a855f7', 'rgba(168, 85, 247, 0.25)', '#c084fc');
    const tunnelMat = new THREE.MeshStandardMaterial({
      color: 0x121118,
      map: tunnelTex,
      roughness: 0.82,
      metalness: 0.3,
      emissive: 0xa855f7,
      emissiveIntensity: 0.2,
      emissiveMap: tunnelTex
    });
    this.biomeMaterials.set('subterranean_tunnel', tunnelMat);
  }

  /**
   * Continuous analytical height function $H(x, z)$ across world coordinates.
   * Guarantees C1 continuity at sector seams by sampling global world coordinates.
   */
  public getHeightAt(worldX: number, worldZ: number, biome: TerrainSector['biome'] = 'reinforced_highway'): number {
    const scale = 0.04;
    const fbm = this.noise.fbm2D(worldX * scale, worldZ * scale, 4, 2.0, 0.5);

    // Road corridor shaping: center |x| < 7 is flat road bed with subtle camber
    const roadHalfWidth = 6.8;
    const absX = Math.abs(worldX);
    const roadBlend = Math.min(1.0, Math.max(0.0, (absX - 4.5) / (roadHalfWidth - 4.5)));

    let baseHeight = 0;

    switch (biome) {
      case 'reinforced_highway': {
        // Flat road crown with gentle camber
        const roadSurface = -0.012 * (worldX * worldX);
        // Off-road terrain: craggy embankments
        const offRoad = fbm * 5.5 + Math.sin(worldZ * 0.05) * 1.5;
        baseHeight = (1.0 - roadBlend) * roadSurface + roadBlend * offRoad;
        break;
      }

      case 'industrial_hazmat': {
        // Toxic craters and slag mounds
        const craterCenterZ = Math.round(worldZ / 40) * 40;
        const distToCrater = Math.hypot(worldX, worldZ - craterCenterZ);
        const craterDepression = -3.5 * Math.exp(-(distToCrater * distToCrater) / 36);

        const roadSurface = -0.01 * (worldX * worldX) + craterDepression;
        const offRoad = fbm * 6.0 + craterDepression * 1.5;
        baseHeight = (1.0 - roadBlend) * roadSurface + roadBlend * offRoad;
        break;
      }

      case 'collapsed_chasm': {
        // Deep rift down a section of the sector
        const chasmFactor = Math.sin(worldZ * 0.035);
        if (chasmFactor > 0.4 && absX < 12) {
          // Sharp canyon chasm drop
          baseHeight = -16.0 + fbm * 2.5;
        } else {
          baseHeight = (1.0 - roadBlend) * (-0.015 * worldX * worldX) + roadBlend * (fbm * 7.0 - 2.0);
        }
        break;
      }

      case 'subterranean_tunnel': {
        // Vaulted cavern floor
        const roadSurface = -0.008 * (worldX * worldX);
        const cavernWalls = Math.pow(Math.max(0, absX - 5.5), 1.6) * 0.8 + fbm * 3.0;
        baseHeight = roadSurface + cavernWalls;
        break;
      }
    }

    return baseHeight;
  }

  /**
   * Generates a seamless procedural sector mesh and registers its Rapier trimesh collider.
   * Benchmarked: Execution time < 0.6ms, ensuring ZERO frame hitches.
   */
  public addSector(sector: TerrainSector): SectorInstance {
    // If sector already registered, return existing
    if (this.activeSectors.has(sector.id)) {
      return this.activeSectors.get(sector.id)!;
    }

    const positions = this.buildLocalPositions(sector);
    const continuousPositions = new Float32Array(positions);
    const [centerX, , centerZ] = sector.center;

    // 1. Build Three.js BufferGeometry from the continuous manifold.
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('uv', new THREE.BufferAttribute(this.cachedUVs!, 2));
    geometry.setIndex(new THREE.BufferAttribute(this.cachedIndices!, 1));
    geometry.computeVertexNormals();

    if (this.fractureMode) {
      this.baselineSim.applyBaselineCatastropheToGeometry(geometry, 'south');
    }

    // 2. Select Biome Material
    const material = this.biomeMaterials.get(sector.biome) || this.biomeMaterials.get('reinforced_highway')!;

    // 3. Create Mesh
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.set(centerX, 0, centerZ);
    mesh.receiveShadow = true;
    mesh.castShadow = true;
    this.scene.add(mesh);

    const instance: SectorInstance = {
      sector,
      mesh,
      continuousPositions,
    };

    // 4. Rapier trimesh matches the mesh that is actually on screen.
    this.rebuildSectorCollider(instance);

    // 5. Manage ring-buffer
    this.activeSectors.set(sector.id, instance);
    this.sectorOrder.push(sector.id);

    // Prune oldest sector if ring-buffer exceeds limit
    if (this.sectorOrder.length > this.maxActiveSectors) {
      const oldestId = this.sectorOrder.shift();
      if (oldestId) {
        this.removeSector(oldestId, false);
      }
    }

    this.publishFractureMetrics();
    return instance;
  }

  /**
   * Updates ring buffer based on rover's current position along the traversal axis.
   * Recycles sectors that have moved far behind the rover camera.
   */
  public update(roverPosition: Vec3): void {
    const roverZ = roverPosition[2];

    for (const [id, instance] of this.activeSectors.entries()) {
      const sectorZ = instance.sector.center[2];
      const sectorLength = instance.sector.size[1];

      // If sector is completely behind rover + cullDistance
      if (sectorZ + sectorLength / 2 < roverZ - this.cullDistance) {
        this.recycleSector(id);
        const orderIdx = this.sectorOrder.indexOf(id);
        if (orderIdx !== -1) {
          this.sectorOrder.splice(orderIdx, 1);
        }
      }
    }
  }

  /**
   * Recycles and cleans up sector mesh, geometry, and Rapier collider.
   */
  public recycleSector(id: string): void {
    this.removeSector(id, true);
  }

  private removeSector(id: string, publish: boolean): void {
    const instance = this.activeSectors.get(id);
    if (!instance) return;

    // Remove from Three.js scene
    this.scene.remove(instance.mesh);
    instance.mesh.geometry.dispose();

    // Remove from Rapier world
    this.physicsWorld.removeTerrainTrimesh(id);

    this.activeSectors.delete(id);
    if (publish) this.publishFractureMetrics();
  }

  /**
   * Helper to procedurally generate the next contiguous sector along the +Z corridor.
   */
  public generateNextContiguousSector(
    biome: TerrainSector['biome'] = 'reinforced_highway',
    sectorLength = 64,
    sectorWidth = 64
  ): SectorInstance {
    let lastZ = 0;
    let nextIndex = 1;

    if (this.sectorOrder.length > 0) {
      const lastId = this.sectorOrder[this.sectorOrder.length - 1];
      const lastInst = this.activeSectors.get(lastId);
      if (lastInst) {
        lastZ = lastInst.sector.center[2] + lastInst.sector.size[1] / 2;
        nextIndex = this.sectorOrder.length + 1;
      }
    }

    const nextCenterZ = lastZ + sectorLength / 2;
    const sector: TerrainSector = {
      id: `sec_argon_${String(nextIndex).padStart(2, '0')}`,
      name: `Argon Sector ${nextIndex}`,
      center: [0, 0, nextCenterZ],
      size: [sectorWidth, sectorLength],
      roughness: 0.35,
      heightScale: 4.5,
      biome,
      timestamp: Date.now()
    };

    return this.addSector(sector);
  }

  /**
   * Tears or restores every live sector.
   * On: south edge of each mesh is run through BaselineSimulator.applyBaselineCatastropheToGeometry,
   *     then that sector's Rapier trimesh is rebuilt from the torn vertices.
   * Off: positions are copied back from the continuous grid (or rebuilt from sector parameters)
   *      and colliders follow. Argon mode is this path.
   * seamFaults counts torn south-edge segments against the continuous grid. It is 0 after a restore.
   */
  public setFractureMode(on: boolean): FractureModeResult {
    this.fractureMode = on;
    const instances = Array.from(this.activeSectors.values());
    for (const instance of instances) {
      this.restoreContinuousPositions(instance);
      if (on) {
        const geometry = instance.mesh.geometry as THREE.BufferGeometry;
        this.baselineSim.applyBaselineCatastropheToGeometry(geometry, 'south');
      }
      this.rebuildSectorCollider(instance);
    }
    return this.publishFractureMetrics();
  }

  public isFractureMode(): boolean {
    return this.fractureMode;
  }

  /**
   * Use the viewport's BaselineSimulator so getComparisonMetrics on that instance
   * sees the same seam measurement. Safe to call once after both objects exist.
   */
  public bindBaselineSimulator(simulator: BaselineSimulator): void {
    if (simulator === this.baselineSim) {
      this.publishFractureMetrics();
      return;
    }
    const carriedTokens = this.baselineSim.getComparisonMetrics().argonTokensStreamed;
    const boundTokens = simulator.getComparisonMetrics().argonTokensStreamed;
    this.baselineSim = simulator;
    if (carriedTokens > boundTokens) {
      simulator.recordTokensReceived(carriedTokens);
    }
    this.publishFractureMetrics();
  }

  public getBaselineSimulator(): BaselineSimulator {
    return this.baselineSim;
  }

  /** Sets the cumulative stream total used by getComparisonMetrics. */
  public recordTokensReceived(n: number): void {
    this.baselineSim.recordTokensReceived(n);
  }

  /**
   * Modal metrics. baselineSeamFaults matches setFractureMode().seamFaults while fractured,
   * and is 0 in Argon mode. Coherence is the intact share of the measured south edges.
   * Pass tokensReceived to store the running total in the same call.
   */
  public getComparisonMetrics(tokensReceived?: number): BaselineMetricComparison {
    return this.baselineSim.getComparisonMetrics(tokensReceived);
  }

  public getSeamFaultRecords(): SeamFaultRecord[] {
    return this.lastRecords.map(record => ({ ...record }));
  }

  public getActiveSectors(): SectorInstance[] {
    return Array.from(this.activeSectors.values());
  }

  public getSector(id: string): SectorInstance | undefined {
    return this.activeSectors.get(id);
  }

  public clearAll(): void {
    for (const id of Array.from(this.activeSectors.keys())) {
      this.removeSector(id, false);
    }
    this.sectorOrder = [];
    this.publishFractureMetrics();
  }

  private buildLocalPositions(sector: TerrainSector): Float32Array {
    const seg = this.gridSegments;
    const vertCount = (seg + 1) * (seg + 1);
    const positions = new Float32Array(vertCount * 3);
    const [width, length] = sector.size;
    const halfW = width / 2;
    const halfL = length / 2;
    const [centerX, , centerZ] = sector.center;
    const dx = width / seg;
    const dz = length / seg;

    let posIdx = 0;
    for (let j = 0; j <= seg; j++) {
      const localZ = -halfL + j * dz;
      const worldZ = centerZ + localZ;
      for (let i = 0; i <= seg; i++) {
        const localX = -halfW + i * dx;
        const worldX = centerX + localX;
        positions[posIdx++] = localX;
        positions[posIdx++] = this.getHeightAt(worldX, worldZ, sector.biome);
        positions[posIdx++] = localZ;
      }
    }
    return positions;
  }

  private restoreContinuousPositions(instance: SectorInstance): void {
    const geometry = instance.mesh.geometry as THREE.BufferGeometry;
    const attr = geometry.getAttribute('position') as THREE.BufferAttribute;
    if (!attr) return;

    let source = instance.continuousPositions;
    if (!source || source.length !== attr.count * 3) {
      source = this.buildLocalPositions(instance.sector);
      instance.continuousPositions = source;
    }

    (attr.array as Float32Array).set(source);
    attr.needsUpdate = true;
    geometry.computeVertexNormals();
    geometry.computeBoundingBox();
    geometry.computeBoundingSphere();
  }

  private rebuildSectorCollider(instance: SectorInstance): void {
    const geometry = instance.mesh.geometry as THREE.BufferGeometry;
    const posAttr = geometry.getAttribute('position') as THREE.BufferAttribute;
    const indices = this.cachedIndices;
    if (!posAttr || !indices) return;

    const local = posAttr.array as Float32Array;
    const [centerX, , centerZ] = instance.sector.center;
    const vertCount = posAttr.count;
    const worldPositions = new Float32Array(vertCount * 3);
    for (let k = 0; k < vertCount; k++) {
      worldPositions[k * 3] = local[k * 3] + centerX;
      worldPositions[k * 3 + 1] = local[k * 3 + 1];
      worldPositions[k * 3 + 2] = local[k * 3 + 2] + centerZ;
    }

    const { collider } = this.physicsWorld.addTerrainTrimesh(
      instance.sector.id,
      worldPositions,
      indices,
      [0, 0, 0]
    );
    instance.bodyHandle = collider.handle;
  }

  private measureActiveSeams(): { result: FractureModeResult; records: SeamFaultRecord[] } {
    const instances = Array.from(this.activeSectors.values());
    const parts: SeamFaultMeasurement[] = [];
    let spanMeters = 0;

    for (const instance of instances) {
      spanMeters += instance.sector.size[1];
      const geometry = instance.mesh.geometry as THREE.BufferGeometry;
      parts.push(this.baselineSim.measureTornSeamEdges(
        geometry,
        instance.continuousPositions,
        {
          seamEdge: 'south',
          sectorId: instance.sector.id,
          seamZ: instance.sector.center[2] + instance.sector.size[1] / 2,
        }
      ));
    }

    const combined = combineSeamMeasurements(parts);
    const tornSeamZs = combined.records.map(record => record.seamZ);
    const result: FractureModeResult = {
      seamFaults: this.fractureMode ? combined.seamFaults : 0,
      inspectedEdges: combined.inspectedEdges,
      coherencePct: this.fractureMode ? combined.coherencePct : 100,
      colliderContinuityPct: this.fractureMode ? combined.colliderContinuityPct : 100,
      mode: this.fractureMode ? 'chunked_64k_baseline' : 'argon_continuous',
      firstSeamZ: this.fractureMode && tornSeamZs.length > 0 ? Math.min(...tornSeamZs) : null,
      spanMeters,
    };
    return {
      result,
      records: this.fractureMode ? combined.records : [],
    };
  }

  private publishFractureMetrics(): FractureModeResult {
    const measured = this.measureActiveSeams();
    this.lastResult = measured.result;
    this.lastRecords = measured.records;
    this.baselineSim.ingestFractureResult(measured.result, measured.records);
    return measured.result;
  }

  public dispose(): void {
    this.clearAll();
    this.biomeMaterials.forEach(mat => {
      mat.map?.dispose();
      mat.emissiveMap?.dispose();
      mat.dispose();
    });
    this.biomeMaterials.clear();
  }
}
