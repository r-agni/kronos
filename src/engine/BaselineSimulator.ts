import * as THREE from 'three';
import { BaselineMetricComparison, TerrainSector } from '../types/simulation';

export interface SeamFaultRecord {
  id: string;
  seamZ: number;
  heightDeltaMeters: number;
  gapDistanceMeters: number;
  type: 'height_cliff' | 'mesh_tear' | 'missing_collider' | 'disconnected_bridge';
  severity: 'minor' | 'major' | 'fatal';
  description: string;
}

export interface DetailedBaselineReport {
  mode: 'argon_continuous' | 'chunked_64k_baseline';
  metrics: BaselineMetricComparison;
  navigableRouteDistanceMeters: number;
  fatalObstacleDetected: boolean;
  seamFaults: SeamFaultRecord[];
  coherenceHistory: Array<{ timestamp: number; argonCoherence: number; baselineCoherence: number }>;
}

/** Displacements at or below this are treated as a continuous edge. */
export const SEAM_TEAR_THRESHOLD_METERS = 0.05;

/** Legacy chunk window. Baseline token totals reported to the modal never exceed this. */
export const BASELINE_CHUNK_TOKEN_LIMIT = 65536;

export interface SeamFaultMeasurement {
  /** South-edge segments whose endpoints moved off the continuous edge. */
  seamFaults: number;
  /** South-edge segments inspected, torn or not. */
  inspectedEdges: number;
  /** 100 * intact segments / inspected segments. 100 when nothing is torn. */
  coherencePct: number;
  /** Same intact ratio as coherencePct. This is the measured mesh, not the Argon column. */
  colliderContinuityPct: number;
  records: SeamFaultRecord[];
}

export interface FractureModeResult {
  seamFaults: number;
  inspectedEdges: number;
  coherencePct: number;
  colliderContinuityPct: number;
  mode: 'argon_continuous' | 'chunked_64k_baseline';
  /** Smallest nominal world-Z among torn south edges. Null when no segment is torn. */
  firstSeamZ: number | null;
  /** Sum of active sector lengths along Z. */
  spanMeters: number;
}

type MetricListener = (metrics: BaselineMetricComparison) => void;

export interface SeamMeasureOptions {
  seamEdge?: 'north' | 'south';
  thresholdMeters?: number;
  sectorId?: string;
  /** Nominal world-Z of this edge before the horizontal tear is applied. */
  seamZ?: number;
}

/**
 * Intact south-edge ratio, as a percentage.
 * Zero torn segments scores 100. A fully torn edge scores 0.
 */
export function coherenceFromSeamFaults(seamFaults: number, inspectedEdges: number): number {
  if (!Number.isFinite(seamFaults) || seamFaults <= 0) return 100;
  if (!Number.isFinite(inspectedEdges) || inspectedEdges <= 0) return 0;
  const intact = Math.max(0, inspectedEdges - seamFaults);
  return +((100 * intact) / inspectedEdges).toFixed(2);
}

export function combineSeamMeasurements(parts: SeamFaultMeasurement[]): SeamFaultMeasurement {
  let seamFaults = 0;
  let inspectedEdges = 0;
  const records: SeamFaultRecord[] = [];
  for (const part of parts) {
    seamFaults += part.seamFaults;
    inspectedEdges += part.inspectedEdges;
    records.push(...part.records);
  }
  const coherencePct = coherenceFromSeamFaults(seamFaults, inspectedEdges);
  return {
    seamFaults,
    inspectedEdges,
    coherencePct,
    colliderContinuityPct: coherencePct,
    records,
  };
}

function emptySeamMeasurement(): SeamFaultMeasurement {
  return {
    seamFaults: 0,
    inspectedEdges: 0,
    coherencePct: 100,
    colliderContinuityPct: 100,
    records: [],
  };
}

/**
 * Segment count for the square sector grids SectorManager builds.
 * Vertex count is (segments + 1)^2. Returns 0 when the buffer is not a grid.
 */
export function seamGridSegments(vertexCount: number): number {
  if (!Number.isFinite(vertexCount) || vertexCount <= 1) return 0;
  const side = Math.round(Math.sqrt(vertexCount));
  if (side * side !== vertexCount || side < 2) return 0;
  return side - 1;
}

/**
 * Compares Argon (continuous edges) with the 64K baseline (torn south edges).
 * Fault counts and coherence come from mesh measurements. Token totals come from
 * recordTokensReceived or the optional argument to getComparisonMetrics.
 */
export class BaselineSimulator {
  private baselineActive = false;
  private argonTokens = 0;
  private storedFaults = 0;
  private storedEdges = 0;
  private storedRecords: SeamFaultRecord[] = [];
  private storedFirstSeamZ: number | null = null;
  private continuousSpanMeters = 0;

  private history: Array<{ timestamp: number; argonCoherence: number; baselineCoherence: number }> = [];
  private listeners: Set<MetricListener> = new Set();

  constructor() {
    this.recordHistoryPoint();
  }

  /**
   * Selects which column the last mesh measurement is reported on.
   * Does not move vertices. SectorManager.setFractureMode owns the mesh.
   */
  public setBaselineMode(enabled: boolean): void {
    this.baselineActive = enabled;
    this.recordHistoryPoint();
    this.notifyListeners();
  }

  public isBaselineActive(): boolean {
    return this.baselineActive;
  }

  /**
   * Sets the cumulative token count actually received from the Argon stream.
   * This replaces the previous total; pass the running total, not a per-chunk delta.
   */
  public recordTokensReceived(n: number): void {
    if (!Number.isFinite(n)) return;
    const next = Math.max(0, Math.floor(n));
    if (next === this.argonTokens) return;
    this.argonTokens = next;
    this.recordHistoryPoint();
    this.notifyListeners();
  }

  /**
   * Stores a mesh measurement produced by SectorManager.setFractureMode.
   * While mode is argon_continuous, baseline seam faults report as 0.
   */
  public ingestFractureResult(result: FractureModeResult, records: SeamFaultRecord[] = []): void {
    this.baselineActive = result.mode === 'chunked_64k_baseline';
    this.continuousSpanMeters = Number.isFinite(result.spanMeters) ? Math.max(0, result.spanMeters) : 0;
    if (this.baselineActive) {
      this.storedFaults = Math.max(0, result.seamFaults);
      this.storedEdges = Math.max(0, result.inspectedEdges);
      this.storedRecords = records.map(record => ({ ...record }));
      this.storedFirstSeamZ = result.firstSeamZ;
    } else {
      this.storedFaults = 0;
      this.storedEdges = Math.max(0, result.inspectedEdges);
      this.storedRecords = [];
      this.storedFirstSeamZ = null;
    }
    this.recordHistoryPoint();
    this.notifyListeners();
  }

  public getSeamFaultRecords(): SeamFaultRecord[] {
    return this.storedRecords.map(record => ({ ...record }));
  }

  /**
   * Kept so existing callers can tick the simulator.
   * Tokens and coherence are not synthesized here.
   */
  public update(_deltaTime: number, _roverTraversedMeters = 0): void {
  }

  /**
   * Drops or shifts one edge of a sector grid to show a 64K chunk seam.
   * Callers that need to toggle the tear must restore the continuous positions first;
   * a second call offsets the edge again.
   */
  public applyBaselineCatastropheToGeometry(
    geometry: THREE.BufferGeometry,
    seamEdge: 'north' | 'south' = 'south',
    heightOffset = 1.84,
    gapOffset = 0.95
  ): void {
    const posAttr = geometry.getAttribute('position') as THREE.BufferAttribute;
    if (!posAttr) return;

    const seg = seamGridSegments(posAttr.count);
    if (seg <= 0) return;

    const positions = posAttr.array as Float32Array;
    const targetRow = seamEdge === 'south' ? seg : 0;

    for (let i = 0; i <= seg; i++) {
      const vertIdx = targetRow * (seg + 1) + i;
      const posIndex = vertIdx * 3;

      const centerFactor = 1.0 - Math.abs(i - seg / 2) / (seg / 2);
      positions[posIndex + 1] += heightOffset * (0.8 + 0.4 * centerFactor);

      if (seamEdge === 'south') {
        positions[posIndex + 2] -= gapOffset;
      } else {
        positions[posIndex + 2] += gapOffset;
      }
    }

    posAttr.needsUpdate = true;
    geometry.computeVertexNormals();
    geometry.computeBoundingBox();
    geometry.computeBoundingSphere();
  }

  /**
   * Counts south-edge (or north-edge) segments that differ from a continuous reference.
   * One fault is one grid segment along that edge, not a fixed benchmark constant.
   */
  public measureTornSeamEdges(
    geometry: THREE.BufferGeometry,
    referencePositions: ArrayLike<number>,
    options: SeamMeasureOptions = {}
  ): SeamFaultMeasurement {
    const posAttr = geometry.getAttribute('position') as THREE.BufferAttribute;
    if (!posAttr) return emptySeamMeasurement();

    const vertexCount = posAttr.count;
    const seg = seamGridSegments(vertexCount);
    if (seg <= 0 || referencePositions.length < vertexCount * 3) {
      return emptySeamMeasurement();
    }

    const seamEdge = options.seamEdge ?? 'south';
    const threshold = options.thresholdMeters ?? SEAM_TEAR_THRESHOLD_METERS;
    const current = posAttr.array as ArrayLike<number>;
    const targetRow = seamEdge === 'south' ? seg : 0;
    const sectorId = options.sectorId ?? 'sector';
    const seamZ = options.seamZ ?? 0;
    const records: SeamFaultRecord[] = [];

    for (let i = 0; i < seg; i++) {
      const a = (targetRow * (seg + 1) + i) * 3;
      const b = a + 3;
      const heightDelta = Math.max(
        Math.abs(current[a + 1] - referencePositions[a + 1]),
        Math.abs(current[b + 1] - referencePositions[b + 1])
      );
      const gap = Math.max(
        Math.abs(current[a + 2] - referencePositions[a + 2]),
        Math.abs(current[b + 2] - referencePositions[b + 2])
      );
      if (heightDelta <= threshold && gap <= threshold) continue;

      const tornInHeight = heightDelta > threshold;
      const tornInGap = gap > threshold;
      records.push({
        id: `${sectorId}_${seamEdge}_edge_${i}`,
        seamZ,
        heightDeltaMeters: +heightDelta.toFixed(3),
        gapDistanceMeters: +gap.toFixed(3),
        type: tornInHeight && tornInGap ? 'mesh_tear' : tornInHeight ? 'height_cliff' : 'mesh_tear',
        severity: heightDelta >= 1.5 || gap >= 0.75 ? 'fatal' : heightDelta >= 0.3 || gap >= 0.2 ? 'major' : 'minor',
        description: `${seamEdge} edge segment ${i} on ${sectorId} is torn: height Δ ${heightDelta.toFixed(2)} m, gap ${gap.toFixed(2)} m.`,
      });
    }

    const coherencePct = coherenceFromSeamFaults(records.length, seg);
    return {
      seamFaults: records.length,
      inspectedEdges: seg,
      coherencePct,
      colliderContinuityPct: coherencePct,
      records,
    };
  }

  /**
   * Benchmark modal payload.
   * Argon columns stay continuous (0 faults, 100% coherence, 100% collider continuity).
   * Baseline fault count and coherence follow the ingested mesh measurement while fracture mode is on.
   * Pass tokensReceived to set the cumulative stream total in the same call.
   */
  public getComparisonMetrics(tokensReceived?: number): BaselineMetricComparison {
    if (tokensReceived !== undefined) {
      this.recordTokensReceived(tokensReceived);
    }

    const tokens = this.argonTokens;
    const baselineFaults = this.baselineActive ? this.storedFaults : 0;
    const baselineCoherence = this.baselineActive
      ? coherenceFromSeamFaults(this.storedFaults, this.storedEdges)
      : 100;

    return {
      argonTokensStreamed: tokens,
      argonCoherencePct: 100,
      argonSeamFaults: 0,
      baselineTokensGenerated: Math.min(BASELINE_CHUNK_TOKEN_LIMIT, tokens),
      baselineCoherencePct: baselineCoherence,
      baselineSeamFaults: baselineFaults,
      colliderContinuityPct: 100,
    };
  }

  public getDetailedReport(): DetailedBaselineReport {
    const showingFaults = this.baselineActive && this.storedFaults > 0;
    return {
      mode: this.baselineActive ? 'chunked_64k_baseline' : 'argon_continuous',
      metrics: this.getComparisonMetrics(),
      navigableRouteDistanceMeters: showingFaults && this.storedFirstSeamZ != null
        ? Math.max(0, this.storedFirstSeamZ)
        : this.continuousSpanMeters,
      fatalObstacleDetected: showingFaults,
      seamFaults: showingFaults ? this.storedRecords.map(record => ({ ...record })) : [],
      coherenceHistory: [...this.history],
    };
  }

  /**
   * Compares two sampled seam polylines. Coherence falls as the max step grows.
   */
  public evaluateSeamCoherence(
    _sectorA: TerrainSector,
    _sectorB: TerrainSector,
    seamHeightsA: number[],
    seamHeightsB: number[]
  ): { coherencePct: number; maxStepCliff: number; isContinuous: boolean } {
    let maxDiff = 0;
    const len = Math.min(seamHeightsA.length, seamHeightsB.length);
    for (let i = 0; i < len; i++) {
      const diff = Math.abs(seamHeightsA[i] - seamHeightsB[i]);
      if (diff > maxDiff) maxDiff = diff;
    }

    const isContinuous = maxDiff < SEAM_TEAR_THRESHOLD_METERS;
    const coherencePct = Math.max(0, 100 - maxDiff * 4.0);

    return {
      coherencePct: +coherencePct.toFixed(2),
      maxStepCliff: +maxDiff.toFixed(3),
      isContinuous,
    };
  }

  public onMetricUpdate(callback: MetricListener): () => void {
    this.listeners.add(callback);
    callback(this.getComparisonMetrics());
    return () => this.listeners.delete(callback);
  }

  public resetTokens(): void {
    if (this.argonTokens === 0) return;
    this.argonTokens = 0;
    this.notifyListeners();
  }

  private recordHistoryPoint(): void {
    const metrics = this.previewMetrics();
    this.history.push({
      timestamp: Date.now(),
      argonCoherence: metrics.argonCoherencePct,
      baselineCoherence: metrics.baselineCoherencePct,
    });
    if (this.history.length > 60) {
      this.history.shift();
    }
  }

  private previewMetrics(): BaselineMetricComparison {
    const baselineFaults = this.baselineActive ? this.storedFaults : 0;
    return {
      argonTokensStreamed: this.argonTokens,
      argonCoherencePct: 100,
      argonSeamFaults: 0,
      baselineTokensGenerated: Math.min(BASELINE_CHUNK_TOKEN_LIMIT, this.argonTokens),
      baselineCoherencePct: this.baselineActive
        ? coherenceFromSeamFaults(this.storedFaults, this.storedEdges)
        : 100,
      baselineSeamFaults: baselineFaults,
      colliderContinuityPct: 100,
    };
  }

  private notifyListeners(): void {
    const metrics = this.getComparisonMetrics();
    this.listeners.forEach(cb => {
      try {
        cb(metrics);
      } catch (err) {
        console.error('[BaselineSimulator] Error in metric listener:', err);
      }
    });
  }
}
