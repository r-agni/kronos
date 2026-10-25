import * as THREE from 'three';
import { LiDARPoint, Vec3 } from '../types/simulation';

export interface LiDARVisualizerOptions {
  beamCount?: number;            // Default 32 channels
  maxRange?: number;             // Default 35 meters
  beamColor?: number;            // Default neon cyan 0x00f0ff
  hitPointColor?: number;        // Default 0x00f0ff
  maxPersistentPoints?: number;  // Default 128 instanced hit markers
}

export class LiDARVisualizer {
  private group: THREE.Group;
  private beamCount: number = 32;
  private maxRange: number = 35.0;

  // Laser beam line segments
  private beamGeometry: THREE.BufferGeometry;
  private beamPositions: Float32Array;
  private beamColors: Float32Array;
  private beamLines: THREE.LineSegments;

  // Impact hit point instanced markers
  private maxHitInstances: number = 128;
  private hitInstancedMesh: THREE.InstancedMesh;
  private dummyTransform: THREE.Object3D = new THREE.Object3D();
  private hitPositions: THREE.Vector3[] = [];
  private hitColors: THREE.Color[] = [];
  private hitTimes: number[] = [];
  private activeHitCount: number = 0;

  // Procedural sweeping state (when external rayHits are not continuously provided)
  private sweepAzimuth: number = 0;
  private sweepElevationAngles: number[] = [];
  private elapsedTime: number = 0;

  // Materials & Colors
  private cyanColor: THREE.Color = new THREE.Color(0x00f0ff);
  private hazardColor: THREE.Color = new THREE.Color(0xff3355);
  private countermeasureColor: THREE.Color = new THREE.Color(0xc084fc);
  private structureColor: THREE.Color = new THREE.Color(0xfbbf24);

  constructor(options: LiDARVisualizerOptions = {}) {
    this.group = new THREE.Group();
    this.group.name = 'LiDARVisualizer';

    this.beamCount = options.beamCount ?? 32;
    this.maxRange = options.maxRange ?? 35.0;
    this.maxHitInstances = options.maxPersistentPoints ?? 128;

    // 1. Setup 32 Channel Elevation Angles (-22° to +12° tactical fan)
    const minElev = (-22 * Math.PI) / 180;
    const maxElev = (12 * Math.PI) / 180;
    for (let i = 0; i < this.beamCount; i++) {
      const t = i / (this.beamCount - 1);
      this.sweepElevationAngles.push(minElev + t * (maxElev - minElev));
    }

    // 2. Setup Dynamic LineSegments for Laser Beams
    // 32 beams = 64 vertices = 192 coordinates
    const vertexCount = this.beamCount * 2;
    this.beamPositions = new Float32Array(vertexCount * 3);
    this.beamColors = new Float32Array(vertexCount * 3);

    this.beamGeometry = new THREE.BufferGeometry();
    this.beamGeometry.setAttribute('position', new THREE.BufferAttribute(this.beamPositions, 3));
    this.beamGeometry.setAttribute('color', new THREE.BufferAttribute(this.beamColors, 3));

    // Neon cyan additive laser beam material
    const beamMaterial = new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.88,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      linewidth: 1.5,
    });

    this.beamLines = new THREE.LineSegments(this.beamGeometry, beamMaterial);
    this.beamLines.frustumCulled = false;
    this.group.add(this.beamLines);

    // 3. Setup InstancedMesh for Glowing Impact Hit Points
    // Faceted diamond/octahedron hit marker
    const markerGeometry = new THREE.OctahedronGeometry(0.12, 0);
    const markerMaterial = new THREE.MeshBasicMaterial({
      color: options.hitPointColor ?? 0x00f0ff,
      transparent: true,
      opacity: 0.9,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });

    this.hitInstancedMesh = new THREE.InstancedMesh(
      markerGeometry,
      markerMaterial,
      this.maxHitInstances
    );
    this.hitInstancedMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.hitInstancedMesh.frustumCulled = false;

    // Initialize instance arrays
    const tempMatrix = new THREE.Matrix4().makeScale(0, 0, 0);
    for (let i = 0; i < this.maxHitInstances; i++) {
      this.hitInstancedMesh.setMatrixAt(i, tempMatrix);
      this.hitInstancedMesh.setColorAt(i, this.cyanColor);
      this.hitPositions.push(new THREE.Vector3(0, -999, 0));
      this.hitColors.push(this.cyanColor.clone());
      this.hitTimes.push(0);
    }
    this.hitInstancedMesh.instanceMatrix.needsUpdate = true;
    if (this.hitInstancedMesh.instanceColor) {
      this.hitInstancedMesh.instanceColor.needsUpdate = true;
    }
    this.group.add(this.hitInstancedMesh);
  }

  /**
   * Updates the 32 laser beams and glowing hit points in real-time.
   * Can accept external rayHits or run in high-fidelity procedural sweep mode.
   *
   * @param roverPosition - Current position of the rover or turret center
   * @param rayHits - Optional array of active LiDAR hits from physics / sensor engine
   */
  public update(
    roverPosition: THREE.Vector3 | Vec3,
    rayHits?: LiDARPoint[]
  ): void {
    const origin = roverPosition instanceof THREE.Vector3
      ? roverPosition
      : new THREE.Vector3(roverPosition[0], roverPosition[1], roverPosition[2]);

    const dt = 0.016; // approximate delta if not passed
    this.elapsedTime += dt;
    this.sweepAzimuth += 14.0 * dt; // Rapid 360-degree rotation

    if (rayHits && rayHits.length > 0) {
      this.updateFromRayHits(origin, rayHits);
    } else {
      this.updateProceduralCone(origin);
    }

    this.updateHitMarkers();
  }

  /**
   * Updates laser lines and hit points from active ray cast hits
   */
  private updateFromRayHits(origin: THREE.Vector3, hits: LiDARPoint[]): void {
    const count = Math.min(this.beamCount, hits.length);

    for (let i = 0; i < count; i++) {
      const hit = hits[i];
      const startIdx = i * 6;

      // Start vertex (LiDAR Turret Origin)
      this.beamPositions[startIdx] = origin.x;
      this.beamPositions[startIdx + 1] = origin.y;
      this.beamPositions[startIdx + 2] = origin.z;

      // Bright white-cyan core at origin
      this.beamColors[startIdx] = 0.6;
      this.beamColors[startIdx + 1] = 1.0;
      this.beamColors[startIdx + 2] = 1.0;

      // End vertex (Hit Point)
      const hitX = hit.hitPoint[0];
      const hitY = hit.hitPoint[1];
      const hitZ = hit.hitPoint[2];

      this.beamPositions[startIdx + 3] = hitX;
      this.beamPositions[startIdx + 4] = hitY;
      this.beamPositions[startIdx + 5] = hitZ;

      // Color code by entity type
      const col = this.getEntityColor(hit.entityType);
      this.beamColors[startIdx + 3] = col.r;
      this.beamColors[startIdx + 4] = col.g;
      this.beamColors[startIdx + 5] = col.b;

      // Register or update impact hit point
      this.recordHitPoint(new THREE.Vector3(hitX, hitY, hitZ), col);
    }

    // Hide remaining beams if fewer than 32 hits passed
    for (let i = count; i < this.beamCount; i++) {
      const startIdx = i * 6;
      for (let j = 0; j < 6; j++) {
        this.beamPositions[startIdx + j] = origin.x;
      }
    }

    this.beamGeometry.attributes.position.needsUpdate = true;
    this.beamGeometry.attributes.color.needsUpdate = true;
  }

  /**
   * High-fidelity procedural sweeping mode:
   * 32 beams sweep radially in a 360-degree cone from the rover turret
   */
  private updateProceduralCone(origin: THREE.Vector3): void {
    const baseHeading = this.sweepAzimuth;

    for (let i = 0; i < this.beamCount; i++) {
      const elev = this.sweepElevationAngles[i];
      // Distribute azimuthal spread around current sweeping direction
      const azimuthOffset = ((i - this.beamCount / 2) * (Math.PI / 64));
      const azimuth = baseHeading + azimuthOffset;

      const dirX = Math.cos(elev) * Math.sin(azimuth);
      const dirY = Math.sin(elev);
      const dirZ = Math.cos(elev) * Math.cos(azimuth);

      const startIdx = i * 6;

      // Origin
      this.beamPositions[startIdx] = origin.x;
      this.beamPositions[startIdx + 1] = origin.y;
      this.beamPositions[startIdx + 2] = origin.z;

      // White cyan at emitter
      this.beamColors[startIdx] = 0.5;
      this.beamColors[startIdx + 1] = 1.0;
      this.beamColors[startIdx + 2] = 1.0;

      // Compute hit distance (intersect ground at y = 0 or obstacle distance)
      let beamDist = this.maxRange;
      if (dirY < -0.01) {
        // Intersect ground level (y = 0)
        const groundDist = -origin.y / dirY;
        if (groundDist > 0 && groundDist < this.maxRange) {
          beamDist = groundDist;
        }
      }

      const endX = origin.x + dirX * beamDist;
      const endY = origin.y + dirY * beamDist;
      const endZ = origin.z + dirZ * beamDist;

      this.beamPositions[startIdx + 3] = endX;
      this.beamPositions[startIdx + 4] = endY;
      this.beamPositions[startIdx + 5] = endZ;

      // Neon cyan tip
      this.beamColors[startIdx + 3] = this.cyanColor.r;
      this.beamColors[startIdx + 4] = this.cyanColor.g;
      this.beamColors[startIdx + 5] = this.cyanColor.b;

      // Record hit if it struck ground or near range
      if (beamDist < this.maxRange * 0.95) {
        this.recordHitPoint(new THREE.Vector3(endX, endY, endZ), this.cyanColor);
      }
    }

    this.beamGeometry.attributes.position.needsUpdate = true;
    this.beamGeometry.attributes.color.needsUpdate = true;
  }

  /**
   * Stores an impact hit point into circular instanced buffer
   */
  private recordHitPoint(pos: THREE.Vector3, color: THREE.Color): void {
    const slot = this.activeHitCount % this.maxHitInstances;
    this.hitPositions[slot].copy(pos);
    this.hitColors[slot].copy(color);
    this.hitTimes[slot] = this.elapsedTime;
    this.activeHitCount++;
  }

  /**
   * Animates glowing impact hit markers with pulsing scale and fade
   */
  private updateHitMarkers(): void {
    const totalSlots = Math.min(this.activeHitCount, this.maxHitInstances);

    for (let i = 0; i < this.maxHitInstances; i++) {
      if (i < totalSlots) {
        const age = this.elapsedTime - this.hitTimes[i];
        const pos = this.hitPositions[i];

        // Glowing pulsing scale: harmonic expansion
        const pulse = 1.0 + 0.35 * Math.sin(this.elapsedTime * 9.0 + i * 0.5);
        const fade = Math.max(0.1, 1.0 - Math.min(age * 0.4, 0.9));
        const finalScale = 0.18 * pulse * fade;

        this.dummyTransform.position.copy(pos);
        this.dummyTransform.rotation.set(
          this.elapsedTime * 2.0 + i,
          this.elapsedTime * 3.0 + i,
          0
        );
        this.dummyTransform.scale.setScalar(finalScale);
        this.dummyTransform.updateMatrix();

        this.hitInstancedMesh.setMatrixAt(i, this.dummyTransform.matrix);
        this.hitInstancedMesh.setColorAt(i, this.hitColors[i]);
      } else {
        // Inactive instances scaled to 0
        this.dummyTransform.scale.set(0, 0, 0);
        this.dummyTransform.updateMatrix();
        this.hitInstancedMesh.setMatrixAt(i, this.dummyTransform.matrix);
      }
    }

    this.hitInstancedMesh.instanceMatrix.needsUpdate = true;
    if (this.hitInstancedMesh.instanceColor) {
      this.hitInstancedMesh.instanceColor.needsUpdate = true;
    }
  }

  /**
   * Entity color mapper for multi-spectral hazard discrimination
   */
  private getEntityColor(entityType?: string): THREE.Color {
    switch (entityType) {
      case 'hazard':
        return this.hazardColor;
      case 'countermeasure':
        return this.countermeasureColor;
      case 'structure':
        return this.structureColor;
      case 'terrain':
      default:
        return this.cyanColor;
    }
  }

  public getGroup(): THREE.Group {
    return this.group;
  }

  public setVisible(visible: boolean): void {
    this.group.visible = visible;
  }

  public setBeamOpacity(opacity: number): void {
    const mat = this.beamLines.material as THREE.LineBasicMaterial;
    mat.opacity = opacity;
  }

  public dispose(): void {
    this.beamGeometry.dispose();
    if (this.beamLines.material instanceof THREE.Material) {
      this.beamLines.material.dispose();
    }

    this.hitInstancedMesh.geometry.dispose();
    if (this.hitInstancedMesh.material instanceof THREE.Material) {
      this.hitInstancedMesh.material.dispose();
    }
  }
}
