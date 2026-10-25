import RAPIER from '@dimforge/rapier3d-compat';
import type { Mesh } from 'three';
import { Vec3, LiDARPoint } from '../types/simulation';

export type PhysicsEntityType = 'terrain' | 'hazard' | 'structure' | 'countermeasure' | 'rover';

export interface RaycastHitResult {
  distance: number;
  hitPoint: Vec3;
  hitNormal: Vec3;
  colliderHandle: number;
  entityType: PhysicsEntityType;
  bodyId?: string;
}

export interface BodyTransform {
  position: Vec3;
  rotation: [number, number, number, number]; // [x, y, z, w]
  linearVelocity?: Vec3;
  angularVelocity?: Vec3;
}

export interface DynamicBodyOptions {
  mass?: number;
  friction?: number;
  restitution?: number;
  linearDamping?: number;
  angularDamping?: number;
  entityType?: PhysicsEntityType;
}

/**
 * High-performance Rapier 3D physics engine manager for procedural terrain sectors,
 * dynamic debris/obstacles, autonomous rover kinematics/dynamics, and LiDAR raycasting.
 */
export class PhysicsWorld {
  private static isInitialized = false;
  private static initPromise: Promise<void> | null = null;

  public world: RAPIER.World;
  private staticBodies: Map<string, { body: RAPIER.RigidBody; collider: RAPIER.Collider }> = new Map();
  private dynamicBodies: Map<string, { body: RAPIER.RigidBody; collider: RAPIER.Collider }> = new Map();
  private dynamicMeshes: Map<string, Mesh> = new Map();
  private colliderMetadata: Map<number, { id: string; entityType: PhysicsEntityType }> = new Map();

  private roverBody: RAPIER.RigidBody | null = null;
  private roverId: string | null = null;

  private accumulator: number = 0;
  private readonly fixedTimeStep: number = 1 / 60; // 60 Hz physics loop

  /**
   * Initializes Rapier WASM bindings. Safe to call multiple times.
   */
  public static async init(): Promise<void> {
    if (PhysicsWorld.isInitialized) return;
    if (!PhysicsWorld.initPromise) {
      PhysicsWorld.initPromise = RAPIER.init().then(() => {
        PhysicsWorld.isInitialized = true;
      });
    }
    return PhysicsWorld.initPromise;
  }

  /**
   * Factory method ensuring Rapier is initialized before instantiating the world.
   */
  public static async create(gravity: Vec3 = [0, -9.81, 0]): Promise<PhysicsWorld> {
    await PhysicsWorld.init();
    return new PhysicsWorld(gravity);
  }

  /**
   * Constructs Rapier world. Note: PhysicsWorld.init() must have been awaited before calling new PhysicsWorld().
   */
  constructor(gravity: Vec3 = [0, -9.81, 0]) {
    const rapierGravity = new RAPIER.Vector3(gravity[0], gravity[1], gravity[2]);
    this.world = new RAPIER.World(rapierGravity);
  }

  /**
   * Steps the physics simulation at fixed 60 Hz rate using an accumulator to prevent spiral-of-death.
   */
  public step(deltaTime: number): void {
    // Clamp excessive delta time (e.g. browser tab backgrounding)
    const clampedDelta = Math.min(deltaTime, 0.1);
    this.accumulator += clampedDelta;

    while (this.accumulator >= this.fixedTimeStep) {
      this.world.step();
      this.accumulator -= this.fixedTimeStep;
    }
  }

  /**
   * Executes a single physics step immediately.
   */
  public stepOnce(): void {
    this.world.step();
  }

  // ==========================================================================
  // Terrain Sector Trimesh Colliders
  // ==========================================================================

  /**
   * Registers a static triangle mesh collider for a procedural terrain sector.
   * Enables seamless collision with mathematically generated terrain manifolds.
   */
  public addTerrainTrimesh(
    id: string,
    vertices: Float32Array,
    indices: Uint32Array,
    position: Vec3 = [0, 0, 0]
  ): { body: RAPIER.RigidBody; collider: RAPIER.Collider } {
    // Remove previous sector registration if re-generating
    this.removeTerrainTrimesh(id);

    const bodyDesc = RAPIER.RigidBodyDesc.fixed().setTranslation(
      position[0],
      position[1],
      position[2]
    );
    const body = this.world.createRigidBody(bodyDesc);

    const colliderDesc = RAPIER.ColliderDesc.trimesh(vertices, indices)
      .setFriction(0.85)
      .setRestitution(0.05);

    const collider = this.world.createCollider(colliderDesc, body);

    this.staticBodies.set(id, { body, collider });
    this.colliderMetadata.set(collider.handle, { id, entityType: 'terrain' });

    return { body, collider };
  }

  /**
   * Removes a terrain sector trimesh and its Rapier rigid body.
   */
  public removeTerrainTrimesh(id: string): void {
    const entry = this.staticBodies.get(id);
    if (!entry) return;

    this.colliderMetadata.delete(entry.collider.handle);
    this.world.removeCollider(entry.collider, false);
    this.world.removeRigidBody(entry.body);
    this.staticBodies.delete(id);
  }

  // ==========================================================================
  // Dynamic Rigid Bodies (Hazards, Falling Debris, Spheres)
  // ==========================================================================

  /**
   * Adds a dynamic cuboid rigid body (e.g. falling concrete block, collapsed bridge segment).
   */
  public addDynamicCuboid(
    id: string,
    position: Vec3,
    halfExtents: Vec3,
    options: DynamicBodyOptions = {}
  ): RAPIER.RigidBody {
    this.removeRigidBody(id);

    const mass = options.mass ?? 10.0;
    const bodyDesc = RAPIER.RigidBodyDesc.dynamic()
      .setTranslation(position[0], position[1], position[2])
      .setAdditionalMass(mass)
      .setLinearDamping(options.linearDamping ?? 0.1)
      .setAngularDamping(options.angularDamping ?? 0.2);

    const body = this.world.createRigidBody(bodyDesc);

    const colliderDesc = RAPIER.ColliderDesc.cuboid(
      halfExtents[0],
      halfExtents[1],
      halfExtents[2]
    )
      .setFriction(options.friction ?? 0.6)
      .setRestitution(options.restitution ?? 0.2);

    const collider = this.world.createCollider(colliderDesc, body);

    this.dynamicBodies.set(id, { body, collider });
    this.colliderMetadata.set(collider.handle, {
      id,
      entityType: options.entityType ?? 'hazard'
    });

    return body;
  }

  /**
   * Adds a dynamic sphere rigid body (e.g. rolling boulder, seismic debris).
   */
  public addDynamicSphere(
    id: string,
    position: Vec3,
    radius: number,
    options: DynamicBodyOptions = {}
  ): RAPIER.RigidBody {
    this.removeRigidBody(id);

    const mass = options.mass ?? 15.0;
    const bodyDesc = RAPIER.RigidBodyDesc.dynamic()
      .setTranslation(position[0], position[1], position[2])
      .setAdditionalMass(mass)
      .setLinearDamping(options.linearDamping ?? 0.05)
      .setAngularDamping(options.angularDamping ?? 0.15);

    const body = this.world.createRigidBody(bodyDesc);

    const colliderDesc = RAPIER.ColliderDesc.ball(radius)
      .setFriction(options.friction ?? 0.7)
      .setRestitution(options.restitution ?? 0.3);

    const collider = this.world.createCollider(colliderDesc, body);

    this.dynamicBodies.set(id, { body, collider });
    this.colliderMetadata.set(collider.handle, {
      id,
      entityType: options.entityType ?? 'hazard'
    });

    return body;
  }

  /**
   * Adds a static structure or countermeasure collider (e.g. carbon ramp, deployable bridge).
   */
  public addStaticStructure(
    id: string,
    position: Vec3,
    halfExtents: Vec3,
    entityType: PhysicsEntityType = 'structure',
    rotationQuat?: [number, number, number, number]
  ): { body: RAPIER.RigidBody; collider: RAPIER.Collider } {
    this.removeRigidBody(id);

    const bodyDesc = RAPIER.RigidBodyDesc.fixed().setTranslation(
      position[0],
      position[1],
      position[2]
    );

    if (rotationQuat) {
      bodyDesc.setRotation(
        new RAPIER.Quaternion(rotationQuat[0], rotationQuat[1], rotationQuat[2], rotationQuat[3])
      );
    }

    const body = this.world.createRigidBody(bodyDesc);

    const colliderDesc = RAPIER.ColliderDesc.cuboid(
      halfExtents[0],
      halfExtents[1],
      halfExtents[2]
    )
      .setFriction(0.9)
      .setRestitution(0.05);

    const collider = this.world.createCollider(colliderDesc, body);

    this.staticBodies.set(id, { body, collider });
    this.colliderMetadata.set(collider.handle, { id, entityType });

    return { body, collider };
  }

  /**
   * Removes a dynamic or static rigid body and unregisters its collider metadata.
   */
  public removeRigidBody(id: string): void {
    this.dynamicMeshes.delete(id);
    if (id === this.roverId) {
      this.roverBody = null;
      this.roverId = null;
    }

    if (this.dynamicBodies.has(id)) {
      const entry = this.dynamicBodies.get(id)!;
      this.colliderMetadata.delete(entry.collider.handle);
      this.world.removeCollider(entry.collider, false);
      this.world.removeRigidBody(entry.body);
      this.dynamicBodies.delete(id);
    }

    if (this.staticBodies.has(id)) {
      const entry = this.staticBodies.get(id)!;
      this.colliderMetadata.delete(entry.collider.handle);
      this.world.removeCollider(entry.collider, false);
      this.world.removeRigidBody(entry.body);
      this.staticBodies.delete(id);
    }
  }

  /**
   * Retrieves world-space transform and velocities for Three.js synchronization.
   */
  public getBodyTransform(id: string): BodyTransform | null {
    const entry = this.dynamicBodies.get(id) || this.staticBodies.get(id);
    if (!entry) return null;

    const t = entry.body.translation();
    const r = entry.body.rotation();
    const lv = entry.body.isDynamic() ? entry.body.linvel() : { x: 0, y: 0, z: 0 };
    const av = entry.body.isDynamic() ? entry.body.angvel() : { x: 0, y: 0, z: 0 };

    return {
      position: [t.x, t.y, t.z],
      rotation: [r.x, r.y, r.z, r.w],
      linearVelocity: [lv.x, lv.y, lv.z],
      angularVelocity: [av.x, av.y, av.z]
    };
  }

  /**
   * Binds a Three.js mesh to an existing dynamic body (sphere, cuboid, rover).
   * Call after addDynamicSphere / addDynamicCuboid / createRover. Bodies with no
   * registered mesh are left untouched by syncDynamicMeshes.
   */
  public registerDynamicMesh(id: string, mesh: Mesh): void {
    this.dynamicMeshes.set(id, mesh);
  }

  /**
   * Copies each dynamic body's Rapier translation and rotation onto its registered mesh.
   */
  public syncDynamicMeshes(): void {
    for (const [id, entry] of this.dynamicBodies) {
      const mesh = this.dynamicMeshes.get(id);
      if (!mesh) continue;

      const t = entry.body.translation();
      const r = entry.body.rotation();
      mesh.position.set(t.x, t.y, t.z);
      mesh.quaternion.set(r.x, r.y, r.z, r.w);
    }
  }

  // ==========================================================================
  // Autonomous Rover Controller
  // ==========================================================================

  /**
   * Configures the autonomous rover rigid body.
   * Can be dynamic (physics-driven suspension/forces) or kinematic (direct trajectory positioning).
   */
  public createRover(
    id: string,
    initialPos: Vec3,
    halfExtents: Vec3 = [0.9, 0.45, 1.6],
    isKinematic: boolean = false
  ): RAPIER.RigidBody {
    this.removeRigidBody(id);

    let bodyDesc: RAPIER.RigidBodyDesc;
    if (isKinematic) {
      bodyDesc = RAPIER.RigidBodyDesc.kinematicPositionBased().setTranslation(
        initialPos[0],
        initialPos[1],
        initialPos[2]
      );
    } else {
      bodyDesc = RAPIER.RigidBodyDesc.dynamic()
        .setTranslation(initialPos[0], initialPos[1], initialPos[2])
        .setAdditionalMass(750) // 750 kg exploratory rover chassis
        .setLinearDamping(0.6)
        .setAngularDamping(2.5);
    }

    const body = this.world.createRigidBody(bodyDesc);

    // Rover chassis cuboid collider with chamfered bottom for step clearance
    const colliderDesc = RAPIER.ColliderDesc.cuboid(
      halfExtents[0],
      halfExtents[1],
      halfExtents[2]
    )
      .setFriction(0.95)
      .setRestitution(0.02);

    const collider = this.world.createCollider(colliderDesc, body);

    this.roverBody = body;
    this.roverId = id;
    this.dynamicBodies.set(id, { body, collider });
    this.colliderMetadata.set(collider.handle, { id, entityType: 'rover' });

    return body;
  }

  /**
   * Returns the existing rover body, or creates a kinematic one on first use.
   * Does not replace a rover that was already created.
   */
  public ensureRoverKinematic(
    id: string,
    halfExtents: Vec3,
    initialPosition: Vec3
  ): RAPIER.RigidBody {
    if (this.roverBody && this.roverId && this.dynamicBodies.has(this.roverId)) {
      return this.roverBody;
    }
    return this.createRover(id, initialPosition, halfExtents, true);
  }

  /**
   * Updates kinematic rover position and orientation.
   */
  public updateRoverKinematic(
    position: Vec3,
    rotationQuat: [number, number, number, number]
  ): void {
    if (!this.roverBody) return;
    this.roverBody.setNextKinematicTranslation(
      new RAPIER.Vector3(position[0], position[1], position[2])
    );
    this.roverBody.setNextKinematicRotation(
      new RAPIER.Quaternion(rotationQuat[0], rotationQuat[1], rotationQuat[2], rotationQuat[3])
    );
  }

  /**
   * Applies drive forces and steering torques to dynamic rover chassis.
   */
  public applyRoverForce(forceVec: Vec3, torqueY: number): void {
    if (!this.roverBody || !this.roverBody.isDynamic()) return;

    this.roverBody.applyImpulse(
      new RAPIER.Vector3(forceVec[0], forceVec[1], forceVec[2]),
      true
    );
    this.roverBody.applyTorqueImpulse(
      new RAPIER.Vector3(0, torqueY, 0),
      true
    );
  }

  public getRoverTransform(): BodyTransform | null {
    if (!this.roverId) return null;
    return this.getBodyTransform(this.roverId);
  }

  // ==========================================================================
  // LiDAR Raycasting & Proximity Queries
  // ==========================================================================

  /**
   * Casts a single LiDAR beam and computes hit distance, hit point, and surface normal.
   */
  public castRay(
    origin: Vec3,
    direction: Vec3,
    maxDistance: number = 100
  ): RaycastHitResult | null {
    const len = Math.hypot(direction[0], direction[1], direction[2]);
    if (len < 1e-6) return null;

    const dirX = direction[0] / len;
    const dirY = direction[1] / len;
    const dirZ = direction[2] / len;

    const ray = new RAPIER.Ray(
      new RAPIER.Vector3(origin[0], origin[1], origin[2]),
      new RAPIER.Vector3(dirX, dirY, dirZ)
    );

    // Cast ray against all solid colliders in world
    const hit = this.world.castRayAndGetNormal(ray, maxDistance, true);
    if (!hit) return null;

    const distance = hit.timeOfImpact;
    const hitPoint: Vec3 = [
      origin[0] + dirX * distance,
      origin[1] + dirY * distance,
      origin[2] + dirZ * distance
    ];

    const hitNormal: Vec3 = [hit.normal.x, hit.normal.y, hit.normal.z];
    const colliderHandle = hit.collider.handle;
    const meta = this.colliderMetadata.get(colliderHandle);

    return {
      distance,
      hitPoint,
      hitNormal,
      colliderHandle,
      entityType: meta?.entityType ?? 'terrain',
      bodyId: meta?.id
    };
  }

  /**
   * Downward ray from (x, rayStartY, z). Returns the highest hit surface Y
   * among terrain, ramps, and debris. The rover collider is excluded.
   * Returns null when the ray hits nothing.
   */
  public sampleSupportHeight(x: number, z: number, rayStartY: number = 100): number | null {
    // Colliders added since the last step are invisible to queries until this runs.
    this.world.updateSceneQueries();

    const ray = new RAPIER.Ray(
      new RAPIER.Vector3(x, rayStartY, z),
      new RAPIER.Vector3(0, -1, 0)
    );

    const roverCollider = this.roverId
      ? this.dynamicBodies.get(this.roverId)?.collider
      : undefined;

    let highestY: number | null = null;
    const maxDistance = Math.abs(rayStartY) + 500;

    this.world.intersectionsWithRay(
      ray,
      maxDistance,
      true,
      (intersect) => {
        const meta = this.colliderMetadata.get(intersect.collider.handle);
        if (meta?.entityType === 'rover') return true;

        const y = rayStartY - intersect.timeOfImpact;
        if (highestY === null || y > highestY) highestY = y;
        return true;
      },
      undefined,
      undefined,
      roverCollider,
      this.roverBody ?? undefined
    );

    return highestY;
  }

  /**
   * Casts a single beam and returns a typed LiDARPoint matching simulation data contracts.
   */
  public castLiDARPoint(
    origin: Vec3,
    direction: Vec3,
    maxDistance: number = 100
  ): LiDARPoint | null {
    const hit = this.castRay(origin, direction, maxDistance);
    if (!hit) return null;

    const len = Math.hypot(direction[0], direction[1], direction[2]);
    const dirNorm: Vec3 = len > 1e-6
      ? [direction[0] / len, direction[1] / len, direction[2] / len]
      : [0, 0, 1];

    let entityType: LiDARPoint['entityType'] = 'terrain';
    if (hit.entityType === 'hazard') entityType = 'hazard';
    else if (hit.entityType === 'structure') entityType = 'structure';
    else if (hit.entityType === 'countermeasure') entityType = 'countermeasure';

    return {
      origin,
      direction: dirNorm,
      hitPoint: hit.hitPoint,
      distance: hit.distance,
      hitNormal: hit.hitNormal,
      entityType
    };
  }

  /**
   * Sweeps an array of LiDAR rays across horizontal & vertical field-of-view from the rover mast.
   */
  public sweepLiDARScan(
    origin: Vec3,
    roverHeadingRad: number,
    numBeams: number = 32,
    hFovDeg: number = 110,
    pitchRad: number = -0.05,
    maxDistance: number = 80
  ): LiDARPoint[] {
    const points: LiDARPoint[] = [];
    const halfFovRad = (hFovDeg * Math.PI) / 360;
    const stepRad = (2 * halfFovRad) / Math.max(1, numBeams - 1);

    for (let i = 0; i < numBeams; i++) {
      const angleOffset = -halfFovRad + i * stepRad;
      const angle = roverHeadingRad + angleOffset;

      const dirX = Math.sin(angle) * Math.cos(pitchRad);
      const dirY = Math.sin(pitchRad);
      const dirZ = Math.cos(angle) * Math.cos(pitchRad);

      const pt = this.castLiDARPoint(origin, [dirX, dirY, dirZ], maxDistance);
      if (pt) {
        points.push(pt);
      }
    }

    return points;
  }

  // ==========================================================================
  // Lifecycle & Memory Management
  // ==========================================================================

  /**
   * Cleans up all Rapier rigid bodies, colliders, and frees WASM memory.
   */
  public dispose(): void {
    this.staticBodies.clear();
    this.dynamicBodies.clear();
    this.dynamicMeshes.clear();
    this.colliderMetadata.clear();
    this.roverBody = null;
    this.roverId = null;
    this.world.free();
  }
}
