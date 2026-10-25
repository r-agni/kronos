import * as THREE from 'three';
import type { RoverMorphConfig, RoverMorphologyMode } from '../types/simulation';

export interface RoverOptions {
  scale?: number;
  enableSpotlights?: boolean;
  spotlightIntensity?: number;
  initialPosition?: THREE.Vector3;
}

interface WheelAssembly {
  pivotGroup: THREE.Group;       // Vertical suspension compression
  steerGroup?: THREE.Group;      // Y-axis steering (front wheels)
  spinGroup: THREE.Group;       // X-axis wheel rolling
  isFront: boolean;
  isLeft: boolean;
  baseY: number;
  baseX: number;
}

export class RoverModel {
  private group: THREE.Group;
  private chassisGroup: THREE.Group;
  private turretGroup: THREE.Group;
  private lidarOriginMarker: THREE.Object3D;
  private thrusterPlumeGroup: THREE.Group;
  private activeMorphology: RoverMorphologyMode = 'standard_cruiser';
  private targetTrackScale: number = 1.0;
  private currentTrackScale: number = 1.0;
  private targetSuspensionLift: number = 0.0;
  private currentSuspensionLift: number = 0.0;
  private thrusterBoostActive: boolean = false;
  
  // Wheel tracking
  private wheelAssemblies: WheelAssembly[] = [];
  private wheelRadius: number = 0.35;
  private totalWheelSpin: number = 0;
  private currentSteerAngle: number = 0;

  // Spotlights and optics
  private leftSpotlight!: THREE.SpotLight;
  private rightSpotlight!: THREE.SpotLight;
  private leftSpotTarget!: THREE.Object3D;
  private rightSpotTarget!: THREE.Object3D;
  private lightCones: THREE.Mesh[] = [];
  private lensFlares: THREE.Mesh[] = [];

  // Animation timing
  private elapsedTime: number = 0;

  // Materials (kept for disposal & dynamic tuning)
  private materials: THREE.Material[] = [];

  constructor(options: RoverOptions = {}) {
    this.group = new THREE.Group();
    this.group.name = 'TacticalRover';

    this.chassisGroup = new THREE.Group();
    this.group.add(this.chassisGroup);

    // Build the procedural rover components
    this.buildChassis();
    this.buildSuspensionAndWheels();
    this.buildSensorTurret();
    this.buildSpotlights(options.spotlightIntensity ?? 18.0);
    this.buildAuxiliaryDetails();

    if (options.scale) {
      this.group.scale.setScalar(options.scale);
    }
    if (options.initialPosition) {
      this.group.position.copy(options.initialPosition);
    }
  }

  /**
   * Procedural Tactical Armored Chassis with Carbon Composite & Titanium
   */
  private buildChassis(): void {
    // 1. Materials
    const carbonCompositeMat = new THREE.MeshStandardMaterial({
      color: 0x111827, // Dark carbon / slate
      roughness: 0.45,
      metalness: 0.65,
    });
    this.materials.push(carbonCompositeMat);

    const titaniumPlateMat = new THREE.MeshStandardMaterial({
      color: 0x334155, // Titanium ballistic plating
      roughness: 0.28,
      metalness: 0.85,
    });
    this.materials.push(titaniumPlateMat);

    const hazardDecalMat = new THREE.MeshStandardMaterial({
      color: 0xf59e0b, // Tactical Amber / Hazard Yellow
      emissive: 0xb45309,
      emissiveIntensity: 0.45,
      roughness: 0.35,
      metalness: 0.2,
    });
    this.materials.push(hazardDecalMat);

    const glassVisorMat = new THREE.MeshStandardMaterial({
      color: 0x050c1a,
      roughness: 0.1,
      metalness: 0.95,
      envMapIntensity: 1.5,
    });
    this.materials.push(glassVisorMat);

    // Lower Skid Plate / Belly Hull (Beveled Armor)
    const bellyGeo = new THREE.BoxGeometry(1.05, 0.22, 2.3);
    const bellyMesh = new THREE.Mesh(bellyGeo, titaniumPlateMat);
    bellyMesh.position.set(0, 0.35, 0);
    bellyMesh.castShadow = true;
    bellyMesh.receiveShadow = true;
    this.chassisGroup.add(bellyMesh);

    // Main Upper Armored Hull
    const mainHullGeo = new THREE.BoxGeometry(0.96, 0.38, 2.1);
    const mainHullMesh = new THREE.Mesh(mainHullGeo, carbonCompositeMat);
    mainHullMesh.position.set(0, 0.58, 0.05);
    mainHullMesh.castShadow = true;
    mainHullMesh.receiveShadow = true;
    this.chassisGroup.add(mainHullMesh);

    // Sloped Front Bumper / Glacis Plate
    const frontGlacisGeo = new THREE.BoxGeometry(0.92, 0.28, 0.45);
    const frontGlacisMesh = new THREE.Mesh(frontGlacisGeo, titaniumPlateMat);
    frontGlacisMesh.position.set(0, 0.45, 1.15);
    frontGlacisMesh.rotation.x = -Math.PI / 7;
    frontGlacisMesh.castShadow = true;
    this.chassisGroup.add(frontGlacisMesh);

    // Front Tow & Push Bullbar
    const bullBarGeo = new THREE.CylinderGeometry(0.035, 0.035, 0.92, 12);
    bullBarGeo.rotateZ(Math.PI / 2);
    const bullBarMesh = new THREE.Mesh(bullBarGeo, titaniumPlateMat);
    bullBarMesh.position.set(0, 0.42, 1.34);
    bullBarMesh.castShadow = true;
    this.chassisGroup.add(bullBarMesh);

    // Sloped Sensor Avionics Cabin / Hood
    const avionicsGeo = new THREE.BoxGeometry(0.72, 0.24, 1.1);
    const avionicsMesh = new THREE.Mesh(avionicsGeo, carbonCompositeMat);
    avionicsMesh.position.set(0, 0.82, 0.1);
    avionicsMesh.castShadow = true;
    this.chassisGroup.add(avionicsMesh);

    // Dark Sensor Visor Panel (Front Windshield / Optronics Sensor Array)
    const visorGeo = new THREE.BoxGeometry(0.66, 0.14, 0.35);
    const visorMesh = new THREE.Mesh(visorGeo, glassVisorMat);
    visorMesh.position.set(0, 0.82, 0.62);
    visorMesh.rotation.x = -Math.PI / 6;
    this.chassisGroup.add(visorMesh);

    // Hazard Amber Accent Stripes (Flanks)
    const stripeGeoL = new THREE.BoxGeometry(0.04, 0.08, 1.7);
    const stripeMeshL = new THREE.Mesh(stripeGeoL, hazardDecalMat);
    stripeMeshL.position.set(-0.49, 0.65, 0.05);
    this.chassisGroup.add(stripeMeshL);

    const stripeMeshR = new THREE.Mesh(stripeGeoL, hazardDecalMat);
    stripeMeshR.position.set(0.49, 0.65, 0.05);
    this.chassisGroup.add(stripeMeshR);

    // Front Warning Chevrons on Bumper
    const chevronGeo = new THREE.BoxGeometry(0.35, 0.06, 0.03);
    const chevronL = new THREE.Mesh(chevronGeo, hazardDecalMat);
    chevronL.position.set(-0.25, 0.48, 1.3);
    chevronL.rotation.z = Math.PI / 12;
    this.chassisGroup.add(chevronL);

    const chevronR = new THREE.Mesh(chevronGeo, hazardDecalMat);
    chevronR.position.set(0.25, 0.48, 1.3);
    chevronR.rotation.z = -Math.PI / 12;
    this.chassisGroup.add(chevronR);

    // Exoskeleton Tubular Rollcage Bars (Titanium Rollhoops)
    const tubeMat = new THREE.MeshStandardMaterial({
      color: 0x475569,
      roughness: 0.3,
      metalness: 0.85,
    });
    this.materials.push(tubeMat);

    const rollbarPoints = [
      new THREE.Vector3(-0.44, 0.65, -0.75),
      new THREE.Vector3(-0.44, 1.05, -0.65),
      new THREE.Vector3(0.44, 1.05, -0.65),
      new THREE.Vector3(0.44, 0.65, -0.75),
    ];
    this.createTubularBar(rollbarPoints, 0.03, tubeMat);

    const frontRollbarPoints = [
      new THREE.Vector3(-0.44, 0.65, 0.45),
      new THREE.Vector3(-0.42, 1.02, 0.35),
      new THREE.Vector3(0.42, 1.02, 0.35),
      new THREE.Vector3(0.44, 0.65, 0.45),
    ];
    this.createTubularBar(frontRollbarPoints, 0.03, tubeMat);

    // Longitudinal Spine Bars
    const spineGeo = new THREE.CylinderGeometry(0.025, 0.025, 1.02, 8);
    const spineL = new THREE.Mesh(spineGeo, tubeMat);
    spineL.position.set(-0.43, 1.035, -0.15);
    spineL.rotation.x = Math.PI / 2;
    this.chassisGroup.add(spineL);

    const spineR = new THREE.Mesh(spineGeo, tubeMat);
    spineR.position.set(0.43, 1.035, -0.15);
    spineR.rotation.x = Math.PI / 2;
    this.chassisGroup.add(spineR);
  }

  /**
   * Helper to create tactical tubular rollcage geometry
   */
  private createTubularBar(points: THREE.Vector3[], radius: number, material: THREE.Material): void {
    const curve = new THREE.CatmullRomCurve3(points);
    const geo = new THREE.TubeGeometry(curve, 20, radius, 8, false);
    const mesh = new THREE.Mesh(geo, material);
    mesh.castShadow = true;
    this.chassisGroup.add(mesh);
  }

  /**
   * 6 Rugged Wheels with Off-road Treads on Suspension Pivots
   */
  private buildSuspensionAndWheels(): void {
    // Wheel Material: MeshStandardMaterial with roughness 0.8, metalness 0.2 (per requirement)
    const wheelTireMat = new THREE.MeshStandardMaterial({
      color: 0x14181d, // Deep rugged rubber
      roughness: 0.8,
      metalness: 0.2,
    });
    this.materials.push(wheelTireMat);

    // Titanium Beadlock Rim
    const rimMat = new THREE.MeshStandardMaterial({
      color: 0x334155,
      roughness: 0.3,
      metalness: 0.85,
    });
    this.materials.push(rimMat);

    // Hub center bolt / brake rotor
    const hubMat = new THREE.MeshStandardMaterial({
      color: 0x64748b,
      roughness: 0.2,
      metalness: 0.9,
    });
    this.materials.push(hubMat);

    // Suspension Linkage Arm Material
    const armMat = new THREE.MeshStandardMaterial({
      color: 0x1e293b,
      roughness: 0.4,
      metalness: 0.7,
    });
    this.materials.push(armMat);

    // Spring Damper Material (Hazard yellow/amber spring)
    const springMat = new THREE.MeshStandardMaterial({
      color: 0xd97706,
      roughness: 0.3,
      metalness: 0.5,
    });
    this.materials.push(springMat);

    // Positions for 6 wheels: Front (Z = +0.82), Mid (Z = 0.0), Rear (Z = -0.82)
    const zOffsets = [0.82, 0.0, -0.82];
    const wheelTrackX = 0.68; // Center offset from chassis spine
    const baseY = 0.34;

    for (let i = 0; i < 3; i++) {
      const zPos = zOffsets[i];
      const isFront = i === 0;

      // Left wheel (-X) and Right wheel (+X)
      for (const isLeft of [true, false]) {
        const xPos = isLeft ? -wheelTrackX : wheelTrackX;

        // 1. Suspension Pivot Group (moves up/down during bobbing)
        const pivotGroup = new THREE.Group();
        pivotGroup.position.set(xPos, baseY, zPos);
        this.group.add(pivotGroup);

        // Suspension A-arm wishbone linking chassis to wheel hub
        const armLength = 0.22;
        const armGeo = new THREE.BoxGeometry(armLength, 0.05, 0.08);
        const armMesh = new THREE.Mesh(armGeo, armMat);
        armMesh.position.set(isLeft ? armLength / 2 : -armLength / 2, 0.04, 0);
        armMesh.castShadow = true;
        pivotGroup.add(armMesh);

        // Shock Absorber Cylinder & Spring
        const damperGeo = new THREE.CylinderGeometry(0.024, 0.024, 0.26, 8);
        const damperMesh = new THREE.Mesh(damperGeo, springMat);
        damperMesh.position.set(isLeft ? 0.08 : -0.08, 0.12, 0);
        damperMesh.rotation.z = isLeft ? -0.35 : 0.35;
        pivotGroup.add(damperMesh);

        // Steering knuckle group (only for front wheels)
        let steerGroup: THREE.Group | undefined;
        let attachTarget: THREE.Group = pivotGroup;

        if (isFront) {
          steerGroup = new THREE.Group();
          pivotGroup.add(steerGroup);
          attachTarget = steerGroup;
        }

        // 2. Wheel Spin Group (rolls forward/backward along X axis)
        const spinGroup = new THREE.Group();
        attachTarget.add(spinGroup);

        // Procedural Rugged Tire Mesh
        const wheelMesh = this.createRuggedWheelMesh(wheelTireMat, rimMat, hubMat, isLeft);
        spinGroup.add(wheelMesh);

        this.wheelAssemblies.push({
          pivotGroup,
          steerGroup,
          spinGroup,
          isFront,
          isLeft,
          baseY,
          baseX: xPos,
        });
      }
    }
  }

  /**
   * Procedural Tire with aggressive tread lugs and beadlock rim
   */
  private createRuggedWheelMesh(
    tireMat: THREE.Material,
    rimMat: THREE.Material,
    hubMat: THREE.Material,
    isLeft: boolean
  ): THREE.Group {
    const wheelGroup = new THREE.Group();
    const tireRadius = this.wheelRadius;
    const tireWidth = 0.26;

    // Main tire toroidal/cylindrical tread body
    // Orient cylinder along X axis (wheel axle)
    const tireGeo = new THREE.CylinderGeometry(tireRadius, tireRadius, tireWidth, 24);
    tireGeo.rotateZ(Math.PI / 2);
    const tireMesh = new THREE.Mesh(tireGeo, tireMat);
    tireMesh.castShadow = true;
    wheelGroup.add(tireMesh);

    // Aggressive knobby tread blocks arranged radially
    const treadCount = 14;
    const lugWidth = tireWidth * 0.9;
    const lugHeight = 0.04;
    const lugThickness = 0.055;
    const lugGeo = new THREE.BoxGeometry(lugWidth, lugHeight, lugThickness);

    for (let i = 0; i < treadCount; i++) {
      const angle = (i / treadCount) * Math.PI * 2;
      const lugMesh = new THREE.Mesh(lugGeo, tireMat);
      lugMesh.position.set(
        0,
        Math.cos(angle) * (tireRadius + lugHeight * 0.4),
        Math.sin(angle) * (tireRadius + lugHeight * 0.4)
      );
      lugMesh.rotation.x = -angle;
      wheelGroup.add(lugMesh);
    }

    // Outer Titanium Beadlock Rim
    const rimRadius = tireRadius * 0.65;
    const rimGeo = new THREE.CylinderGeometry(rimRadius, rimRadius, tireWidth + 0.01, 16);
    rimGeo.rotateZ(Math.PI / 2);
    const rimMesh = new THREE.Mesh(rimGeo, rimMat);
    wheelGroup.add(rimMesh);

    // Planetary Hub cap and beadlock ring
    const hubGeo = new THREE.CylinderGeometry(0.09, 0.09, tireWidth + 0.03, 12);
    hubGeo.rotateZ(Math.PI / 2);
    const hubMesh = new THREE.Mesh(hubGeo, hubMat);
    wheelGroup.add(hubMesh);

    // Decorative Hub Bolts on the outer face
    const outerX = isLeft ? -tireWidth / 2 - 0.015 : tireWidth / 2 + 0.015;
    const boltCount = 6;
    const boltGeo = new THREE.CylinderGeometry(0.015, 0.015, 0.02, 6);
    boltGeo.rotateZ(Math.PI / 2);

    for (let b = 0; b < boltCount; b++) {
      const bAngle = (b / boltCount) * Math.PI * 2;
      const boltMesh = new THREE.Mesh(boltGeo, hubMat);
      boltMesh.position.set(
        outerX,
        Math.cos(bAngle) * 0.14,
        Math.sin(bAngle) * 0.14
      );
      wheelGroup.add(boltMesh);
    }

    return wheelGroup;
  }

  /**
   * Top Sensor Turret with Rotating LiDAR Puck Cylinder and Glowing Cyan LED Ring
   */
  private buildSensorTurret(): void {
    // Pedestal base mounted on chassis roof
    const pedestalMat = new THREE.MeshStandardMaterial({
      color: 0x1e293b,
      roughness: 0.35,
      metalness: 0.8,
    });
    this.materials.push(pedestalMat);

    const pedestalGeo = new THREE.CylinderGeometry(0.12, 0.15, 0.14, 16);
    const pedestalMesh = new THREE.Mesh(pedestalGeo, pedestalMat);
    pedestalMesh.position.set(0, 0.98, 0.18);
    pedestalMesh.castShadow = true;
    this.chassisGroup.add(pedestalMesh);

    // Turret Group (Spins 360 deg)
    this.turretGroup = new THREE.Group();
    this.turretGroup.position.set(0, 1.08, 0.18);
    this.group.add(this.turretGroup);

    // Anodized tactical matte black aluminum LiDAR puck cylinder
    const puckMat = new THREE.MeshStandardMaterial({
      color: 0x090d16,
      roughness: 0.35,
      metalness: 0.75,
    });
    this.materials.push(puckMat);

    const puckGeo = new THREE.CylinderGeometry(0.125, 0.125, 0.16, 24);
    const puckMesh = new THREE.Mesh(puckGeo, puckMat);
    puckMesh.castShadow = true;
    this.turretGroup.add(puckMesh);

    // Ribbed cooling fins on puck
    const finGeo = new THREE.CylinderGeometry(0.132, 0.132, 0.012, 24);
    for (const y of [-0.04, 0.04]) {
      const finMesh = new THREE.Mesh(finGeo, puckMat);
      finMesh.position.y = y;
      this.turretGroup.add(finMesh);
    }

    // Glowing Cyan LED Ring (Emissive 0x00f0ff per requirement)
    const ringMat = new THREE.MeshStandardMaterial({
      color: 0x00f0ff,
      emissive: 0x00f0ff,
      emissiveIntensity: 2.8,
      roughness: 0.2,
      metalness: 0.1,
    });
    this.materials.push(ringMat);

    const ringGeo = new THREE.CylinderGeometry(0.13, 0.13, 0.024, 24);
    const ringMesh = new THREE.Mesh(ringGeo, ringMat);
    ringMesh.position.y = 0.0;
    this.turretGroup.add(ringMesh);

    // Top Aperture Lens
    const lensMat = new THREE.MeshStandardMaterial({
      color: 0x001122,
      roughness: 0.05,
      metalness: 0.95,
    });
    this.materials.push(lensMat);

    const topCapGeo = new THREE.CylinderGeometry(0.09, 0.11, 0.02, 16);
    const topCapMesh = new THREE.Mesh(topCapGeo, lensMat);
    topCapMesh.position.y = 0.09;
    this.turretGroup.add(topCapMesh);

    // LiDAR Origin Marker Object (used by LiDAR visualizer for exact ray origin)
    this.lidarOriginMarker = new THREE.Object3D();
    this.lidarOriginMarker.position.set(0, 0, 0);
    this.turretGroup.add(this.lidarOriginMarker);
  }

  /**
   * Dual High-Intensity Forward Spotlights with volumetric light cones and lens flare meshes
   */
  private buildSpotlights(initialIntensity: number): void {
    const spotlightHousingMat = new THREE.MeshStandardMaterial({
      color: 0x1e293b,
      roughness: 0.4,
      metalness: 0.8,
    });
    this.materials.push(spotlightHousingMat);

    const lensGlowMat = new THREE.MeshStandardMaterial({
      color: 0xf8fafc,
      emissive: 0x93c5fd,
      emissiveIntensity: 3.5,
      roughness: 0.1,
    });
    this.materials.push(lensGlowMat);

    // Additive volumetric beam cone material
    const coneMat = new THREE.MeshBasicMaterial({
      color: 0x60a5fa,
      transparent: true,
      opacity: 0.12,
      blending: THREE.AdditiveBlending,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    this.materials.push(coneMat);

    const flareMat = new THREE.MeshBasicMaterial({
      color: 0xdbeafe,
      transparent: true,
      opacity: 0.75,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    this.materials.push(flareMat);

    const xOffsets = [-0.38, 0.38];
    const yPos = 0.58;
    const zPos = 1.25;

    // Left spotlight
    this.leftSpotlight = new THREE.SpotLight(0xf0fdf4, initialIntensity);
    this.leftSpotlight.position.set(xOffsets[0], yPos, zPos);
    this.leftSpotlight.angle = Math.PI / 7;
    this.leftSpotlight.penumbra = 0.45;
    this.leftSpotlight.distance = 65;
    this.leftSpotlight.castShadow = true;
    this.leftSpotlight.shadow.mapSize.width = 1024;
    this.leftSpotlight.shadow.mapSize.height = 1024;
    this.leftSpotlight.shadow.bias = -0.001;
    this.leftSpotlight.shadow.camera.near = 0.2;
    this.leftSpotlight.shadow.camera.far = 70;

    this.leftSpotTarget = new THREE.Object3D();
    this.leftSpotTarget.position.set(xOffsets[0], yPos - 0.5, zPos + 25);
    this.chassisGroup.add(this.leftSpotTarget);
    this.leftSpotlight.target = this.leftSpotTarget;
    this.chassisGroup.add(this.leftSpotlight);

    // Right spotlight
    this.rightSpotlight = new THREE.SpotLight(0xf0fdf4, initialIntensity);
    this.rightSpotlight.position.set(xOffsets[1], yPos, zPos);
    this.rightSpotlight.angle = Math.PI / 7;
    this.rightSpotlight.penumbra = 0.45;
    this.rightSpotlight.distance = 65;
    this.rightSpotlight.castShadow = true;
    this.rightSpotlight.shadow.mapSize.width = 1024;
    this.rightSpotlight.shadow.mapSize.height = 1024;
    this.rightSpotlight.shadow.bias = -0.001;
    this.rightSpotlight.shadow.camera.near = 0.2;
    this.rightSpotlight.shadow.camera.far = 70;

    this.rightSpotTarget = new THREE.Object3D();
    this.rightSpotTarget.position.set(xOffsets[1], yPos - 0.5, zPos + 25);
    this.chassisGroup.add(this.rightSpotTarget);
    this.rightSpotlight.target = this.rightSpotTarget;
    this.chassisGroup.add(this.rightSpotlight);

    // Build visible physical light housings, lens discs, and volumetric cones
    for (let i = 0; i < 2; i++) {
      const x = xOffsets[i];

      // Physical housing cylinder
      const housingGeo = new THREE.CylinderGeometry(0.08, 0.08, 0.14, 16);
      housingGeo.rotateX(Math.PI / 2);
      const housingMesh = new THREE.Mesh(housingGeo, spotlightHousingMat);
      housingMesh.position.set(x, yPos, zPos);
      this.chassisGroup.add(housingMesh);

      // Glowing lens disc
      const lensGeo = new THREE.CircleGeometry(0.075, 16);
      const lensMesh = new THREE.Mesh(lensGeo, lensGlowMat);
      lensMesh.position.set(x, yPos, zPos + 0.072);
      this.chassisGroup.add(lensMesh);

      // Lens flare halo plane
      const flareGeo = new THREE.PlaneGeometry(0.35, 0.35);
      const flareMesh = new THREE.Mesh(flareGeo, flareMat);
      flareMesh.position.set(x, yPos, zPos + 0.08);
      this.chassisGroup.add(flareMesh);
      this.lensFlares.push(flareMesh);

      // Volumetric beam cone
      // Cone height 10m, radius 2.6m
      const coneLength = 9.0;
      const coneRadius = 2.4;
      const coneGeo = new THREE.ConeGeometry(coneRadius, coneLength, 16, 1, true);
      // Move cone apex to origin and orient along +Z
      coneGeo.translate(0, -coneLength / 2, 0);
      coneGeo.rotateX(-Math.PI / 2);
      const coneMesh = new THREE.Mesh(coneGeo, coneMat);
      coneMesh.position.set(x, yPos, zPos + 0.07);
      this.chassisGroup.add(coneMesh);
      this.lightCones.push(coneMesh);
    }
  }

  /**
   * Auxiliary details: rear brake/tail lights, antenna mast, radiator vents
   */
  private buildAuxiliaryDetails(): void {
    // Rear Brake / Tail Lights (Emissive Red)
    const tailMat = new THREE.MeshStandardMaterial({
      color: 0xef4444,
      emissive: 0xef4444,
      emissiveIntensity: 2.2,
      roughness: 0.3,
    });
    this.materials.push(tailMat);

    const tailLightGeo = new THREE.BoxGeometry(0.14, 0.06, 0.04);
    for (const x of [-0.4, 0.4]) {
      const tailMesh = new THREE.Mesh(tailLightGeo, tailMat);
      tailMesh.position.set(x, 0.58, -1.02);
      this.chassisGroup.add(tailMesh);
    }

    // Rear Radiator Heat Sinks (slotted fins with faint crimson thermal glow)
    const heatGlowMat = new THREE.MeshStandardMaterial({
      color: 0x450a0a,
      emissive: 0x7f1d1d,
      emissiveIntensity: 0.6,
      roughness: 0.5,
    });
    this.materials.push(heatGlowMat);

    const ventBackplate = new THREE.BoxGeometry(0.55, 0.22, 0.04);
    const ventMesh = new THREE.Mesh(ventBackplate, heatGlowMat);
    ventMesh.position.set(0, 0.52, -1.01);
    this.chassisGroup.add(ventMesh);

    // Comms Mast & Antenna
    const mastMat = new THREE.MeshStandardMaterial({
      color: 0x475569,
      roughness: 0.3,
      metalness: 0.8,
    });
    this.materials.push(mastMat);

    const mastGeo = new THREE.CylinderGeometry(0.015, 0.02, 0.75, 8);
    const mastMesh = new THREE.Mesh(mastGeo, mastMat);
    mastMesh.position.set(0.36, 1.25, -0.6);
    this.chassisGroup.add(mastMesh);

    // Flashing Beacon LED atop antenna
    const beaconMat = new THREE.MeshBasicMaterial({
      color: 0xf59e0b,
    });
    this.materials.push(beaconMat);

    const beaconGeo = new THREE.SphereGeometry(0.035, 8, 8);
    const beaconMesh = new THREE.Mesh(beaconGeo, beaconMat);
    beaconMesh.position.set(0.36, 1.63, -0.6);
    this.chassisGroup.add(beaconMesh);

    // Plasma Jump Thruster Nozzles & Plumes (for Thruster Hopper Morphology)
    this.thrusterPlumeGroup = new THREE.Group();
    this.thrusterPlumeGroup.visible = false;
    const plumeMat = new THREE.MeshBasicMaterial({
      color: 0x00f0ff,
      transparent: true,
      opacity: 0.78,
    });
    this.materials.push(plumeMat);
    const plumeGeo = new THREE.ConeGeometry(0.16, 0.75, 12);
    plumeGeo.rotateX(Math.PI);
    for (const [tx, tz] of [[-0.52, 0.55], [0.52, 0.55], [-0.52, -0.55], [0.52, -0.55]]) {
      const plume = new THREE.Mesh(plumeGeo, plumeMat);
      plume.position.set(tx, 0.08, tz);
      this.thrusterPlumeGroup.add(plume);
    }
    this.chassisGroup.add(this.thrusterPlumeGroup);
  }

  public applyMorphology(config: RoverMorphConfig): void {
    this.activeMorphology = config.mode;
    this.targetTrackScale = Math.max(0.85, Math.min(1.6, config.trackWidthScale));
    this.targetSuspensionLift = Math.max(0, Math.min(0.65, (config.suspensionHeight - 1.0) * 0.55));
    this.thrusterBoostActive = Boolean(config.thrusterBoost);
    if (this.thrusterPlumeGroup) {
      this.thrusterPlumeGroup.visible = this.thrusterBoostActive;
    }
  }

  public getActiveMorphology(): RoverMorphologyMode {
    return this.activeMorphology;
  }

  /**
   * Updates rover state every frame:
   * - Rotates wheels proportional to speed
   * - Steers front wheels according to steerAngle
   * - Bobs suspension slightly based on speed and time
   * - Spins the LiDAR turret
   */
  public update(deltaTime: number, speed: number, steerAngle: number): void {
    this.elapsedTime += deltaTime;

    // 1. Wheel Rolling Rotation proportional to speed
    const distanceDelta = speed * deltaTime;
    const spinDelta = distanceDelta / this.wheelRadius;
    this.totalWheelSpin += spinDelta;

    // 2. Smooth Steering Angle for front wheels
    const steerLerpSpeed = Math.min(deltaTime * 12.0, 1.0);
    this.currentSteerAngle += (steerAngle - this.currentSteerAngle) * steerLerpSpeed;

    // 3. Subtle Suspension Bobbing based on speed and vibration
    const speedMagnitude = Math.abs(speed);
    const bobFrequency = 10.0 + speedMagnitude * 1.5;
    const baseBobAmplitude = Math.min(speedMagnitude * 0.014, 0.035);

    // Smoothly interpolate live chassis morphology
    const morphLerp = Math.min(deltaTime * 6.0, 1.0);
    this.currentTrackScale += (this.targetTrackScale - this.currentTrackScale) * morphLerp;
    this.currentSuspensionLift += (this.targetSuspensionLift - this.currentSuspensionLift) * morphLerp;

    // Chassis micro-pitch/roll based on speed & acceleration + morphology lift
    this.chassisGroup.position.y = this.currentSuspensionLift + Math.sin(this.elapsedTime * bobFrequency) * baseBobAmplitude;
    this.chassisGroup.rotation.x = Math.sin(this.elapsedTime * 6.0) * (baseBobAmplitude * 0.4);
    this.turretGroup.position.y = 1.08 + this.currentSuspensionLift;

    if (this.thrusterPlumeGroup) {
      this.thrusterPlumeGroup.visible = this.thrusterBoostActive;
      if (this.thrusterBoostActive) {
        const pulse = 0.85 + 0.3 * Math.sin(this.elapsedTime * 28.0);
        this.thrusterPlumeGroup.scale.set(1, pulse, 1);
      }
    }

    // Apply rotation, track width morphing, and suspension to all 6 wheels
    for (let i = 0; i < this.wheelAssemblies.length; i++) {
      const assembly = this.wheelAssemblies[i];

      // Roll the wheel around its local X-axis
      assembly.spinGroup.rotation.x = this.totalWheelSpin;

      // Steer front wheels around Y-axis
      if (assembly.isFront && assembly.steerGroup) {
        assembly.steerGroup.rotation.y = -this.currentSteerAngle;
      }

      // Individual suspension bounce with phase offset per axle + track width morph
      const axlePhase = assembly.isFront ? 0.0 : assembly.isLeft ? 1.2 : 2.4;
      const wheelBob = Math.cos(this.elapsedTime * bobFrequency + axlePhase) * baseBobAmplitude * 0.6;
      assembly.pivotGroup.position.x = assembly.baseX * this.currentTrackScale;
      assembly.pivotGroup.position.y = assembly.baseY + wheelBob;
    }

    // 4. Spin LiDAR Turret (360 degrees rotation, ~600 RPM tactical sweep)
    const turretSpinSpeed = 14.0; // rad/s
    this.turretGroup.rotation.y += turretSpinSpeed * deltaTime;

    // 5. Subtle flicker/pulse on light flares
    const flarePulse = 0.75 + 0.15 * Math.sin(this.elapsedTime * 20.0);
    for (const flare of this.lensFlares) {
      flare.scale.setScalar(flarePulse);
    }
  }

  /**
   * Returns the world position of the LiDAR puck optical center
   */
  public getLiDAROrigin(): THREE.Vector3 {
    const worldPos = new THREE.Vector3();
    this.lidarOriginMarker.getWorldPosition(worldPos);
    return worldPos;
  }

  /**
   * Returns the Three.js root object for adding to scenes
   */
  public getGroup(): THREE.Group {
    return this.group;
  }

  public setPosition(x: number, y: number, z: number): void {
    this.group.position.set(x, y, z);
  }

  public setRotation(x: number, y: number, z: number): void {
    this.group.rotation.set(x, y, z);
  }

  public getHeading(): number {
    return this.group.rotation.y;
  }

  /**
   * Toggles spotlights on or off
   */
  public setSpotlights(enabled: boolean, intensity: number = 18.0): void {
    this.leftSpotlight.visible = enabled;
    this.rightSpotlight.visible = enabled;
    this.leftSpotlight.intensity = enabled ? intensity : 0;
    this.rightSpotlight.intensity = enabled ? intensity : 0;

    for (const cone of this.lightCones) {
      cone.visible = enabled;
    }
    for (const flare of this.lensFlares) {
      flare.visible = enabled;
    }
  }

  public getSpotlights(): THREE.SpotLight[] {
    return [this.leftSpotlight, this.rightSpotlight];
  }

  /**
   * Clean up all geometries and materials on teardown
   */
  public dispose(): void {
    this.group.traverse((child) => {
      if (child instanceof THREE.Mesh) {
        child.geometry.dispose();
      }
    });

    for (const mat of this.materials) {
      mat.dispose();
    }
    this.materials = [];
  }
}
