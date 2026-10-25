import * as THREE from 'three';

export interface SceneManagerOptions {
  container?: HTMLElement;
  fogColor?: number;
  fogDensity?: number;
  ambientColor?: number;
  ambientIntensity?: number;
  exposure?: number;
  enableShadows?: boolean;
}

export type CameraViewMode = 'chase' | 'orbital' | 'tactical_iso' | 'first_person';

export class SceneManager {
  private container: HTMLElement;
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private renderer: THREE.WebGLRenderer;
  private ambientLight: THREE.AmbientLight;
  private facilityFloodlight: THREE.DirectionalLight;
  private floodlightTarget: THREE.Object3D;
  private rimLight: THREE.DirectionalLight;
  private fog: THREE.FogExp2;

  // Camera tracking state
  private viewMode: CameraViewMode = 'chase';
  private cameraTarget: THREE.Vector3 = new THREE.Vector3();
  private currentCameraPos: THREE.Vector3 = new THREE.Vector3(0, 12, -22);
  private resizeHandler: () => void;

  constructor(container: HTMLElement, options: SceneManagerOptions = {}) {
    this.container = container;

    const width = container.clientWidth || window.innerWidth || 800;
    const height = container.clientHeight || window.innerHeight || 600;

    // 1. Scene setup with atmospheric exponential fog
    this.scene = new THREE.Scene();
    const fogColorHex = options.fogColor ?? 0x06090e;
    const fogDensity = options.fogDensity ?? 0.015;

    this.scene.background = new THREE.Color(fogColorHex);
    this.fog = new THREE.FogExp2(fogColorHex, fogDensity);
    this.scene.fog = this.fog;

    // 2. Perspective Camera (Tactical 60 deg field of view)
    this.camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 1000);
    this.camera.position.copy(this.currentCameraPos);
    this.camera.lookAt(0, 0, 0);

    // 3. WebGLRenderer with cinema-grade ACES Filmic tonemapping & soft shadows
    this.renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: false,
      powerPreference: 'high-performance',
      stencil: false,
    });
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

    // Tonemapping
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = options.exposure ?? 1.1;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;

    // Shadows
    if (options.enableShadows !== false) {
      this.renderer.shadowMap.enabled = true;
      this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    }

    this.container.appendChild(this.renderer.domElement);

    // 4. Industrial Atmospheric Lighting
    // Ambient light: Dark blue/charcoal industrial tint
    const ambientHex = options.ambientColor ?? 0x1e293b;
    const ambientInt = options.ambientIntensity ?? 1.2;
    this.ambientLight = new THREE.AmbientLight(ambientHex, ambientInt);
    this.scene.add(this.ambientLight);

    // Directional light: Distant hazmat facility floodlight with soft shadows
    this.facilityFloodlight = new THREE.DirectionalLight(0xdbeafe, 2.4);
    this.facilityFloodlight.position.set(-45, 65, -35);
    this.facilityFloodlight.castShadow = true;

    // High fidelity shadow frustum
    this.facilityFloodlight.shadow.mapSize.width = 2048;
    this.facilityFloodlight.shadow.mapSize.height = 2048;
    this.facilityFloodlight.shadow.camera.near = 0.5;
    this.facilityFloodlight.shadow.camera.far = 250;
    const d = 40;
    this.facilityFloodlight.shadow.camera.left = -d;
    this.facilityFloodlight.shadow.camera.right = d;
    this.facilityFloodlight.shadow.camera.top = d;
    this.facilityFloodlight.shadow.camera.bottom = -d;
    this.facilityFloodlight.shadow.bias = -0.0005;
    this.facilityFloodlight.shadow.normalBias = 0.02;

    this.floodlightTarget = new THREE.Object3D();
    this.floodlightTarget.position.set(0, 0, 0);
    this.scene.add(this.floodlightTarget);
    this.facilityFloodlight.target = this.floodlightTarget;
    this.scene.add(this.facilityFloodlight);

    // Subtle cyan-tinted secondary industrial rim/fill light for rich volumetric depth
    this.rimLight = new THREE.DirectionalLight(0x0284c7, 0.7);
    this.rimLight.position.set(40, 20, 45);
    this.scene.add(this.rimLight);

    // Setup window resize listener
    this.resizeHandler = () => this.handleResize();
    window.addEventListener('resize', this.resizeHandler);
  }

  /**
   * Adds an object to the scene
   */
  public add(object: THREE.Object3D): void {
    this.scene.add(object);
  }

  /**
   * Removes an object from the scene
   */
  public remove(object: THREE.Object3D): void {
    this.scene.remove(object);
  }

  /**
   * Renders the current scene with the camera
   */
  public render(): void {
    this.renderer.render(this.scene, this.camera);
  }

  /**
   * Handles window or container resizing
   */
  public handleResize(width?: number, height?: number): void {
    const w = width ?? this.container.clientWidth ?? window.innerWidth;
    const h = height ?? this.container.clientHeight ?? window.innerHeight;

    if (w <= 0 || h <= 0) return;

    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();

    this.renderer.setSize(w, h);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  }

  /**
   * Updates camera position and orientation smoothly to follow a target (e.g. the Rover)
   */
  public updateCamera(
    targetPosition: THREE.Vector3,
    delta: number,
    headingAngle: number = 0,
    pitchAngle: number = 0
  ): void {
    const smoothLerpFactor = 1.0 - Math.exp(-delta * 6.0);

    // Smooth target tracking
    this.cameraTarget.lerp(targetPosition, smoothLerpFactor);

    if (this.viewMode === 'chase') {
      // Third-person tactical chase camera behind and slightly elevated
      const chaseDistance = 14.0;
      const chaseHeight = 6.5;

      // Position behind vehicle based on heading
      const offsetX = -Math.sin(headingAngle) * chaseDistance;
      const offsetZ = -Math.cos(headingAngle) * chaseDistance;

      const idealPos = new THREE.Vector3(
        this.cameraTarget.x + offsetX,
        this.cameraTarget.y + chaseHeight,
        this.cameraTarget.z + offsetZ
      );

      this.currentCameraPos.lerp(idealPos, smoothLerpFactor * 0.85);
      this.camera.position.copy(this.currentCameraPos);

      // Look slightly above the rover center
      const lookTarget = this.cameraTarget.clone().add(new THREE.Vector3(0, 1.2, 0));
      this.camera.lookAt(lookTarget);
    } else if (this.viewMode === 'tactical_iso') {
      // Elevated isometric observation view
      const isoOffset = new THREE.Vector3(-18, 22, -18);
      const idealPos = this.cameraTarget.clone().add(isoOffset);
      this.currentCameraPos.lerp(idealPos, smoothLerpFactor);
      this.camera.position.copy(this.currentCameraPos);
      this.camera.lookAt(this.cameraTarget);
    } else if (this.viewMode === 'first_person') {
      // Mounted directly on rover sensor mast
      const mastOffset = new THREE.Vector3(
        Math.sin(headingAngle) * 0.6,
        1.1,
        Math.cos(headingAngle) * 0.6
      );
      this.camera.position.copy(targetPosition).add(mastOffset);

      const lookDist = 20.0;
      const lookTarget = targetPosition.clone().add(
        new THREE.Vector3(
          Math.sin(headingAngle) * lookDist,
          Math.tan(pitchAngle) * lookDist,
          Math.cos(headingAngle) * lookDist
        )
      );
      this.camera.lookAt(lookTarget);
    } else if (this.viewMode === 'orbital') {
      // Gentle orbit around rover
      const time = performance.now() * 0.0003;
      const radius = 16.0;
      const height = 8.0;
      this.camera.position.set(
        this.cameraTarget.x + Math.sin(time) * radius,
        this.cameraTarget.y + height,
        this.cameraTarget.z + Math.cos(time) * radius
      );
      this.camera.lookAt(this.cameraTarget);
    }

    // Keep facility floodlight tracking around the rover area for optimal shadow detail
    this.floodlightTarget.position.copy(this.cameraTarget);
    this.facilityFloodlight.position.set(
      this.cameraTarget.x - 45,
      this.cameraTarget.y + 65,
      this.cameraTarget.z - 35
    );
  }

  public setViewMode(mode: CameraViewMode): void {
    this.viewMode = mode;
  }

  public getViewMode(): CameraViewMode {
    return this.viewMode;
  }

  public setFogDensity(density: number): void {
    this.fog.density = density;
  }

  public setFogColor(color: number | THREE.Color): void {
    if (typeof color === 'number') {
      this.fog.color.setHex(color);
      if (this.scene.background instanceof THREE.Color) {
        this.scene.background.setHex(color);
      }
    } else {
      this.fog.color.copy(color);
      if (this.scene.background instanceof THREE.Color) {
        this.scene.background.copy(color);
      }
    }
  }

  public setExposure(exposure: number): void {
    this.renderer.toneMappingExposure = exposure;
  }

  public setFloodlightIntensity(intensity: number): void {
    this.facilityFloodlight.intensity = intensity;
  }

  public getScene(): THREE.Scene {
    return this.scene;
  }

  public getCamera(): THREE.PerspectiveCamera {
    return this.camera;
  }

  public getRenderer(): THREE.WebGLRenderer {
    return this.renderer;
  }

  public getDomElement(): HTMLCanvasElement {
    return this.renderer.domElement;
  }

  public dispose(): void {
    window.removeEventListener('resize', this.resizeHandler);
    this.renderer.dispose();
    if (this.renderer.domElement.parentElement) {
      this.renderer.domElement.parentElement.removeChild(this.renderer.domElement);
    }
  }
}
