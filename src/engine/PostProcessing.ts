import * as THREE from 'three';

export type VisionMode = 'standard' | 'night_vision' | 'thermal' | 'crt_terminal';

export interface PostProcessingConfig {
  vignetteIntensity?: number;   // 0.0 to 1.0 (default: 0.6)
  vignetteRadius?: number;      // 0.0 to 1.0 (default: 0.75)
  scanlinesEnabled?: boolean;   // default: true
  scanlineIntensity?: number;   // default: 0.18
  scanlineCount?: number;       // default: 800.0
  noiseIntensity?: number;      // default: 0.06
  chromaticAberration?: number; // default: 0.003
  visionMode?: VisionMode;      // default: 'standard'
}

/**
 * Screen-space GLSL Shaders for Tactical Industrial Hazmat Rendering
 */
const PostProcessShader = {
  vertexShader: /* glsl */ `
    varying vec2 vUv;
    void main() {
      vUv = uv;
      gl_Position = vec4(position.xy, 0.0, 1.0);
    }
  `,
  fragmentShader: /* glsl */ `
    uniform sampler2D tDiffuse;
    uniform vec2 uResolution;
    uniform float uTime;
    uniform float uVignetteIntensity;
    uniform float uVignetteRadius;
    uniform float uScanlineIntensity;
    uniform float uScanlineCount;
    uniform float uNoiseIntensity;
    uniform float uChromaticAberration;
    uniform float uGlitchIntensity;
    uniform int uVisionMode; // 0: standard, 1: night_vision, 2: thermal, 3: crt_terminal

    varying vec2 vUv;

    // High performance pseudorandom noise
    float random(vec2 p) {
      return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
    }

    // Thermal pseudo-color ramp
    vec3 thermalRamp(float val) {
      val = clamp(val, 0.0, 1.0);
      vec3 c0 = vec3(0.05, 0.02, 0.18); // deep blue/indigo (cold)
      vec3 c1 = vec3(0.55, 0.05, 0.45); // magenta
      vec3 c2 = vec3(0.95, 0.35, 0.05); // amber orange
      vec3 c3 = vec3(1.0, 0.95, 0.4);   // bright hot yellow
      vec3 c4 = vec3(1.0, 1.0, 1.0);    // white hot

      if (val < 0.25) return mix(c0, c1, val * 4.0);
      if (val < 0.50) return mix(c1, c2, (val - 0.25) * 4.0);
      if (val < 0.75) return mix(c2, c3, (val - 0.50) * 4.0);
      return mix(c3, c4, (val - 0.75) * 4.0);
    }

    void main() {
      vec2 uv = vUv;

      // 1. Glitch / EMP shock displacement
      if (uGlitchIntensity > 0.001) {
        float slice = floor(uv.y * 32.0);
        float sliceNoise = random(vec2(slice, floor(uTime * 18.0)));
        if (sliceNoise > 0.7) {
          uv.x += (random(vec2(uTime, slice)) - 0.5) * 0.06 * uGlitchIntensity;
        }
      }

      // 2. Chromatic aberration sampling
      float chroma = uChromaticAberration + uGlitchIntensity * 0.015;
      vec2 distFromCenter = uv - 0.5;
      vec2 rUv = uv + distFromCenter * chroma;
      vec2 bUv = uv - distFromCenter * chroma;

      vec4 baseCol = texture2D(tDiffuse, uv);
      float r = texture2D(tDiffuse, rUv).r;
      float g = baseCol.g;
      float b = texture2D(tDiffuse, bUv).b;
      vec3 color = vec3(r, g, b);

      // 3. Vision Mode Color Grading
      if (uVisionMode == 1) {
        // Night Vision: Phosphor monochrome green boost
        float lum = dot(color, vec3(0.299, 0.587, 0.114));
        color = vec3(lum * 0.2, lum * 1.35, lum * 0.45);
      } else if (uVisionMode == 2) {
        // FLIR Thermal: Infrared false color
        float lum = dot(color, vec3(0.299, 0.587, 0.114));
        color = thermalRamp(lum * 1.2);
      } else if (uVisionMode == 3) {
        // CRT Terminal: High-contrast amber phosphor tint
        float lum = dot(color, vec3(0.299, 0.587, 0.114));
        color = vec3(lum * 1.25, lum * 0.78, lum * 0.15);
      }

      // 4. CRT Scanline Grid
      if (uScanlineIntensity > 0.001) {
        float scanline = sin(uv.y * uScanlineCount * 3.14159);
        scanline = (scanline + 1.0) * 0.5; // 0 to 1
        float scanFactor = mix(1.0, scanline, uScanlineIntensity);
        color *= scanFactor;
      }

      // 5. Film Grain / Hazmat Sensor Noise
      if (uNoiseIntensity > 0.001) {
        float grain = (random(uv * 2.5 + fract(uTime * 35.0)) - 0.5) * uNoiseIntensity;
        color += grain;
      }

      // 6. Tactical Camera Vignette
      if (uVignetteIntensity > 0.001) {
        float dist = length(uv - 0.5);
        float vig = smoothstep(uVignetteRadius, uVignetteRadius - 0.4, dist);
        color *= mix(1.0 - uVignetteIntensity, 1.0, vig);
      }

      gl_FragColor = vec4(color, baseCol.a);
    }
  `,
};

export class PostProcessingManager {
  private config: Required<PostProcessingConfig>;
  private renderTarget: THREE.WebGLRenderTarget;
  private postScene: THREE.Scene;
  private postCamera: THREE.OrthographicCamera;
  private postMaterial: THREE.ShaderMaterial;
  private quadMesh: THREE.Mesh;

  // Glitch animation state
  private glitchDecayRate: number = 2.5; // decays per second
  private currentGlitch: number = 0.0;
  private elapsedTime: number = 0;

  constructor(
    width: number = window.innerWidth,
    height: number = window.innerHeight,
    userConfig: PostProcessingConfig = {}
  ) {
    this.config = {
      vignetteIntensity: userConfig.vignetteIntensity ?? 0.6,
      vignetteRadius: userConfig.vignetteRadius ?? 0.75,
      scanlinesEnabled: userConfig.scanlinesEnabled ?? true,
      scanlineIntensity: userConfig.scanlineIntensity ?? 0.18,
      scanlineCount: userConfig.scanlineCount ?? 800.0,
      noiseIntensity: userConfig.noiseIntensity ?? 0.06,
      chromaticAberration: userConfig.chromaticAberration ?? 0.003,
      visionMode: userConfig.visionMode ?? 'standard',
    };

    // 1. Offscreen Render Target matching screen resolution
    this.renderTarget = new THREE.WebGLRenderTarget(width, height, {
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      format: THREE.RGBAFormat,
      type: THREE.UnsignedByteType,
    });

    // 2. Fullscreen Orthographic Camera & Scene
    this.postScene = new THREE.Scene();
    this.postCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

    // 3. Shader Material with uniforms
    this.postMaterial = new THREE.ShaderMaterial({
      vertexShader: PostProcessShader.vertexShader,
      fragmentShader: PostProcessShader.fragmentShader,
      uniforms: {
        tDiffuse: { value: this.renderTarget.texture },
        uResolution: { value: new THREE.Vector2(width, height) },
        uTime: { value: 0 },
        uVignetteIntensity: { value: this.config.vignetteIntensity },
        uVignetteRadius: { value: this.config.vignetteRadius },
        uScanlineIntensity: {
          value: this.config.scanlinesEnabled ? this.config.scanlineIntensity : 0.0,
        },
        uScanlineCount: { value: this.config.scanlineCount },
        uNoiseIntensity: { value: this.config.noiseIntensity },
        uChromaticAberration: { value: this.config.chromaticAberration },
        uGlitchIntensity: { value: 0.0 },
        uVisionMode: { value: this.getVisionModeId(this.config.visionMode) },
      },
      depthTest: false,
      depthWrite: false,
    });

    // 4. Fullscreen Quad
    const quadGeo = new THREE.PlaneGeometry(2, 2);
    this.quadMesh = new THREE.Mesh(quadGeo, this.postMaterial);
    this.postScene.add(this.quadMesh);
  }

  /**
   * Renders the 3D scene through the post-processing pipeline onto the canvas
   */
  public render(
    renderer: THREE.WebGLRenderer,
    scene: THREE.Scene,
    camera: THREE.Camera,
    deltaTime: number = 0.016
  ): void {
    this.elapsedTime += deltaTime;

    // Decay glitch pulse
    if (this.currentGlitch > 0.001) {
      this.currentGlitch = Math.max(0.0, this.currentGlitch - deltaTime * this.glitchDecayRate);
    }

    // Update uniform values
    this.postMaterial.uniforms.uTime.value = this.elapsedTime;
    this.postMaterial.uniforms.uGlitchIntensity.value = this.currentGlitch;

    // Pass 1: Render 3D Scene into offscreen render target
    const prevRenderTarget = renderer.getRenderTarget();
    renderer.setRenderTarget(this.renderTarget);
    renderer.clear();
    renderer.render(scene, camera);

    // Pass 2: Render fullscreen post quad to screen canvas
    renderer.setRenderTarget(prevRenderTarget);
    renderer.render(this.postScene, this.postCamera);
  }

  /**
   * Handles canvas/window resize
   */
  public handleResize(width: number, height: number): void {
    if (width <= 0 || height <= 0) return;
    this.renderTarget.setSize(width, height);
    this.postMaterial.uniforms.uResolution.value.set(width, height);
  }

  /**
   * Adjusts vignette darkness and spread
   */
  public setVignette(intensity: number, radius: number = 0.75): void {
    this.config.vignetteIntensity = intensity;
    this.config.vignetteRadius = radius;
    this.postMaterial.uniforms.uVignetteIntensity.value = intensity;
    this.postMaterial.uniforms.uVignetteRadius.value = radius;
  }

  /**
   * Toggles scanlines and adjusts frequency
   */
  public setScanlines(enabled: boolean, intensity: number = 0.18, count: number = 800.0): void {
    this.config.scanlinesEnabled = enabled;
    this.config.scanlineIntensity = intensity;
    this.config.scanlineCount = count;
    this.postMaterial.uniforms.uScanlineIntensity.value = enabled ? intensity : 0.0;
    this.postMaterial.uniforms.uScanlineCount.value = count;
  }

  /**
   * Adjusts sensor ISO noise intensity
   */
  public setNoise(intensity: number): void {
    this.config.noiseIntensity = intensity;
    this.postMaterial.uniforms.uNoiseIntensity.value = intensity;
  }

  /**
   * Changes tactical vision mode ('standard' | 'night_vision' | 'thermal' | 'crt_terminal')
   */
  public setVisionMode(mode: VisionMode): void {
    this.config.visionMode = mode;
    this.postMaterial.uniforms.uVisionMode.value = this.getVisionModeId(mode);
  }

  /**
   * Triggers an EMP or hazard impact glitch wave
   */
  public triggerGlitch(intensity: number = 1.0, decayRate: number = 2.5): void {
    this.currentGlitch = Math.min(1.0, intensity);
    this.glitchDecayRate = decayRate;
  }

  public getRenderTarget(): THREE.WebGLRenderTarget {
    return this.renderTarget;
  }

  private getVisionModeId(mode: VisionMode): number {
    switch (mode) {
      case 'night_vision':
        return 1;
      case 'thermal':
        return 2;
      case 'crt_terminal':
        return 3;
      case 'standard':
      default:
        return 0;
    }
  }

  public dispose(): void {
    this.renderTarget.dispose();
    this.quadMesh.geometry.dispose();
    this.postMaterial.dispose();
  }
}
