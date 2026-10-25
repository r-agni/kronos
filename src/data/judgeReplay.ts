import type { HazardEvent, TerrainSector } from '../types/simulation';

/**
 * Keyless, append-only Argon DSL transcript for Judge Mode and offline sector requests.
 * Syntax matches parseDSLDirectives in server/geminiArgon.ts (quoted keys, brace blocks).
 * No API keys or credentials.
 */
export interface WorldDelta {
  sector?: TerrainSector;
  structures: Array<{
    type: string;
    position: [number, number, number];
    scale: [number, number, number];
    material: string;
  }>;
  hazards: HazardEvent[];
  rawDirectives: string;
}

const SECTOR_INDUSTRIAL_HAZMAT = `// === ARGON PROCEDURAL SECTOR 01 ===
// Lead World Architect: continuous collider boundary for industrial_hazmat
ADD_SECTOR {
  id: "sec_01_industrial_hazmat",
  name: "Sector 01 - INDUSTRIAL HAZMAT",
  biome: "industrial_hazmat",
  center: [0, 0, 50],
  size: [36, 60],
  heightScale: 2.20,
  roughness: 0.47,
  hasBridge: false,
  hasTunnel: false
}

SPAWN_STRUCTURE {
  type: "pipe_rack",
  position: [-10.5, 2.0, 40],
  scale: [2.0, 7.5, 2.0],
  material: "galvanized_industrial_steel"
}

SPAWN_STRUCTURE {
  type: "retaining_wall",
  position: [12.0, 1.5, 58],
  scale: [1.2, 3.4, 18.0],
  material: "reinforced_concrete"
}

TRIGGER_HAZARD {
  id: "hz_01_leak",
  type: "radioactive_leak",
  position: [4.0, 0.0, 66],
  radius: 12.5,
  severity: "warning"
}

DEPLOY_COUNTERMEASURE {
  id: "cm_01_ramp",
  type: "carbon_ramp",
  position: [0.0, 0.2, 72],
  rotation: [0.0, 0.0, 0.0],
  dimensions: [3.2, 0.4, 8.0]
}

MUTATE_PHYSICS {
  id: "phys_01_hazmat_gust",
  gravity: -9.81,
  friction: 0.58,
  windX: 8.5,
  fogDensity: 0.011,
  label: "Corrosive Crosswind Regime"
}
`;

const SECTOR_COLLAPSED_CHASM = `// === ARGON PROCEDURAL SECTOR 02 ===
// Lead World Architect: span the collapsed_chasm without breaking the seam
ADD_SECTOR {
  id: "sec_02_collapsed_chasm",
  name: "Sector 02 - COLLAPSED CHASM",
  biome: "collapsed_chasm",
  center: [0, 0, 110],
  size: [36, 60],
  heightScale: 3.40,
  roughness: 0.62,
  hasBridge: true,
  hasTunnel: false
}

SPAWN_STRUCTURE {
  type: "monolithic_arch",
  position: [0.0, 4.2, 104],
  scale: [14.0, 6.0, 3.0],
  material: "carbon_nanotube_reinforcement"
}

SPAWN_STRUCTURE {
  type: "catwalk",
  position: [-8.0, 3.1, 118],
  scale: [2.4, 1.1, 16.0],
  material: "grated_steel"
}

TRIGGER_HAZARD {
  id: "hz_02_chasm",
  type: "chasm_rupture",
  position: [2.5, 0.0, 124],
  radius: 14.0,
  severity: "critical"
}

DEPLOY_COUNTERMEASURE {
  id: "cm_02_bridge",
  type: "deployable_bridge",
  position: [0.0, 0.4, 128],
  rotation: [0.0, 0.0, 0.0],
  dimensions: [6.0, 0.6, 18.0]
}

MUTATE_PHYSICS {
  id: "phys_02_lunar_chasm",
  gravity: -1.62,
  friction: 0.35,
  windX: -12.0,
  fogDensity: 0.006,
  label: "Low-G 0.16g Seismic Fissure"
}

MORPH_ROVER {
  id: "morph_02_hopper",
  mode: "thruster_hopper",
  suspensionHeight: 1.42,
  trackWidthScale: 1.28,
  thrusterBoost: true
}
`;

const SECTOR_SUBTERRANEAN_TUNNEL = `// === ARGON PROCEDURAL SECTOR 03 ===
// Lead World Architect: bore the subterranean_tunnel and vent the steam main
ADD_SECTOR {
  id: "sec_03_subterranean_tunnel",
  name: "Sector 03 - SUBTERRANEAN TUNNEL",
  biome: "subterranean_tunnel",
  center: [0, 0, 170],
  size: [36, 60],
  heightScale: 1.80,
  roughness: 0.38,
  hasBridge: false,
  hasTunnel: true
}

SPAWN_STRUCTURE {
  type: "catwalk",
  position: [6.5, 2.4, 162],
  scale: [2.0, 1.2, 22.0],
  material: "anodized_aluminum"
}

SPAWN_STRUCTURE {
  type: "pipe_rack",
  position: [-9.0, 3.6, 176],
  scale: [1.8, 5.5, 12.0],
  material: "insulated_process_pipe"
}

TRIGGER_HAZARD {
  id: "hz_03_steam",
  type: "steam_pipe_burst",
  position: [-3.0, 1.0, 182],
  radius: 9.5,
  severity: "catastrophic"
}

DEPLOY_COUNTERMEASURE {
  id: "cm_03_scrubber",
  type: "air_scrubber",
  position: [1.5, 1.2, 186],
  rotation: [0.0, 1.57, 0.0],
  dimensions: [2.2, 2.8, 2.2]
}
`;

const SECTOR_REINFORCED_HIGHWAY = `// === ARGON PROCEDURAL SECTOR 04 ===
// Lead World Architect: reinforced_highway deck under overpass and interference
ADD_SECTOR {
  id: "sec_04_reinforced_highway",
  name: "Sector 04 - REINFORCED HIGHWAY",
  biome: "reinforced_highway",
  center: [0, 0, 230],
  size: [36, 60],
  heightScale: 1.40,
  roughness: 0.29,
  hasBridge: true,
  hasTunnel: false
}

SPAWN_STRUCTURE {
  type: "cooling_tower",
  position: [14.0, 8.0, 222],
  scale: [8.0, 16.0, 8.0],
  material: "cast_concrete"
}

SPAWN_STRUCTURE {
  type: "retaining_wall",
  position: [-13.0, 2.2, 238],
  scale: [1.4, 4.0, 20.0],
  material: "highway_barrier_concrete"
}

TRIGGER_HAZARD {
  id: "hz_04_overpass",
  type: "structural_overpass_fall",
  position: [0.0, 6.0, 244],
  radius: 16.0,
  severity: "critical"
}

TRIGGER_HAZARD {
  id: "hz_04_emp",
  type: "emp_interference",
  position: [5.0, 1.0, 250],
  radius: 20.0,
  severity: "warning"
}

DEPLOY_COUNTERMEASURE {
  id: "cm_04_beacon",
  type: "sensor_beacon",
  position: [-2.0, 0.5, 236],
  rotation: [0.0, 0.0, 0.0],
  dimensions: [0.6, 2.4, 0.6]
}
`;

/** Ordered append-only sectors. Each entry parses as its own WORLD_DELTA. */
export const JUDGE_REPLAY_SECTORS: readonly string[] = [
  SECTOR_INDUSTRIAL_HAZMAT,
  SECTOR_COLLAPSED_CHASM,
  SECTOR_SUBTERRANEAN_TUNNEL,
  SECTOR_REINFORCED_HIGHWAY
];

/** Full cached transcript, sectors in append order. */
export const JUDGE_REPLAY_TRANSCRIPT = JUDGE_REPLAY_SECTORS.join('');

export function judgeReplaySectorAt(index: number): string {
  const count = JUDGE_REPLAY_SECTORS.length;
  const wrapped = ((index % count) + count) % count;
  return JUDGE_REPLAY_SECTORS[wrapped];
}

/** Line chunks that concatenate back to the sector script. Local stream only. */
export function replayStreamChunks(script: string): string[] {
  if (!script) return [];
  const lines = script.split('\n');
  const chunks: string[] = [];
  for (let i = 0; i < lines.length; i++) {
    const chunk = i < lines.length - 1 ? `${lines[i]}\n` : lines[i];
    if (chunk.length > 0) chunks.push(chunk);
  }
  return chunks;
}

/**
 * Same extraction rules as parseDSLDirectives in server/geminiArgon.ts.
 * DEPLOY_COUNTERMEASURE stays in rawDirectives; it is not a separate WORLD_DELTA field.
 */
export function parseReplayDirectives(fullText: string): WorldDelta {
  const result: WorldDelta = {
    structures: [],
    hazards: [],
    rawDirectives: fullText
  };

  try {
    const sectorMatch = fullText.match(/ADD_SECTOR\s*\{([\s\S]*?)\}/);
    if (sectorMatch && sectorMatch[1]) {
      const block = sectorMatch[1];
      const id = block.match(/id:\s*['"]([^'"]+)['"]/)?.[1] || `sec_${Date.now()}`;
      const name = block.match(/name:\s*['"]([^'"]+)['"]/)?.[1] || 'Argon Procedural Sector';
      const biome = (block.match(/biome:\s*['"]([^'"]+)['"]/)?.[1] || 'industrial_hazmat') as TerrainSector['biome'];

      const centerMatch = block.match(/center:\s*\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]/);
      const center: [number, number, number] = centerMatch
        ? [parseFloat(centerMatch[1]), parseFloat(centerMatch[2]), parseFloat(centerMatch[3])]
        : [0, 0, 50];

      const sizeMatch = block.match(/size:\s*\[\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\]/);
      const size: [number, number] = sizeMatch
        ? [parseFloat(sizeMatch[1]), parseFloat(sizeMatch[2])]
        : [36, 60];

      const heightScale = parseFloat(block.match(/heightScale:\s*(\d+\.?\d*)/)?.[1] || '2.0');
      const roughness = parseFloat(block.match(/roughness:\s*(\d+\.?\d*)/)?.[1] || '0.4');
      const hasBridge = block.includes('hasBridge: true');
      const hasTunnel = block.includes('hasTunnel: true');

      result.sector = {
        id,
        name,
        biome,
        center,
        size,
        heightScale,
        roughness,
        hasBridge,
        hasTunnel,
        timestamp: Date.now()
      };
    }

    const structureRegex = /SPAWN_STRUCTURE\s*\{([\s\S]*?)\}/g;
    let sMatch: RegExpExecArray | null;
    while ((sMatch = structureRegex.exec(fullText)) !== null) {
      const block = sMatch[1];
      const type = block.match(/type:\s*['"]([^'"]+)['"]/)?.[1] || 'pipe_rack';
      const posMatch = block.match(/position:\s*\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]/);
      const position: [number, number, number] = posMatch
        ? [parseFloat(posMatch[1]), parseFloat(posMatch[2]), parseFloat(posMatch[3])]
        : [0, 0, 0];

      const scaleMatch = block.match(/scale:\s*\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]/);
      const scale: [number, number, number] = scaleMatch
        ? [parseFloat(scaleMatch[1]), parseFloat(scaleMatch[2]), parseFloat(scaleMatch[3])]
        : [1, 1, 1];

      const material = block.match(/material:\s*['"]([^'"]+)['"]/)?.[1] || 'steel';
      result.structures.push({ type, position, scale, material });
    }

    const hazardRegex = /TRIGGER_HAZARD\s*\{([\s\S]*?)(\}|$)/g;
    let hMatch: RegExpExecArray | null;
    while ((hMatch = hazardRegex.exec(fullText)) !== null) {
      const block = hMatch[1];
      const id = block.match(/id:\s*['"]([^'"]+)['"]/)?.[1] || `hz_${Date.now()}`;
      const type = (block.match(/type:\s*['"]([^'"]+)['"]/)?.[1] || 'seismic_collapse') as HazardEvent['type'];
      const posMatch = block.match(/position:\s*\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]/);
      const position: [number, number, number] = posMatch
        ? [parseFloat(posMatch[1]), parseFloat(posMatch[2]), parseFloat(posMatch[3])]
        : [0, 0, 0];
      const radius = parseFloat(block.match(/radius:\s*(\d+\.?\d*)/)?.[1] || '10');
      const severity = (block.match(/severity:\s*['"]([^'"]+)['"]/)?.[1] || 'warning') as HazardEvent['severity'];

      result.hazards.push({
        id,
        type,
        title: `Hazard: ${type.replace('_', ' ').toUpperCase()}`,
        position,
        radius,
        severity,
        timeRemainingSec: 25,
        triggerSource: 'vlm_adversary',
        active: true
      });
    }
  } catch (err) {
    console.error('[Judge Replay] Error parsing DSL directives:', err);
  }

  return result;
}
