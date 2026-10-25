import { Vec3, HazardType, StreamToken, RoverMorphologyMode } from '../types/simulation';

export type DSLCommandType =
  | 'ADD_SECTOR'
  | 'EXTEND_RAMP'
  | 'SPAWN_STRUCTURE'
  | 'TRIGGER_HAZARD'
  | 'DEPLOY_COUNTERMEASURE'
  | 'MUTATE_PHYSICS'
  | 'MORPH_ROVER';

export interface AddSectorCommand {
  type: 'ADD_SECTOR';
  id: string;
  center: Vec3;
  size: [number, number]; // [width, length]
  roughness: number;
  heightScale: number;
  biome: 'industrial_hazmat' | 'subterranean_tunnel' | 'collapsed_chasm' | 'reinforced_highway';
  rawSnippet?: string;
}

export interface ExtendRampCommand {
  type: 'EXTEND_RAMP';
  id: string;
  start: Vec3;
  end: Vec3;
  width: number;
  material: string;
  rawSnippet?: string;
}

export interface SpawnStructureCommand {
  type: 'SPAWN_STRUCTURE';
  id: string;
  structureType: 'bridge' | 'tunnel' | 'overpass' | 'tower';
  position: Vec3;
  scale: Vec3;
  rawSnippet?: string;
}

export interface TriggerHazardCommand {
  type: 'TRIGGER_HAZARD';
  id: string;
  hazardType: HazardType;
  position: Vec3;
  radius: number;
  severity: 'warning' | 'critical' | 'catastrophic' | number;
  rawSnippet?: string;
}

export interface DeployCountermeasureCommand {
  type: 'DEPLOY_COUNTERMEASURE';
  id: string;
  countermeasureType: 'carbon_ramp' | 'deployable_bridge' | 'sensor_beacon' | 'air_scrubber' | 'blast_barrier';
  position: Vec3;
  rotation?: Vec3;
  dimensions?: Vec3;
  rawSnippet?: string;
}

export interface MutatePhysicsCommand {
  type: 'MUTATE_PHYSICS';
  id: string;
  gravityY: number;
  surfaceFriction: number;
  lateralWindX: number;
  fogDensity: number;
  environmentLabel: string;
  rawSnippet?: string;
}

export interface MorphRoverCommand {
  type: 'MORPH_ROVER';
  id: string;
  mode: RoverMorphologyMode;
  suspensionHeight: number;
  trackWidthScale: number;
  thrusterBoost: boolean;
  rawSnippet?: string;
}

export type DSLCommand =
  | AddSectorCommand
  | ExtendRampCommand
  | SpawnStructureCommand
  | TriggerHazardCommand
  | DeployCountermeasureCommand
  | MutatePhysicsCommand
  | MorphRoverCommand;

type CommandCallback = (command: DSLCommand) => void;
type TokenCallback = (token: StreamToken) => void;

/**
 * High-speed append-only streaming parser designed for Gemini 4 Argon procedural generation.
 * Ingests partial tokens and chunks safely, extracts AST commands as soon as balanced payloads
 * close, and classifies lexical tokens for real-time cybernetic HUD visualization.
 */
export class ProceduralDSLParser {
  private buffer: string = '';
  private parsedCommands: DSLCommand[] = [];
  private streamTokens: StreamToken[] = [];
  private commandListeners: Set<CommandCallback> = new Set();
  private tokenListeners: Set<TokenCallback> = new Set();

  private static readonly KEYWORDS = new Set([
    'ADD_SECTOR',
    'EXTEND_RAMP',
    'SPAWN_STRUCTURE',
    'TRIGGER_HAZARD',
    'DEPLOY_COUNTERMEASURE',
    'MUTATE_PHYSICS',
    'MORPH_ROVER',
    'const',
    'let',
    'world',
    'createSector',
    'appendContinuousSector',
    'return',
    'function'
  ]);

  private static readonly GEOMETRY_IDENTIFIERS = new Set([
    'center',
    'size',
    'start',
    'end',
    'width',
    'position',
    'scale',
    'rotation',
    'dimensions',
    'elevation',
    'heightScale',
    'bounds',
    'suspensionHeight',
    'trackWidthScale'
  ]);

  private static readonly PHYSICS_IDENTIFIERS = new Set([
    'roughness',
    'material',
    'colliders',
    'trimesh',
    'retaining_barrier',
    'friction',
    'surfaceFriction',
    'restitution',
    'mass',
    'radius',
    'severity',
    'hazardType',
    'countermeasureType',
    'gravity',
    'gravityY',
    'windX',
    'lateralWindX',
    'fogDensity',
    'thrusterBoost',
    'mode'
  ]);

  /**
   * Appends an incoming chunk from the Gemini 4 Argon streaming stream.
   * Parses complete AST commands immediately, buffers incomplete constructs,
   * and dispatches tokens to listeners.
   */
  public pushChunk(chunk: string): DSLCommand[] {
    if (!chunk || chunk.length === 0) return [];

    // 1. Lexical classification of new tokens
    this.lexChunk(chunk);

    // 2. Append to continuous buffer
    this.buffer += chunk;

    // 3. Extract all complete AST commands
    const newCommands: DSLCommand[] = [];
    let parsed: DSLCommand | null;

    while ((parsed = this.extractNextCommand()) !== null) {
      newCommands.push(parsed);
      this.parsedCommands.push(parsed);
      this.commandListeners.forEach(cb => {
        try {
          cb(parsed!);
        } catch (err) {
          console.error('[ProceduralDSL] Error in command listener callback:', err);
        }
      });
    }

    // Safeguard buffer size if conversational commentary accumulates
    if (this.buffer.length > 32768) {
      const lastCmdIndex = Math.max(
        this.buffer.lastIndexOf('ADD_SECTOR'),
        this.buffer.lastIndexOf('EXTEND_RAMP'),
        this.buffer.lastIndexOf('SPAWN_STRUCTURE'),
        this.buffer.lastIndexOf('TRIGGER_HAZARD'),
        this.buffer.lastIndexOf('DEPLOY_COUNTERMEASURE'),
        this.buffer.lastIndexOf('MUTATE_PHYSICS'),
        this.buffer.lastIndexOf('MORPH_ROVER')
      );
      if (lastCmdIndex > 0) {
        this.buffer = this.buffer.slice(lastCmdIndex);
      } else if (this.buffer.length > 65536) {
        this.buffer = this.buffer.slice(32768);
      }
    }

    return newCommands;
  }

  /**
   * Parses an entire document or completed script in a single pass.
   */
  public parseComplete(fullText: string): DSLCommand[] {
    this.reset();
    return this.pushChunk(fullText);
  }

  /**
   * Scans buffer for complete command blocks.
   * Handles balanced braces, partial JSON objects, and single-quote / unquoted keys.
   */
  private extractNextCommand(): DSLCommand | null {
    const cmdRegex = /\b(ADD_SECTOR|EXTEND_RAMP|SPAWN_STRUCTURE|TRIGGER_HAZARD|DEPLOY_COUNTERMEASURE|MUTATE_PHYSICS|MORPH_ROVER)\b/g;
    const match = cmdRegex.exec(this.buffer);

    if (!match) {
      // Also look for alternative JS/TS functional syntax if emitted by the model
      return this.extractAlternativeFunctionalCommand();
    }

    const commandType = match[1] as DSLCommandType;
    const startIdx = match.index;

    // Locate the opening brace after the command keyword
    const openBraceIdx = this.buffer.indexOf('{', startIdx + commandType.length);
    if (openBraceIdx === -1) {
      // Opening brace hasn't arrived yet
      return null;
    }

    // Find the matching closing brace, accounting for string literals and escapes
    const closeBraceIdx = this.findMatchingBrace(this.buffer, openBraceIdx);
    if (closeBraceIdx === -1) {
      // Incomplete payload: waiting for more streaming chunks
      return null;
    }

    // Complete command payload isolated!
    const jsonStr = this.buffer.slice(openBraceIdx, closeBraceIdx + 1);
    const rawSnippet = this.buffer.slice(startIdx, closeBraceIdx + 1);

    // Parse the payload safely
    const payload = this.parseLenientJSON(jsonStr);

    let command: DSLCommand | null = null;
    if (payload) {
      command = this.mapPayloadToCommand(commandType, payload, rawSnippet);
    }

    // Advance buffer past this command
    this.buffer = this.buffer.slice(closeBraceIdx + 1);

    return command;
  }

  /**
   * Fallback extractor for code syntax like:
   * const sector = world.createSector({ id: "sec_01", center: [0,0,40], roughness: 0.38 ... });
   */
  private extractAlternativeFunctionalCommand(): DSLCommand | null {
    const createSectorIdx = this.buffer.indexOf('createSector(');
    if (createSectorIdx !== -1) {
      const openBraceIdx = this.buffer.indexOf('{', createSectorIdx);
      if (openBraceIdx !== -1) {
        const closeBraceIdx = this.findMatchingBrace(this.buffer, openBraceIdx);
        if (closeBraceIdx !== -1) {
          const jsonStr = this.buffer.slice(openBraceIdx, closeBraceIdx + 1);
          const rawSnippet = this.buffer.slice(createSectorIdx, closeBraceIdx + 2);
          const payload = this.parseLenientJSON(jsonStr);
          this.buffer = this.buffer.slice(closeBraceIdx + 1);

          if (payload) {
            return this.mapPayloadToCommand('ADD_SECTOR', payload, rawSnippet);
          }
        }
      }
    }
    return null;
  }

  /**
   * Tracks balanced braces ignoring braces that occur inside strings.
   */
  private findMatchingBrace(text: string, openIndex: number): number {
    let depth = 0;
    let inString: string | null = null;
    let isEscaped = false;

    for (let i = openIndex; i < text.length; i++) {
      const ch = text[i];

      if (isEscaped) {
        isEscaped = false;
        continue;
      }

      if (ch === '\\') {
        isEscaped = true;
        continue;
      }

      if (inString !== null) {
        if (ch === inString) {
          inString = null;
        }
        continue;
      }

      if (ch === '"' || ch === "'" || ch === '`') {
        inString = ch;
        continue;
      }

      if (ch === '{') {
        depth++;
      } else if (ch === '}') {
        depth--;
        if (depth === 0) {
          return i;
        }
      }
    }

    return -1; // Unclosed brace: partial token
  }

  /**
   * Extremely resilient JSON parser that handles:
   * - Unquoted object keys ({ center: [0,0,0] } -> { "center": [0,0,0] })
   * - Single-quoted strings ('carbon_ramp' -> "carbon_ramp")
   * - Trailing commas ({ a: 1, } -> { a: 1 })
   * - JS comments (// ...)
   * - Regex extraction fallback if standard JSON.parse fails.
   */
  private parseLenientJSON(raw: string): Record<string, any> | null {
    try {
      // 1. Remove comments
      let clean = raw.replace(/\/\/.*$/gm, '').replace(/\/\*[\s\S]*?\*\//g, '');

      // 2. Normalize single quotes to double quotes (except escaped ones)
      clean = clean.replace(/'([^'\\]*(?:\\.[^'\\]*)*)'/g, '"$1"');

      // 3. Quote unquoted identifier keys
      clean = clean.replace(/([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:/g, '$1"$2":');

      // 4. Strip trailing commas
      clean = clean.replace(/,\s*([}\]])/g, '$1');

      return JSON.parse(clean);
    } catch {
      // Fallback: heuristic regex key-value extraction for partial or non-standard payloads
      return this.heuristicRegexExtractor(raw);
    }
  }

  /**
   * Robust regex fallback extractor to guarantee zero crashes on LLM formatting quirks.
   */
  private heuristicRegexExtractor(text: string): Record<string, any> {
    const result: Record<string, any> = {};

    // Match id
    const idMatch = text.match(/id\s*[:=]\s*["']?([a-zA-Z0-9_-]+)["']?/i);
    if (idMatch) result.id = idMatch[1];

    // Match array coordinates like center, position, start, end, scale, size
    const arrayRegex = /(center|position|start|end|scale|size|rotation|dimensions)\s*[:=]\s*\[\s*([-0-9.]+)\s*,\s*([-0-9.]+)(?:\s*,\s*([-0-9.]+))?\s*\]/gi;
    let arrMatch: RegExpExecArray | null;
    while ((arrMatch = arrayRegex.exec(text)) !== null) {
      const key = arrMatch[1];
      const vals: number[] = [parseFloat(arrMatch[2]), parseFloat(arrMatch[3])];
      if (arrMatch[4] !== undefined) vals.push(parseFloat(arrMatch[4]));
      result[key] = vals;
    }

    // Match numeric properties
    const numRegex = /(roughness|heightScale|width|radius|severity|friction|restitution)\s*[:=]\s*([-0-9.]+)/gi;
    let nMatch: RegExpExecArray | null;
    while ((nMatch = numRegex.exec(text)) !== null) {
      result[nMatch[1]] = parseFloat(nMatch[2]);
    }

    // Match string properties like biome, type, material
    const strRegex = /(biome|type|material|countermeasureType|hazardType|structureType)\s*[:=]\s*["']?([a-zA-Z0-9_-]+)["']?/gi;
    let sMatch: RegExpExecArray | null;
    while ((sMatch = strRegex.exec(text)) !== null) {
      result[sMatch[1]] = sMatch[2];
    }

    return result;
  }

  /**
   * Maps lenient payload to strongly typed AST DSLCommand.
   */
  private mapPayloadToCommand(
    commandType: DSLCommandType,
    payload: Record<string, any>,
    rawSnippet: string
  ): DSLCommand {
    const fallbackId = `cmd_${Date.now()}_${Math.floor(Math.random() * 1000)}`;

    switch (commandType) {
      case 'ADD_SECTOR': {
        const center: Vec3 = Array.isArray(payload.center) && payload.center.length >= 3
          ? [Number(payload.center[0]), Number(payload.center[1]), Number(payload.center[2])]
          : [0, 0, 0];

        const size: [number, number] = Array.isArray(payload.size) && payload.size.length >= 2
          ? [Number(payload.size[0]), Number(payload.size[1])]
          : [64, 64];

        let biome: AddSectorCommand['biome'] = 'reinforced_highway';
        const candidateBiome = payload.biome || payload.material;
        if (candidateBiome === 'industrial_hazmat' || candidateBiome === 'hazmat') {
          biome = 'industrial_hazmat';
        } else if (candidateBiome === 'subterranean_tunnel' || candidateBiome === 'tunnel') {
          biome = 'subterranean_tunnel';
        } else if (candidateBiome === 'collapsed_chasm' || candidateBiome === 'chasm') {
          biome = 'collapsed_chasm';
        }

        return {
          type: 'ADD_SECTOR',
          id: String(payload.id || fallbackId),
          center,
          size,
          roughness: typeof payload.roughness === 'number' ? payload.roughness : 0.35,
          heightScale: typeof payload.heightScale === 'number' ? payload.heightScale : 4.0,
          biome,
          rawSnippet
        };
      }

      case 'EXTEND_RAMP': {
        const start: Vec3 = Array.isArray(payload.start) && payload.start.length >= 3
          ? [Number(payload.start[0]), Number(payload.start[1]), Number(payload.start[2])]
          : [0, 0, 0];

        const end: Vec3 = Array.isArray(payload.end) && payload.end.length >= 3
          ? [Number(payload.end[0]), Number(payload.end[1]), Number(payload.end[2])]
          : [0, 2, 20];

        return {
          type: 'EXTEND_RAMP',
          id: String(payload.id || fallbackId),
          start,
          end,
          width: typeof payload.width === 'number' ? payload.width : 6.0,
          material: String(payload.material || 'carbon_fiber'),
          rawSnippet
        };
      }

      case 'SPAWN_STRUCTURE': {
        const position: Vec3 = Array.isArray(payload.position) && payload.position.length >= 3
          ? [Number(payload.position[0]), Number(payload.position[1]), Number(payload.position[2])]
          : [0, 0, 0];

        const scale: Vec3 = Array.isArray(payload.scale) && payload.scale.length >= 3
          ? [Number(payload.scale[0]), Number(payload.scale[1]), Number(payload.scale[2])]
          : [1, 1, 1];

        let structureType: SpawnStructureCommand['structureType'] = 'bridge';
        const typeStr = String(payload.type || payload.structureType || 'bridge').toLowerCase();
        if (typeStr.includes('tunnel')) structureType = 'tunnel';
        else if (typeStr.includes('overpass')) structureType = 'overpass';
        else if (typeStr.includes('tower')) structureType = 'tower';

        return {
          type: 'SPAWN_STRUCTURE',
          id: String(payload.id || fallbackId),
          structureType,
          position,
          scale,
          rawSnippet
        };
      }

      case 'TRIGGER_HAZARD': {
        const position: Vec3 = Array.isArray(payload.position) && payload.position.length >= 3
          ? [Number(payload.position[0]), Number(payload.position[1]), Number(payload.position[2])]
          : [0, 0, 0];

        let hazardType: HazardType = 'seismic_collapse';
        const hTypeStr = String(payload.type || payload.hazardType || 'seismic_collapse').toLowerCase();
        if (hTypeStr.includes('chasm')) hazardType = 'chasm_rupture';
        else if (hTypeStr.includes('radioactive')) hazardType = 'radioactive_leak';
        else if (hTypeStr.includes('steam')) hazardType = 'steam_pipe_burst';
        else if (hTypeStr.includes('overpass')) hazardType = 'structural_overpass_fall';
        else if (hTypeStr.includes('emp')) hazardType = 'emp_interference';

        let severity: TriggerHazardCommand['severity'] = 'critical';
        if (typeof payload.severity === 'number') {
          severity = payload.severity;
        } else if (payload.severity === 'warning' || payload.severity === 'catastrophic') {
          severity = payload.severity;
        }

        return {
          type: 'TRIGGER_HAZARD',
          id: String(payload.id || fallbackId),
          hazardType,
          position,
          radius: typeof payload.radius === 'number' ? payload.radius : 15.0,
          severity,
          rawSnippet
        };
      }

      case 'DEPLOY_COUNTERMEASURE': {
        const position: Vec3 = Array.isArray(payload.position) && payload.position.length >= 3
          ? [Number(payload.position[0]), Number(payload.position[1]), Number(payload.position[2])]
          : [0, 0, 0];

        let cmType: DeployCountermeasureCommand['countermeasureType'] = 'carbon_ramp';
        const cmStr = String(payload.type || payload.countermeasureType || 'carbon_ramp').toLowerCase();
        if (cmStr.includes('bridge')) cmType = 'deployable_bridge';
        else if (cmStr.includes('beacon') || cmStr.includes('sensor')) cmType = 'sensor_beacon';
        else if (cmStr.includes('scrubber') || cmStr.includes('air')) cmType = 'air_scrubber';
        else if (cmStr.includes('barrier') || cmStr.includes('blast')) cmType = 'blast_barrier';

        let rotation: Vec3 | undefined;
        if (Array.isArray(payload.rotation) && payload.rotation.length >= 3) {
          rotation = [Number(payload.rotation[0]), Number(payload.rotation[1]), Number(payload.rotation[2])];
        }

        let dimensions: Vec3 | undefined;
        if (Array.isArray(payload.dimensions) && payload.dimensions.length >= 3) {
          dimensions = [Number(payload.dimensions[0]), Number(payload.dimensions[1]), Number(payload.dimensions[2])];
        }

        return {
          type: 'DEPLOY_COUNTERMEASURE',
          id: String(payload.id || fallbackId),
          countermeasureType: cmType,
          position,
          rotation,
          dimensions,
          rawSnippet
        };
      }

      case 'MUTATE_PHYSICS': {
        const gravityY = typeof payload.gravityY === 'number'
          ? payload.gravityY
          : typeof payload.gravity === 'number'
            ? payload.gravity
            : -9.81;
        const surfaceFriction = typeof payload.surfaceFriction === 'number'
          ? payload.surfaceFriction
          : typeof payload.friction === 'number'
            ? payload.friction
            : 0.75;
        const lateralWindX = typeof payload.lateralWindX === 'number'
          ? payload.lateralWindX
          : typeof payload.windX === 'number'
            ? payload.windX
            : 0;
        const fogDensity = typeof payload.fogDensity === 'number'
          ? payload.fogDensity
          : 0.008;
        const environmentLabel = String(
          payload.label || payload.environmentLabel || 'Mutated Argon Regime'
        );

        return {
          type: 'MUTATE_PHYSICS',
          id: String(payload.id || fallbackId),
          gravityY,
          surfaceFriction,
          lateralWindX,
          fogDensity,
          environmentLabel,
          rawSnippet
        };
      }

      case 'MORPH_ROVER': {
        const rawMode = String(payload.mode || payload.chassisMode || 'standard_cruiser').toLowerCase();
        let mode: RoverMorphologyMode = 'standard_cruiser';
        if (rawMode.includes('thruster') || rawMode.includes('hopper') || rawMode.includes('jump')) {
          mode = 'thruster_hopper';
        } else if (rawMode.includes('crawler') || rawMode.includes('clearance') || rawMode.includes('lift')) {
          mode = 'high_clearance_crawler';
        } else if (rawMode.includes('wide') || rawMode.includes('heavy') || rawMode.includes('track')) {
          mode = 'wide_track_heavy';
        }

        const suspensionHeight = typeof payload.suspensionHeight === 'number'
          ? payload.suspensionHeight
          : mode === 'high_clearance_crawler'
            ? 1.5
            : mode === 'thruster_hopper'
              ? 1.35
              : 1.0;

        const trackWidthScale = typeof payload.trackWidthScale === 'number'
          ? payload.trackWidthScale
          : mode === 'wide_track_heavy'
            ? 1.38
            : mode === 'high_clearance_crawler'
              ? 1.2
              : 1.0;

        const thrusterBoost = payload.thrusterBoost !== undefined
          ? Boolean(payload.thrusterBoost)
          : mode === 'thruster_hopper';

        return {
          type: 'MORPH_ROVER',
          id: String(payload.id || fallbackId),
          mode,
          suspensionHeight,
          trackWidthScale,
          thrusterBoost,
          rawSnippet
        };
      }
    }
  }

  /**
   * Lexical analysis of chunk for real-time syntax highlighted token stream HUD.
   */
  private lexChunk(chunk: string): void {
    const tokenRegex = /([a-zA-Z_][a-zA-Z0-9_]*|[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?|"[^"]*"|'[^']*'|[{}\[\],:;()=]|\s+|\/\/.*|\S)/g;
    let m: RegExpExecArray | null;

    while ((m = tokenRegex.exec(chunk)) !== null) {
      const tok = m[0];
      if (/^\s+$/.test(tok)) continue; // skip pure whitespace in token stream

      let kind: StreamToken['kind'] = 'meta';

      if (ProceduralDSLParser.KEYWORDS.has(tok)) {
        kind = 'keyword';
      } else if (ProceduralDSLParser.GEOMETRY_IDENTIFIERS.has(tok)) {
        kind = 'geometry';
      } else if (ProceduralDSLParser.PHYSICS_IDENTIFIERS.has(tok)) {
        kind = 'physics';
      } else if (/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(tok)) {
        kind = 'identifier';
      }

      const streamToken: StreamToken = {
        token: tok,
        kind,
        timestamp: Date.now()
      };

      this.streamTokens.push(streamToken);
      this.tokenListeners.forEach(cb => {
        try {
          cb(streamToken);
        } catch (e) {
          console.error('[ProceduralDSL] Error in token listener:', e);
        }
      });
    }
  }

  // ==========================================================================
  // Event Subscriptions & Queries
  // ==========================================================================

  public onCommand(callback: CommandCallback): () => void {
    this.commandListeners.add(callback);
    return () => this.commandListeners.delete(callback);
  }

  public onToken(callback: TokenCallback): () => void {
    this.tokenListeners.add(callback);
    return () => this.tokenListeners.delete(callback);
  }

  public getParsedCommands(): DSLCommand[] {
    return [...this.parsedCommands];
  }

  public getStreamTokens(): StreamToken[] {
    return [...this.streamTokens];
  }

  public getBuffer(): string {
    return this.buffer;
  }

  public reset(): void {
    this.buffer = '';
    this.parsedCommands = [];
    this.streamTokens = [];
  }
}
