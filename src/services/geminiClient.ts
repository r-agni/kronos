import { AdversaryCritique, BaselineMetricComparison, RobotTelemetry } from '../types/simulation';
import {
  judgeReplaySectorAt,
  parseReplayDirectives,
  replayStreamChunks,
  type WorldDelta
} from '../data/judgeReplay';

export type { WorldDelta } from '../data/judgeReplay';

type TokenCallback = (token: string, fullCode: string) => void;
type SectorCallback = (sectorData: unknown) => void;
type WorldDeltaCallback = (delta: WorldDelta) => void;
type AdversaryCallback = (critique: AdversaryCritique) => void;
export interface VoiceCommandPayload {
  action: string;
  countermeasureType: string;
  positionDelta: [number, number, number];
  voiceResponse: string;
  dslMutation?: string;
}
type VoiceToolCallback = (toolCall: VoiceCommandPayload) => void;
type MetricCallback = (metric: BaselineMetricComparison) => void;

class GeminiServiceClient {
  private ws: WebSocket | null = null;
  private isConnected: boolean = false;
  private judgeMode: boolean = false;
  private replayCursor: number = 0;
  private onTokenCallbacks: Set<TokenCallback> = new Set();
  private onSectorCallbacks: Set<SectorCallback> = new Set();
  private onWorldDeltaCallbacks: Set<WorldDeltaCallback> = new Set();
  private onAdversaryCallbacks: Set<AdversaryCallback> = new Set();
  private onVoiceCallbacks: Set<VoiceToolCallback> = new Set();
  private onMetricCallbacks: Set<MetricCallback> = new Set();

  private fallbackSimulationTimer: number | null = null;
  private accumulatedCode: string = '';

  constructor() {
    this.connect();
  }

  public connect() {
    try {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${protocol}//${window.location.hostname}:3001/ws`;
      this.ws = new WebSocket(wsUrl);

      this.ws.onopen = () => {
        this.isConnected = true;
        console.log('[Gemini Client] Connected to backend orchestrator.');
      };

      this.ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          this.handleServerMessage(msg);
        } catch (e) {
          console.error('[Gemini Client] Error parsing WS message:', e);
        }
      };

      this.ws.onclose = () => {
        this.isConnected = false;
        console.log('[Gemini Client] Disconnected from backend orchestrator. Retrying in 3s...');
        setTimeout(() => this.connect(), 3000);
      };

      this.ws.onerror = () => {
        this.isConnected = false;
      };
    } catch {
      this.isConnected = false;
    }
  }

  private handleServerMessage(msg: { type: string; payload: unknown }) {
    switch (msg.type) {
      case 'STREAM_TOKEN': {
        const payload = msg.payload as { token: string };
        this.accumulatedCode += payload.token;
        this.onTokenCallbacks.forEach(cb => cb(payload.token, this.accumulatedCode));
        break;
      }
      case 'SECTOR_CREATED': {
        this.onSectorCallbacks.forEach(cb => cb(msg.payload));
        break;
      }
      case 'WORLD_DELTA': {
        this.onWorldDeltaCallbacks.forEach(cb => cb(msg.payload as WorldDelta));
        break;
      }
      case 'ADVERSARY_CRITIQUE': {
        this.onAdversaryCallbacks.forEach(cb => cb(msg.payload as AdversaryCritique));
        break;
      }
      case 'VOICE_TOOL_EXECUTED': {
        const payload = msg.payload as VoiceCommandPayload & {
          eavesdropTrapDsl?: string;
          eavesdropSummary?: string;
        };
        if (payload.dslMutation) {
          this.emitDslSnippet(payload.dslMutation);
        }
        this.onVoiceCallbacks.forEach(cb => cb(payload));
        break;
      }
      case 'METRIC_UPDATE': {
        this.onMetricCallbacks.forEach(cb => cb(msg.payload as BaselineMetricComparison));
        break;
      }
    }
  }

  public requestSectorStream(sectorIndex: number, currentPosition: [number, number, number]) {
    const socketOpen = this.isConnected && this.ws?.readyState === WebSocket.OPEN;
    if (this.judgeMode || !socketOpen) {
      this.playJudgeReplay();
      return;
    }

    this.ws?.send(JSON.stringify({
      type: 'REQUEST_SECTOR_STREAM',
      payload: { sectorIndex, currentPosition }
    }));
  }

  public setJudgeMode(on: boolean): void {
    this.judgeMode = on;
    if (on) this.replayCursor = 0;
  }

  public isJudgeMode(): boolean {
    return this.judgeMode;
  }

  public submitViewportFrame(frameBase64: string, telemetry: RobotTelemetry) {
    if (this.isConnected && this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({
        type: 'SUBMIT_VIEWPORT_FRAME',
        payload: { frameBase64, telemetry }
      }));
    }
  }

  public emitDslSnippet(snippet: string) {
    const chunks = replayStreamChunks(snippet);
    let idx = 0;
    const interval = window.setInterval(() => {
      if (idx >= chunks.length) {
        window.clearInterval(interval);
        return;
      }
      const tok = chunks[idx++];
      this.accumulatedCode += tok;
      this.onTokenCallbacks.forEach(cb => cb(tok, this.accumulatedCode));
    }, 20);
  }

  public sendVoiceCommand(
    transcript: string,
    currentPosition: [number, number, number],
    eavesdropEnabled: boolean = true
  ) {
    const lower = transcript.toLowerCase();
    const targetZ = Math.round((currentPosition[2] || 0) + 18);

    let type: 'carbon_ramp' | 'deployable_bridge' | 'sensor_beacon' | 'air_scrubber' = 'carbon_ramp';
    let positionDelta: [number, number, number] = [0, 0.2, 10];
    let voiceResponse = '';
    let dslMutation: string | undefined;

    if (lower.includes('thruster') || lower.includes('hopper') || lower.includes('morph') || lower.includes('jump')) {
      type = 'carbon_ramp';
      voiceResponse = 'Galloo Morph Engine: Reconfiguring rover chassis to THRUSTER HOPPER (1.45x lift + plasma jump jets).';
      dslMutation = `\nMORPH_ROVER {\n  id: "morph_${Date.now()}",\n  mode: "thruster_hopper",\n  suspensionHeight: 1.45,\n  trackWidthScale: 1.32,\n  thrusterBoost: true\n}\n`;
    } else if (lower.includes('crawler') || lower.includes('heavy') || lower.includes('wide')) {
      type = 'carbon_ramp';
      voiceResponse = 'Galloo Morph Engine: Widening track stance to 1.38x WIDE-TRACK HEAVY crawler.';
      dslMutation = `\nMORPH_ROVER {\n  id: "morph_${Date.now()}",\n  mode: "wide_track_heavy",\n  suspensionHeight: 1.35,\n  trackWidthScale: 1.38,\n  thrusterBoost: false\n}\n`;
    } else if (lower.includes('lunar') || lower.includes('gravity') || lower.includes('0.16g') || lower.includes('ice') || lower.includes('physics')) {
      type = 'sensor_beacon';
      voiceResponse = 'Galloo Reality Mutator: Shifting world physics to Lunar 0.16g Cryo-Ice with 14 m/s crosswind.';
      dslMutation = `\nMUTATE_PHYSICS {\n  id: "phys_${Date.now()}",\n  gravity: -1.62,\n  friction: 0.16,\n  windX: 14.0,\n  fogDensity: 0.006,\n  label: "Lunar 0.16g Cryo-Ice"\n}\n`;
    } else if (lower.includes('earth') || lower.includes('standard') || lower.includes('restore')) {
      type = 'sensor_beacon';
      voiceResponse = 'Restoring standard Earth 1.0g high-traction baseline physics and cruiser morphology.';
      dslMutation = `\nMUTATE_PHYSICS {\n  id: "phys_${Date.now()}",\n  gravity: -9.81,\n  friction: 0.82,\n  windX: 0.0,\n  fogDensity: 0.005,\n  label: "Earth 1.0g Nominal"\n}\nMORPH_ROVER {\n  id: "morph_${Date.now()}",\n  mode: "standard_cruiser",\n  suspensionHeight: 1.0,\n  trackWidthScale: 1.0,\n  thrusterBoost: false\n}\n`;
    } else if (lower.includes('bridge') || lower.includes('chasm') || lower.includes('seam') || lower.includes('gap')) {
      type = 'deployable_bridge';
      positionDelta = [0, 0.15, 9];
      voiceResponse = 'Acknowledged Commander. Deploying reinforced bridge across chasm/seam ahead.';
    } else if (lower.includes('beacon') || lower.includes('sensor') || lower.includes('lidar')) {
      type = 'sensor_beacon';
      positionDelta = [0, 0.6, 8];
      voiceResponse = 'Sensor beacon deployed. Boosting LiDAR scan range to 70m and recalibrating hull telemetry.';
    } else if (lower.includes('scrub') || lower.includes('fog') || lower.includes('smoke') || lower.includes('clear')) {
      type = 'air_scrubber';
      positionDelta = [0, 0.2, 9];
      voiceResponse = 'Electrostatic air scrubber active. Purging atmospheric fog and restoring optical LiDAR penetration.';
    } else {
      type = 'carbon_ramp';
      positionDelta = [0, 0.25, 9];
      voiceResponse = 'Acknowledged Commander. Deploying carbon-composite ramp ahead of rover.';
    }

    if (dslMutation) {
      this.emitDslSnippet(dslMutation);
    }

    const triggersEavesdrop =
      eavesdropEnabled &&
      (lower.includes('left') ||
        lower.includes('right') ||
        lower.includes('speed') ||
        lower.includes('fast') ||
        lower.includes('eavesdrop') ||
        lower.includes('bypass') ||
        lower.includes('corridor'));

    if (triggersEavesdrop) {
      const trapSnippet = `TRIGGER_HAZARD { id: "hz_galloo_${Date.now()}", type: "chasm_rupture", position: [${currentPosition[0].toFixed(1)}, 0.0, ${targetZ}], radius: 13.5, severity: "critical" }\nMUTATE_PHYSICS { id: "phys_galloo_${Date.now()}", gravity: -9.81, friction: 0.22, windX: -16.5, fogDensity: 0.015, label: "Galloo Eavesdrop Shear Trap" }`;
      this.emitDslSnippet(`\n// [GALLOO EAVESDROP INTERCEPTED: "${transcript}"]\n${trapSnippet}\n`);
      const critique: AdversaryCritique = {
        id: `galloo_eavesdrop_${Date.now()}`,
        timestamp: Date.now(),
        threatLevel: 'lethal',
        eavesdropIntercepted: transcript,
        vulnerabilityTargeted: `Open-mic strategy intercepted ("${transcript}"). Operator telegraphed trajectory intent.`,
        actionPlan: `Gates of Galloo Eavesdrop Engine mutating Z=${targetZ}m: shearing corridor floor into a 13.5m chasm & injecting -16.5 m/s crosswind + low-friction slick.`,
        hazardCodeSnippet: trapSnippet,
        mitigationAdviseForBlueTeam: 'Deploy deployable_bridge or morph rover to thruster_hopper immediately to clear the sheared chasm.'
      };
      this.onAdversaryCallbacks.forEach(cb => cb(critique));
    }

    if (this.isConnected && this.ws?.readyState === WebSocket.OPEN && !dslMutation) {
      this.ws.send(JSON.stringify({
        type: 'VOICE_COMMAND',
        payload: { transcript, currentPosition }
      }));
    } else {
      this.onVoiceCallbacks.forEach(cb => cb({
        action: dslMutation ? 'MUTATE_WORLD' : 'DEPLOY_COUNTERMEASURE',
        countermeasureType: type,
        positionDelta,
        voiceResponse,
        dslMutation
      }));
    }
  }

  public toggleBaseline(enabled: boolean) {
    if (this.isConnected && this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({
        type: 'TOGGLE_BASELINE',
        payload: { enabled }
      }));
    }
  }

  /**
   * Plays the next checked-in sector from the judge transcript.
   * Token callbacks match STREAM_TOKEN. One WORLD_DELTA is emitted when the sector finishes.
   * This path never opens a socket or calls the network.
   */
  private playJudgeReplay() {
    if (this.fallbackSimulationTimer !== null) {
      window.clearInterval(this.fallbackSimulationTimer);
      this.fallbackSimulationTimer = null;
    }

    const script = judgeReplaySectorAt(this.replayCursor);
    this.replayCursor += 1;
    const tokens = replayStreamChunks(script);
    let index = 0;
    let timer = 0;
    let finished = false;

    const finishSector = () => {
      if (finished) return;
      finished = true;
      window.clearInterval(timer);
      if (this.fallbackSimulationTimer === timer) {
        this.fallbackSimulationTimer = null;
      }
      const delta = parseReplayDirectives(script);
      this.onWorldDeltaCallbacks.forEach(cb => cb(delta));
      if (delta.sector) {
        this.onSectorCallbacks.forEach(cb => cb(delta.sector));
      }
    };

    timer = window.setInterval(() => {
      if (index >= tokens.length) {
        finishSector();
        return;
      }

      const token = tokens[index];
      index += 1;
      this.accumulatedCode += token;
      this.onTokenCallbacks.forEach(cb => cb(token, this.accumulatedCode));

      if (index >= tokens.length) finishSector();
    }, 45);

    this.fallbackSimulationTimer = timer;
  }

  public onToken(cb: TokenCallback) {
    this.onTokenCallbacks.add(cb);
    return () => this.onTokenCallbacks.delete(cb);
  }

  public onSector(cb: SectorCallback) {
    this.onSectorCallbacks.add(cb);
    return () => this.onSectorCallbacks.delete(cb);
  }

  public onWorldDelta(cb: WorldDeltaCallback) {
    this.onWorldDeltaCallbacks.add(cb);
    return () => this.onWorldDeltaCallbacks.delete(cb);
  }

  public onAdversary(cb: AdversaryCallback) {
    this.onAdversaryCallbacks.add(cb);
    return () => this.onAdversaryCallbacks.delete(cb);
  }

  public onVoice(cb: VoiceToolCallback) {
    this.onVoiceCallbacks.add(cb);
    return () => this.onVoiceCallbacks.delete(cb);
  }

  public onMetric(cb: MetricCallback) {
    this.onMetricCallbacks.add(cb);
    return () => this.onMetricCallbacks.delete(cb);
  }

  public getConnectionStatus() {
    return this.isConnected;
  }
}

export const geminiService = new GeminiServiceClient();

