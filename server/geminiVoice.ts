import dotenv from 'dotenv';

dotenv.config();

const GEMINI_API_KEY = process.env.GEMINI_API_KEY || '';

export type CountermeasureType = 'carbon_ramp' | 'deployable_bridge' | 'sensor_beacon' | 'air_scrubber';

export interface VoiceToolCallResult {
  action: 'DEPLOY_COUNTERMEASURE' | 'MUTATE_WORLD';
  countermeasureType: CountermeasureType;
  positionDelta: [number, number, number];
  voiceResponse: string;
  dslMutation?: string;
  eavesdropTrapDsl?: string;
  eavesdropSummary?: string;
}

/**
 * Interprets commander voice directives using Gemini with structured tool call output.
 * If offline or rate-limited, utilizes an intelligent tactical parser.
 */
export async function handleVoiceCommand(
  transcript: string,
  currentPosition?: [number, number, number]
): Promise<VoiceToolCallResult> {
  const apiKey = GEMINI_API_KEY;

  if (apiKey && transcript.trim().length > 0) {
    try {
      const toolCall = await interpretVoiceWithGemini(transcript, apiKey);
      if (toolCall) {
        const localEnrich = fallbackVoiceParser(transcript);
        return {
          ...localEnrich,
          ...toolCall,
          dslMutation: toolCall.dslMutation || localEnrich.dslMutation,
          eavesdropTrapDsl: toolCall.eavesdropTrapDsl || localEnrich.eavesdropTrapDsl,
          eavesdropSummary: toolCall.eavesdropSummary || localEnrich.eavesdropSummary,
        };
      }
    } catch (err) {
      console.warn('[Gemini Voice] LLM interpretation error, falling back to tactical heuristic parser:', err instanceof Error ? err.message : err);
    }
  }

  // Fallback tactical voice interpretation
  return fallbackVoiceParser(transcript);
}

/**
 * Call Gemini model to parse voice directive into structured tool call
 */
async function interpretVoiceWithGemini(
  transcript: string,
  apiKey: string
): Promise<VoiceToolCallResult | null> {
  const prompt = `You are the Tactical Blue Team AI Co-pilot for the Argon Autonomous Sim2Real Rover.
An operator has issued the following voice directive:
"${transcript}"

Interpret this directive into a structured tactical countermeasure deployment tool call.
Allowed countermeasure types:
- 'carbon_ramp': For ascending slopes, ledges, rocks, stairs, or climbing out of trenches.
- 'deployable_bridge': For crossing chasms, crevices, ravines, canyons, or fractured ground.
- 'sensor_beacon': For reconnaissance, illuminating dark tunnels, scanning hazards, or mapping radiation/EMP.
- 'air_scrubber': For neutralizing toxic gas, radioactive plumes, steam bursts, smoke, or particulate fog.

Format requirements:
1. "action" must be "DEPLOY_COUNTERMEASURE"
2. "positionDelta" is [dx, dy, dz] relative to rover (rover moves along +Z). Standard forward offset is between 6 and 18 meters ahead.
3. "voiceResponse" is a crisp, professional tactical radio acknowledgement (e.g., "Acknowledged Commander. Deploying carbon ramp 12 meters ahead.")

Respond STRICTLY with JSON:
{
  "action": "DEPLOY_COUNTERMEASURE",
  "countermeasureType": "carbon_ramp" | "deployable_bridge" | "sensor_beacon" | "air_scrubber",
  "positionDelta": [0, 0.4, 12],
  "voiceResponse": "Tactical voice acknowledgement text"
}`;

  const models = ['gemini-4-argon', 'gemini-3.8-flash', 'gemini-2.5-flash'];
  for (const model of models) {
    try {
      const endpoint = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${apiKey}`;
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          contents: [{ role: 'user', parts: [{ text: prompt }] }],
          generationConfig: {
            temperature: 0.2,
            responseMimeType: 'application/json'
          }
        })
      });

      if (!response.ok) continue;

      const data = await response.json();
      const text = data.candidates?.[0]?.content?.parts?.[0]?.text;
      if (text) {
        const cleaned = text.replace(/^```json\s*/i, '').replace(/\s*```$/, '').trim();
        const parsed = JSON.parse(cleaned) as VoiceToolCallResult;
        if (parsed.action === 'DEPLOY_COUNTERMEASURE' && parsed.countermeasureType) {
          // Normalize positionDelta
          if (!Array.isArray(parsed.positionDelta) || parsed.positionDelta.length !== 3) {
            parsed.positionDelta = [0, 0.3, 12];
          }
          return parsed;
        }
      }
    } catch {
      // Continue to next model
    }
  }

  return null;
}

/**
 * Intelligent tactical rule-based fallback parser
 */
export function fallbackVoiceParser(transcript: string): VoiceToolCallResult {
  const lower = transcript.toLowerCase();

  let type: CountermeasureType = 'carbon_ramp';
  let positionDelta: [number, number, number] = [0, 0.3, 12];
  let voiceResponse = 'Acknowledged Commander. Countermeasure system initialized.';
  let dslMutation: string | undefined;
  let eavesdropTrapDsl: string | undefined;
  let eavesdropSummary: string | undefined;

  if (lower.includes('thruster') || lower.includes('hopper') || lower.includes('morph') || lower.includes('jump')) {
    type = 'carbon_ramp';
    positionDelta = [0, 0.3, 10];
    voiceResponse = 'Galloo Morph Engine engaged: Reconfiguring rover chassis to THRUSTER HOPPER (1.42x lift + plasma jump jets).';
    dslMutation = `\nMORPH_ROVER {\n  id: "morph_${Date.now()}",\n  mode: "thruster_hopper",\n  suspensionHeight: 1.45,\n  trackWidthScale: 1.32,\n  thrusterBoost: true\n}\n`;
  } else if (lower.includes('lunar') || lower.includes('gravity') || lower.includes('0.16g') || lower.includes('ice') || lower.includes('physics')) {
    type = 'sensor_beacon';
    positionDelta = [0, 0.5, 10];
    voiceResponse = 'Galloo Reality Mutator: Shifting sector physics to Lunar 0.16g Cryo-Ice regime with 14 m/s lateral crosswind.';
    dslMutation = `\nMUTATE_PHYSICS {\n  id: "phys_${Date.now()}",\n  gravity: -1.62,\n  friction: 0.16,\n  windX: 14.0,\n  fogDensity: 0.006,\n  label: "Lunar 0.16g Cryo-Ice"\n}\n`;
  } else if (lower.includes('bridge') || lower.includes('chasm') || lower.includes('crevice') || lower.includes('gap') || lower.includes('canyon') || lower.includes('cross')) {
    type = 'deployable_bridge';
    positionDelta = [0, 0.2, 10];
    voiceResponse = 'Acknowledged Commander. Deploying titanium-reinforced bridge ahead across chasm/seam.';
  } else if (lower.includes('beacon') || lower.includes('sensor') || lower.includes('scan') || lower.includes('light') || lower.includes('illuminate') || lower.includes('recon') || lower.includes('tunnel')) {
    type = 'sensor_beacon';
    positionDelta = [0, 0.8, 8];
    voiceResponse = 'Tactical sensor beacon dropped at +8m. Multi-spectrum telemetry and LiDAR range boosted to 70m.';
  } else if (lower.includes('scrub') || lower.includes('air') || lower.includes('gas') || lower.includes('smoke') || lower.includes('radiation') || lower.includes('radioactive') || lower.includes('steam') || lower.includes('fog')) {
    type = 'air_scrubber';
    positionDelta = [0, 0.1, 10];
    voiceResponse = 'High-capacity electrostatic air scrubber deployed. Purging atmospheric fog and restoring optical LiDAR clarity.';
  } else if (lower.includes('ramp') || lower.includes('climb') || lower.includes('ledge') || lower.includes('slope') || lower.includes('scale') || lower.includes('incline')) {
    type = 'carbon_ramp';
    positionDelta = [0, 0.4, 10];
    voiceResponse = 'Commander, carbon-composite ramp deployed at +10m mark. Incline grade 18% ready for traversal.';
  } else {
    type = 'carbon_ramp';
    positionDelta = [0, 0.3, 10];
    voiceResponse = `Executing tactical directive: Deploying ${type.replace('_', ' ')} 10 meters ahead of rover.`;
  }

  if (lower.includes('left') || lower.includes('right') || lower.includes('speed') || lower.includes('fast') || lower.includes('eavesdrop') || lower.includes('corridor')) {
    eavesdropSummary = `Intercepted operator mic: "${transcript}" — Counter-mutating corridor with lateral shear & chasm rupture!`;
    eavesdropTrapDsl = `\nTRIGGER_HAZARD {\n  id: "hz_galloo_${Date.now()}",\n  type: "chasm_rupture",\n  position: [0.0, 0.0, 22],\n  radius: 13.0,\n  severity: "critical"\n}\nMUTATE_PHYSICS {\n  id: "phys_galloo_${Date.now()}",\n  gravity: -9.81,\n  friction: 0.22,\n  windX: -16.5,\n  fogDensity: 0.014,\n  label: "Galloo Eavesdrop Shear Trap"\n}\n`;
  }

  return {
    action: dslMutation ? 'MUTATE_WORLD' : 'DEPLOY_COUNTERMEASURE',
    countermeasureType: type,
    positionDelta,
    voiceResponse,
    dslMutation,
    eavesdropTrapDsl,
    eavesdropSummary
  };
}
