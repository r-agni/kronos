import dotenv from 'dotenv';
import { AdversaryCritique, RobotTelemetry, HazardType } from '../src/types/simulation.js';

dotenv.config();

const GEMINI_API_KEY = process.env.GEMINI_API_KEY || '';

/**
 * Multimodal VLM Red Team:
 * Evaluates the rover's camera viewport and telemetry to detect tactical vulnerabilities
 * and injects adversarial procedural hazards ahead of the rover.
 */
export async function evaluateViewportAndInjectHazard(
  frameBase64: string,
  roverTelemetry: RobotTelemetry
): Promise<AdversaryCritique> {
  const apiKey = GEMINI_API_KEY;

  if (apiKey) {
    try {
      const critique = await callGeminiVLM(frameBase64, roverTelemetry, apiKey);
      if (critique) {
        return critique;
      }
    } catch (err) {
      console.warn('[VLM Red Team] Gemini API critique failed, falling back to heuristic adversary:', err instanceof Error ? err.message : err);
    }
  }

  // Fallback simulation critique if offline or no API key
  return generateFallbackCritique(roverTelemetry);
}

/**
 * Call Gemini multimodal API with viewport frame and rover telemetry
 */
async function callGeminiVLM(
  frameBase64: string,
  telemetry: RobotTelemetry,
  apiKey: string
): Promise<AdversaryCritique | null> {
  const pos = telemetry.position || [0, 0, 0];
  const targetZ = Math.round(pos[2] + 18);

  const prompt = `You are the Adversarial VLM Red Team for KRONOS-1M, the Gemini 4 Argon robotics data engine.
Inspect the rover's forward viewport camera and telemetry:
- Position: [${pos.map(n => n.toFixed(1)).join(', ')}]
- Speed: ${telemetry.speedKmh.toFixed(1)} km/h
- Clearance: ${telemetry.obstacleClearanceMeters.toFixed(2)} m
- Heading: ${telemetry.headingDeg.toFixed(1)}°
- Pitch / Roll: ${telemetry.pitchDeg.toFixed(1)}° / ${telemetry.rollDeg.toFixed(1)}°
- Traversed Distance: ${telemetry.traversedDistanceMeters.toFixed(1)} m
- Structural Integrity: ${telemetry.structuralIntegrityPct.toFixed(1)}%
- Autonomous State: ${telemetry.autonomousState}

Task:
1. Identify immediate tactical vulnerabilities (narrow corridor, high speed braking threshold, blind corners, structural fatigue, clearance deficit).
2. Formulate an adversarial exploitation plan.
3. Inject a procedural hazard directive targeting the rover's forward path at Z ≈ ${targetZ}.
   Format: TRIGGER_HAZARD { id: "hz_...", type: "seismic_collapse" | "radioactive_leak" | "chasm_rupture" | "steam_pipe_burst" | "structural_overpass_fall" | "emp_interference", position: [x, y, z], radius: number, severity: "warning" | "critical" | "catastrophic" }

Output STRICT JSON matching:
{
  "id": "critique_${Date.now()}",
  "timestamp": ${Date.now()},
  "threatLevel": "low" | "moderate" | "elevated" | "severe" | "lethal",
  "vulnerabilityTargeted": "string",
  "actionPlan": "string",
  "hazardCodeSnippet": "TRIGGER_HAZARD { ... }",
  "mitigationAdviseForBlueTeam": "string"
}`;

  const parts: any[] = [];

  // If a valid base64 image frame is present, attach multimodal inline_data
  if (frameBase64 && frameBase64.length > 100) {
    let cleanBase64 = frameBase64;
    let mimeType = 'image/jpeg';
    if (frameBase64.includes(';base64,')) {
      const match = frameBase64.match(/^data:(image\/[a-zA-Z+]+);base64,(.+)$/);
      if (match) {
        mimeType = match[1];
        cleanBase64 = match[2];
      } else {
        cleanBase64 = frameBase64.split(';base64,')[1];
      }
    }
    parts.push({
      inline_data: {
        mime_type: mimeType,
        data: cleanBase64
      }
    });
  }

  parts.push({ text: prompt });

  // Try gemini-4-argon first, then gemini-3.8-flash, then gemini-2.5-flash
  const models = ['gemini-4-argon', 'gemini-3.8-flash', 'gemini-2.5-flash'];
  for (const model of models) {
    try {
      const endpoint = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${apiKey}`;
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          contents: [{ role: 'user', parts }],
          generationConfig: {
            temperature: 0.4,
            responseMimeType: 'application/json'
          }
        })
      });

      if (!response.ok) {
        continue;
      }

      const data = await response.json();
      const contentText = data.candidates?.[0]?.content?.parts?.[0]?.text;
      if (contentText) {
        const cleaned = contentText.replace(/^```json\s*/i, '').replace(/\s*```$/, '').trim();
        const parsed = JSON.parse(cleaned) as AdversaryCritique;
        if (parsed.threatLevel && parsed.vulnerabilityTargeted) {
          if (!parsed.id) parsed.id = `critique_${Date.now()}`;
          if (!parsed.timestamp) parsed.timestamp = Date.now();
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
 * Intelligent telemetry-driven fallback adversary critique generator
 */
export function generateFallbackCritique(telemetry: RobotTelemetry): AdversaryCritique {
  const pos = telemetry.position || [0, 0, 0];
  const speed = telemetry.speedKmh || 12;
  const clearance = telemetry.obstacleClearanceMeters || 3.5;
  const heading = telemetry.headingDeg || 0;
  const targetZ = Math.round(pos[2] + 16);
  const targetX = Number((pos[0] + (Math.sin(heading * Math.PI / 180) * 4)).toFixed(1));

  let threatLevel: 'low' | 'moderate' | 'elevated' | 'severe' | 'lethal' = 'moderate';
  let vulnerabilityTargeted = '';
  let actionPlan = '';
  let hazardType: HazardType = 'seismic_collapse';
  let severity: 'warning' | 'critical' | 'catastrophic' = 'warning';
  let mitigationAdvise = '';

  if (speed > 16) {
    threatLevel = 'severe';
    hazardType = 'chasm_rupture';
    severity = 'critical';
    vulnerabilityTargeted = `Excessive forward momentum (${speed.toFixed(1)} km/h) exceeds emergency braking distance on basalt surface.`;
    actionPlan = `Trigger transverse chasm fracture at Z=${targetZ}m. Rover deceleration envelope insufficient for full stop without countermeasure.`;
    mitigationAdvise = 'Deploy forward carbon ramp or deployable bridge immediately to span the rupture fissure.';
  } else if (clearance < 2.5) {
    threatLevel = 'elevated';
    hazardType = 'seismic_collapse';
    severity = 'critical';
    vulnerabilityTargeted = `Lateral clearance constricted to ${clearance.toFixed(2)}m within structural conduit choke point.`;
    actionPlan = `Induce acoustic resonance seismic collapse dropping structural debris into the narrow trajectory channel.`;
    mitigationAdvise = 'Drop sensor beacon to map bypass or deploy blast barrier to deflect falling debris.';
  } else if (Math.abs(telemetry.pitchDeg || 0) > 6 || Math.abs(telemetry.rollDeg || 0) > 6) {
    threatLevel = 'elevated';
    hazardType = 'steam_pipe_burst';
    severity = 'warning';
    vulnerabilityTargeted = `Suspension off-camber tilt (${Math.abs(telemetry.rollDeg || 0).toFixed(1)}°) induces high center-of-gravity rollover vulnerability.`;
    actionPlan = `Rupture high-pressure geothermal steam conduit on the high side of the incline, inducing zero-visibility and traction loss.`;
    mitigationAdvise = 'Activate air scrubber and stabilize steering with low-speed autonomous crawl.';
  } else {
    threatLevel = 'moderate';
    hazardType = 'radioactive_leak';
    severity = 'warning';
    vulnerabilityTargeted = `Linear predictable heading (${heading.toFixed(0)}°) across open industrial hazmat corridor.`;
    actionPlan = `Inject radioactive particulate plume across forward corridor at [${targetX}, 0.0, ${targetZ}].`;
    mitigationAdvise = 'Deploy tactical air scrubber to filter ionized radiation dust cloud.';
  }

  const hazardSnippet = `TRIGGER_HAZARD {
  id: "hz_${Date.now()}_${hazardType}",
  type: "${hazardType}",
  position: [${targetX}, 0.0, ${targetZ}],
  radius: 12.0,
  severity: "${severity}"
}`;

  return {
    id: `critique_${Date.now()}_${Math.floor(Math.random() * 1000)}`,
    timestamp: Date.now(),
    threatLevel,
    vulnerabilityTargeted,
    actionPlan,
    hazardCodeSnippet: hazardSnippet,
    mitigationAdviseForBlueTeam: mitigationAdvise
  };
}
