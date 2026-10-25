import {
  CounterfactualBranch,
  DenseProcessRewardTick,
  MultiverseForkEvent,
  RobotTelemetry,
  RubricScoreCard,
  TemporalSegment,
  Vec3,
} from '../types/simulation';

/**
 * KRONOS-1M Autonomous Robotics Data & Counterfactual Multiverse Engine
 *
 * Unifies 6 capabilities in a single continuous 1M-context stream:
 * 01. Temporal Sub-Task Segmentation (continuous 1M vs fractured 64K window stitches)
 * 02. Contextual VLA Action Captioning (past + future trajectory conditioned)
 * 03. Episode Outcome Scoring (1.0 - 5.0 fleet deployment monitor)
 * 04. Dense 0.5-Second Process Rewards (NORMAL | SUBOPTIMAL | FAILURE)
 * 05. Live Natural-Language Test-Time Rubric Evaluation
 * 06. 4-Branch Counterfactual Multiverse DPO Forking (Alpha, Beta, Gamma, Omega)
 */
export class KronosEngine {
  private startTimeMs: number = Date.now();
  private lastTickSec: number = -0.5;
  private segments: TemporalSegment[] = [];
  private rewardTicks: DenseProcessRewardTick[] = [];
  private forkEvents: MultiverseForkEvent[] = [];
  private rubricText: string =
    'Maintain wheel slip < 30%, obstacle clearance > 2.2m, and zero unbridged seam halts.';
  private maxWheelSlipLimit: number = 0.30;
  private minClearanceLimit: number = 2.2;
  private lastAutoForkSec: number = -10;

  public reset(): void {
    this.startTimeMs = Date.now();
    this.lastTickSec = -0.5;
    this.segments = [];
    this.rewardTicks = [];
    this.forkEvents = [];
    this.lastAutoForkSec = -10;
  }

  public setRubric(rubricText: string): void {
    if (!rubricText.trim()) return;
    this.rubricText = rubricText.trim();
    const slipMatch = rubricText.match(/(\d+(?:\.\d+)?)\s*%/);
    if (slipMatch) {
      this.maxWheelSlipLimit = Math.max(0.05, Math.min(0.9, parseFloat(slipMatch[1]) / 100));
    }
    const clearMatch = rubricText.match(/(\d+(?:\.\d+)?)\s*m\b/i);
    if (clearMatch) {
      this.minClearanceLimit = Math.max(0.5, Math.min(15, parseFloat(clearMatch[1])));
    }
  }

  public getRubric(): string {
    return this.rubricText;
  }

  public stepTelemetry(
    telemetry: RobotTelemetry,
    isFractured64kMode: boolean
  ): {
    newTick: DenseProcessRewardTick | null;
    shouldTriggerAutoFork: boolean;
  } {
    const elapsedSec = +((Date.now() - this.startTimeMs) / 1000).toFixed(2);
    if (elapsedSec - this.lastTickSec < 0.5) {
      return { newTick: null, shouldTriggerAutoFork: false };
    }
    this.lastTickSec = elapsedSec;

    const slip = telemetry.wheelSlipRatio ?? 0.08;
    const clearance = telemetry.obstacleClearanceMeters;
    const state = telemetry.autonomousState;

    let rating: DenseProcessRewardTick['rating'] = 'normal';
    let score = 0.94;
    let reason = `Nominal trajectory (${telemetry.speedKmh.toFixed(1)} km/h, slip ${(slip * 100).toFixed(0)}%)`;

    if (state === 'route_blocked' || telemetry.speedKmh < 1.2) {
      rating = 'failure';
      score = 0.12;
      reason = isFractured64kMode
        ? '64K context seam cliff halt (unbridged boundary tear ΔY = -2.4m)'
        : 'Corridor blocked by unbridged chasm rupture ahead';
    } else if (slip > this.maxWheelSlipLimit || clearance < this.minClearanceLimit || state === 'avoiding_hazard') {
      rating = 'suboptimal';
      score = 0.58;
      if (slip > this.maxWheelSlipLimit) {
        reason = `Rubric violation: wheel slip ${(slip * 100).toFixed(0)}% > ${(this.maxWheelSlipLimit * 100).toFixed(0)}% threshold`;
      } else if (clearance < this.minClearanceLimit) {
        reason = `Rubric violation: LiDAR clearance ${clearance.toFixed(2)}m < ${this.minClearanceLimit.toFixed(1)}m threshold`;
      } else {
        reason = `Reactive LiDAR avoidance maneuver (clearance ${clearance.toFixed(1)}m)`;
      }
    } else if (state === 'scaling_ramp') {
      rating = 'normal';
      score = 0.98;
      reason = `Optimal counterfactual traversal (${telemetry.activeMorphology || 'bridged ascent'})`;
    }

    const tick: DenseProcessRewardTick = {
      id: `tick_${this.rewardTicks.length + 1}`,
      timestampSec: elapsedSec,
      rating,
      score,
      reason,
      position: [...telemetry.position] as Vec3,
      wheelSlipRatio: slip,
      clearanceMeters: clearance,
    };

    this.rewardTicks.push(tick);
    if (this.rewardTicks.length > 160) {
      this.rewardTicks.shift();
    }

    this.updateTemporalSegmentation(elapsedSec, telemetry, tick, isFractured64kMode);

    const shouldTriggerAutoFork =
      rating === 'failure' && elapsedSec - this.lastAutoForkSec > 7.5;

    return { newTick: tick, shouldTriggerAutoFork };
  }

  private updateTemporalSegmentation(
    elapsedSec: number,
    telemetry: RobotTelemetry,
    tick: DenseProcessRewardTick,
    isFractured64kMode: boolean
  ): void {
    const current = this.segments[this.segments.length - 1];
    const stitchStatus: TemporalSegment['stitchStatus'] = isFractured64kMode
      ? 'window_fractured'
      : 'continuous_1m';

    const stateChanged =
      !current ||
      current.dominantState !== telemetry.autonomousState ||
      current.stitchStatus !== stitchStatus ||
      elapsedSec - current.startSec >= 8.0;

    if (stateChanged) {
      if (current) {
        current.endSec = elapsedSec;
      }
      const idx = this.segments.length + 1;
      const { label, caption } = this.buildSegmentLabelAndCaption(
        telemetry,
        stitchStatus,
        idx
      );
      this.segments.push({
        id: `seg_${String(idx).padStart(2, '0')}`,
        index: idx,
        label,
        startSec: elapsedSec,
        endSec: +(elapsedSec + 0.5).toFixed(2),
        instructionCaption: caption,
        stitchStatus,
        dominantState: telemetry.autonomousState,
        meanReward: tick.score,
      });
      if (this.segments.length > 24) {
        this.segments.shift();
      }
    } else if (current) {
      current.endSec = elapsedSec;
      current.meanReward = +((current.meanReward * 0.8 + tick.score * 0.2)).toFixed(2);
    }
  }

  private buildSegmentLabelAndCaption(
    telemetry: RobotTelemetry,
    stitchStatus: TemporalSegment['stitchStatus'],
    idx: number
  ): { label: string; caption: string } {
    const z = telemetry.position[2].toFixed(1);
    if (stitchStatus === 'window_fractured') {
      return {
        label: `SEG-${String(idx).padStart(2, '0')}: 64K WINDOW STITCH`,
        caption: `[STITCH ARTIFACT @ Z=${z}m] Lost prior boundary elevation context; temporal sub-task horizon truncated.`,
      };
    }
    switch (telemetry.autonomousState) {
      case 'route_blocked':
        return {
          label: `SEG-${String(idx).padStart(2, '0')}: SEAM / CHASM HALT`,
          caption: `Halt forward drive at Z=${z}m cliff edge and trigger 4-branch counterfactual DPO synthesis.`,
        };
      case 'avoiding_hazard':
        return {
          label: `SEG-${String(idx).padStart(2, '0')}: LIDAR SLALOM`,
          caption: `Execute 32-beam LiDAR fan evasion around hazard at Z=${z}m while maintaining lateral stability.`,
        };
      case 'scaling_ramp':
        return {
          label: `SEG-${String(idx).padStart(2, '0')}: COUNTERFACTUAL ASCENT`,
          caption: `Traverse chasm/seam at Z=${z}m via synthesized ${telemetry.activeMorphology || 'deployable bridge'} morphology.`,
        };
      case 'voice_override':
        return {
          label: `SEG-${String(idx).padStart(2, '0')}: TELEOP OVERRIDE`,
          caption: `Follow human operator direct kinematic steering input along Z=${z}m corridor.`,
        };
      default:
        return {
          label: `SEG-${String(idx).padStart(2, '0')}: CONTINUOUS CRUISE`,
          caption: `Maintain ${telemetry.speedKmh.toFixed(1)} km/h center-corridor alignment under ${telemetry.physicsRegime || 'Earth 1.0g'} regime.`,
        };
    }
  }
  public triggerMultiverseFork(
    originPos: Vec3,
    originHeading: number,
    triggerReason: string,
    physicsEnv?: {
      sampleSupportHeight?: (x: number, z: number) => number;
      surfaceFriction?: number;
      lateralWindX?: number;
      isFractured64k?: boolean;
      hazardZones?: Array<{ pos: Vec3; radius: number; type: string }>;
    }
  ): MultiverseForkEvent {
    const elapsedSec = +((Date.now() - this.startTimeMs) / 1000).toFixed(2);
    this.lastAutoForkSec = elapsedSec;

    const [ox, oy, oz] = originPos;
    const steps = 16;
    const dt = 0.15;
    const friction = physicsEnv?.surfaceFriction ?? 0.72;
    const windX = physicsEnv?.lateralWindX ?? 0;
    const sampleH = physicsEnv?.sampleSupportHeight;
    const hazards = physicsEnv?.hazardZones ?? [];

    const simBranch = (
      branchId: CounterfactualBranch['branchId'],
      label: string,
      policyName: string,
      colorHex: number,
      cssColor: string,
      speed: number,
      steerFn: (u: number) => number,
      lift: number,
      jump: boolean,
      bridge: boolean,
      actionDsl: string
    ): CounterfactualBranch => {
      const pts: Vec3[] = [[+ox.toFixed(2), +oy.toFixed(2), +oz.toFixed(2)]];
      let [x, y, z, h] = [ox, oy, oz, originHeading];
      let [rewardSum, halted, slipViolations, hazardHits] = [0, false, 0, 0];

      for (let i = 1; i <= steps; i++) {
        const u = i / steps;
        if (!halted) {
          const steer = steerFn(u);
          h += steer * dt;
          const slip = Math.min(0.92, Math.max(0.04, (1 - friction) * 0.55 + Math.abs(steer) * 0.42 + Math.abs(windX) * 0.01));
          x = Math.max(-11.5, Math.min(11.5, x + Math.sin(h) * speed * dt + windX * 0.04 * (1.1 - friction) * dt));
          z += Math.cos(h) * speed * dt;
          const groundY = sampleH ? sampleH(x, z) : oy;
          const inChasm =
            !bridge &&
            !jump &&
            (oy - groundY > 1.05 ||
              (physicsEnv?.isFractured64k && i >= 7) ||
              hazards.some(hz => hz.type === 'chasm_rupture' && Math.hypot(hz.pos[0] - x, hz.pos[2] - z) < hz.radius * 0.65));
          const hitSolid = hazards.some(hz => hz.type !== 'chasm_rupture' && Math.hypot(hz.pos[0] - x, hz.pos[2] - z) < Math.max(2.5, hz.radius * 0.35));

          if (inChasm) {
            halted = true;
            y = Math.max(0.1, groundY - 1.1);
            rewardSum += 0.05;
          } else {
            y = Math.max(0.2, groundY + lift * 0.25 + (jump ? Math.sin(u * Math.PI) * 2.4 : 0));
            if (hitSolid) {
              hazardHits++;
              rewardSum += 0.35;
            } else if (slip > this.maxWheelSlipLimit) {
              slipViolations++;
              rewardSum += 0.52;
            } else {
              rewardSum += 0.96 * Math.min(1.0, speed / 8.5);
            }
          }
        } else {
          rewardSum += 0.05;
        }
        pts.push([+x.toFixed(2), +y.toFixed(2), +z.toFixed(2)]);
      }

      const processRewardMean = +(rewardSum / steps).toFixed(2);
      const dist = z - oz;
      const rawOut = halted ? 1.0 + Math.min(0.6, dist / 30) : 1.5 + 3.5 * processRewardMean - slipViolations * 0.08 - hazardHits * 0.25;
      const outcomeScore1to5 = +Math.max(1.0, Math.min(5.0, rawOut)).toFixed(1);
      const rationale = halted
        ? `Rollout halted at Z=${z.toFixed(1)}m due to unbridged elevation drop.`
        : slipViolations > 0
          ? `Traversed ${dist.toFixed(1)}m with ${slipViolations} wheel-slip violations.`
          : `Traversed ${dist.toFixed(1)}m (${processRewardMean} mean reward, 0 violations).`;

      return { branchId, label, policyName, colorHex, cssColor, dpoRole: 'rejected', outcomeScore1to5, processRewardMean, trajectory: pts, actionDsl, rationale };
    };

    const branches: CounterfactualBranch[] = [
      simBranch('alpha_fail', 'GHOST α // Baseline Policy (Unmodified)', 'vla_baseline_64k', 0xff2244, '#ff2244', 8.5, () => 0.02, 0.0, false, false, `// REJECTED: Unmodified forward velocity into seam/chasm @ Z=${(oz + 12).toFixed(1)}m`),
      simBranch('beta_swerve', 'GHOST β // Hard LiDAR Swerve Branch', 'reactive_lidar_swerve', 0xf59e0b, '#f59e0b', 7.8, u => (u < 0.5 ? -0.65 : 0.35), 0.1, false, true, `// REJECTED: High-slip lateral swerve (steer=-0.65 rad)`),
      simBranch('gamma_crawl', 'GHOST γ // Wide-Track Heavy Crawl', 'codesign_widetrack_crawl', 0xa855f7, '#a855f7', 5.2, u => (u < 0.5 ? 0.28 : -0.18), 0.35, false, true, `MORPH_ROVER { mode: "wide_track_heavy", suspensionHeight: 1.35, trackWidthScale: 1.38, thrusterBoost: false }`),
      simBranch('omega_optimal', 'GHOST Ω // Argon 1M Optimal Co-Design', 'argon_1m_counterfactual_optimal', 0x00f0ff, '#00f0ff', 9.4, () => -ox * 0.02, 0.45, true, true, `MORPH_ROVER { mode: "thruster_hopper", suspensionHeight: 1.45, trackWidthScale: 1.32, thrusterBoost: true }\nDEPLOY_COUNTERMEASURE { type: "deployable_bridge", position: [0, 0.3, ${(oz + 10).toFixed(1)}], dimensions: [6.0, 0.5, 18.0] }`),
    ];

    let bestIdx = 0;
    for (let i = 1; i < branches.length; i++) {
      if (branches[i].outcomeScore1to5 >= branches[bestIdx].outcomeScore1to5) bestIdx = i;
    }
    branches[bestIdx].dpoRole = 'chosen';

    const forkEvent: MultiverseForkEvent = {
      id: `fork_${Date.now()}`,
      timestamp: Date.now(),
      simTimeSec: elapsedSec,
      triggerReason,
      originPosition: [ox, oy, oz],
      originHeading,
      thinkingLevel: 'high',
      branches,
      chosenBranchId: branches[bestIdx].branchId,
    };

    this.forkEvents.unshift(forkEvent);
    if (this.forkEvents.length > 15) this.forkEvents.pop();
    return forkEvent;
  }
  public getScoreCard(): RubricScoreCard {
    let normal = 0;
    let suboptimal = 0;
    let failure = 0;
    const violations = new Set<string>();

    for (const tick of this.rewardTicks) {
      if (tick.rating === 'normal') normal++;
      else if (tick.rating === 'suboptimal') {
        suboptimal++;
        if (tick.reason.includes('Rubric')) violations.add(tick.reason);
      } else {
        failure++;
        violations.add(tick.reason);
      }
    }

    const total = Math.max(1, normal + suboptimal + failure);
    const compliancePct = +(((normal + suboptimal * 0.4) / total) * 100).toFixed(1);
    const rawOutcome = 1.0 + 4.0 * ((normal + suboptimal * 0.55 - failure * 0.6) / total);
    const overallOutcomeScore1to5 = +Math.max(1.0, Math.min(5.0, rawOutcome)).toFixed(2);

    return {
      rubricText: this.rubricText,
      overallOutcomeScore1to5,
      compliancePct,
      normalStepsCount: normal,
      suboptimalStepsCount: suboptimal,
      failureStepsCount: failure,
      activeViolations: Array.from(violations).slice(0, 3),
    };
  }

  public getSegments(): TemporalSegment[] {
    return this.segments;
  }

  public getRecentTicks(count: number = 36): DenseProcessRewardTick[] {
    return this.rewardTicks.slice(-count);
  }

  public getForkEvents(): MultiverseForkEvent[] {
    return this.forkEvents;
  }

  public exportDpoDatasetJson(telemetry: RobotTelemetry): string {
    const scoreCard = this.getScoreCard();
    const payload = {
      schema: 'kronos-1m-robotics-dpo-v1',
      generatedAt: new Date().toISOString(),
      orchestrator: {
        managedAgent: 'antigravity-preview-09-2026',
        model: 'gemini-4-argon',
        contextWindowTokens: 1000000,
      },
      rubricEvaluation: scoreCard,
      temporalSegments: this.segments,
      denseProcessRewards0_5s: this.rewardTicks,
      counterfactualDpoPreferencePairs: this.forkEvents.map(fork => {
        const chosen = fork.branches.find(b => b.dpoRole === 'chosen')!;
        const rejected = fork.branches.filter(b => b.dpoRole === 'rejected');
        return {
          forkId: fork.id,
          simTimeSec: fork.simTimeSec,
          triggerReason: fork.triggerReason,
          stateVector: {
            position: fork.originPosition,
            headingRad: fork.originHeading,
            physicsRegime: telemetry.physicsRegime || 'Earth 1.0g Nominal',
          },
          chosen: {
            policy: chosen.policyName,
            outcomeScore: chosen.outcomeScore1to5,
            actionDsl: chosen.actionDsl,
            trajectory: chosen.trajectory,
          },
          rejected: rejected.map(r => ({
            policy: r.policyName,
            outcomeScore: r.outcomeScore1to5,
            actionDsl: r.actionDsl,
            trajectory: r.trajectory,
            rejectionReason: r.rationale,
          })),
        };
      }),
    };
    return JSON.stringify(payload, null, 2);
  }

  public exportMuJoCoXml(telemetry: RobotTelemetry): string {
    const morph = telemetry.activeMorphology || 'standard_cruiser';
    const trackHalf = morph === 'wide_track_heavy' ? 0.76 : morph === 'thruster_hopper' ? 0.72 : 0.55;
    const clearanceZ = morph === 'high_clearance_crawler' ? 0.68 : morph === 'thruster_hopper' ? 0.62 : 0.45;
    return `<?xml version="1.0" encoding="utf-8"?>
<mujoco model="kronos_1m_sim2real_${morph}">
  <compiler angle="radian" coordinate="local" inertiafromgeom="true"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="RK4"/>
  <default>
    <geom condim="3" friction="0.78 0.005 0.0001" rgba="0.2 0.8 0.9 1"/>
  </default>
  <worldbody>
    <light directional="true" diffuse="0.8 0.8 0.8" pos="0 0 25" dir="0 0 -1"/>
    <geom name="continuous_argon_sector_floor" type="plane" size="25 200 0.1" rgba="0.08 0.11 0.16 1"/>
    <body name="rover_chassis_${morph}" pos="${telemetry.position[0].toFixed(2)} ${telemetry.position[2].toFixed(2)} ${clearanceZ}">
      <freejoint name="rover_root"/>
      <geom name="hull" type="box" size="0.55 1.15 0.22" mass="48" rgba="0.05 0.94 1.0 1"/>
      <geom name="wheel_fl" type="cylinder" pos="-${trackHalf} 0.75 -0.18" size="0.35 0.12" euler="0 1.5708 0" mass="4.5"/>
      <geom name="wheel_fr" type="cylinder" pos="${trackHalf} 0.75 -0.18" size="0.35 0.12" euler="0 1.5708 0" mass="4.5"/>
      <geom name="wheel_rl" type="cylinder" pos="-${trackHalf} -0.75 -0.18" size="0.35 0.12" euler="0 1.5708 0" mass="4.5"/>
      <geom name="wheel_rr" type="cylinder" pos="${trackHalf} -0.75 -0.18" size="0.35 0.12" euler="0 1.5708 0" mass="4.5"/>
    </body>
  </worldbody>
</mujoco>`;
  }

  public exportOpenUsdAscii(telemetry: RobotTelemetry): string {
    const morph = telemetry.activeMorphology || 'standard_cruiser';
    return `#usda 1.0
(
    defaultPrim = "KronosSim2RealWorld"
    metersPerUnit = 1
    upAxis = "Y"
)

def Xform "KronosSim2RealWorld"
{
    def PhysicsScene "PhysicsScene"
    {
        vector3f physics:gravityDirection = (0, -1, 0)
        float physics:gravityMagnitude = 9.81
    }

    def Xform "ArgonRover_${morph}" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI"]
    )
    {
        double3 xformOp:translate = (${telemetry.position[0].toFixed(2)}, ${telemetry.position[1].toFixed(2)}, ${telemetry.position[2].toFixed(2)})
        uniform token[] xformOpOrder = ["xformOp:translate"]
        string kronos:physicsRegime = "${telemetry.physicsRegime || 'Earth 1.0g Nominal'}"
    }
}
`;
  }
}

export const kronosEngine = new KronosEngine();


