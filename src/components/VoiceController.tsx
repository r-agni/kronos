import React, { useState, useEffect, useRef } from 'react';
import { Mic, MicOff, Radio, ShieldCheck, Sliders, Send } from 'lucide-react';
import { soundEffects } from '../services/soundEffects';

interface VoiceControllerProps {
  onVoiceCommand: (transcript: string) => void;
  onUpdateRubric?: (rubric: string) => void;
  activeRubric?: string;
  lastVoiceResponse: string | null;
}

export const VoiceController: React.FC<VoiceControllerProps> = ({
  onVoiceCommand,
  onUpdateRubric,
  activeRubric = 'Maintain wheel slip < 30%, obstacle clearance > 2.2m, and zero unbridged seam halts.',
  lastVoiceResponse,
}) => {
  const [isListening, setIsListening] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [audioLevel, setAudioLevel] = useState(0);
  const recognitionRef = useRef<unknown>(null);
  const animFrameRef = useRef<number | null>(null);

  const submitCommand = (raw: string) => {
    const cmd = raw.trim();
    if (!cmd) return;
    soundEffects.playCountermeasureDeploy();
    if (cmd.toLowerCase().includes('rubric:') || cmd.toLowerCase().includes('slip <') || cmd.toLowerCase().includes('penalize')) {
      onUpdateRubric?.(cmd.replace(/^rubric:\s*/i, ''));
    }
    onVoiceCommand(cmd);
  };

  useEffect(() => {
    // Check SpeechRecognition browser API
    const SpeechRecognition = (window as unknown as { SpeechRecognition?: any; webkitSpeechRecognition?: any }).SpeechRecognition ||
                              (window as unknown as { SpeechRecognition?: any; webkitSpeechRecognition?: any }).webkitSpeechRecognition;

    if (SpeechRecognition) {
      const recognition = new SpeechRecognition();
      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.lang = 'en-US';

      recognition.onresult = (event: any) => {
        let currentTranscript = '';
        for (let i = event.resultIndex; i < event.results.length; i++) {
          currentTranscript += event.results[i][0].transcript;
        }
        setTranscript(currentTranscript);

        if (event.results[event.results.length - 1].isFinal) {
          const finalCommand = currentTranscript.trim();
          if (finalCommand) {
            onVoiceCommand(finalCommand);
            soundEffects.playCountermeasureDeploy();
          }
        }
      };

      recognition.onerror = () => {
        setIsListening(false);
      };

      recognition.onend = () => {
        setIsListening(false);
      };

      recognitionRef.current = recognition;
    }

    return () => {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    };
  }, [onVoiceCommand]);

  const toggleMic = () => {
    if (!recognitionRef.current) {
      // Mock toggle for systems without Web Speech API
      const nextState = !isListening;
      setIsListening(nextState);
      if (nextState) {
        soundEffects.playLidarPing();
        // Simulate voice command after 2 seconds for demo perfection
        setTimeout(() => {
          const sampleCommands = [
            "Argon, deploy reinforced ramp over the chasm!",
            "Deploy sensor beacon at waypoint Bravo",
            "Clear toxic fog with emergency ventilation"
          ];
          const chosen = sampleCommands[Math.floor(Math.random() * sampleCommands.length)];
          setTranscript(chosen);
          onVoiceCommand(chosen);
          soundEffects.playCountermeasureDeploy();
          setTimeout(() => setIsListening(false), 1200);
        }, 1800);
      }
      return;
    }

    const rec = recognitionRef.current as { start: () => void; stop: () => void };
    if (isListening) {
      rec.stop();
      setIsListening(false);
    } else {
      try {
        rec.start();
        setIsListening(true);
        soundEffects.playLidarPing();
      } catch (err) {
        console.error(err);
      }
    }
  };

  // Simulated waveform animation when listening
  useEffect(() => {
    if (!isListening) {
      setAudioLevel(0);
      return;
    }
    const interval = setInterval(() => {
      setAudioLevel(Math.random() * 80 + 20);
    }, 80);
    return () => clearInterval(interval);
  }, [isListening]);

  return (
    <div className="bg-[#0b1017]/95 backdrop-blur-md border border-gray-800/90 rounded-lg p-3 shadow-xl font-mono">
      <div className="flex items-center justify-between pb-2 mb-2 border-b border-gray-800 text-xs">
        <div className="flex items-center space-x-2">
          <Sliders className="w-4 h-4 text-cyan-400" />
          <span className="font-bold tracking-wider text-cyan-400 uppercase">
            TEST-TIME RUBRIC &amp; CO-DESIGN MUTATOR
          </span>
        </div>
        <div className="flex items-center space-x-1.5 text-[10px] text-emerald-400">
          <Radio className="w-3 h-3 animate-pulse" />
          <span>GALLOO OPEN-MIC ACTIVE</span>
        </div>
      </div>

      <div className="text-[10px] text-gray-300 bg-black/50 border border-gray-800 rounded px-2 py-1 mb-2 truncate">
        <span className="text-cyan-400 font-bold mr-1">ACTIVE RUBRIC:</span>
        <span>{activeRubric}</span>
      </div>

      <div className="flex items-center space-x-2">
        <button
          onClick={toggleMic}
          className={`w-9 h-9 rounded-lg border flex-shrink-0 transition-all flex items-center justify-center ${
            isListening
              ? 'bg-cyan-500/20 border-cyan-400 text-cyan-300 shadow-[0_0_15px_rgba(0,240,255,0.4)] animate-pulse'
              : 'bg-gray-900 border-gray-700 text-gray-400 hover:text-cyan-400 hover:border-cyan-500/50'
          }`}
          title="Speak rubric, morphology co-design, or world mutation"
        >
          {isListening ? <Mic className="w-4 h-4" /> : <MicOff className="w-4 h-4" />}
        </button>

        <input
          type="text"
          value={transcript}
          onChange={e => setTranscript(e.target.value)}
          onKeyDown={e => {
            if (e.key === 'Enter') submitCommand(transcript);
          }}
          placeholder='Speak/type rubric ("Slip < 18%"), morph ("Thruster Hopper"), or physics...'
          className="flex-1 bg-black/70 border border-gray-800 focus:border-cyan-500 rounded px-2.5 py-1.5 text-[11px] text-gray-100 outline-none"
        />

        <button
          onClick={() => submitCommand(transcript)}
          className="px-2.5 py-1.5 bg-cyan-600 hover:bg-cyan-500 text-black font-bold rounded text-[11px] flex items-center space-x-1 transition"
        >
          <Send className="w-3 h-3" />
          <span>APPLY</span>
        </button>
      </div>

      {lastVoiceResponse && (
        <div className="mt-1.5 flex items-center space-x-1.5 text-[10px] text-emerald-400 bg-emerald-950/20 px-2 py-0.5 rounded border border-emerald-800/30 truncate">
          <ShieldCheck className="w-3 h-3 flex-shrink-0" />
          <span className="truncate">{lastVoiceResponse}</span>
        </div>
      )}

      <div className="mt-2 pt-2 border-t border-gray-800/60 flex items-center space-x-1.5 text-[10px] text-gray-400 overflow-x-auto">
        <span className="flex-shrink-0 text-cyan-500 font-bold">PRESETS:</span>
        <button
          onClick={() => {
            const cmd = 'Morph rover to thruster hopper with jump jets';
            setTranscript(cmd);
            submitCommand(cmd);
          }}
          className="px-2 py-0.5 bg-cyan-950/60 hover:bg-cyan-900 border border-cyan-700/70 text-cyan-200 rounded whitespace-nowrap transition"
        >
          + Morph: Thruster Hopper
        </button>
        <button
          onClick={() => {
            const cmd = 'Mutate physics to Lunar 0.16g cryo-ice with crosswind';
            setTranscript(cmd);
            submitCommand(cmd);
          }}
          className="px-2 py-0.5 bg-violet-950/60 hover:bg-violet-900 border border-violet-700/70 text-violet-200 rounded whitespace-nowrap transition"
        >
          + Physics: Lunar 0.16g Ice
        </button>
        <button
          onClick={() => {
            const cmd = 'Deploy reinforced bridge across chasm';
            setTranscript(cmd);
            submitCommand(cmd);
          }}
          className="px-2 py-0.5 bg-gray-900 hover:bg-gray-800 border border-gray-700 hover:border-cyan-500 text-gray-300 rounded whitespace-nowrap transition"
        >
          + Deploy Bridge
        </button>
        <button
          onClick={() => {
            const cmd = 'Rubric: Penalize wheel slip > 15% and require clearance > 3.5m';
            setTranscript(cmd);
            submitCommand(cmd);
          }}
          className="px-2 py-0.5 bg-amber-950/60 hover:bg-amber-900 border border-amber-700/70 text-amber-200 rounded whitespace-nowrap transition"
        >
          + Strict Slip Rubric (15%)
        </button>
      </div>
    </div>
  );
};

