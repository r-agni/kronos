import React, { useEffect, useRef } from 'react';
import { Terminal, Cpu, Zap, Activity } from 'lucide-react';

interface CodeStreamHUDProps {
  code: string;
  totalTokens: number;
  tokensPerSec: number;
  isStreaming: boolean;
}

export const CodeStreamHUD: React.FC<CodeStreamHUDProps> = ({
  code,
  totalTokens,
  tokensPerSec,
  isStreaming
}) => {
  const containerRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom as tokens arrive
  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [code]);

  return (
    <div className="h-full flex flex-col bg-[#0b1017]/90 backdrop-blur-md border border-gray-800/80 rounded-lg overflow-hidden shadow-2xl">
      {/* HUD Header */}
      <div className="px-3 py-2 bg-[#0e1622] border-b border-gray-800 flex items-center justify-between text-xs">
        <div className="flex items-center space-x-2">
          <Terminal className="w-3.5 h-3.5 text-cyan-400" />
          <span className="font-bold tracking-wider text-cyan-400 uppercase font-mono">
            GEMINI-4-ARGON // 1M OUTPUT STREAM
          </span>
        </div>
        <div className="flex items-center space-x-3 text-gray-400 font-mono">
          <div className="flex items-center space-x-1">
            <Activity className={`w-3 h-3 ${isStreaming ? 'text-emerald-400 animate-pulse' : 'text-gray-500'}`} />
            <span>{isStreaming ? `${tokensPerSec.toFixed(1)} tok/s` : 'IDLE'}</span>
          </div>
          <div className="flex items-center space-x-1 text-cyan-300">
            <Zap className="w-3 h-3" />
            <span>{totalTokens.toLocaleString()} TOKENS</span>
          </div>
        </div>
      </div>

      {/* Code Stream Output */}
      <div
        ref={containerRef}
        className="flex-1 p-3 font-mono text-[11px] leading-relaxed overflow-y-auto text-gray-300 select-text whitespace-pre-wrap break-all"
        style={{ scrollBehavior: 'smooth' }}
      >
        {code ? (
          <div>
            <span className="text-gray-500">// Append-only procedural scene AST</span>
            <br />
            {code}
            {isStreaming && (
              <span className="inline-block w-2 h-3.5 ml-1 bg-cyan-400 animate-pulse align-middle" />
            )}
          </div>
        ) : (
          <div className="h-full flex flex-col items-center justify-center text-gray-500 space-y-2">
            <Cpu className="w-6 h-6 animate-pulse text-cyan-500/60" />
            <p className="text-xs tracking-wider">AWAITING ARGON AST STREAM...</p>
          </div>
        )}
      </div>

      {/* Footer Status Bar */}
      <div className="px-3 py-1.5 bg-[#090d14] border-t border-gray-800/80 text-[10px] font-mono text-gray-400 flex justify-between items-center">
        <div className="flex items-center space-x-2">
          <span className="w-1.5 h-1.5 rounded-full bg-cyan-400"></span>
          <span>PIPELINE: HOT-RELOAD AST TO THREE.JS SHADER NODES</span>
        </div>
        <span className="text-cyan-500 font-bold">1M CONTEXT WINDOW</span>
      </div>
    </div>
  );
};

