import { useState, useRef, useEffect } from "react";

export default function Stopwatch() {
  const [elapsedMs, setElapsedMs] = useState(0);
  const [running, setRunning] = useState(false);
  const [laps, setLaps] = useState([]);
  const startRef = useRef(null);
  const rafRef = useRef(null);

  useEffect(() => {
    if (!running) return;
    const tick = () => {
      setElapsedMs(Date.now() - startRef.current);
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [running]);

  const start = () => {
    startRef.current = Date.now() - elapsedMs;
    setRunning(true);
  };

  const stop = () => {
    setRunning(false);
  };

  const reset = () => {
    setRunning(false);
    setElapsedMs(0);
    setLaps([]);
  };

  const lap = () => {
    setLaps((prev) => [elapsedMs, ...prev]);
  };

  const format = (ms) => {
    const totalCentis = Math.floor(ms / 10);
    const centis = totalCentis % 100;
    const totalSeconds = Math.floor(ms / 1000);
    const seconds = totalSeconds % 60;
    const minutes = Math.floor(totalSeconds / 60) % 60;
    const hours = Math.floor(totalSeconds / 3600);
    const pad = (n, l = 2) => String(n).padStart(l, "0");
    return {
      main: `${hours > 0 ? pad(hours) + ":" : ""}${pad(minutes)}:${pad(seconds)}`,
      centis: pad(centis),
    };
  };

  const { main, centis } = format(elapsedMs);

  let bestIdx = -1;
  let worstIdx = -1;
  if (laps.length > 1) {
    const splits = laps.map((t, i) =>
      i === laps.length - 1 ? t : laps[i] - laps[i + 1]
    );
    const min = Math.min(...splits);
    const max = Math.max(...splits);
    bestIdx = splits.indexOf(min);
    worstIdx = splits.indexOf(max);
  }

  return (
    <div
      style={{
        minHeight: "100vh",
        width: "100%",
        background: "#0B0E11",
        color: "#EDEFF1",
        fontFamily:
          "'IBM Plex Mono', 'Courier New', monospace",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        padding: "48px 20px",
        boxSizing: "border-box",
      }}
    >
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
        .sw-btn {
          border: none;
          cursor: pointer;
          font-family: 'IBM Plex Mono', monospace;
          font-weight: 600;
          font-size: 14px;
          letter-spacing: 0.08em;
          text-transform: uppercase;
          border-radius: 4px;
          padding: 14px 28px;
          transition: transform 0.08s ease, filter 0.15s ease;
        }
        .sw-btn:active { transform: scale(0.96); }
        .sw-btn:hover { filter: brightness(1.1); }
        .sw-lap-row {
          display: flex;
          justify-content: space-between;
          padding: 10px 4px;
          border-bottom: 1px solid #1E2328;
          font-size: 14px;
        }
        .sw-scroll::-webkit-scrollbar { width: 6px; }
        .sw-scroll::-webkit-scrollbar-thumb { background: #2A3038; border-radius: 3px; }
      `}</style>

      <div
        style={{
          fontFamily: "'IBM Plex Sans', sans-serif",
          fontSize: 12,
          letterSpacing: "0.25em",
          textTransform: "uppercase",
          color: "#6B7280",
          marginBottom: 40,
        }}
      >
        Stopwatch
      </div>

      {/* Dial */}
      <div
        style={{
          position: "relative",
          width: 280,
          height: 280,
          borderRadius: "50%",
          border: `2px solid ${running ? "#E8703A" : "#252A31"}`,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          boxShadow: running
            ? "0 0 0 1px rgba(232,112,58,0.15), 0 0 40px rgba(232,112,58,0.12)"
            : "none",
          transition: "border-color 0.3s ease, box-shadow 0.3s ease",
        }}
      >
        {/* tick marks */}
        {Array.from({ length: 60 }).map((_, i) => {
          const isMajor = i % 5 === 0;
          const angle = (i / 60) * 360;
          return (
            <div
              key={i}
              style={{
                position: "absolute",
                top: "50%",
                left: "50%",
                width: isMajor ? 2 : 1,
                height: isMajor ? 10 : 5,
                background: "#2E343C",
                transform: `rotate(${angle}deg) translate(0, -128px)`,
                transformOrigin: "center",
              }}
            />
          );
        })}

        <div
          style={{
            fontSize: 46,
            fontWeight: 600,
            fontVariantNumeric: "tabular-nums",
            letterSpacing: "0.02em",
          }}
        >
          {main}
        </div>
        <div
          style={{
            fontSize: 18,
            color: "#E8703A",
            fontVariantNumeric: "tabular-nums",
            marginTop: 4,
          }}
        >
          .{centis}
        </div>
      </div>

      {/* Controls */}
      <div style={{ display: "flex", gap: 14, marginTop: 44 }}>
        {!running ? (
          <button
            className="sw-btn"
            onClick={start}
            style={{ background: "#E8703A", color: "#0B0E11" }}
          >
            {elapsedMs > 0 ? "Resume" : "Start"}
          </button>
        ) : (
          <button
            className="sw-btn"
            onClick={stop}
            style={{ background: "#EDEFF1", color: "#0B0E11" }}
          >
            Stop
          </button>
        )}

        <button
          className="sw-btn"
          onClick={lap}
          disabled={!running}
          style={{
            background: "transparent",
            color: running ? "#EDEFF1" : "#3A4048",
            border: `1px solid ${running ? "#3A4048" : "#20242A"}`,
            cursor: running ? "pointer" : "not-allowed",
          }}
        >
          Lap
        </button>

        <button
          className="sw-btn"
          onClick={reset}
          disabled={elapsedMs === 0 && !running}
          style={{
            background: "transparent",
            color: elapsedMs > 0 || running ? "#EDEFF1" : "#3A4048",
            border: `1px solid ${elapsedMs > 0 || running ? "#3A4048" : "#20242A"}`,
            cursor: elapsedMs > 0 || running ? "pointer" : "not-allowed",
          }}
        >
          Reset
        </button>
      </div>

      {/* Laps */}
      {laps.length > 0 && (
        <div
          className="sw-scroll"
          style={{
            width: "100%",
            maxWidth: 340,
            marginTop: 40,
            maxHeight: 260,
            overflowY: "auto",
          }}
        >
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              fontSize: 11,
              letterSpacing: "0.15em",
              textTransform: "uppercase",
              color: "#6B7280",
              padding: "0 4px 10px",
              fontFamily: "'IBM Plex Sans', sans-serif",
            }}
          >
            <span>Lap</span>
            <span>Split</span>
            <span>Total</span>
          </div>
          {laps.map((t, idx) => {
            const lapNumber = laps.length - idx;
            const split = idx === laps.length - 1 ? t : t - laps[idx + 1];
            const isBest = laps.length > 1 && idx === bestIdx;
            const isWorst = laps.length > 1 && idx === worstIdx;
            const color = isBest ? "#4ADE80" : isWorst ? "#E8703A" : "#EDEFF1";
            return (
              <div className="sw-lap-row" key={idx}>
                <span style={{ color: "#6B7280", width: 40 }}>
                  {String(lapNumber).padStart(2, "0")}
                </span>
                <span style={{ color, fontVariantNumeric: "tabular-nums" }}>
                  {format(split).main}.{format(split).centis}
                </span>
                <span
                  style={{
                    color: "#6B7280",
                    fontVariantNumeric: "tabular-nums",
                  }}
                >
                  {format(t).main}.{format(t).centis}
                </span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
