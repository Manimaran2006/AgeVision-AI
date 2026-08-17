"""
V3 Training Progress Monitor
Reads the LIVE task log and prints clean status every 5 seconds
"""
import time, re, os

# Point directly to the running task log
LOG_PATH = r"C:\Users\LOQ\.gemini\antigravity-ide\brain\446fb425-4e13-4c09-b948-b884d108a56a\.system_generated\tasks\task-522.log"

print("=" * 90)
print("  V3 TRAINING MONITOR — Live Stream")
print(f"  Log: {LOG_PATH}")
print("  (Updates every 5s · Ctrl+C to stop)")
print("=" * 90 + "\n")

pattern = re.compile(
    r"Epoch\s+(\d+)\s+\[([^\]]+)\]:\s+(\d+)%\|.*?\|\s*(\d+)/(\d+)\s+\[([0-9:]+)<([0-9:]+).*?"
    r"loss=([0-9.]+),\s*MAE=([0-9.]+)y?,?\s*img/s=([0-9.]+)"
)

while True:
    try:
        if os.path.exists(LOG_PATH):
            with open(LOG_PATH, "r", encoding="utf-8", errors="ignore") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - 8000))
                content = f.read()

            matches = pattern.findall(content)
            if matches:
                epoch, phase, pct, cur_b, tot_b, elapsed, eta, loss, mae, speed = matches[-1]
                now = time.strftime("%H:%M:%S")
                bar_len = 25
                filled = int(bar_len * int(pct) / 100)
                bar = "█" * filled + "░" * (bar_len - filled)

                print(f"[{now}] Ep {epoch}/3 [{phase}] [{bar}] {pct:>3}% | "
                      f"{int(cur_b):>6,}/{int(tot_b):,} | Loss:{float(loss):>8.4f} | "
                      f"MAE:{float(mae.replace('y','')):>6.2f} yrs | {float(speed):>5.1f} img/s | {elapsed} < {eta}")
            else:
                # Show last few lines of log for non-tqdm output
                lines = content.strip().split("\n")
                last = [l.strip() for l in lines[-3:] if l.strip()]
                if last:
                    now = time.strftime("%H:%M:%S")
                    for l in last:
                        if "INFO" in l:
                            print(f"[{now}] {l}")
        else:
            print(f"Waiting for log file: {LOG_PATH}")

        time.sleep(5)
    except KeyboardInterrupt:
        print("\nMonitor stopped.")
        break
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(5)
