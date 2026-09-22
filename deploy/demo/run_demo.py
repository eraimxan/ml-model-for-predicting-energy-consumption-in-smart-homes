from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT.parent
HOURLY_PATH = DATA_ROOT / "outputs" / "data" / "hourly_clean.parquet"
OUT_PATH = Path(__file__).resolve().parent / "demo_output.json"

PORT = 8001
BASE_URL = f"http://127.0.0.1:{PORT}"


def _wait_for_health(timeout_s: float = 15.0) -> None:
    deadline = time.time() + timeout_s
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=1.0)
            if r.status_code == 200 and r.json().get("model_loaded"):
                return
        except Exception as e:
            last_err = e
        time.sleep(0.5)
    raise TimeoutError(f"server did not become healthy within {timeout_s}s ({last_err})")


def _build_readings() -> list[dict]:
    df = pd.read_parquet(HOURLY_PATH).iloc[-168:]
    out: list[dict] = []
    for ts, row in df.iterrows():
        out.append(
            {
                "timestamp": ts.isoformat(),
                "Global_active_power": float(row["Global_active_power"]),
                "Global_reactive_power": float(row["Global_reactive_power"]),
                "Voltage": float(row["Voltage"]),
                "Global_intensity": float(row["Global_intensity"]),
                "Sub_metering_1": float(row["Sub_metering_1"]),
                "Sub_metering_2": float(row["Sub_metering_2"]),
                "Sub_metering_3": float(row["Sub_metering_3"]),
            }
        )
    return out


def main() -> int:
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "api.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(PORT),
        "--log-level",
        "warning",
    ]
    proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT))
    try:
        _wait_for_health()
        payload = {"readings": _build_readings()}
        r = requests.post(f"{BASE_URL}/predict", json=payload, timeout=15.0)
        r.raise_for_status()
        data = r.json()

        print(json.dumps(data, indent=2))
        print()
        line = (
            f"Predicted: {data['predicted_kw']:.4f} kW for {data['timestamp_predicted']}"
            f" | Model RMSE: {data['rmse_kw']:.4f} kW | R^2: {data['r2']:.4f}"
        )
        try:
            print(line)
        except UnicodeEncodeError:
            sys.stdout.buffer.write(line.encode("utf-8", errors="replace"))
            sys.stdout.buffer.write(b"\n")

        with open(OUT_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
