# Embedded AI Predictive Maintenance System

A real-time, edge-deployed fault detection pipeline for rotating machinery. A quantized TensorFlow Lite model runs directly on a Raspberry Pi, classifying vibration signals into five fault categories with sub-50 ms inference latency — no cloud subscription required.

---

## Business Problem

Unplanned equipment failure is one of the largest cost drivers in manufacturing, energy, and process industries. Industry estimates put unplanned downtime at **$50,000–$250,000 per hour** for continuous-process plants, with rotating machinery (motors, pumps, compressors, conveyors) responsible for the majority of failures.

Conventional approaches have two failure modes:

| Approach | Problem |
|---|---|
| **Reactive maintenance** — fix it when it breaks | Catastrophic failure, safety risk, long lead times for replacement parts |
| **Time-based preventive maintenance** — service on a schedule | Over-maintenance (replacing healthy components), high labour cost, no reduction in surprise failures |

What neither approach provides is *early warning* — detecting the specific fault developing inside a machine weeks before it causes a breakdown.

Cloud-connected condition monitoring systems exist but carry significant barriers: per-device licensing fees (₹5,000–20,000/month per asset), mandatory internet connectivity, data sovereignty concerns in regulated industries, and latency incompatible with fast-response shutdowns.

---

## Proposed Solution

This system implements **edge AI predictive maintenance**: the inference model lives on the same device that reads the sensor, eliminating cloud dependency entirely.

**How it solves the problem:**

- **Early fault detection** — vibration FFT features catch bearing wear, imbalance, and misalignment weeks before mechanical failure, enabling planned maintenance during scheduled downtime.
- **Zero cloud cost** — the INT8 quantized TFLite model runs in < 50 ms on a ₹2,000 Raspberry Pi. No SaaS subscription, no connectivity requirement, no data leaving the plant floor.
- **Actionable alerts** — when confidence exceeds 75%, a Telegram push notification reaches the maintenance engineer in seconds, with fault type and confidence score. Every event is persisted to SQLite for trend analysis and audit trails.
- **Scalable and open** — MQTT-based architecture means any number of machines can publish to the same broker. The Node-RED dashboard aggregates them. Swapping the Pi for an STM32 or NVIDIA Jetson requires only a new TFLite runtime; the rest of the pipeline is unchanged.

**Cost comparison:**

| | Commercial IIoT Platform | This System |
|---|---|---|
| Hardware per asset | ₹15,000–40,000 | ₹2,500–4,000 |
| Monthly subscription | ₹5,000–20,000 | ₹0 |
| Cloud dependency | Required | None |
| Inference latency | 1–10 s (round-trip) | < 50 ms (on-device) |
| Data sovereignty | Vendor cloud | On-premises |

---

## Industry Deployment

### Target use cases

| Industry | Asset | Fault monitored | Typical consequence of missed fault |
|---|---|---|---|
| **Manufacturing** | CNC spindle motors, conveyor drives | Bearing wear, imbalance | Spindle crash, scrap batch, 4–8 h unplanned downtime |
| **Energy / utilities** | Cooling tower fans, pump sets | Misalignment, cavitation proxy | Pump seizure, process trip, regulatory penalty |
| **HVAC** | Chiller compressors, AHU fans | Bearing wear, imbalance | Building comfort loss, SLA breach, emergency call-out cost |
| **Agriculture** | Grain augers, irrigation pumps | Imbalance, misalignment | Harvest delay, crop loss during time-critical window |
| **Automotive** | Test-bench dynamometers | All five fault classes | Invalid test data, retesting cost, programme delay |

---

### Phase 1 — Site assessment

Before installing any hardware, a site survey determines sensor placement and network topology.

1. **Identify critical assets** — rank machines by failure impact (downtime cost × failure frequency). Start with the top 3–5 assets; expand after validating the pipeline.
2. **Characterise each asset** — record shaft speed (RPM), bearing model number, drive-end vs. free-end mounting constraints, and ambient temperature range. Bearing fault frequencies (BPFO, BPFI, BSF) are calculated from the bearing catalogue and used to validate model sensitivity.
3. **Assess mounting surface** — the MPU6050 must be mounted rigidly (no rubber gaskets) on the bearing housing or motor end-cap. A loose mount attenuates high-frequency content and degrades detection accuracy.
4. **Plan cable routes** — I2C bus (SDA/SCL) maximum reliable length is ~1 m. For longer runs, use a 3.3 V I2C buffer (e.g., PCA9600) or switch to a CAN/RS-485 front-end. Plan conduit paths that avoid VFD cable trays (EMI source).
5. **Network survey** — confirm Wi-Fi or Ethernet coverage at each machine location. The Pi only needs LAN connectivity to the Mosquitto broker; it does not require internet access.

> **Phase 1 checkpoint** ✓ Asset list ranked by risk · bearing fault frequencies calculated · sensor mounting location identified · cable route and network coverage confirmed

---

### Phase 2 — Hardware installation

#### Bill of materials (per monitored asset)

| Component | Spec | Est. Cost (INR) |
|---|---|---|
| Raspberry Pi 4 (2 GB) / Pi Zero 2W | Edge compute | ₹2,000–3,500 |
| MPU6050 module | 3-axis accelerometer/gyroscope, ±16 g, 400 Hz ODR via I2C | ₹80–120 |
| DHT22 | Ambient/winding temperature & humidity | ₹80–120 |
| ACS712 (30 A variant) | Phase current monitoring via SPI ADC (MCP3008) | ₹70–100 |
| DIN-rail enclosure (IP54) | Protects Pi + PCB from dust, coolant mist | ₹300–600 |
| 24 V → 5 V DIN PSU | Panel-mount power supply, UL-listed | ₹400–700 |
| M6 stainless mounting stud + epoxy | Rigid sensor mount on bearing housing | ₹50–80 |

> **Total bill of materials per asset: ₹3,000–5,200**

#### Wiring

```
Bearing housing
  └─ MPU6050 (M6 stud, epoxy bonded)
       ├─ VCC  → Pi 3.3 V (Pin 1)
       ├─ GND  → Pi GND  (Pin 6)
       ├─ SDA  → Pi GPIO 2 (Pin 3)   ← I2C data
       └─ SCL  → Pi GPIO 3 (Pin 5)   ← I2C clock

Motor terminal box
  └─ ACS712 (in series with phase L1)
       └─ Analogue out → MCP3008 CH0 → Pi SPI (GPIO 8/9/10/11)

Motor body / control panel
  └─ DHT22
       ├─ VCC  → Pi 3.3 V
       ├─ GND  → Pi GND
       └─ DATA → Pi GPIO 4 (Pin 7)
```

Enable I2C and SPI on the Pi:
```bash
sudo raspi-config   # Interface Options → I2C → Enable
                    # Interface Options → SPI → Enable
sudo reboot
i2cdetect -y 1      # should show 0x68 (MPU6050)
```

> **Phase 2 checkpoint** ✓ `i2cdetect -y 1` shows `0x68` · DHT22 reads ambient temperature · ACS712 analogue output in expected voltage range · enclosure sealed and PSU voltage confirmed at 5.0 V

---

### Phase 3 — Software deployment

#### 3a. OS and environment

```bash
# Flash Raspberry Pi OS Lite (64-bit) to SD card using Raspberry Pi Imager
# Enable SSH and set hostname/Wi-Fi credentials in the Imager advanced options

ssh pi@<hostname>.local
sudo apt update && sudo apt install -y python3-pip python3-venv git mosquitto

python3 -m venv /opt/predmaint/venv
source /opt/predmaint/venv/bin/activate
pip install -r requirements.txt
```

#### 3b. Deploy model and application files

```bash
# From development machine — copy only the runtime files (no training code)
scp model_int8.tflite feature_norm.npy \
    inference_engine.py alert_manager.py \
    pi@<hostname>.local:/opt/predmaint/
```

#### 3c. Configure environment variables

Create `/etc/predmaint.env` — **never commit credentials to source control**:

```ini
MQTT_BROKER=192.168.1.50        # IP of the shared Mosquitto broker
MQTT_PORT=1883
TELEGRAM_BOT_TOKEN=<token>      # from BotFather — set once per deployment
TELEGRAM_CHAT_ID=<chat_id>      # maintenance team group chat ID
ALERT_DB_PATH=/opt/predmaint/alerts.db
```

#### 3d. Create systemd services

`/etc/systemd/system/inference-engine.service`:
```ini
[Unit]
Description=Vibration Fault Inference Engine
After=network.target mosquitto.service

[Service]
EnvironmentFile=/etc/predmaint.env
WorkingDirectory=/opt/predmaint
ExecStart=/opt/predmaint/venv/bin/python inference_engine.py
Restart=on-failure
RestartSec=5
User=pi

[Install]
WantedBy=multi-user.target
```

`/etc/systemd/system/alert-manager.service` — same structure, `ExecStart` points to `alert_manager.py`.

```bash
sudo systemctl daemon-reload
sudo systemctl enable inference-engine alert-manager
sudo systemctl start  inference-engine alert-manager
sudo systemctl status inference-engine   # confirm Active: running
```

> **Phase 3 checkpoint** ✓ `systemctl status inference-engine` shows `Active: running` · `mosquitto_sub -t inference/result` returns valid JSON · no import errors in service journal (`journalctl -u inference-engine -n 20`)

---

### Phase 4 — Commissioning and baseline

1. **Confirm sensor data** — run `i2cdetect` and check MQTT traffic with `mosquitto_sub -t sensor/vibration` to verify the hardware reader is publishing at the expected rate.
2. **Capture a healthy baseline** — with the machine running under normal load, record 30–60 minutes of `normal` class data. If the model consistently predicts `normal` at > 90% confidence, the sensor placement and normalisation are correct.
3. **Simulate fault response** — temporarily introduce a known imbalance (attach a small mass to the shaft coupling) and confirm the dashboard switches to `imbalance` within 2–3 windows (< 2 s). Remove the mass and confirm recovery to `normal`.
4. **Set alert thresholds** — the default 75% confidence threshold may need tuning per asset. Noisy environments (high background vibration from adjacent machines) may require raising to 80–85% to reduce false positives.
5. **Document baseline signature** — export the first hour of `alerts.db` data as the machine's healthy reference. This becomes the comparison baseline for trend reporting.

> **Phase 4 checkpoint** ✓ `normal` class predicted at ≥ 90% confidence under healthy load · imbalance test triggers correct fault class within 2 s · Telegram alert received on mobile · first row written to `alerts.db`

---

### Phase 5 — Ongoing operations

| Activity | Frequency | Responsible |
|---|---|---|
| Review alert history dashboard | Daily | Maintenance supervisor |
| Confirm `normal` baseline drift has not occurred | Weekly | Reliability engineer |
| Re-run `train_cwru_model.py` if a new fault type is added | As needed | ML engineer |
| Replace SD card (wear levelling) | Every 12–18 months | Maintenance technician |
| Validate sensor mounting integrity (torque check) | Every 6 months | Maintenance technician |
| Archive `alerts.db` and rotate to new file | Monthly | Automated via cron |

> **Phase 5 checkpoint** ✓ Cron job confirmed running · at least one weekly baseline review completed · SD card replacement date logged · sensor torque check recorded in maintenance register

---

### Deployment architecture (production)

```
Plant floor (per asset)               LAN / control room
──────────────────────────────────    ──────────────────────────────────────
 [Motor]──[MPU6050 @ bearing]          [Mosquitto broker]
   │       [DHT22 @ winding]    MQTT         │
   └──[Raspberry Pi]  ──────────────►  ┌─────┴──────────────────┐
      inference_engine.py              │                        │
      alert_manager.py           [Node-RED dashboard]   [SQLite alerts.db]
      hardware_reader.py          http://broker:1880/ui   trend reports
                                        │
                                  [Telegram push]
                                   maintenance team
```

For **multi-asset deployments**, each Pi publishes to a shared broker using namespaced topics:

```
sensor/vibration/motor-01   →  inference/result/motor-01
sensor/vibration/pump-03    →  inference/result/pump-03
```

Node-RED uses one tab per asset; `alert_manager.py` subscribes to `inference/result/#` and tags each alert with the machine ID.

---

### Scaling beyond one machine

| Asset count | Recommended broker | Notes |
|---|---|---|
| 1–10 | Mosquitto on a Pi or small Linux VM | Zero licence cost; adequate throughput |
| 10–50 | EMQX Community Edition | Web dashboard, built-in metrics, clustering |
| 50+ | HiveMQ or EMQX Enterprise | HA clustering, TLS mutual auth, audit logging |

- Topic namespacing: `sensor/vibration/<machine_id>` and `inference/result/<machine_id>`.
- Node-RED dashboard: one tab per machine; shared broker connection.
- `alert_manager.py` subscribes to `inference/result/#`; the `machine_id` from the topic is included in every Telegram alert and SQLite row.
- No changes to `inference_engine.py` or the TFLite model are needed when adding assets.

---

## Architecture

```
┌─────────────────────┐      MQTT       ┌──────────────────────┐      MQTT       ┌───────────────────────┐
│  Layer 1 · Sensors  │ ──────────────► │  Layer 2 · Edge AI   │ ──────────────► │  Layer 3 · Alerting   │
│                     │  sensor/        │                       │  inference/     │                       │
│  MPU6050 (vibe)     │  vibration      │  FFT feature extract  │  result         │  Node-RED dashboard   │
│  DHT22  (temp/hum)  │                 │  INT8 TFLite model    │                 │  Telegram Bot alerts  │
│  ACS712 (current)   │                 │  inference_engine.py  │                 │  SQLite audit log     │
└─────────────────────┘                 └──────────────────────┘                 └───────────────────────┘
```

**Fault classes detected:** `normal` · `bearing_wear` · `imbalance` · `misalignment` · `critical`

---

## Repository Structure

```
emb-ai-pred-main/
├── inference_engine.py          # MQTT subscriber → feature extraction → TFLite inference → publish result
├── sensor_simulator.py          # Drop-in MPU6050 replacement — generates synthetic fault signals over MQTT
├── alert_manager.py             # Subscribes to inference results; fires Telegram alerts + SQLite logging
├── train_model.py               # Generates synthetic training data, trains Keras model, exports model_int8.tflite
├── model_int8.tflite            # INT8 quantized model (8.7 KB, <50 ms on Pi 4) — run train_model.py to regenerate
├── feature_norm.npy             # Feature mean/std from training set — used by inference_engine for normalisation
├── test_inference.py            # Unit tests for inference_engine (builds a dummy TFLite model on-the-fly)
├── test_simulator.py            # Unit tests for sensor_simulator
├── requirements.txt             # Python dependencies
├── nodered-dashboard-flow.json  # Importable Node-RED flow (waveform chart, fault gauge, alert table)
├── 1. mosquito-mqtt-startup.txt # Mosquitto broker setup & test commands
├── 2. start-node-red-dash.txt   # Node-RED startup instructions
├── 3. verify-full-stack.txt     # End-to-end smoke-test checklist
├── 4. full-pipeline-run.md      # Step-by-step guide to run all 5 processes
└── cwru-simulation/             # Real-data replay using CWRU bearing dataset
    ├── download_cwru.py         # Downloads CWRU .mat files, resamples 12 kHz → 400 Hz, saves .npy
    ├── cwru_replay_simulator.py # Streams real CWRU windows over MQTT — drop-in for sensor_simulator.py
    ├── train_cwru_model.py      # Retrains the classifier on real CWRU features; overwrites model_int8.tflite
    ├── README.md                # Setup and usage instructions for the CWRU pipeline
    └── data/                    # Downloaded .mat and resampled .npy files (generated by download_cwru.py)
```

---

## Hardware

| Component | Purpose | Est. Cost (INR) |
|---|---|---|
| Raspberry Pi 4 (2 GB) / Pi Zero 2W | Edge compute | ₹2,000–3,500 |
| MPU6050 (accelerometer/gyroscope) | 3-axis vibration @ up to 1 kHz via I2C | ₹80–120 |
| DHT22 | Temperature & humidity | ₹80–120 |
| ACS712 | Motor current / load monitoring | ₹70–100 |
| Small DC / servo motor | Test machine (attach coin for imbalance simulation) | ₹150 |

> **Total bill of materials: ₹2,500–4,000**

---

## Signal Processing & Model

- **Sampling rate:** 400 Hz · **Window:** 512 samples (1.28 s)
- **Feature vector (6 values):** RMS, peak amplitude, spectral mean, dominant frequency (Hz), spectral energy, kurtosis
- **Model format:** INT8 quantized TFLite, 8.6 KB, < 50 ms on Raspberry Pi 4
- **Training data:** Real CWRU drive-end accelerometer recordings (high-overlap windowing + augmentation → ~2,870 windows across 4 fault classes) plus 600 synthetic `critical` windows. Trained via `cwru-simulation/train_cwru_model.py`. The original synthetic-only trainer (`train_model.py`) is retained for reference.
- **Why CWRU-trained?** The synthetic model misclassified every real recording due to domain gap (sinusoids vs. impulsive, non-Gaussian bearing signals). Retraining on real CWRU features achieved 100% validation accuracy.
- **Feature normalisation:** `feature_norm.npy` (mean/std per feature, fit on CWRU training windows) is produced at training time and loaded by `inference_engine.py` at startup

---

## Quick Start

### 1. Prerequisites

```bash
# Install Mosquitto MQTT broker (macOS)
brew install mosquitto
brew services start mosquitto

# Install Node-RED
npm install -g node-red

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Train the model

Run once to generate `model_int8.tflite` and `feature_norm.npy`:

```bash
python train_model.py
# Dataset: 2550 train / 450 val — 5 classes
# Best val accuracy: 100.0%
# Saved model_int8.tflite (8.7 KB)
```

> Re-run any time you change the fault profiles or feature extraction logic.

### 3. Run the pipeline

Open four terminal windows:

```bash
# Terminal 1 — MQTT broker
mosquitto

# Terminal 2 — Sensor simulator (replace with real MPU6050 reader on hardware)
python sensor_simulator.py --fault normal          # healthy baseline
python sensor_simulator.py --fault bearing_wear    # BPFO fault @ 105.9 Hz
python sensor_simulator.py --fault imbalance       # 1× RPM fault @ 29.95 Hz
python sensor_simulator.py --fault misalignment    # 2×/4× RPM harmonics
python sensor_simulator.py --fault critical        # multiple fault modes

# Terminal 3 — Inference engine (subscribes to sensor data, publishes predictions)
python inference_engine.py

# Terminal 4 — Alert manager (optional — requires Telegram credentials)
python alert_manager.py
```

### 4. Dashboard

```bash
node-red &
# 1. Open http://localhost:1880
# 2. Hamburger menu → Import → select nodered-dashboard-flow.json → Deploy
# 3. Open http://localhost:1880/ui
```

**Dashboard widgets:** live vibration waveform chart · fault classification (colour-coded) · confidence gauge · motor temperature · phase current · alert history table

---

## CWRU Real-Data Replay

The `cwru-simulation/` folder provides an end-to-end pipeline for streaming **real** CWRU bearing accelerometer recordings through the same MQTT → inference engine → Node-RED stack, replacing synthetic signals with the highest-fidelity simulation possible short of physical hardware.

### Dataset

| Label | Fault | CWRU file | Key frequency |
|---|---|---|---|
| `normal` | None — healthy baseline | 97.mat | — |
| `bearing_wear` | 0.007″ outer-race fault | 105.mat | BPFO ≈ 105.9 Hz |
| `imbalance` | 0.007″ inner-race fault | 118.mat | BPFI ≈ 162.2 Hz |
| `misalignment` | 0.007″ ball fault | 130.mat | BSF ≈ 68.6 Hz |

Files are downloaded directly from the public [CWRU Bearing Data Center](https://engineering.case.edu/bearingdatacenter) — no account or special tooling required. The drive-end (DE) accelerometer channel is extracted and resampled from 12 kHz to **400 Hz** to match the inference engine's `SAMPLE_RATE`.

### Setup

```bash
# Step 1 — Download and prepare the dataset (one-time)
cd cwru-simulation
python download_cwru.py
# [normal]       Loaded 'X097_DE_time' — 243,938 samples @ 12,000 Hz
# [normal]       Resampled → 8,132 samples @ 400 Hz
# ...
# All files processed successfully.

# Step 2 — Retrain the model on real CWRU data (one-time, ~15 s)
#   Overwrites ../model_int8.tflite and ../feature_norm.npy
python train_cwru_model.py
# Best val accuracy: 100.0%
# Saved model_int8.tflite (8.6 KB)

# Step 3 — Start the full stack (if not already running)
mosquitto                              # Terminal 1 — MQTT broker
python inference_engine.py             # Terminal 2 — inference engine (project root)
node-red                               # Terminal 3 — dashboard

# Step 4 — Start CWRU replay (cwru-simulation/)
python cwru_replay_simulator.py --fault normal
python cwru_replay_simulator.py --fault bearing_wear
python cwru_replay_simulator.py --fault imbalance
python cwru_replay_simulator.py --fault misalignment
```

> **Important:** The synthetic-trained `train_model.py` misclassifies real CWRU signals due to domain gap (sinusoidal training data vs. impulsive real signals). Always run `train_cwru_model.py` after `download_cwru.py`.

Open **http://localhost:1880/ui** — the **Data Source & Ground Truth** panel shows `📡 REAL — CWRU Real Data` alongside the ground-truth label; the **Fault Classification** panel shows the model's live prediction for direct comparison.

### Payload schema

The replay simulator publishes to `sensor/vibration` using the exact same JSON schema as `sensor_simulator.py`, so `inference_engine.py` and the Node-RED flow require no changes:

```json
{
  "vibration_x":    [512 floats at 400 Hz],
  "temperature":    42.1,
  "current_a":      3.19,
  "fault_injected": "bearing_wear",
  "source":         "cwru_real_data"
}
```

The `source` field is the only addition — it drives the dashboard's data-source badge and is ignored by the inference engine.

### Pipeline

```
download_cwru.py
  └─ CWRU .mat  →  resample 12 kHz → 400 Hz  →  cwru-simulation/data/<label>.npy

cwru_replay_simulator.py
  └─ data/<label>.npy
       └─ 512-sample windows (50 % overlap, 0.64 s stride)
            └─ MQTT  sensor/vibration  →  inference_engine.py
                                               └─ MQTT  inference/result  →  Node-RED
```

---

## Configuration

Alert manager settings are controlled via environment variables — no secrets in source code.

| Variable | Default | Description |
|---|---|---|
| `MQTT_BROKER` | `localhost` | Mosquitto broker hostname |
| `MQTT_PORT` | `1883` | Mosquitto broker port |
| `TELEGRAM_BOT_TOKEN` | *(required)* | BotFather token |
| `TELEGRAM_CHAT_ID` | *(required)* | Target chat / group ID |
| `ALERT_DB_PATH` | `alerts.db` | SQLite database file path |

Alerts fire when model confidence exceeds **75%** on any non-normal fault class.

---

## Testing

```bash
python test_inference.py    # builds a dummy TFLite model on-the-fly; no pre-trained file needed
python test_simulator.py
```

---

## MQTT Topics

| Topic | Publisher | Payload |
|---|---|---|
| `sensor/vibration` | `sensor_simulator.py` / `cwru_replay_simulator.py` / real hardware | `{"vibration_x": [<float>×512], "temperature": <°C>, "current_a": <A>, "fault_injected": <str>, "source": <str>}` |
| `inference/result` | `inference_engine.py` | `{"class": "<fault>", "confidence": <0–1>}` |

> The `source` field (`"cwru_real_data"` or absent for synthetic) is optional — consumed by the Node-RED **Data Source & Ground Truth** panel and ignored by the inference engine.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Edge ML | TensorFlow Lite (INT8 quantized) |
| Signal processing | NumPy, SciPy (FFT) |
| IoT messaging | MQTT — Mosquitto broker, Paho Python client |
| Dashboard | Node-RED |
| Alerting | Telegram Bot API |
| Data persistence | SQLite |
| Model training | TensorFlow / Keras, scikit-learn |

---

## Roadmap

- [ ] Real MPU6050 I2C reader (`hardware_reader.py`)
- [ ] Jupyter notebook: FFT plots, model training, confusion matrix
- [ ] ONNX export path for non-Pi deployment targets
- [ ] Estimated hours-to-failure trending in dashboard
- [ ] Telegram Bot integration — wire `alert_manager.py` to a BotFather token; send fault class, confidence score, and timestamp as a push notification when confidence exceeds 75% on any non-normal class
- [ ] SQLite data persistence — log every `inference/result` event (timestamp, fault class, confidence, source) to `alerts.db` via `alert_manager.py`; expose a `/history` query endpoint for trend reports and maintenance audit trails
- [x] Real CWRU bearing dataset replay — `cwru-simulation/` (download, resample, stream over MQTT)
- [x] Retrain on real CWRU data — `cwru-simulation/train_cwru_model.py` (100% val accuracy, bearing_wear detected at 72–84% confidence)

---

## License

MIT

---

## ⭐ Star this repo

If this project saved you time, helped you learn edge AI, or gave you a working starting point for a real deployment — consider starring the repository.

Stars help other engineers and students discover the project, and every one is appreciated.

> **[⭐ Star on GitHub](https://github.com/manishkumarai/Embedded-AI-Predictive-Maintenance/)** — takes 2 seconds and means a lot.

Feedback, issues, and pull requests are equally welcome.

---

## Abbreviations

| Abbreviation | Full Form |
|---|---|
| AI | Artificial Intelligence |
| AHU | Air Handling Unit |
| API | Application Programming Interface |
| BPFI | Ball Pass Frequency, Inner Race |
| BPFO | Ball Pass Frequency, Outer Race |
| BSF | Ball Spin Frequency |
| CNC | Computer Numerical Control |
| CPU | Central Processing Unit |
| CWRU | Case Western Reserve University |
| DB | Database |
| DC | Direct Current |
| DHT | Digital Humidity & Temperature (sensor family) |
| DIN | Deutsches Institut für Normung (German standards body; DIN-rail mounting) |
| EMI | Electromagnetic Interference |
| FFT | Fast Fourier Transform |
| GPIO | General Purpose Input/Output |
| HA | High Availability |
| HVAC | Heating, Ventilation, and Air Conditioning |
| I2C | Inter-Integrated Circuit (serial communication bus) |
| IIoT | Industrial Internet of Things |
| INR | Indian Rupee |
| INT8 | 8-bit Integer (quantization format) |
| IoT | Internet of Things |
| IP | Ingress Protection (enclosure rating, e.g. IP54) |
| JSON | JavaScript Object Notation |
| LAN | Local Area Network |
| MAFAULDA | Machinery Fault Database |
| ML | Machine Learning |
| MQTT | Message Queuing Telemetry Transport |
| ODR | Output Data Rate |
| ONNX | Open Neural Network Exchange |
| OS | Operating System |
| Pi | Raspberry Pi |
| PSU | Power Supply Unit |
| RMS | Root Mean Square |
| RPM | Revolutions Per Minute |
| SaaS | Software as a Service |
| SCL | Serial Clock Line (I2C) |
| SD | Secure Digital (SD card) |
| SDA | Serial Data Line (I2C) |
| SLA | Service Level Agreement |
| SPI | Serial Peripheral Interface |
| SQLite | Self-contained, serverless SQL database engine |
| SSH | Secure Shell |
| TFLite | TensorFlow Lite |
| TLS | Transport Layer Security |
| UI | User Interface |
| UL | Underwriters Laboratories (UL-listed certification) |
| URL | Uniform Resource Locator |
| UTC | Coordinated Universal Time |
| venv | Python Virtual Environment |
| VFD | Variable Frequency Drive |
| VM | Virtual Machine |
