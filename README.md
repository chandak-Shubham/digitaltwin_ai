# Automotive Digital Twin Synthetic Dataset

## Overview

This repository contains a process-informed synthetic dataset and generation toolset for an **Automotive Assembly Digital Twin**. The dataset models vehicles moving sequentially through 10 manufacturing stations, capturing physical sensor telemetry, operational durations, ambient factory conditions, and annotated fault/root-cause attributes for machine learning (anomaly detection, predictive maintenance, and root cause identification).

> **Disclaimer**: Vehicle models (`Hyundai i20`, `Hyundai Venue`, `Hyundai Creta`, `Hyundai Verna`) are used as hypothetical simulation categories only. The synthetic data and sensor parameters do not represent actual OEM manufacturing specifications.

---

## Dataset Architecture

- **Total Vehicles**: 10,000 unique vehicles (`VEH-00001` to `VEH-10000`)
- **Total Stations**: 10 stations per vehicle
- **Total Rows**: Exactly **100,000 rows** (10,000 vehicles × 10 stations)
- **Primary Data File**: `dataset.csv`
- **Vehicle Anomaly Rate**: ~6.4% (642 anomalous vehicles)

---

## Station Workflow ($S1 \rightarrow S10$)

1. **`S1`**: Body Preparation
2. **`S2`**: Underbody Assembly
3. **`S3`**: Suspension and Brake Assembly
4. **`S4`**: Powertrain Installation
5. **`S5`**: Electrical Installation
6. **`S6`**: Interior Assembly
7. **`S7`**: Door and Glass Assembly
8. **`S8`**: Wheel and Tyre Assembly
9. **`S9`**: Fluid Filling
10. **`S10`**: Final Inspection

---

## 24-Column Schema Specification

| # | Column Name | Data Type | Physical Unit | Description | Unmeasured Sensors |
|---|---|---|---|---|---|
| 1 | `vehicle_id` | `string` | N/A | Unique vehicle ID (`VEH-00001` .. `VEH-10000`) | |
| 2 | `vehicle_model` | `string` | N/A | Model category (`Hyundai i20`, `Venue`, `Creta`, `Verna`) | |
| 3 | `vehicle_variant` | `string` | N/A | Trim variant (`Era`, `Magna`, `Sportz`, `Asta`, `SX`, etc.) | |
| 4 | `station_id` | `string` | N/A | Station identifier (`S1` to `S10`) | |
| 5 | `station_name` | `string` | N/A | Human-readable station name | |
| 6 | `timestamp` | `datetime` | ISO 8601 | Station completion timestamp | |
| 7 | `cycle_time_sec` | `float` | seconds | Processing duration spent at station | |
| 8 | `torque_nm` | `float` | $N \cdot m$ | Fastening torque applied | `NaN` if unmeasured |
| 9 | `temperature_c` | `float` | $^\circ\text{C}$ | Operating process / tool temperature | `NaN` if unmeasured |
| 10 | `vibration_rms` | `float` | $g$ | RMS vibration level | `NaN` if unmeasured |
| 11 | `pressure_bar` | `float` | bar | Pneumatic / hydraulic pressure | `NaN` if unmeasured |
| 12 | `force_n` | `float` | $N$ | Pressing / joining force | `NaN` if unmeasured |
| 13 | `position_error_mm` | `float` | $mm$ | Alignment gap/flush deviation | `NaN` if unmeasured |
| 14 | `voltage_v` | `float` | $V$ | Electrical supply / testing voltage | `NaN` if unmeasured |
| 15 | `current_a` | `float` | $A$ | Operating current draw | `NaN` if unmeasured |
| 16 | `flow_rate_lpm` | `float` | $L/min$ | Fluid filling flow rate | `NaN` if unmeasured |
| 17 | `queue_time_sec` | `float` | seconds | Waiting time before station entry | |
| 18 | `ambient_temperature_c` | `float` | $^\circ\text{C}$ | Factory floor ambient temperature | |
| 19 | `humidity_pct` | `float` | $\%$ | Factory floor relative humidity | |
| 20 | `shift` | `string` | N/A | Production shift (`Shift_1`, `Shift_2`, `Shift_3`) | |
| 21 | `production_batch` | `string` | N/A | Weekly batch tracking code | |
| 22 | `rework_flag` | `integer` | $0$ / $1$ | Indicator if vehicle required rework | |
| 23 | `anomaly_flag` | `integer` | $0$ / $1$ | Operational fault indicator at this station | |
| 24 | `root_cause_station` | `string` | N/A | Ground-truth root cause station ID (`NONE` if normal) | |

> **Note on NaN Usage**: A global sensor column layout is used. If a sensor metric is not measured at a specific station, its value is **`NaN`** (never `0`).

---

## Fault Scenarios & Propagation Physics

1. **Local Station Fault**: Component or tool failure originating at a specific station (e.g. `S3` torque drop or `S9` coolant leak).
2. **Non-Deterministic Downstream Propagation**: An upstream fault (e.g., `S1` body prep misalignment) has a probabilistic chance (e.g., 55%) of propagating downstream to induce secondary deviations (e.g., `S7` door alignment fit errors).
3. **Synthetic Ground Truth (`root_cause_station`)**:
   - For normal vehicles: `root_cause_station` = `"NONE"`
   - For anomalous vehicles: `root_cause_station` = originating station (e.g. `"S1"`), even if downstream stations also show secondary `anomaly_flag = 1`.

---


