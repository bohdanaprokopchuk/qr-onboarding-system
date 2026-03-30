# QR Onboarding System

A research-oriented and application-ready QR code reading system for reliable decoding under challenging camera and environmental conditions.

The project extends conventional QR scanning with adaptive preprocessing, enhanced decoding strategies, ROI tracking, multi-frame stabilization, payload-aware optimization, and evaluation tooling. It is designed both as a practical computer vision solution and as a structured experimental platform for studying robust QR decoding pipelines.
The system follows a modular QR reading workflow and is organized as two complementary layers: an operational layer for practical decoding and live processing, and an experimental layer for benchmarking, ablation studies, validation, and analytical export.

---

## Overview

Standard QR readers perform well in clean conditions, but their reliability often drops when the input is degraded by low light, blur, weak contrast, reflections, unstable framing, small QR size, or complex backgrounds. This project addresses those limitations through a modular architecture that improves decoding robustness for both still images and live camera streams.

The repository combines:

- practical QR scanning workflows
- enhanced preprocessing and recovery logic
- comparative binarization support
- live camera adaptation and ROI tracking
- payload-level optimization and split QR workflows
- benchmarking, validation, and statistical evaluation utilities

This makes the project suitable for:

- research work in computer vision
- comparative evaluation of QR recovery pipelines
- experiments on difficult real-world camera input
- demonstration of robust QR decoding under adverse conditions

---

## Target Conditions

The system is intended to improve QR decoding performance in conditions such as:

- low illumination
- motion blur and defocus blur
- small QR code footprint in the frame
- perspective distortion
- glare and reflections
- low contrast
- background clutter
- unstable live camera input
- payload complexity that reduces decoding robustness

---

## Key Capabilities

- QR code reading from still images
- QR code reading from live camera streams
- baseline and enhanced decoding pipelines
- adaptive preprocessing and threshold control
- multiple binarization strategies for difficult scenes
- ROI tracking for frame-to-frame focus refinement
- multi-frame support for unstable live input
- payload-aware QR processing and optimization
- split QR generation and reconstruction workflows
- statistical evaluation and benchmarking utilities
- modular components for console, API, and integration scenarios

---

## Implemented Methods and Subsystems

### 1. Baseline and Enhanced QR Decoding Pipelines
The repository contains both a baseline pipeline and an enhanced pipeline for more difficult visual conditions, which supports direct comparison between standard and improved processing flows.

Implemented through:

- `pipeline.py`
- `enhanced_pipeline.py`
- `qr_decoder.py`

### 2. Image Preprocessing for QR Recovery
The system includes preprocessing logic designed to improve visual separability of QR structures before decoding.

Implemented through:

- `preprocessing.py`

### 3. Adaptive Threshold Selection
Threshold adaptation is used to better match changing input conditions instead of relying on a single fixed configuration.

Implemented through:

- `adaptive_thresholds.py`

### 4. Comparative Binarization Support
A dedicated binarization subsystem enables experiments with multiple thresholding strategies and comparative evaluation on difficult inputs.

Implemented through:

- `binarization.py`
- `tools/benchmark_binarization_methods.py`

### 5. Adaptive Camera Runtime Logic
For live scanning scenarios, the project includes adaptive camera handling intended to improve acquisition stability and real-time decoding robustness.

Implemented through:

- `camera.py`
- `adaptive_camera.py`

### 6. ROI Tracking Pipeline
To reduce unnecessary full-frame processing and maintain focus on the relevant visual region, the system includes region-of-interest tracking across frames.

Implemented through:

- `roi_tracking.py`

### 7. Multi-Frame Decoding Support
When a single frame is insufficient for reliable decoding, the system can use information across multiple frames to improve practical scan success.

Implemented through:

- `multi_frame.py`

### 8. Payload-Aware QR Optimization
The project includes payload-oriented logic for preparing, optimizing, and evaluating encoded data at the payload level in addition to the image level.

Implemented through:

- `payload_optimizer.py`
- `payload_codecs.py`

### 9. Split QR Workflows
The repository supports split QR generation and reconstruction workflows for structured or extended payload delivery experiments.

Implemented through:

- `split_qr.py`
- `qr_generation.py`
- `tools/generate_split_qr.py`

### 10. Evaluation and Statistical Assessment
The system includes research-oriented evaluation and benchmark tooling for measuring, validating, and comparing pipeline behavior.

Implemented through:

- `evaluation.py`
- `pipeline_stats.py`
- `tools/benchmark_pipeline.py`
- `tools/evaluate_research_pipeline.py`
- `tools/run_validation_suite.py`

### 11. Research Extensions and ML-Related Components
The project also includes ML-related and research enhancement modules for experimentation with advanced recovery or classification logic.

Implemented through:

- `ml_models.py`
- `tools/train_research_enhancer.py`

### 12. Integration-Oriented Components
Beyond local experimentation, the repository contains modules that support console usage, web integration, cloud-related workflows, persistence, and provisioning logic.

Implemented through:

- `desktop_console.py`
- `web_api.py`
- `cloud_service.py`
- `persistence.py`
- `provisioning.py`
- `consent.py`
- `crypto_utils.py`

### 13. Experimental Benchmarking
The project includes a separate experimental benchmarking component for evaluating QR decoding performance across static-image, streaming, and split-QR scenarios. It supports method comparison, ablation studies, metric aggregation, and export of final result tables and plots.

Implemented through:

- `tools/run_static_benchmark.py`
- `tools/run_stream_benchmark.py`
- `tools/run_split_benchmark.py`
- `tools/aggregate_benchmarks.py`
- `tools/export_final_tables.py`

## Experimental Layer

Alongside the main QR decoding pipeline, the project includes an experimental layer for systematic testing and comparison of different processing configurations.

Its role is not only to check whether a QR code can be decoded, but also to examine how different parts of the system affect the result under difficult conditions such as blur, low light, unstable framing, and payload complexity. This makes the repository useful not only as an implementation, but also as a research environment for controlled evaluation.

### What the Experimental Layer Covers

The experimental layer supports:

- comparison of baseline and enhanced decoding pipelines
- evaluation of preprocessing and thresholding methods
- ablation analysis of adaptive subsystems
- benchmarking for static-image, stream, and split-QR scenarios
- aggregation of final metrics and export of result artifacts

### Benchmark Directions

The latest stage of the project includes three main benchmark directions.

#### 1. Static-image benchmark
This benchmark evaluates decoding quality on still images under controlled degradation scenarios. It is used to compare methods under blur, low illumination, contrast loss, glare, perspective distortion, and similar visual challenges.

#### 2. Stream benchmark
This benchmark evaluates performance on sequential camera-like input, where decoding depends not only on image quality, but also on temporal stability, motion, noise, and ROI persistence. It is intended to reflect more realistic live scanning conditions.

#### 3. Split-QR benchmark
This benchmark evaluates workflows in which payload data is distributed across multiple QR fragments and then reconstructed. It extends the project beyond standard single-code decoding and supports experiments with structured payload delivery.

### Comparative and Ablation Evaluation

The experimental layer is designed to compare multiple decoding modes rather than only one final pipeline. In the current evaluation setup, the project benchmarks the following configurations:

- `raw_pyzbar`
- `raw_opencv`
- `raw_combined`
- `fixed:proposed_integral`
- `adaptive_full`
- `adaptive_no_roi`
- `adaptive_no_ml`
- `adaptive_no_switch`
- `adaptive_no_quality`
- `adaptive_ml_only`

### Place of the Experimental Layer in the Architecture

From an architectural perspective, the repository consists of two connected parts:

- **Operational layer**  
  handles image acquisition, preprocessing, localization, decoding, ROI tracking, and live execution

- **Experimental layer**  
  handles benchmarking, ablation analysis, metric aggregation, validation workflows, and export of final results

### Experimental Workflow

The experimental workflow follows these steps:

1. prepare datasets and manifests for static, stream, or split scenarios  
2. run benchmark scripts for selected decoding modes  
3. collect raw CSV results for each benchmark type  
4. aggregate the results into summary tables  
5. export tables and plots for analysis and presentation

### Final Outputs

The exported outputs include:

- `overall_metrics.csv`
- `per_scenario_metrics.csv`
- `per_severity_metrics.csv`
- `split_metrics.csv`
- `stream_metrics.csv`
- plots in `.png` format

The final aggregation stage also prepares result directories such as `final_ready` and `final_export`.

### Research Value

The experimental layer strengthens the research value of the project because it supports:

- systematic comparison of recovery strategies
- controlled evaluation under difficult visual conditions
- ablation-based analysis of subsystem contribution
- experiments with advanced payload and split-QR workflows

---

## Repository Structure

```text
qr_onboarding_system/
│
├── assets/
│   ├── datasets/                  # Datasets for experiments and batch evaluation
│   ├── test_qr_set/               # Test QR samples
│   └── results/                   # Exported outputs and evaluation artifacts
│
├── src/
│   └── qr_onboarding/
│       ├── __init__.py
│       ├── adaptive_camera.py
│       ├── adaptive_thresholds.py
│       ├── binarization.py
│       ├── camera.py
│       ├── cli.py
│       ├── cloud_service.py
│       ├── consent.py
│       ├── crypto_utils.py
│       ├── desktop_console.py
│       ├── enhanced_pipeline.py
│       ├── evaluation.py
│       ├── ml_models.py
│       ├── models.py
│       ├── multi_frame.py
│       ├── overlay.py
│       ├── payload_codecs.py
│       ├── payload_optimizer.py
│       ├── persistence.py
│       ├── pipeline.py
│       ├── pipeline_stats.py
│       ├── preprocessing.py
│       ├── provisioning.py
│       ├── qr_decoder.py
│       ├── qr_generation.py
│       ├── roi_tracking.py
│       ├── split_qr.py
│       └── web_api.py
│
├── tests/
│   ├── conftest.py
│   ├── test_adaptive_extensions.py
│   ├── test_binarization_methods.py
│   ├── test_camera_runtime.py
│   ├── test_cloud_and_provisioning.py
│   ├── test_consent_and_persistence.py
│   ├── test_enhanced_pipeline.py
│   ├── test_ml_models.py
│   ├── test_new_improvements.py
│   ├── test_payload_codecs.py
│   ├── test_persistent_cloud_service.py
│   ├── test_research_payloads.py
│   ├── test_split_qr.py
│   └── test_web_api_e2e.py
│
├── tools/
│   ├── benchmark_binarization_methods.py
│   ├── benchmark_pipeline.py
│   ├── evaluate_research_pipeline.py
│   ├── generate_compat_qr.py
│   ├── generate_split_qr.py
│   ├── mock_cloud_context_server.py
│   ├── run_validation_suite.py
│   └── train_research_enhancer.py
│
├── README.md
├── requirements.txt
└── run_desktop_console.bat
```

---

## Architecture Summary

The system follows a modular QR reading workflow decomposed into distinct stages:

1. **Image acquisition**  
   Static image input or live camera frame capture

2. **Preprocessing**  
   Frame enhancement and preparation for downstream recovery

3. **Thresholding and binarization**  
   Adaptive and comparative strategies for improving QR separability

4. **Localization and ROI refinement**  
   Region selection and frame-to-frame tracking

5. **Decoding**  
   Baseline or enhanced QR extraction logic

6. **Temporal support**  
   Multi-frame stabilization and aggregation behavior

7. **Payload-level handling**  
   Encoding, optimization, splitting, and compatibility workflows

8. **Evaluation and benchmarking**  
   Statistical assessment, validation, and comparative measurement

This separation makes the repository suitable both for practical use and for research-grade experimentation.

---

## Prerequisites

Before running the project, make sure the following are available:

- Python 3.10 or newer
- a working virtual environment tool such as `venv`
- a webcam if you plan to test live camera functionality
- system support required by the QR decoding backend used by the project, if applicable on your platform

If you work on Windows, run all commands from the project root after activating the virtual environment.

---

## How to Get the Project

### 1. Clone the repository

```bash
git clone git@github.com:bohdanaprokopchuk/qr-onboarding-system.git
cd qr-onboarding-system
```

### 2. Create and activate a virtual environment

#### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

#### Windows CMD

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

#### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the project source path

This repository uses a `src`-based layout. Before running the project, make sure Python can see the `src` directory.

#### Windows PowerShell

```powershell
$env:PYTHONPATH = "src"
```

#### Windows CMD

```cmd
set PYTHONPATH=src
```

#### Linux / macOS

```bash
export PYTHONPATH=src
```

### 5. Verify the installation

```bash
python -m qr_onboarding.cli --help
```

If this command runs successfully, the project is installed correctly.

---

## Quick Start

### Show CLI help

```bash
python -m qr_onboarding.cli --help
```

### Run batch processing on a dataset

#### Windows PowerShell or CMD

```powershell
python -m qr_onboarding.cli batch ".\assets\datasets" --pattern "**\*.png" --csv ".\results\batch_results.csv"
```

#### Linux / macOS

```bash
python -m qr_onboarding.cli batch ./assets/datasets --pattern "**/*.png" --csv ./results/batch_results.csv
```

### Run the desktop console on Windows

```cmd
run_desktop_console.bat
```
or

```
.\run_desktop_console.bat
```

Expected result:

- the CLI should execute without import errors
- batch processing should generate a CSV report at the path you specify
- desktop mode should open the local console interface for interactive use

---

## Main Entry Points

The most important modules are:

- `src/qr_onboarding/cli.py` for command-line execution
- `src/qr_onboarding/pipeline.py` for the baseline pipeline
- `src/qr_onboarding/enhanced_pipeline.py` for the enhanced pipeline
- `src/qr_onboarding/camera.py` and `src/qr_onboarding/adaptive_camera.py` for camera workflows
- `src/qr_onboarding/roi_tracking.py` for ROI-based frame refinement
- `src/qr_onboarding/evaluation.py` and `src/qr_onboarding/pipeline_stats.py` for evaluation logic
- `tools/run_validation_suite.py` for structured validation

---

## Usage

### Still-image experiments

1. Place test images into `assets/datasets` or `assets/test_qr_set`
2. Run the CLI pipeline or a benchmarking script
3. Save outputs into `assets/results` or `results`
4. Compare success rate, decoding stability, and exported metrics

### Live camera experiments

1. Connect a camera
2. Launch the camera-enabled workflow or desktop console
3. Observe how adaptive runtime logic, ROI tracking, and multi-frame support affect decoding quality
4. Record performance under different lighting, motion, and framing conditions

### Research evaluation

1. Prepare a structured dataset
2. Run benchmark and evaluation utilities from `tools`
3. Export metrics and CSV summaries
4. Compare preprocessing, thresholding, and decoding configurations

---

## Research and Utility Scripts

The `tools` directory contains scripts for experiments, benchmarking, validation, and dataset-oriented workflows.

### Available scripts

#### `benchmark_binarization_methods.py`
Compares binarization methods on selected datasets or samples.

#### `benchmark_pipeline.py`
Benchmarks the end-to-end QR pipeline.

#### `evaluate_research_pipeline.py`
Runs evaluation logic for research-oriented analysis.

#### `generate_compat_qr.py`
Generates QR samples for compatibility-related testing.

#### `generate_split_qr.py`
Generates split QR payloads for structured experiments.

#### `mock_cloud_context_server.py`
Provides a mock environment for testing cloud-related workflows.

#### `run_validation_suite.py`
Runs a structured validation workflow across the project.

#### `train_research_enhancer.py`
Supports ML-oriented or research enhancement experiments.

---

## Running Tests

Run all tests from the project root:

```bash
pytest
```

Minimal output mode:

```bash
pytest -q
```

If import errors occur, make sure the `src` directory is visible through `PYTHONPATH`.

---

## Testing Scope

The repository includes automated tests for:

- adaptive extensions
- binarization methods
- live camera runtime behavior
- cloud and provisioning workflows
- consent and persistence logic
- enhanced pipeline behavior
- ML-related modules
- payload codecs
- split QR workflows
- end-to-end web API behavior

This improves reproducibility and supports engineering validation.

---

## Relevance

This project demonstrates not only a working QR decoding system, but also a structured experimental framework for evaluating robustness under difficult real-world conditions. The repository combines implementation, modular architecture, testing, and evaluation utilities in a way that supports both practical demonstration and comparative analysis.

The value comes from:
- extending standard QR decoding with enhanced recovery logic
- evaluating performance under adverse camera conditions
- comparing preprocessing and binarization strategies
- integrating ROI tracking and multi-frame stabilization
- supporting reproducible validation through scripts and automated tests

---

## Troubleshooting

### `ModuleNotFoundError: No module named qr_onboarding`
Set the project source path before running commands:

#### Windows PowerShell

```powershell
$env:PYTHONPATH = "src"
```

#### Windows CMD

```cmd
set PYTHONPATH=src
```

#### Linux / macOS

```bash
export PYTHONPATH=src
```

### CLI works, but the camera is unstable
Check the following:

- no other application is using the camera
- lighting is sufficient for live testing
- the camera is connected and available to the operating system
- you are launching the workflow from the project root with the virtual environment activated

### Dependencies install, but decoding fails on some systems
This can happen if the platform is missing external runtime support required by the QR decoding backend. In that case, verify that all project dependencies and any platform-specific decoder requirements are installed correctly.

---

## Conclusion

QR Onboarding System provides a modular and extensible framework for enhanced QR code reading from images and live camera streams. It combines practical computer vision methods with evaluation, benchmarking, and research tooling, making it suitable for real-world experimentation of robust QR decoding under challenging conditions.
