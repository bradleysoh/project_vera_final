# 📘 VERA-X 3nm G5-Nanosheet Process Design Manual
## Version 1.0 - Restricted Confidential

---

## 1. Executive Overview

The **VERA-X 3nm G5-Nanosheet** process represents the industry's first fifth-generation Gate-All-Around (GAA) technology node, specifically optimized for high-performance computing (HPC) and energy-efficient mobile AI applications.

### 1.1 Technology Node Specifications
- **Project Name:** VERA-X (G5-3N)
- **Node Classification:** 3nm Class (Sub-5nm Feature Sizes)
- **Architecture:** Gate-All-Around (GAA) / Nanosheet (NSFET)
- **Substrate:** 300mm Bulk Si with advanced Strain Engineering

### 1.2 Target Applications
- **AI/ML Accelerators:** High bandwidth, low voltage operations.
- **HPC Platforms:** Next-generation data center CPUs/GPUs.
- **5G/6G SoC:** Advanced RF/Digital integration.
- **Automotive (Grade 1):** Extended reliability and thermal cycling.

### 1.3 PPA Goals (v1.0 Baseline)
| Metric | Targeting Performance (vs 5nm Gen 3) |
| :--- | :--- |
| **Power (P)** | 45% reduction at iso-performance |
| **Performance (P)** | 28% increase at iso-power |
| **Area (A)** | 0.42x scaling (SRAM Density: 31.2 Mb/mm²) |

### 1.4 Reliability and Lifetime Targets
- **Operating Temperature:** -40°C to 125°C (Junction)
- **Lifetime:** 10 Years Continuous Workload
- **FIT Rate:** < 10 FIT per 100M Gates

---

## 2. Process Technology Description

### 2.1 FEOL Process Flow
1. **Shallow Trench Isolation (STI):** High-aspect ratio trench with ALD Oxide fill.
2. **Channel Stack Epitaxy:** Multi-layer Si/SiGe sacrificial superlattice deposition.
3. **Inner Spacer Formation:** Atomic Layer Etching (ALE) for precise cavity control.
4. **Source/Drain (S/D) Epitaxy:** In-situ doped Si:C (n-type) and SiGe:B (p-type).
5. **Sacrificial SiGe Removal:** Selective isotropic etching to release nanosheets.

### 2.2 BEOL Stack Structure
The BEOL stack consists of 16 metal layers (M0 to M15) using a combination of Cu, Ru, and Co.

| Layer Category | Layers | Material | Pitch (nm) | Dielectric |
| :--- | :--- | :--- | :--- | :--- |
| **Contact (M0)** | M0, M1 | Cobalt (Co) | 18 | Low-k |
| **Lower Global** | M2 - M7 | Copper (Cu) | 24-32 | Ultra Low-k (ULK) |
| **Upper Global** | M8 - M12 | Copper (Cu) | 48-120 | ELK |
| **Top/Power** | M13 - M15 | Al / Cu | 200+ | SiO2/SiN |

### 2.3 Lithography Strategy
- **EUV (13.5nm):** Single patterning for critical layers (M0-M4, Gate, VIA0-VIA3).
- **DUV (193i):** Multi-patterning (LELE / SADP) for upper global layers.
- **Overlay Budget:** < 1.5nm (3-sigma) for critical mask alignments.

---

## 3. Design Rules (Highly Detailed)

### 3.1 Minimum Width and Spacing (M1 - M4)
Standard cell routing relies on strict $G_{p}$ (Grid Pitch) alignment.

| Rule ID | Parameter | Value (nm) | Equation / Constraint |
| :--- | :--- | :--- | :--- |
| **W.1.M1** | Min Metal Width | 9 | $W_{min}$ |
| **S.1.M1** | Min Metal Space | 9 | $S_{min}$ |
| **W.2.VG** | Via Gate Enclosure | 3 | $E_{min} \ge 0.3 \times Via\_Diameter$ |

### 3.2 Double Patterning Rules (SADP)
For M2/M3 layers, Self-Aligned Double Patterning (SADP) is enforced to maintain CDU (Critical Dimension Uniformity).
- **Mandrel Pitch:** 44nm
- **Spacer Width:** 11nm
- **Non-Mandrel Width:** 11nm

### 3.3 Antenna Rules
Max ratio of metal area to gate area must satisfy:
$R_{antenna} = \frac{Area_{Metal}}{Area_{Gate}} \le 250$

---

## 4. Device Models

### 4.1 Nanosheet Transistor Parameters (Typical)
| Parameter | NMOS (L=12nm) | PMOS (L=12nm) | Units |
| :--- | :--- | :--- | :--- |
| **Effective Width ($W_{eff}$)** | $3 \times N_{sheets} \times (W_{sheet} + T_{sheet})$ | - | nm |
| **Threshold Voltage ($V_{th0}$)** | 0.28 / 0.35 / 0.42 | -0.26 / -0.33 / -0.40 | V |
| **Ion (mA/um)** | 1.85 | 1.62 | @ Vdd=0.75V |
| **Ioff (pA/um)** | 15 | 18 | @ Vdd=0.75V |

### 4.2 Multi-Vt Options
- **uLVT:** Ultra Low Vt (Speed optimized)
- **LVT:** Low Vt
- **SVT:** Standard Vt
- **HVT:** High Vt (Leakage optimized)

---

## 5. SRAM / Memory Compilers

### 5.1 Bitcell Architecture
- **6T-SRAM Configuration:** High Density (HD) and High Performance (HP) cells.
- **HD Cell Size:** 0.015 um²
- **HP Cell Size:** 0.021 um²

### 5.2 Layout Constraints
- **Orientation:** Vertical nanosheet orientation only.
- **Butted Contacts:** Prohibited in HD arrays.
- **Write Assist:** Negative Bit-Line (NBL) or Word-Line Boosting required for $V_{min} < 0.6V$.

---

## 6. IO and ESD Design

### 6.1 IO Voltage Domains
- **Core Voltage ($V_{DD}$):** 0.75V / 0.65V
- **IO Voltage ($V_{DDIO}$):** 1.2V / 1.8V
- **Analog Voltage ($V_{DDA}$):** 1.2V

### 6.2 ESD Protection Strategy
- **Clamp Architecture:** Dual-diode with snapback NMOS trigger.
- **CDM (Charged Device Model):** 500V target.
- **HBM (Human Body Model):** 2kV target.

### 6.3 Pad Pitch and Spacing
| Parameter | Value | Units |
| :--- | :--- | :--- |
| Min Pad Pitch | 40 | um |
| Pad Opening Size | 32 x 32 | um |
| Latch-up Guard Ring Width | 2.5 | um |

---

## 7. Reliability

### 7.1 Electromigration (EM) Limits
Current density limits are defined at $T_{j} = 110^{\circ}C$ for 10-year lifetime.
$J_{max} = A \cdot \exp\left(\frac{E_{a}}{k T}\right)$

| Metal Layer | $J_{max}$ (Avg) | $J_{peak}$ (RMS) | Units |
| :--- | :--- | :--- | :--- |
| M0 / M1 (Co) | 4.2 | 12.5 | mA/um² |
| M2 - M7 (Cu) | 1.8 | 6.2 | mA/um² |

### 7.2 Aging and Lifetime Models
- **BTI (Bias Temperature Instability):** Frequency degradation $\Delta F < 5\%$ over 10 years.
- **HCI (Hot Carrier Injection):** $V_{th}$ shift $\Delta V_{th} < 20mV$.
- **TDDB (Time-Dependent Dielectric Breakdown):** Area-scaled lifetime $t_{50} > 100$ years.

---

## 8. Packaging & Advanced Integration

### 8.1 3D Interface Specs (VERA-Link)
- **TSV Pitch:** 10.0 um
- **TSV Diameter:** 2.0 um
- **Microbump Pitch:** 25.0 um

### 8.2 Hybrid Bonding Mock Specs
- **Bonding Pitch:** 4.0 um (Cu-to-Cu)
- **Alignment Tolerance:** < 400nm (3-sigma)

---

## 9. Signoff Flow

### 9.1 DRC / LVS / PEX
- **DRC Engine:** PV-Verify v4.2+
- **LVS:** Schematic-to-Layout isomorphism check with parasitic extraction.
- **PEX Corners:** C-best, C-worst, RC-best, RC-worst, Typical.

### 9.2 Timing and IR Signoff
- **Setup/Hold Corner:** $SSG, 0.68V, -40^{\circ}C$ / $FFG, 0.82V, 125^{\circ}C$.
- **Static IR Drop:** Max 2.5% of $V_{DD}$ (approx. 18.75mV).
- **Dynamic IR Drop:** Max 5.0% of $V_{DD}$ (approx. 37.5mV).

---

## 10. Appendices

### 10.1 Acronym Glossary
- **BEOL:** Back End Of Line
- **CMP:** Chemical Mechanical Polishing
- **EUV:** Extreme Ultraviolet Lithography
- **FEOL:** Front End Of Line
- **GAA:** Gate-All-Around
- **LDE:** Layout Dependent Effects

### 10.2 Layer Naming Conventions
- **GT:** Gate Layer (GDS ID: 15.0)
- **NW:** N-Well (GDS ID: 1.0)
- **M1:** Metal 1 (GDS ID: 31.0)
- **VIA1:** Via 1 (GDS ID: 31.1)

### 10.3 Recommended EDA Flow
1. **Logic Synthesis:** V-Synth Pro
2. **Floorplan & Placement:** P-Architect
3. **Routing:** X-Route Extreme
4. **Physical Verification:** PV-Verify

---
*End of Document - VERA-X 3nm G5-Nanosheet*
