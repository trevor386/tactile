# Sensor and terrain calibration: literature review (2026-10-08)

Purpose: calibrate `somato/sensors/tactile.py` (FSR, capacitive), `sim/contact_model.py` and
`configs/terrains/ice_forms.yaml` for a 16-link snake (link 0.10 m, r 0.025 m, 0.25 kg, 2 mm skin, 24 taxels of
1 cm^2 per link: 8 around x 3 rings, so ~20 mm x ~33 mm pitch) sidewinding on ice and snow.
Recommendations only; nothing in the configs was changed.

Evidence tags: **[D]** manufacturer datasheet; **[M]** measured, read in the paper text; **[S]** seen only in an
abstract or search-result summary, not checked at the source; **[E]** my estimate or arithmetic, not from a source.
Where a number is given without a tag it is [M] or [D] from the cited source.

## 1. FSR skins (Interlink FSR 400, Tekscan, Velostat)

| quantity | value | conditions | source |
|---|---|---|---|
| hysteresis | +10 % avg, (RF+ - RF-)/RF+ | FSR 400, 1 kg [D] | [FSR 400 sheet PDS-10004-C](https://www.teachengineering.org/content/uconn_/activities/uconn-2662-contractions-calculations-force-resistance-activity/uconn-2662-contractions-calculations-FSR400series-datasheet.pdf) |
| hysteresis | 7.6 to 17 % of full-scale output over 16 sensors (maker claims < 4.5 %); modelled as **rate-independent** (Preisach) | Tekscan A201-1, 4.5 N triangles, up to 22.6 N/s, 25 C [M] | [Paredes-Madrid 2018](https://www.redalyc.org/journal/496/49657889025/49657889025.pdf) |
| hysteresis | 12 to 17 % (mean 15 %) | piezoresistive hybrid sensor, 0-15 N at ~0.7 N/s [S] | [arXiv 2304.06204](https://arxiv.org/pdf/2304.06204) |
| rise time | < 3 us (steel-ball test) vs **1-2 ms "mechanical"** | FSR 400 sheet [D]; integration guide [D] | datasheet above; [Integration Guide](https://courses.media.mit.edu/2022spring/mas836/readings/fsrguide.pdf) |
| step response | 4.8 ms to 50 %, 34.5 ms 10-90 % rise (loading only) | rigid sphere dropped on sensor [S] | arXiv 2304.06204 |
| response, Velostat mat | 0.21 s (other materials 0.42-0.50 s) | pneumatic pressure cycles [S] | [Martinez-Cesteros 2025](https://zaguan.unizar.es/record/165278) |
| unloading/recovery | no measured FSR time constant found; vendor "recovery < 15 ms" (test undefined) [S] | | [Pimoroni listing](https://shop.pimoroni.com/en-us/products/fsr-force-sensing-resistor-sensor) |
| drift under load | < 5 % per log10(time); 2.5 kg for 24 h: -5 % in R | 1 kg, 35 days [D] | FSR 400 sheet |
| drift, Velostat | decays asymptotically; ~5.8 % per log(min) over 3 h at 8.1 N [S] | 4x12 mm elements | search summary, source not pinned |
| creep model | Burgers + tunnelling; time constant 500 s was **assumed, not fitted**; 1 h drift only in figures | FlexiForce, FSR 402, 16 each, 2-10 N [M] | [Paredes-Madrid 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5706281/) |
| history memory | 100 g reads 10 kOhm; after hours under 5 kg it reads 6 kOhm and creeps back toward 10 kOhm | [D] | [Sensitronics datasheet](https://cdn.hackaday.io/files/1653337073607072/Sensitronics%20FSR%20Datasheet.pdf) |
| unit spread | +/-2 % single part, +/-6 % part-to-part within one batch | FSR 400 [D] | FSR 400 sheet |
| unit spread | +/-15 to 25 % of nominal resistance part-to-part; single part +/-2-5 % | repeatable actuation [D] | Integration Guide |
| unit spread | sensitivity/hysteresis/drift CV compared across 16 FSR402 and SP200; values only in paper | [S] | [Palacio-Gomez 2022](https://czasopisma.pan.pl/dlibra/publication/142267/edition/124544/content) |
| temperature | -40 C after 1 h: -5 % R; +85 C: -15 % R; usable -30 to +70 C | FSR 400 [D]; guide [D] | datasheet; Integration Guide |
| curvature | bending pre-loads the sensor: reduced range and resistance drift | [D] | Integration Guide |
| power law | response "approximately inverse power law (roughly 1/R)" | FSR 402, 10 mm PU tip [D] | Integration Guide |

Plausibility of the model values:
* `tau_load` 2 ms: consistent with the 1-2 ms mechanical rise time (a first-order tau would be 0.5-1 ms).
  (Editor's correction: tactile readings are sampled at 500 Hz, i.e. 10 samples per 20 ms latent step, and stage 1
  sees all of them; the sensor model runs at that rate. So a 2 ms time constant is not inert; it acts at the
  sample scale.)
* `tau_unload` 20 ms: inside the observed range (< 15 ms vendor, 34.5 ms rise, 0.21 s Velostat) but **unmeasured**
  for FSRs; a 2 mm elastomer skin adds its own recovery (Q3).
* `creep_frac` 6 % / `creep_tau` 1.5 s: the right sign and size for 1-3 s contacts (~5 % per decade of time);
  the real drift is logarithmic and keeps going, a single saturating exponential does not.
* `gain_spread` 0.15 (sigma of ln gain): matches the +/-15-25 % guide; the single-batch datasheet is +/-6 %.
  `fsr_degraded.yaml` (0.35) is beyond anything published for Interlink parts (plausible only for DIY Velostat).
* Missing: **rate-independent hysteresis** of 7-17 % full scale, which varies a lot from sensor to sensor.
  The model's loading/unloading asymmetry is a rate-dependent (viscous) loop, a different thing.

## 2. Capacitive and soft-elastomer skins (Ecoflex, PDMS, Dragon Skin)

| quantity | value | conditions | source |
|---|---|---|---|
| iCub capacitive skin | hysteresis ~5 % of range (9.1 fF at 28.6 kPa); 2.5 fF/kPa over 2-45 kPa; taxel 12.6-15.2 mm^2, 2 mm dielectric; "relaxation tau = 1 h 18 min" | 15 cycles/taxel [M] | [Maiolino et al.](https://arxiv.org/abs/1411.6837) |
| thermal compensation | iCub skin uses 2 thermal taxels, tested 15-40 C; no drift number given | [M] | same |
| hysteresis, silicone capacitive | 4.7 % (folded silicone, repeatability 3.4 %); 6.7 % over 15 kPa (graded PDMS); 9.1 % (carbon-PDMS) | [S] | search summaries (no pinned URLs) |
| creep, pristine Ecoflex | "negligible creep" in a capacitive strain sensor; BaTiO3-filled composite creeps | [S] | [Univ. Bath](https://researchportal.bath.ac.uk/en/publications/studying-the-creep-behaviour-of-strechable-capacitive-sensor-with/) |
| relaxation, Ecoflex | considerable only in virgin (never stretched) samples; slow decay > 12 h; temperature-sensitive between -40 and 140 C | [S] | [Polymer Testing 2020](https://pure.southwales.ac.uk/en/publications/a-comprehensive-thermo-viscoelastic-experimental-investigation-of/) |
| PDMS transitions | Tg about -125 C; melting about -47 C; crystallization near -50 C raises modulus | DMA/DSC [M] | [Cannon app note](https://cannoninstrument.com/media/assets/product/documents/App-Note-DSC-and-DMA-Measurements-of-Silicone-Rubber.pdf); [Sandia Sylgard 184](https://www.sandia.gov/polymer-properties/sylgard-184-shear-modulus-vs-temp/) |
| capacitance vs T | Dragon Skin / Smooth-Sil sensors: C rises from -40 to 0 C, falls from 0 to 80 C; RH also matters [S] | 5-95 % RH | [MRS 2022 abstract](https://mrs.org/meetings-events/presentation/2022_mrs_fall_meeting/2022_mrs_fall_meeting-3784239) |
| permittivity tempco | silicone **oil** about 1000 ppm/K (upper-bound reference for elastomers) [S] | patent | [US 4831492](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/4831492) |
| modulus at -20 C | no measurement found. Above the crystallization range the rubber plateau is flat; ideal-rubber scaling (E ~ T) gives about -14 % from +20 to -20 C [E] | | |
| FSR at cold | -5 % resistance at -40 C (see Q1) [D] | | FSR 400 sheet |

Sub-zero summary (-10 to -20 C): silicone elastomers stay rubbery (Tg about 100 K below). The documented
stiffening starts near -40 to -50 C and is crystallization, so a -20 C modulus change of tens of percent
at most is expected [E]. The bigger sub-zero effects are probably capacitive baseline shift (T and RH), frost or
meltwater on the surface, and cure/aging state of the silicone, not the bulk modulus. Fresh platinum-cure
silicones also stiffen for ~1 day after casting ([S] a Dragon Skin 30 study); calibrate on cured samples.

Plausibility: `modulus` 0.2 MPa is a Dragon Skin 20-30 class bulk value (Ecoflex 00-30 is 0.02-0.08 MPa [S]). For
a **bonded** 2 mm layer the effective compression modulus is larger: E_c ~ E(1 + 2 S^2) with S the shape factor
(Gent; [S] [overview of the pressure method](https://avesis.metu.edu.tr/yayin/6e20f5c0-fdbb-4ede-b554-56849d301305/a-new-formulation-for-the-analysis-of-elastic-layers-bonded-to-rigid-surfaces)),
about 2x for a 3 mm wide ground strip and 4x for a full 10 mm taxel [E]. `visco_frac` 0.2 is about 2x the
measured silicone value (Q3).

## 3. Skin mechanics: viscoelastic contact of a soft skin on hard ground

| quantity | value | conditions | source |
|---|---|---|---|
| silicone relaxation | Prony g_i = 0.015, 0.044, 0.029 at tau_i = 25, 150, 300 ms (**total 8.8 %**); "stabilized immediately" | GLS-40 silicone, Shore A 11, 20-40 % strain [M] | [Cabibihan et al. 2009](https://arxiv.org/pdf/0909.3559) |
| polyurethane relaxation | g_i = 0.167, 0.158, 0.113 at tau_i = 0.1, 1.38, 25.5 s (**total 44 %**) | Shore A 45 [M] | same |
| loop area | silicone 0.17 N mm at 4 N (about 8 % of input work [E]); polyurethane 0.29; human fingertip 2.26 | plate indentation [M] | same |
| bulk PDMS | loss modulus small against storage modulus for Sylgard 184 at room temperature; tan delta independent of cure/aging [S] | DMA 0-40 C | [Johnston et al., Herts](https://uhra.herts.ac.uk/id/eprint/3211/) |
| contact on hard ground | 3 N on a 0.10 m, r 25 mm link, 2 mm skin, E_c 0.2-1 MPa: indentation 25-80 um, strip 2.3-3.9 mm wide, **mean pressure 8-13 kPa** [E] | thin-layer + line-contact arithmetic | consistent with the 10-20 kPa figure in the task |

First-order filter per taxel: a reasonable approximation for the **small-signal, short-time** response, with
two corrections: (i) one tau is a poor fit to a 25-300 ms spectrum plus a minutes-long tail (use 2-3 terms;
`tau_visco` 0.3 s is the long end of the measured silicone spectrum); (ii) it is lossless for rate-independent
hysteresis. Lateral coupling can be ignored: load spreads over roughly the skin thickness (2 mm) while taxel pitch
is 20-33 mm, so per-taxel independence is acceptable. Footprint nonlinearity comes from geometry (`spread` 1 mm is
about 0.5 x thickness, consistent), not from skin dynamics.

**Texture filtering by the skin (estimate [E], elasticity argument, not from a source).** A pressure ripple of
wavelength L at the skin surface is attenuated at the sensing plane by roughly exp(-2 pi t / L), t = 2 mm; a 10 mm
taxel then averages it by roughly |sinc(pi a / L)| (a = 10 mm):

| L (mm) | 2 | 4 | 6 | 10 | 15 |
|---|---|---|---|---|---|
| skin attenuation | 0.002 | 0.04 | 0.12 | 0.28 | 0.43 |
| taxel averaging | 0 | 0.13 | 0.17 | 0 | 0.41 |
| visible fraction | ~0 | 0.005 | 0.02 | ~0 | 0.18 |

`contact_model.py` multiplies pressure by 1 + A * texture(x_k, y_k), sampled at the taxel centre with no spatial
averaging and no skin filter; for the configured 2-6 mm wavelengths the visible amplitude is 1-2 orders of
magnitude smaller than A. Temporally, the stimulus is averaged over each *sensor sample period*, which is 2 ms for
tactile (500 Hz), not the 20 ms latent period. (Editor's correction: the earlier numbers here assumed 20 ms.) A 2 ms
boxcar keeps ~70 % of a 225 Hz vibration (f = v/L for L 2 mm at v 0.45 m/s) and ~97 % at 75 Hz. The spatial
attenuation above is therefore the dominant missing effect, not the temporal one [E].

## 4. Ice and snow friction for a slowly sliding soft skin

| slider / surface | mu | conditions | source |
|---|---|---|---|
| tire-tread rubber on ice | -5 C: 0.14-0.37; -10 C: 0.22-0.53; -13 C: 0.37-0.73 | 0.65 m/s, 0.15-0.45 MPa, 3 compounds, 4 ice types [M] | [Lahayne et al. 2016](https://link.springer.com/article/10.1007/s11249-016-0665-z) |
| rubber on ice | 0.85 at -25 C, 0.005 m/s; ~0.30 for 0.003-2.6 m/s over -33 to -1 C; 0.08 at -0.1 C, ~3 m/s | [S] via review | [Lever et al. 2021](https://www.frontiersin.org/journals/mechanical-engineering/articles/10.3389/fmech.2021.690425/full) |
| mechanism map | near 0 C or fast: water/amorphous film, friction = viscoelastic term; below about -10 C and slow: interfacial shear or ice fragments on rubber, approaching ice-on-ice for T < -20 C; mu > 1 below -20 C is reported elsewhere [S] | review text [M] | [Persson and Xu 2025](https://arxiv.org/abs/2507.18782) |
| speed dependence | friction falls steeply from -10 to 0 C; at -5 C nearly constant over 1-10 cm/s [S] | tread rubber | [KAKEN record](https://kaken.nii.ac.jp/grant/KAKENHI-PROJECT-06650168) |
| low-speed regime | 3 um/s-1 cm/s, -38 to -2 C: ice fragments stuck to rubber matter at low T and low speed; thin premelt film above -10 C or at speed [S] | tread block | [Tada et al. 2023](https://link.springer.com/article/10.1007/s40544-022-0715-5) |
| tires, braking on bare ice | 0.09 (car), 0.06 (truck) at -6 C; locked wheel on ice usually taken as 0.10 | field tests [S] | [SAE 960959](https://saemobilus.sae.org/papers/tire-ice-friction-values-960959) |
| tires on packed snow | 0.35 (car), 0.23 (truck) at -6 C (winter test program, attribution unverified); table value 0.30 static, 0.20 kinetic | [S] | [SAE 960652](https://saemobilus.sae.org/papers/vehicle-traction-experiments-snow-ice-960652); [UAF table](https://ffden-2.phys.uaf.edu/211_fall2002.web.dir/ben_townsend/StaticandKineticFriction.htm) |
| non-rubber sliders on snow | PE sleds 0.03-0.08 (2-3 m/s); glass 0.30; acrylic 0.61; at 4.6 kPa, 69 mm/s, -2 C | [M]/[S] | Lever et al. 2021 |
| fresh vs older snow | fresh, cold snow raises friction (more direct contact, ploughing); old warm dense snow lowers it | Colbeck, as summarised [S] | [Colbeck 1992](https://erdc-library.erdc.dren.mil/items/81b728f7-5e2d-4ef8-e053-411ac80adeb3); [Ski wax](https://en.wikipedia.org/wiki/Ski_wax) |
| static vs kinetic | static friction on ice/snow rises with dwell time ("freezedown"); high static, low kinetic for skis on compacted snow; for rubber mu_s > mu_k via thermally activated aging | [S], attribution of the rubber statement unverified | [SIPRE TR 17](https://erdc-library.erdc.dren.mil/jspui/bitstream/11681/6012/1/SIPRE-Technical-Report-17.pdf); [Persson](https://arxiv.org/pdf/cond-mat/0005531) |
| stick-slip | low sliding speeds give relaxation oscillations, frequency scales with sqrt(k_tangential/m); no rubber-on-ice measurement found | [S] | [UTwente thesis](https://research.utwente.nl/en/publications/stick-slip-behaviour-in-sliding-rubber-contacs/) |
| ice roughness | self-affine (Hurst exponent ~1) over 10 um-150 mm: no preferred wavelength; roughness lowers rubber mu on the tested ices | [M] | Lahayne et al. 2016 |

No PDMS or silicone-elastomer-on-ice measurement was found. Hydrophobic counterfaces show lower friction than
hydrophilic ones (Persson and Xu 2025); a silicone skin is therefore likely at the low end of the rubber data
(unverified inference [E]).

Snow sinkage. Bekker parameters for snow vary by more than an order of magnitude: Wong (2001) sets n = 1.6, 1.6,
1.44, kc = 4.37, 2.49, 10.55 kN/m^(n+1), k_phi = 196.7, 245.9, 66.1 kN/m^(n+2)
([Lines, Elliott, Ray, ISTVS 2021](https://par.nsf.gov/servlets/purl/10304353), Table 1, [M]); a one-day-old-snow
tracked-vehicle set has n = 1.01, k_phi ~ 1244 kN/m^3 [S]. Sinkage of a rigid cylinder (R 25 mm, L 0.1 m, 3 N, width
2 sqrt(2 R z)) solved from p = (kc/b + k_phi) z^n [E]: **1.2-1.4 mm** for the stiff set, **11-18 mm** for the Wong
sets, with mean contact pressure only 0.5-2 kPa (versus 8-13 kPa on ice or concrete) because the footprint
grows as the link sinks. Forcing a fixed 10-20 kPa instead would give ~8-15 mm (stiff set) to 7-17 cm (Wong sets,
loose deep snow) [E]. A CRREL study finds
bearing strength rises and critical sinkage falls as power laws of density (0.3-0.6 g/cm^3) [S]
([Abele](https://erdc-library.erdc.dren.mil/jspui/handle/11681/5815)); none of the sources gives sinkage for
powder (50-100 kg/m^3), which sinks far more [E].

Sidewinding caveat [E]: the contact points that support the body are largely static relative to the ground, so
the static coefficient and its dwell dependence matter, and the **slip speed** is likely 0.01-0.2 m/s, below
the 0.45 m/s progression speed. Verify the tangential taxel velocity in the sim logs before choosing the
speed regime.

## 5. Tactile signal content on ice and snow

| item | finding | source |
|---|---|---|
| Jiang et al. 2024 (ICRA) | COBRA snake, 11 joints, **207 virtual normal-pressure sensors**, Perlin-noise terrain 16 x 16 m, hierarchical RL + CPG; simulation only (sim-to-real listed as future work); whole-body contact detection slows the simulator. No ice or snow, no sensor dynamics. | [arXiv 2312.03225](https://arxiv.org/pdf/2312.03225) [M] |
| TRACEPaw (silicone paw) | camera + microphone; six terrains, **snow, gravel and sand least distinguishable**; best small model 0.779 cross-validated | [arXiv 2311.03855](https://arxiv.org/abs/2311.03855) [S] |
| Multimodal Bayesian inference | icy crossing: without a measured friction coefficient the baseline labelled ice as concrete or snow and fell in 3/3 trials; static gait when expected mu <= 0.25 | [arXiv 2402.05872](https://arxiv.org/pdf/2402.05872) [S] |
| Legged slip/friction | HMM slip estimator on ANYmal on ice; online friction identification | [IROS 2019](https://research-collection.ethz.ch/bitstream/handle/20.500.11850/355281/IROS19___Dynamic_Locomotion_on_Slippery_Ground.pdf), [arXiv 2502.16843](https://arxiv.org/pdf/2502.16843) [S] |
| Proprioceptive classification | Spot: ~97 % on three terrains (labels not checked); RHex/AQUA: snow, ice, linoleum tested without published confusion figures | [arXiv 2508.16504](https://arxiv.org/html/2508.16504v1), [RHex terrain ID](https://kodlab.seas.upenn.edu/uploads/Aaron/terrain-id-SPIE2013.pdf) [S] |
| Vibration on snow | boot accelerometers while skiing: peaks 5-30 Hz, content above 15-20 Hz disappears as snow softens | [Alpine skiing vibration abstract](https://sponet.de/sponet/Record/3042433) [S] |
| Snake on snow/ice | EELS (JPL/CMU) field-tested on snow and ice; no tactile skin | [Science Robotics](https://authors.library.caltech.edu/records/mxgh8-2xa15) [S] |

No study of tactile-skin (or foot-tactile) terrain classification on ice or snow, and no vibration spectrum of
rubber or silicone sliding on ice or snow, was found. The simulated-signal literature for snakes (Jiang et al.)
uses plain normal pressure on a cave-like terrain, so the present benchmark has no published reference to match.

## Recommended parameter ranges for this codebase

Ranges are my proposals, not approved changes. "Basis" uses the tags above.

| parameter | current | recommended | basis |
|---|---|---|---|
| FSR `tau_load` | 2 ms | 1-5 ms (acts at the 500 Hz tactile sample scale) | 1-2 ms mechanical rise [D] |
| FSR `tau_unload` | 20 ms | 20-100 ms, nominal 40; draw per taxel (log-uniform) | no measurement; bounds from vendor 15 ms, 34.5 ms rise [S], skin recovery 25-300 ms |
| FSR `creep_frac`, `creep_tau` | 0.06, 1.5 s | 0.04-0.10; replace by log-time drift: 2-3 exponentials at 0.5, 5, 50 s totalling ~5 % per decade | < 5 %/decade [D] |
| FSR `gain_spread` | 0.15 | 0.06-0.25, nominal 0.15; `fsr_degraded` 0.35 is a stress test only | +/-6 % single batch, +/-15-25 % across batches [D] |
| FSR static hysteresis (new) | none | 7-17 % of full scale, rate-independent, varying per taxel | [M] Paredes-Madrid |
| FSR cold/curvature (new) | none | resistance -5 % at -40 C is negligible; add per-taxel bending preload offset and slow drift | [D] |
| capacitive `modulus` | 0.2 MPa | 0.1-1 MPa as an **effective** bonded-layer compression modulus (1.7-4x bulk) | [E] |
| capacitive `visco_frac` | 0.20 | 0.05-0.12 (measured 0.088) plus a <= 3 % slow tail (10-60 s) | [M] Cabibihan |
| capacitive `tau_visco` | 0.3 s | spectrum 0.025 / 0.15 / 0.3 s, or one tau 0.1-0.2 s | [M] Cabibihan |
| capacitive static hysteresis (new) | none | 4-9 % of range | iCub 5 % [M]; others [S] |
| capacitive baseline (`drift_std`, baseline) | per-taxel random walk | add a common-mode, per-link thermal term of 1-5 % of C/C0 over a 30-40 K swing; keep the random walk small | silicone oil 0.1 %/K bound [S], MRS abstract [S]; magnitude [E] |
| `skin_stiffness` (penetration mode) | 3e7 Pa/m (E_c 0.06 MPa at t 2 mm) | 1e8-5e8 Pa/m (E_c 0.2-1 MPa) | consistency with capacitive modulus [E] |
| friction, glare ice | 0.04-0.10 | 0.04-0.15 near 0 C; **0.2-0.7 below -10 C** if cold ice is in scope | Lahayne [M], Lever [S] |
| friction, rough ice | 0.10-0.22 | 0.12-0.30 | roughness/frost, Marmo via Lever [S]; [E] |
| friction, packed snow | 0.20-0.38 | 0.20-0.40 (keep) | tires 0.23-0.35 [S] |
| friction, fresh snow | 0.15-0.30 | 0.15-0.40; allow overlap with or above packed (ploughing) | Colbeck [S] |
| friction, concrete | 0.50-0.80 | 0.5-1.0 (dry silicone on rough concrete can exceed 1; wet or iced lower) | no source retrieved [E] |
| static/dynamic ratio | 1.1 (PhysX), 1.0 (MuJoCo) | 1.1-1.5, higher after long dwell | [S], [E] |
| sinkage, rough ice | 0-0.2 mm | keep | frost layer, [E] |
| sinkage, packed snow | 0.5-1.5 mm | 0.5-2 mm | Bekker arithmetic 1.4 mm [E] |
| sinkage, fresh snow | 3-8 mm | 3-15 mm (powder deeper) | Bekker arithmetic 11-18 mm [E] |
| `texture_amp`, `texture_wavelength` | amp 0.01-0.40; L 0.5-15 mm | spatially low-pass the texture field (Gaussian, sigma ~ 3 mm for the skin, then average over the 10 mm taxel) before sampling, or set amp as post-filter value: expect visible modulation <= 5 % for L <= 15 mm and ~0 for L <= 6 mm | Section 3 table [E] |

## Gaps: what the models omit, ranked by likely impact on terrain classification

1. **Temperature, speed, pressure and dwell dependence of friction.** Rubber-on-ice friction spans about 0.05 (near
   0 C, fast) to 0.85 (-25 C, slow); the class ranges encode only the near-melting regime. If temperature varies
   in deployment, friction alone cannot separate glare ice, rough ice and snow, and the "friction only" Bayes
   ceiling (0.795 in the investigation log) is a property of the sampling boxes, not of the physics.
2. **Texture cues that real hardware cannot see.** The 2 mm skin and 10 mm taxels (spatial low-pass; the temporal
   2 ms sample averaging matters little) remove most of the 2-6 mm texture the sim passes through; the "texture is solvable" part of the
   task ceiling is optimistic. Sinkage and pressure level (about 1-2 kPa on snow vs 8-13 kPa on ice, [E]) are the
   robust cues.
3. **Snow mechanics beyond footprint width:** ploughing and compaction drag, plastic (non-recoverable) sinkage,
   berm formation against the lateral push of a sidewinding link, pressure-sinkage nonlinearity (n 1.0-1.6) and
   snow-depth/base effects. Only a static footprint widening is modelled.
4. **Stick-slip and static-to-kinetic transitions.** Coulomb friction gives none; real rubber/ice and
   snow contact produces relaxation oscillations and dwell-dependent static friction, a strong time-domain cue.
5. **Rate-independent (Preisach-type) FSR hysteresis, logarithmic drift and hours-long loading memory**, with
   large unit-to-unit variation in the hysteresis itself. Matters for absolute levels and sim-to-real, less for
   within-run patterns.
6. **Cold-environment sensor effects:** common-mode thermal and humidity drift (capacitive), frost or meltwater on
   the skin, snow packing between taxels, curvature preload of FSRs on a 25 mm radius.
7. **Shear and crosstalk.** FSR and capacitive taxels respond to shear and to neighbours through the skin; the sim
   senses none (shear channels exist in the stimulus but `fsr` ignores them).
8. **Load sharing between taxels.** Hybrid mode renormalises the link force over the footprint, so a taxel
   near a 3 mm ground strip can carry the whole link load (~30 kPa for 3 N); a real 2 mm skin spreads load over only
   a few mm and has dead zones between 1 cm^2 taxels on a 20-33 mm grid.

Open verification items: tangential taxel speed during sidewinding (sim logs); whether sensor `dt` is 20 ms in the
training data; any bench measurement of FSR unloading tau and rate-independent hysteresis under a 2 mm skin;
PDMS or Ecoflex friction on ice at -5 to -20 C (none found).
