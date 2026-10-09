# Terrain contact physics for the tactile snake sim, v2: literature review (2026-10-08)

Extends `sensor_terrain_calibration.md` (v1: FSR/capacitive sensors, rubber-ice friction near 0 C, Bekker snow
arithmetic, skin texture filtering). Nothing in v1 is repeated except where a number is needed for arithmetic here.
Robot: 16 links, capsule L 0.10 m, R 25 mm, 0.25 kg (weight 2.45 N on Earth, 0.33 N on Europa), 2 mm elastomer
skin, 24 taxels of 1 cm^2 per link. Recommendations only; no config was changed.

Tags: **[M]** measured, read in the source text; **[S]** only in an abstract, review, search snippet or secondary
citation, not checked in the primary source; **[E]** my estimate or arithmetic. Arithmetic scripts: rigid cylinder,
L 0.10 m, R 25 mm; formulas are given where they matter.

**Five headline findings**
1. On every hard surface (ice at any temperature, concrete, firn, packed snow of hand hardness 1F or harder) the
   per-link normal stiffness is set by the **skin, not the ground**: ~3e4-1.2e5 N/m, indentation 30-90 um, strip
   2.5-4 mm wide, mean pressure 6-10 kPa [E]. Stiffness must therefore not differ between these classes in the sim.
2. Snow and sand deformation under a link is **mostly plastic** (elastic part typically < 5 %) [E]; restitution ~0. A linear spring reproduces the
   loading sinkage but returns energy and lifts the link out of its rut on unloading.
3. **Cold ice is not slippery for a slowly slipping elastomer**: tread rubber gives mu ~0.8 at -32 C [M], up to ~2 on
   polished ice below -10 C [M, secondary]; only near 0 C or at high slip speed does mu fall to 0.05-0.2.
4. Fresh snow and dry sand look alike to a skin (sinkage 1-25 mm vs 1-4 mm, mean pressure 0.5-2 kPa in both) [E].
   Europa ice at 1/7.5 g gives 1.5-2.6 kPa on a compliant skin, the same level as Earth snow or sand [E]: a
   pressure-level cue learned on Earth would call Europa ice "soft terrain".
5. Natural rough ice has cm-scale relief (RMS 6-7.5 mm, correlation length ~65 mm) [M] that a 2-3 cm taxel grid
   resolves as uneven support; mm-scale texture of ice, concrete and snow is invisible through the skin (v1, Sec. 3).

## 1. Contact compliance and dissipation

| quantity | value | conditions | source | tag |
|---|---|---|---|---|
| snow density classes | fresh 70-150 kg/m^3; "compacted" (vehicle) snow 370-560; ice 917 | review | [Shenvi et al. 2022, J. Terramech.](https://par.nsf.gov/servlets/purl/10390372) | [M] |
| snow Young's modulus | 0.2-20 MPa at 100-350 kg/m^3 (low strain rate, Mellor 1975); 20-70 MPa at 210-360 kg/m^3 (dynamic, Sigrist 2006); static tests underestimate (viscous part) | review of lab data | Shenvi et al. 2022 (above) | [M] (secondary) |
| snow P-wave modulus | 10-280 MPa for 150-370 kg/m^3 (talk: 10-340 MPa, 170-370); power law, ~rho^4 | acoustic, lab snow | [Gerling et al., EGU 2016](https://meetingorganizer.copernicus.org/EGU2016/EGU2016-15485.pdf) | [S] |
| hand-hardness pressure | F (fist) 1.5 kPa, 4F 5.6, 1F 25, P 195, K 833 kPa (10-15 N over 82, 22.5, 5, 0.64, 0.15 cm^2); 1 kN/m^2 = 1 kPa | ICSSG test definition | [Geldsetzer & Jamieson 2000, Table 2](https://arc.lib.montana.edu/snow-science/objects/issw-2000-121-127.pdf) | [M] |
| density per hardness | new snow (PP): F 81, 4F 117, 1F 153, P 189 kg/m^3; rounded grains (RG): F 156, 4F 167, 1F 202, P 273, K 393 | regression on >5000 dry layers, W. Canada | same, Table 4 | [M] |
| snow strength per index | index 1 (fist) median ~5 kPa; index 5 (knife) ~1000 kPa, max 3000; ICSSG-1990 gives < 1 kPa for index 1 | instrumented test | [Hoeller & Fromm 2010](https://www.cambridge.org/core/journals/annals-of-glaciology/article/quantification-of-the-hand-hardness-test/E62E451029F311012D3626A71BD356D9) | [M] |
| bearing vs density | bearing strength rises, failure sinkage falls, as power laws of density 0.3-0.6 g/cm^3; stress bulb extends below plate | rigid plates, constant rate | [Abele 1970, CRREL](https://erdc-library.erdc.dren.mil/jspui/handle/11681/5815) | [S] |
| compaction | grain rearrangement first, then grain yield; quasi-static compaction tends to ~550 kg/m^3; well-bonded snow > ~300 kg/m^3 shows elastic response before yield only at higher strain rates | numerics + review | [Annals Glaciol. micromechanics](https://www.cambridge.org/core/journals/annals-of-glaciology/article/preliminary-numerical-investigation-of-the-micromechanics-of-snow-compaction/9BA47D54A0C50EA8A0392697D32F9D09) | [S] |
| unloading of snow | unloading/reloading stiffness controlled by elastic modulus; loading curve by cohesion, friction, hardening | CEL model of plate test | [BIT 2025](https://pure.bit.edu.cn/en/publications/simulation-of-loadsinkage-relationship-and-parameter-inversion-of/) | [S] |
| ice-on-ice restitution | eps_eq 0.79 (d 2 cm), 0.68 (d 2.5 cm), falling with speed; fragmentation above 2.9 / 1.9 m/s | 0.9-6.5 m/s, 256 K, frost-free | [Deckers & Teiser 2016](https://arxiv.org/pdf/1601.04609) | [M] |
| ice restitution vs T | elastic-inelastic critical speed constant from 113 K to ~230 K, ~4x lower at 256 K (Higa 1996) | 1.5 cm spheres on ice blocks | Deckers & Teiser 2016 (citing Higa) | [M] (secondary) |
| frosty ice restitution | e = (v/vc)^-p: Bridges 1984 p 0.234, vc 0.0077 cm/s (= 7.7e-5 m/s), 210 K; Hatzes 1988 p 0.20, vc 0.025 cm/s, 123 K; Supulver 1995 frost-free ~100 K p 0.14, vc 0.01 cm/s; tested only 1.5e-4 to 2e-2 m/s | ring-particle pendulum tests | [Heisselmann et al. 2010](https://arxiv.org/pdf/0908.3424) | [S] |
| same laws at 0.05-0.3 m/s | e = 0.14-0.22 (Bridges), 0.24-0.35 (Hatzes), 0.33-0.42 (Supulver); extrapolated | arithmetic | | [E] |
| silicone rebound resilience | 69 % (ISO 4662, 48 Shore A HCR); typical silicone band 30-70 % | datasheet / vendor article | [Wacker R 865/50 S](https://www.wacker.com/h/de-de/medias/ELASTOSIL-R-86550-S-en-2023.03.16.pdf); [Jehbco](https://jehbco.com.au/unveiling-the-elasticity-dynamics-a-deep-dive-into-silicone-rubbers-rebound-resilience/) | [S] |
| rebound to restitution | e = sqrt(height ratio): 30-70 % gives e 0.55-0.84 | conversion | | [E] |
| cold rubber restitution | two hockey-puck compounds lose 27 % and 35 % of COR from 24 C to -10 C; a third gains 3 % | patent data | [US 5330184](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/5330184) | [S] |
| ice penetration hardness | plastic contact pressure 130 MPa at -90 C, 70 MPa at -30 C; hardness falls sharply above -1.5 C | steel sphere indentation | [Liefferink et al. 2021, PRX](https://physics.aps.org/featured-article-pdf/10.1103/PhysRevX.11.011025) | [M] |

Arithmetic for one link [E]:
* **Skin on rigid ground** (bonded layer, t 2 mm, E_c 0.2-1 MPa; F = (4/3) L (E_c/t) sqrt(2R) delta^1.5): at 2.45 N,
  delta 30-88 um, strip 2.5-4.2 mm, mean pressure 6-10 kPa, secant k = F/delta 2.8e4-8.2e4 N/m, tangent
  4.2e4-1.2e5 N/m. At Europa weight (0.33 N): delta 8-23 um, mean pressure 1.5-2.6 kPa, tangent k 2.1e4-6.3e4 N/m.
  Ice (E ~9 GPa), concrete (~30 GPa) and snow of >= 1F hardness (E >= 1 MPa, strength >= 25 kPa) all act as rigid.
* **Rigid-plastic snow** (footprint grows until the mean pressure equals the snow strength p_y; z = R - sqrt(R^2 - (b/2)^2),
  b = F/(L p_y)): p_y 25 kPa (1F) z ~0.005 mm; 5.6 kPa (4F) 0.10 mm; 1.5 kPa (F) 1.4 mm; 1.0 kPa 3.2 mm; 0.5 kPa
  20 mm; 0.3 kPa footprint exceeds the link diameter (burial, limited by compaction). In the shallow regime
  z ~ F^2 / (8 R L^2 p_y^2): **doubling the load quadruples the sinkage.**
* **Bekker sets** (Wong snow A/B/C, read in v1; pressure integrated over the curved footprint): 16-24 mm at 2.45 N,
  4.6-8.3 mm at 0.33 N (v1 got 11-18 mm with a cruder footprint). The one-day-old stiff set gives 1.2-1.4 mm (v1).
* **Elastic part of snow sinkage**: line contact rebound ~ F/(E L) x (1-3) = 1-75 um for E 1-20 MPa (0.1-0.4 mm if E
  were the 0.2 MPa static minimum), against 1.5-25 mm sinkage in fresh snow: **elastic fraction typically < 5 %**,
  up to ~20 % only in the softest-modulus, shallowest corner [E].
* **Restitution to damping ratio** of a linear spring-dashpot, zeta = -ln e / sqrt(pi^2 + ln^2 e): e 0.9 -> 0.034;
  0.8 -> 0.071; 0.7 -> 0.11; 0.5 -> 0.22; 0.3 -> 0.36; 0.2 -> 0.46; 0.1 -> 0.59; 0.05 -> 0.69; e = 0 needs zeta >= 1.
* **Time scale**: k 3e4-1.2e5 N/m on 0.25 kg gives f_n 55-110 Hz, contact half-period 4.5-9 ms, 1/omega 1.4-2.9 ms.

What this means for the sim:
* Hard classes (glare ice, rough ice, concrete, packed snow, firn, cold ice) must share **one stiffness box**
  (3e4-1.2e5 N/m per link, mildly nonlinear F ~ delta^1.5) and one damping box set by the skin and robot structure,
  not by the ground. Any stiffness difference between them in the sim is an artifact a real skin would not feel.
  If the timestep cannot resolve a 1.5-3 ms time constant, clamp all hard classes to the same softer value.
* Restitution of the skin on rigid ground: e 0.4-0.8 (zeta 0.07-0.3) from the silicone resilience [S] plus losses in the
  robot [E]; no measurement for a skin-covered link exists. Ice-on-ice restitution is irrelevant unless the contact
  is ice-like on both sides (Europa with a glassy skin).
* Snow and sand: use the loading secant stiffness F/z_ss for k (Section 6), zeta >= 1 so nothing rebounds, and accept
  that the rut is not kept. A plastic floor (raise the local ground height by the achieved sinkage) or a heightfield
  update is the better model if the sinkage cue matters; F ~ z^0.5 (snow, shallow) and F ~ z^1.5 (sand) are not linear.
* MuJoCo mapping [E, derived from the docs' relation a_c = A(A+R)^-1 a_ref + R(A+R)^-1 a_u with R = (1-d)/d * A_hat,
  [MuJoCo computation](https://mujoco.readthedocs.io/en/stable/computation/index.html), [M]]: for one isolated contact,
  steady penetration r_ss ~ ((1-d)/d) * g / k_ref, where k_ref is the reference stiffness (acceleration per metre,
  i.e. per unit effective mass). So solref and solimp together set the sinkage; check in the sim by resting one link
  (target: sinkage column of Section 6) and dropping one link from 1-2 cm (target: restitution column).

## 2. Friction of elastomers and polymers on ice and snow, cold to cryogenic

| slider / counterface | mu | conditions | source | tag |
|---|---|---|---|---|
| tread rubber (Tg ~ -45 C) on ice | ~0.8 at -32 C, 0.38 mm/s; ~0.15 at -2 C | p0 0.2 MPa, -38 to -2 C, 3 um/s-1 cm/s, block 6x2 cm | [Tada et al. 2023, Friction](https://www.sciopen.com/article/10.1007/s40544-022-0715-5) | [M] |
| same, mechanism | below ~-15 C and slow: ice fragments stick to rubber and slip on ice, so rubber-ice ~ ice-ice friction; above -10 C or fast: premelted film, viscoelastic friction only; low-T friction peaks near 0.1 mm/s | | same | [M] |
| rubber on polished vs frosty ice | ~1.8 polished vs ~0.5 with hoar frost at 10 mm/s; polished ice frosts over within hours at -30 C (Roberts 1981) | | Tada et al. 2023, citing Roberts | [M] (secondary) |
| rubber on very smooth ice | Gaussian in log v, peak near 10 mm/s; peak ~2 at -20 C falling to 0.2 at -2.5 C (Hemette et al.) | | Tada et al. 2023, citing Hemette | [M] (secondary) |
| rubber vs polished ice adhesion | mu of order 2 at low speed for -10 C > T > Tg | | Tada et al. 2023 | [M] (secondary) |
| pressure effect | natural rubber mu at 0.8 MPa is half that at 0.2 MPa; mu high near -20 to -15 C, ~0.2 from -5 to 0 C | tread rubbers | [KAKEN 62550648](https://kaken.nii.ac.jp/grant/KAKENHI-PROJECT-62550648) | [S] |
| hard sliders (SiC, glass, steel) on ice | Arrhenius mu = 1.5e-4 exp(11.5 kJ/mol / RT); ploughing raises mu above -20 C | 2.5 N, 0.38 mm/s, -110 to 0 C | [Liefferink et al. 2021](https://physics.aps.org/featured-article-pdf/10.1103/PhysRevX.11.011025) | [M] |
| same fit, evaluated | 0.03 at -10 C, 0.045 at -30 C, 0.45 at -100 C; gives > 1 below ~155 K, so it **cannot be extrapolated** to Europa | arithmetic | | [E] |
| SiC on ice | ~0.1 at -32 C, 0.38 mm/s (vs ~0.8 for rubber) | | Tada et al. 2023 citing Liefferink | [M] (secondary) |
| polymers on ice | PTFE mu_k 0.016 -> 0.071, HDPE 0.013 -> 0.079 from -5 to -30 C | speed not checked | [Wieleba 2011, Tribologia](https://yadda.icm.edu.pl/baztech/element/bwmeta1.element.baztech-article-BPS1-0046-0059/c/Tribologia_5_2011_Wieleba.pdf) | [S] |
| ice on ice, -3 to -40 C | ~0.5 at 1e-6 m/s, peak ~0.7 at 5e-5 m/s, ~0.1 at 1e-2 m/s | | [J. Glaciol. record](https://resolve.cambridge.org/core/product/25D7C9C4B48B28B05AC603E90AD9171E/core-reader) | [S] |
| ice on ice, 98-263 K | mu_k 0.15-0.76 (smooth); velocity strengthening below 1e-5-1e-4 m/s at >= 223 K; no clear v-dependence at 133 and 173 K | <= 98 kPa, 5e-8 to 1e-3 m/s (Schulson & Fortt 2012) | [review, arXiv 1502.04173](https://arxiv.org/pdf/1502.04173) | [S] |
| ice on ice, 77-115 K | slope 0.55 (sigma_n <= 5 MPa, intercept 1.0 MPa); 0.20 (>= 10 MPa); T- and v-independent; stick-slip in all tests | triaxial saw-cut (Beeman et al. 1988) | [USGS record](https://pubs.usgs.gov/publication/70207840) | [S] |
| PTFE on steel, cryogenic | no systematic change of mu from 4 to 200 K; PTFE composites slightly lower mu at 77 K (stiffer); vacuum data scarce | cryotribometers | [Tribol. Lett. 2006](https://www.doi.org/10.1007/S11249-006-9115-7); [BAM 2016](https://opus4.kobv.de/opus4-bam/frontdoor/index/index/year/2016/docId/31188) | [S] |
| silicone transitions | Tg ~ -125 C (148 K); crystallization near -40 to -50 C stiffens it | DSC/DMA | v1, Sec. 2 | [M] |

What this means for the sim:
* The friction class boxes must carry a **temperature (and slip-speed) regime**. Near 0 C: glare ice 0.05-0.2. From
  -10 to -40 C at the slow end (0.1-10 mm/s): 0.5-2 for rubber, with frost and wear debris pulling it to ~0.5 and
  polished ice pushing it to ~2. At the sim's slip speeds (1-20 cm/s) no cold-ice rubber data exist; frictional
  heating lowers mu with speed, so 0.4-1.2 is a reasonable cold box [E].
* Our nominal pressure (6-10 kPa) is 20-30x below the tread tests; rubber mu rises at lower pressure [S], so the skin
  is probably at the upper end of these ranges [E]. No silicone/PDMS-on-ice data at any temperature (as in v1).
* Cryogenic (Europa): no elastomer data; any skin is glassy at 80-130 K. Ice-on-ice (0.15-0.76; 0.55 at low normal
  stress, temperature-independent from 77 to 115 K) is the best proxy, since a hard polymer slider carries ice debris
  as rubber does at -30 C. Use 0.3-0.8 [E] and expect stick-slip (seen in all 77-115 K tests [S]).
* Cold friction overlaps snow (0.2-0.6) and concrete (0.6-1.0): if temperature varies at test time, friction alone
  cannot identify ice. Treat this as a physical property of the task, not a sim weakness.

## 3. Europa and Enceladus surface properties (out-of-distribution set)

| quantity | value | status | source | tag |
|---|---|---|---|---|
| surface gravity | Europa 1.315 m/s^2; Enceladus 0.113 m/s^2 | known | standard values | [E] |
| surface temperature | model: equator 96 K, pole 46 K, global annual mean 90 K | model | [Ashkenazy 2016](https://arxiv.org/abs/1608.07372) | [S] |
| surface temperature | modelled from Galileo PPR: 67-148 K | data + model | [Lange et al. 2026](https://arxiv.org/abs/2604.14374) | [S] |
| thermal inertia (cm-depth) | 40-150 J m^-2 K^-1 s^-1/2 (PPR); ALMA global 95, range 40-300; PPR re-fit 56 +/- 17, equatorial band 39 +/- 7 | measured, model-dependent | [Trumbo et al. 2018](https://arxiv.org/pdf/1808.07111); Lange et al. 2026 | [S] |
| porosity / grain size | average porosity 0.61 +/- 0.1, grains um to cm | inferred from thermal data | Lange et al. 2026 | [S] |
| top-millimetre regolith | thermal inertia 9-20 (bulk ice ~2000) needs porosity > 80 %, grains < 1 mm, minimal grain contact; compaction within ~1 cm | inferred, all icy moons | [Mergny et al. 2026](https://arxiv.org/abs/2605.27048) | [S] |
| outer microns | > 95 % void (photopolarimetry); says nothing about depth | inferred | [PSI, Nelson](https://www.psi.edu/blog/europa-and-other-planetary-bodies-may-have-extremely-low-density-surfaces/) | [S] |
| sintering | neck growth fast, densification negligible over Europa's surface age: "cohesive but porous" sintered crust; crushing strength "100s of kPa" (model, slides) | model | [Molaro et al. 2019](https://arxiv.org/abs/1901.04633); [KISS slides](https://www.kiss.caltech.edu:443/workshops/oceanworlds/presentations/Molaro.pdf) | [S] |
| cohesion of fine ice | tensile strength 0.9 +/- 0.7 kPa (2.4 um grains, porosity 0.5, 150 K) | lab (Gundlach et al. 2018) | [review, arXiv 1905.01156](https://arxiv.org/pdf/1905.01156) | [S] |
| fragile vacuum-frozen ice | lab layers 10-20 cm thick from boiling/freezing water; scaled to several m on Europa, ~20 m on Enceladus; a lander could break through | lab + scaling | [Charles Univ. release (EPSL 2026)](https://www.mff.cuni.cz/en/public/news/when-water-boils-and-freezes-at-the-same-time-a-recipe-for-strange-icy-landscapes-in-the-solar-system) | [S] |
| m-scale roughness | Hurst 0.4-0.8 below breakpoints at a few 100 m, 0.2-0.6 above (30 m-5 km); extrapolated RMS 0.4-0.7 m at 1 m, chaos 1-2 m | Galileo stereo; extrapolation | [Steinbruegge et al. 2020](https://experts.arizona.edu/en/publications/the-surface-roughness-of-europa-derived-from-galileo-stereo-image/) | [S] |
| penitentes | up to ~15 m deep, ~7.5 m spacing, equatorial (< 23 deg) | model only; not resolvable in images | [Hobley et al. 2018](https://orca.cardiff.ac.uk/115808/) | [S] |
| non-ice materials | NaCl on leading-hemisphere chaos (450 nm band); hydrated sulfuric acid dominant on trailing hemisphere | measured remotely | [Trumbo et al. 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6561749); [Brown & Hand 2013](https://arxiv.org/pdf/1303.0894) | [S] |
| Enceladus | mean T ~72-75 K; plume fallout grains 0.6-15 um (south polar 20-75 um), up to ~1 mm/yr near vents; deposits 100s of m; porous regolith; sublimated lab analogues ~10 kPa shear and compressive strength (source not pinned) | mixed | EPSC 2026 abstracts, e.g. [EPSC2026-259](https://meetingorganizer.copernicus.org/EPSC2026/EPSC2026-259.html) (abstract per number not pinned); [Science News](https://www.sciencenews.org/article/enceladus-snow-saturn-cassini) | [S] |
| ice friction / restitution at 80-130 K | Section 2 (0.15-0.76; 0.55) and Section 1 (e 0.2-0.4 at 0.05-0.3 m/s for frosty ice) | lab | above | [S]/[E] |

Known vs speculative: gravity and the temperature range are known; thermal inertia is measured but its translation
into porosity and grain size is model-dependent; top-mm porosity > 80 % is a strong inference; cohesion, bearing
strength, mm-cm roughness and penitentes are speculative. No in-situ data exist.

What this means for the sim:
* Per-link weight 0.33 N: on solid ice a compliant skin gives 8-23 um indentation and 1.5-2.6 kPa mean pressure, the
  pressure level of Earth snow and sand [E]. This is the most dangerous OOD cue shift, and a good test of whether the
  encoder relies on absolute pressure. Include solid-ice Europa cases at 1.315 m/s^2.
* Porous regolith at 0.33 N: with strength 0.3-10 kPa the rigid-plastic estimate gives 0.002-0.6 mm; if the top
  layer is uncohesive and > 80 % porous, sinkage could reach several mm (cm-scale compaction) [E, speculative].
  Draw sinkage log-uniform 0.01-8 mm.
* Any elastomer is glassy at 80-130 K (silicone Tg 148 K). Either keep the Earth skin model (assume a heated or
  cryo-rated compliant skin) or switch to a glassy skin (k > 1e6 N/m, contact effectively rigid). Decide which, as
  the two give very different tactile signals.

## 4. Surface roughness at mm-cm scales

| surface | statistic | conditions | source | tag |
|---|---|---|---|---|
| glacier ice (ablation zone, Svalbard) | RMS height 6.2-7.5 mm, correlation length 64-67 mm (exponential ACF) | 1 m photographic transects, 1.3 mm sampling | [Rees & Arnold 2006](https://doi.org/10.3189/172756506781828665) | [M] |
| same, larger scale | RMS 59-75 mm, correlation length 480-570 mm; power-law (fractal) below ~0.1 m and above a few m, distinct break between (70-500 mm, 6-70 mm) | 10-20 m string transects | same | [M] |
| lab ice (polished) | Sq 61 nm over 208 um | confocal | Liefferink et al. 2021 | [M] |
| road/tribology ice | self-affine, H ~1, 10 um-150 mm | | v1 (Lahayne et al. 2016) | [M] |
| snow on glacier | RMS 0.5-9.2 mm; correlation length 0.6-46 cm; resolves 4-50 cm wavelengths, > 1 mm heights | snowmobile laser profiler | [Lacroix et al. 2008](https://www.cambridge.org/core/journals/journal-of-glaciology/article/in-situ-measurements-of-snow-surface-roughness-using-a-laser-profiler/378C43255C608BA15288AE20977ED2A9) | [S] |
| fresh snow | two scaling regimes, crossover at crystal-cluster size; ballistic-deposition-like at larger scales | photographs | [Manes et al. 2008](https://infoscience.epfl.ch/record/170126) | [S] |
| snow across scales | roughness at mm (boards) and m (lidar) not correlated | | [Fassnacht et al. 2023](https://www.usgs.gov/publications/snow-surface-roughness-across-spatio-temporal-scales) | [S] |
| concrete macrotexture | wavelengths 0.5-50 mm, depth 0.2-10 mm (PIARC/ISO 13473-1) | definition | [PIARC](https://www.piarc.org/en/activities/Road-Dictionary-Terminology-Road-Transport/term-sheet/92982-en-macrotexture) | [S] |
| concrete texture depth | broomed 0.2-0.4 mm, hessian 0.3-0.5, tined 0.4-0.7 mm MTD; tined grooves 1.5-4.5 mm; wears 25-35 % in first half-year | new surfaces | [SA DIT TN025](https://dpti.sa.gov.au/__data/assets/word_doc/0006/47526/TN025.doc) (table not pinned); [Iowa DOT](https://ia.iowadot.gov/erl/archiveoct2014/CM/content/CM%209.40.htm); [TRR 602](https://onlinepubs.trb.org/Onlinepubs/trr/1976/602/602-012.pdf) | [S] |

What a 1 cm taxel on a 2-3 cm grid feels [E, using the v1 filter: skin exp(-2 pi t/L) times taxel sinc]:
* L <= 10 mm is invisible (< 2 % of the amplitude): frost, snow grains, concrete micro- and most macrotexture.
* L 15-50 mm passes 20-70 %; the grid (pitch 20-33 mm) samples a static pattern only for L >~ 40-70 mm, but sliding at
  1-20 cm/s and 500 Hz samples 0.02-0.4 mm along the track, so 15-50 mm relief appears as 3-130 Hz load fluctuation.
* So the felt roughness is **unevenness of support** (which taxels carry load) from features of 15-500 mm: rough
  glacier ice (RMS ~7 mm at ~65 mm) is strongly felt; snow surfaces (RMS 0.5-9 mm at cm-dm) are felt but the skin
  sinks into soft snow and smooths them; glare ice and broomed concrete are effectively flat.

## 5. Granular proxies and terrain-classification benchmarks

| quantity | value | conditions | source | tag |
|---|---|---|---|---|
| RFT vertical constant alpha_z = sigma/z | medium sand 2.02 N/cm^3, Mars Mojave simulant 3.05, poppy seeds loose 0.35 / close 0.55 (1 N/cm^3 = 1e6 N/m^3) | horizontal plate, depth <~ 80 mm | [Agarwal et al. 2019, Table 1](https://arxiv.org/pdf/1901.10667) | [M] |
| internal friction angle, cohesion | sand 34 deg, 1.5 kPa; MMS 35 deg, 0.6 kPa; poppy seeds 36 / 45 deg (angle of repose), 0 kPa | direct shear | same | [M] |
| Bekker fit caveat | fitted kc negative for sand; table units garbled (kphi 3.13e6 is only consistent with alpha_z if read as N/m^(n+2)) | | same | [M]/[E] |
| link sinkage, RFT | F = (4/3) alpha L sqrt(2R) z^1.5: 0.9-3.8 mm; secant k 640-2700 N/m, tangent 970-4100 N/m; mean pressure 1-2 kPa | 2.45 N | arithmetic | [E] |
| gravity scaling | for cohesionless media alpha ~ rho g, so self-weight sinkage is g-independent; cohesive or sintered media sink less at low g | | arithmetic | [E] |
| snow-rover caveat | dynamic immobilization in low-cohesion, low-stiffness snow not captured by terramechanics | field | [Lines et al. 2021](https://par.nsf.gov/servlets/purl/10304353) | [S] |
| rubber on concrete (dry) | ~1 at 0.1-10 mm/s, ~0.75 at 5 um/s (tread, 0.1 MPa, 23 C); SBR 0.95 | tribometer | [arXiv 2411.07332](https://arxiv.org/pdf/2411.07332); [arXiv 2501.12561](https://arxiv.org/pdf/2501.12561) | [S] |
| haptic soil classification | foot impact vibration, wavelet + SVM, > 98 % on Mars simulants with disturbances; tested on unknown soils; PALPATE dataset 2600 testbed + 240 ANYmal impacts | single-foot testbed + ANYmal | [Kolvenbach et al. 2019](https://www.research-collection.ethz.ch/items/28160189-5ff8-4f59-a22f-1d5d1c74b666); [PALPATE](https://www.research-collection.ethz.ch/entities/researchdata/54804f8f-350e-42f5-b564-3c7d39192bab) | [S] |
| foot F/T terrain classification | PUTany (ANYmal, Bednarek et al. ICRA 2019) and QCAT datasets; HAPTR2 transformer | real robots | [HAPTR2](https://sin.put.poznan.pl/publications/details/i49710) | [S] |
| proprioceptive, snow/ice | BorealTC: Husky, 116 min IMU + current + odometry, snow, ice, silty loam; merged with a second dataset, Mamba gains when trained on both | real robot | [LaRocque et al. 2024](https://arxiv.org/abs/2403.16877) | [S] |
| unseen speed/terrain | rover proprioception, > 90 % on generalization tasks; CNN beats SVM on extrapolation to unseen velocity or terrain | real rover | [Ugenti et al. 2022](https://iris.poliba.it/handle/11589/249924) | [S] |
| sim + real | humanoid foot IMU/torque, > 98 % in Gazebo and on the robot (separately) | | [IEEE DataPort](https://ieee-dataport.org/open-access/terrain-identification-humanoid-robots) | [S] |
| deformable-terrain simulator | Chrono SCM (Bekker-Wong soil contact) as a second simulator | | [Chrono SCM](https://api.projectchrono.org/structchrono_1_1synchrono_1_1_s_c_m_parameters.html) | [S] |

What this means for the sim: dry sand and fresh snow occupy the same sinkage and pressure range; separate them only by
cues hardware would feel (friction and ploughing drag, temperature, sinkage growth with load: z ~ F^(2/3) in sand vs
~F^2 in strength-limited snow [E]). Benchmarks to mirror: leave-one-terrain-out and leave-one-condition-out
(temperature, gravity) splits as in Kolvenbach and Ugenti; a cross-dataset merge as in BorealTC; a cross-simulator
test (MuJoCo vs Chrono SCM or DEM for granular classes). None of these uses a tactile skin on ice or snow.

## 6. Recommended simulation parameter boxes

Per link, 2.45 N on Earth unless noted. k is the loading stiffness in N/m (convert to MuJoCo via Section 1). zeta is
the damping ratio of a linear spring-dashpot (e = restitution). Roughness: RMS height and the wavelengths that reach
a 1 cm taxel through a 2 mm skin (>= 15 mm); shorter texture should be filtered out or omitted.

| terrain | friction mu (regime) | sinkage (mm) | k (N/m) | zeta (e) | roughness felt (RMS @ wavelength) | support |
|---|---|---|---|---|---|---|
| glare ice, -5 to 0 C | 0.05-0.2 (premelt) | 0.03-0.09 (skin only) | 3e4-1.2e5 | 0.07-0.3 (0.4-0.8) | < 0.05 mm at < 10 mm (invisible); 0.1-1 mm at 50-500 mm | mu: good; k: good [E] physics; zeta: guess; roughness: guess |
| glare ice, -10 to -20 C | 0.2-1.0 (higher if polished, lower if frosted) | same | same | same | same | mu: tread rubber [M]; silicone unmeasured |
| rough / frosted ice | 0.1-0.3 near 0 C; 0.4-0.8 cold | 0-0.3 (frost crushing) | 1e4-1.2e5 | 0.1-0.4 (0.3-0.7) | 1-7 mm at 30-150 mm (weathered ice: 6-7.5 mm, l 65 mm) | roughness: [M] for glacier ice; rest [E] |
| packed snow, 300-500 kg/m^3 (1F-K) | 0.2-0.4 near 0 C; 0.3-0.6 below -15 C | 0.01-1.5 | 2e3-1e5 (skin-limited if the snow does not yield) | >= 0.7 if yielding (e <= 0.05), else as ice | 0.5-5 mm at 40-500 mm | sinkage: [E] from hardness and Bekker; mu: tires [S] + guess |
| fresh snow, 50-200 kg/m^3 (F-4F) | 0.15-0.4 + ploughing drag | 1.5-25 (shallow: ~F^2) | 100-2000 (secant, loading) | >= 1 (e ~ 0; mostly plastic) | surface 1-10 mm at 10-500 mm, smoothed by sinkage | order of magnitude only [E] |
| concrete, dry | 0.6-1.0 | 0.03-0.09 | 3e4-1.2e5 | 0.07-0.3 (0.4-0.8) | MTD 0.2-0.7 mm mostly at < 15 mm (invisible); joints and cracks | mu, texture [S]; k [E] |
| OOD: cold arctic ice, -30 C | 0.4-1.5 (frosty ~0.5, polished to ~2); stick-slip | 0.03-0.1 | 3e4-3e5 (silicone may crystallize below -40 C) | 0.1-0.4 | as glare or rough ice | mu: [M] tread rubber at 0.2 MPa, slow; extrapolated to skin and speed |
| OOD: Europa-like ice, 80-130 K, g 1.315 | 0.3-0.8 (ice-ice proxy) | solid ice 0.008-0.023 (compliant skin) or ~0 (glassy); porous regolith 0.01-8 | compliant skin 1.4e4-6e4; glassy skin > 1e6; regolith 50-6000 | ice 0.25-0.5 (e 0.2-0.4); regolith >= 1 | mm-cm unknown: use rough-ice spectra x1-3; m-scale RMS 0.4-0.7 m | guess except g and T |
| OOD: dry sand | 0.4-0.8 (tan phi 0.67-0.73 plus skin-grain) | 0.9-3.8 (g-independent) | 600-2700 (secant) | >= 1 (e 0-0.1) | grains invisible; ripples (if any) ~5-10 mm at 50-150 mm | sinkage [E] from [M] alpha_z; mu, ripples guess |
| OOD: gravel, D 4-30 mm | 0.5-1.0 (with interlock) | 0-5 (rearrangement) | 1e4-5e4 (few point contacts) | 0.2-0.6 | RMS ~0.2-0.3 D at ~D (visible for D >= 10 mm) | guess |

Cross-checks for the sampler [E]: mean contact pressure on hard ground 6-10 kPa (Earth), 1.5-2.6 kPa (Europa);
fresh snow and sand 0.5-2 kPa. Link load in sidewinding varies ~0.5-2x weight: sinkage should scale ~F^(2/3) on hard
ground and sand and ~F^2 on strength-limited snow, which a linear spring cannot reproduce (it gives ~F).

## 7. Biggest uncertainties

1. **No elastomer friction data on ice below -40 C, and none for silicone/PDMS at any temperature**; nothing for any
   polymer on ice at 80-130 K in vacuum. Europa friction (0.3-0.8) is an ice-on-ice proxy.
2. **Cold rubber-ice friction varies 2-4x with surface state** (frost ~0.5 vs polished ~2) and peaks at 0.1-10 mm/s;
   our 1-20 cm/s slip at 6-10 kPa is outside all measured conditions.
3. **Snow strength for a small light cylinder**: hand-hardness pressures disagree by ~5x (fist < 1 vs ~5 kPa), Bekker
   parameters come from 0.1-0.2 m plates, so fresh-snow sinkage is known only to within ~10x (1.5-25 mm).
4. **Restitution of a skin-covered link** is unmeasured; it depends on the robot structure as much as the skin.
5. **Europa near-surface mechanics**: porosity (> 80 % at the top mm vs 0.61 average), cohesion (0.1 kPa to 100s of
   kPa), mm-m roughness and penitentes are inferred or modelled, not measured.
6. **Skin state in the cold**: silicone crystallization below ~-40 C and the glass transition at 148 K change the
   contact stiffness by orders of magnitude; no data for our skin.
7. **Lateral loading**: sidewinding pushes links sideways; berms in snow and sand, shear strength and ploughing drag
   are not covered by the normal-load numbers above.
8. **Roughness of real snow and ice under the skin load** at 15-150 mm: only one glacier-ice data set ([M]) and one
   snow profiler data set ([S]); no data for city ice (refrozen ruts, slush) or sea ice at these scales.
