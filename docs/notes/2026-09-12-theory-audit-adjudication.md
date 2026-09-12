# CyFM theoretical audit (ofurman, 2026-09-12) — adjudication ledger
Status: in progress, claim-by-claim. No manuscript edits during this pass.

## G1 — "complex signals inherently reside on a cylinder"
Locations: 00_abstract.tex:2, 01_introduction.tex:23, 04_method.tex:4
Oleksii verdict: Incorrect.
Proposed fix: "For nonzero complex values we study amplitude-phase coordinates on
(0,inf) x S^1 equipped with a chosen product metric."

Cryo: AGREE on diagnosis, DISAGREE on the proposed sentence.
- The topological half is TRUE: C* = C\{0} is diffeomorphic to (0,inf) x S^1 via
  z -> (|z|, z/|z|). Oleksii's flat "Incorrect" is slightly over-graded.
- The false word is "inherently", which smuggles in the METRIC. Inherited metric
  in polar coords is dA^2 + A^2 dtheta^2; the paper uses dA^2 + dtheta^2. That is
  a replacement, not an inheritance. Paper admits this in 03_preliminaries.tex
  (polar metric "describes a flat Euclidean plane, curvature is zero").
- His replacement is too defensive: it deletes the motivation and reads like a
  methods note in the abstract's opening slot.
- Owning the choice STRENGTHENS the paper: it is an empirical comparison of two
  geometries. If the geometry were inherent there would be nothing to compare.

Cryo's additional catches, not in G1 as pasted:
- "topological defects/pathologies" of Euclidean modeling is the wrong word.
  R^2 has no topological defect. Real issues are: polar COORDINATE map singular
  at origin, induced angular velocity O(A^-1), linear paths crossing the origin.
  Coordinate/metric/path problems, not topology.
- 03_preliminaries presents pi_1 = Z (non-trivial fundamental group) as a feature
  ("Crucially ... yet possesses"). The wrap is a COST they handle, not a benefit.
- R^+ is used ambiguously ((0,inf) vs [0,inf)). Never pinned down.
- "each spatial pixel possesses a non-zero amplitude" (03_preliminaries) is false
  for real MRI background. Measure-zero in theory, but numerically the background
  phase is pure noise, and the product metric dA^2 + dtheta^2 weights background
  phase error EQUALLY to signal phase error. This is exactly what the amplitude
  weighting in 04_method was trying to repair, and that weighting demonstrably
  breaks the fixed point (sliced W2 0.121 weighted-L1 vs 0.038 unweighted-L2).
  => G1 is cosmetic for synthetic fields, LOAD-BEARING for the fastMRI v2 track.

Open: claim text truncated at "(z\mapsto(" in the paste; assumed z -> (|z|, z/|z|).

### G1 resolution status (2026-09-12)
DECISION: Marcel agrees; owning the metric choice strengthens novelty. Reframing
approved and pushed for the G1-named sites. Appendix proof parked, not shipped.

Pushed to Overleaf:
- 00_abstract.tex:2   "inherently reside" -> chart + explicit metric substitution
- 01_introduction.tex "severe topological and geometric pathologies" -> the plane
                      is well behaved, the path is not
- 04_method.tex:4     "topological defects" -> "remove the amplitude factor"

Parked: docs/notes/appendix-metric-substitution-draft.tex
  Lemma (chart + pullback dA^2 + A^2 dtheta^2), Remark (both metrics flat, not
  globally isometric), Corollary (circumference 2*pi*r vs 2*pi unifies
  thm:angular_divergence with lem:bounded_velocity).
  Decision taken: do NOT prove the diffeomorphism itself, it reads as padding.

STILL OPEN, same claim, not yet touched:
- 01_introduction.tex:23  caption "treats the signal natively on R^+ x S^1"
                          (my rewrite was REJECTED, needs new wording)
- 03_preliminaries.tex:30 pi_1 = Z sold as a feature; flatness sold as advantage
                          though both metrics are flat
- 05_experiments.tex:9    "fundamental topological differences"
- 05_experiments.tex:35   "topological necessity of the cylindrical product
                          manifold"  <-- strongest overclaim in the paper
- 07_appendix.tex:7       "geometric and topological theorems ... pathology"
These four now CONTRADICT the new abstract. Must close before submission.

### G1 CLOSED (2026-09-12), verified on origin/main
Appendix subsection sec:app_metric_choice SHIPPED (Marcel approved after the
"both metrics are flat" point). It then contradicted four remaining sites,
all now fixed in one commit:
- 03_preliminaries: both flat; difference is circumference 2*pi vs 2*pi*A,
  xref to the appendix; pi_1 = Z framed as a cost shared by both
- 04_method: flatness beats CURVED formulations, explicitly not the Cartesian
- 05_experiments: dropped "fundamental topological differences",
  "geometric pathologies", and "topological necessity" (grep "topological" = 0)
- 07_appendix: retitled to "Path-Level Consequences of the Euclidean Metric"

Build verified locally (pdflatex+bibtex, 3 passes): 0 undefined refs,
0 undefined cites, 0 overfull boxes. Conclusion and References both land on
p.9 before AND after, so the ICLR 9-page main-text limit is untouched; the
extra pages are bibliography + appendix, both unlimited.
Marcel notes the limit is moot for the arXiv build anyway.

STILL OPEN (only G1 item left):
- 01_introduction.tex:23 caption "treats the signal natively on R^+ x S^1".
  Edit was rejected once; original wording is what is live. A ready version
  sits on local branch `backup-with-caption`.

TOOLING GOTCHA (cost us 9 phantom pushes today):
  overleaf_cache/<id>/ IS the MCP server's own clone. edit_file commits there;
  when branches diverge the push silently fails, and read_file then serves the
  UNPUSHED local text back, so verification passes falsely. Verify with
  `git show origin/main:<path>`, never with mcp read_file.
  Also: compile_project returns 400 on this project (4 submissions, no root-doc
  param). Build locally instead: git archive HEAD | tar -x, then pdflatex.

### G1 fully closed (2026-09-12) — the naive/native pair
Marcel's catch, and my error: I called "naive" a value judgement. It is not.
It is standard technical register (naive Bayes, naive algorithm, naive set
theory) and asserts nothing about geometry. My "either both or neither"
conclusion rested on that false premise and was dropped.
The two words are NOT symmetric:
  - "naive"    describes a way of proceeding -> no conflict with G1 -> KEPT
  - "natively" asserts the cylinder is the signal's own domain, i.e. exactly
    the deleted "inherently reside" -> REMOVED (caption 01_introduction:23)
Also: 02_related_work objected to "decoupled Cartesian components", but
decoupling is OUR construction. Marcel's call: the objection belongs on
"Cartesian" and the inherited plane metric. Reworded.
Verified on origin/main: natively=0, naive=1, "decoupled Cartesian"=0,
naively=1. Build clean (0/0/0), conclusion still lands on p.9.
No G1 sites remain open.

## G2 — "C* = R+ x S^1" and "[-pi,pi) = S^1"
Location: 03_preliminaries.tex:20
Oleksii verdict: Qualify. Wants R+ defined as (0,inf); says [-pi,pi) is only a
representative set; wants phase as q = e^{i theta}; wants the treatment of exact
zero voxels stated.

Cryo: AGREE on all four, but the grade is TOO LENIENT on one sub-point.
- "[-pi,pi) =~ S^1" is FALSE, not underspecified. [-pi,pi) with the subspace
  topology is not compact, S^1 is; compactness is a homeomorphism invariant.
  Second argument: [-pi,pi) is contractible (pi_1 = 0), S^1 is not (pi_1 = Z).
  It is a set bijection / fundamental domain, never a homeomorphism.
  Correct form: S^1 =~ R/2piZ, with [-pi,pi) a set of representatives.
- C* =~ R+ x S^1 IS fine once R+ := (0,inf). That half is Qualify, correctly.
- SELF-CONTRADICTION: 04_method (Continuous Trigonometric Embedding) says the
  scalar angle in [-pi,pi) "exhibits an artificial coordinate discontinuity
  across the negative real axis". The method already knows [-pi,pi) is not the
  circle -- that is the entire justification for the (A, cos, sin) embedding.
  If preliminaries:20 were true, section 4.4 would have no reason to exist.
  Same species as the flatness contradiction from G1.

### CODE CHECK (audit could not do this -- it had no repo access)
Zero voxels are NOT excluded. They are manufactured.
  src/cfm/flow/solver.py:99-100
      # Amplitude clamp >= 0
      m_next = torch.clamp(m_t + v_m * dt, min=0.0)
  - Docstring one line above claims "projected onto R+ x S^1", but the clamp is
    min=0.0, i.e. the projection target is [0,inf) x S^1. The origin, which the
    whole paper excludes, is admitted and is a fixed point of the clamp.
  - The clamp exists BECAUSE the learned amplitude ODE does not preserve
    positivity. Positivity is enforced by truncation, not by the flow.
  - At A = 0 the phase is meaningless but the state still carries (cos, sin).
  - Inconsistent floors: solver uses 0.0; models/pointwise_mlp.py:152 uses
    _AMPLITUDE_FLOOR = 1e-8; evaluate.py:117 uses _ANGULAR_FLOOR = 1e-12.
  - NOT QUANTIFIED: how often the clamp fires is unknown without running eval.

### NFE convention -- audit flag CLEARED, no action
flow/solver.py HeunODESolver.sample does 2 model calls per step except the last,
so k steps = 2k-1 NFE. Both geometries inherit the SAME sample() loop
(EuclideanODESolver extends HeunODESolver), so the geometry comparison is fair.
And 05_experiments already states "k steps correspond to 2k-1 function
evaluations". Paper is correct here.
Minor foot-gun only: evaluate.py:310 reads a config key named `nfe` whose values
[1,2,4,8,16,32,64,100] are actually STEP counts, passed to num_steps.

### G2 CLOSED (2026-09-12), verified on origin/main
Marcel's correction, accepted: on the product metric dA^2 + dtheta^2 the origin
is NOT a singularity. The polar metric dA^2 + A^2 dtheta^2 degenerates at A=0
(the circle collapses to a point); ours keeps circumference 2*pi there, so
amplitude does not kill phase. My earlier "the origin is admitted" framing was
too strong and was withdrawn. Consequence: [0,inf) x S^1 is a clean manifold
with boundary and stating it COSTS NOTHING and favours the cylinder.

Residual point (kept, different species): [0,inf) x S^1 is not diffeomorphic to
C*, because decoding A*e^{i theta} sends the whole circle {0} x S^1 to 0 in C.
Two model states differing only in phase at A=0 decode to the same complex
field, so the model can carry phase no complex-valued metric can see.

Shipped:
- 03_preliminaries:20 rewritten. Phase is q = e^{i theta} in S^1, with
  [-pi,pi) named as a set of representatives for R/2piZ. R^+ := (0,inf),
  C* = open cylinder, CyFM on the closure [0,inf) x S^1.
- Manifold renamed to [0,inf) x S^1 at every site naming the model's state
  space: preliminaries (def + geodesics), method (product + geodesics),
  introduction, related work, conclusion.
- R^+ survives only where correct: prelim:20 (twice, the definition and C*)
  and related_work:8 (prior work's R^+ x S^{d-1}). Verified: exactly 3.
- related_work: "metric has a non-trivial fundamental group" was a category
  error (manifolds have pi_1, metrics do not) -> "whose phase factor is a
  circle at every pixel".
Build clean (0 refs / 0 cites / 0 overfull), conclusion still p.9.

OPEN, not a paper edit: instrument how often solver.py's clamp at min=0.0
fires. It is a training-quality signal (v_A pushing amplitude negative),
not a geometry defect. Unquantified.

### G3 CLOSED (2026-09-12), verified on origin/main
Location: 03_preliminaries.tex:22-26. Oleksii verdict: Verified with wording fix.
Cryo: AGREE fully, no dispute.
The polar parametrisation restricted to (0,inf) x S^1 is a diffeomorphism onto
C* and is regular THERE; the origin is not in its domain, so the map does not
"become singular" at it. What degenerates is the attempted extension to A = 0:
the differential drops rank and the whole circle of phases collapses to a point.
Same species of error as G1 and G2: a defect attributed to an object that does
not have it.
Shipped:
- "induce the Riemannian metric" -> "induce the inherited Euclidean metric"
  (matches the abstract and Appendix A.1 wording)
- "the polar coordinate map itself becomes singular at the origin" ->
  "This metric is flat for A > 0, and the parametrisation is nonsingular on its
   domain. What degenerates is its attempted extension to A = 0: the
   differential drops rank there, and the entire circle of phases collapses to
   a single point."
This now stands as the direct contrast to the G2 sentence three lines earlier
(product metric stays non-degenerate at A=0). The two halves finally meet.
Build clean 0/0/0, conclusion still p.9.

NOT done deliberately: Oleksii suggests Petersen, Riemannian Geometry as the
source for the wording. No bib entry added -- the fact is standard and needs no
citation, and this repo already deleted one uncited entry (bose2023equivariant)
for exactly that reason. Flagged to Marcel, his call.

### G4 + G5 CLOSED (2026-09-12), verified on origin/main
Both graded Verified by Oleksii; maths never in question, only the surrounding
claims. Cryo AGREES with both, no dispute.

G4 (03_preliminaries:26-30, 04_method:12)
Note: the target sentence had ALREADY been rewritten in the G1 pass, so the
audit was reading the pre-fix text. Merged rather than overwritten: kept my
"not an advantage over the Cartesian baseline, which is flat as well" and added
Oleksii's list of what survives flatness -> "flatness removes neither the 2pi
wrap, nor the cut locus at antipodal phase, nor the boundary at A = 0".
Also narrowed parallelizability in the preliminaries: it gives a global frame
and therefore no parallel transport, but NOT a global angular chart. The old
"completely eliminating the need for expensive parallel transport or chart
transitions" conflated the two; the branch cut is removed by the trigonometric
embedding, not by parallelizability. Now cross-refs sec:method.

G5 (03_preliminaries:30, 02_related_work:18)
pi_1 stated at field scale in 04_method, right after the product manifold:
pi_1 of a product is the product of the factors', each factor is homotopy
equivalent to S^1, hence pi_1(M^{HxW}) = Z^{HW}, i.e. every pixel wraps
independently. Note [0,inf) is contractible so the closure does not change the
answer from the audit's (0,inf) version.
The related_work half of G5 was already moot: "non-trivial fundamental group at
every pixel" was deleted in the G2 pass (it was also a category error).
Build clean 0/0/0, conclusion still p.9.

### NEW TENSION I INTRODUCED -- handle with the cut-locus item
04_method:12 now names "the cut locus at antipodal phase", but 04_method section
"Continuous Trigonometric Embedding" still opens with "Although the angular
velocity u_theta is continuous on the circle, ...". That is the claim the audit's
executive verdict flags ("calls the circle logarithm continuous despite its
antipodal cut locus"), and it IS false: u_theta = atan2(sin d, cos d) jumps from
+pi to -pi as the phase difference crosses pi. The two sentences now sit three
subsections apart and disagree. Flagged to Marcel, awaiting that audit item.

### Cut-locus item CLOSED (2026-09-12), verified on origin/main
(Executive-verdict item "calls the circle logarithm continuous despite its
antipodal cut locus". Handled early because the G4 pass created the
contradiction; the numbered audit entry has not arrived yet.)

The old opening of 04_method "Continuous Trigonometric Embedding" said
"Although the angular velocity u_theta is continuous on the circle, the scalar
angle exhibits an artificial coordinate discontinuity". Backwards on both
counts. Now split into the two distinct objects:
  - theta in [-pi,pi) jumping at the negative real axis = COORDINATE ARTEFACT.
    Position depends on where the interval is cut; the (cos,sin) embedding
    removes it entirely.
  - log map on S^1 discontinuous at the antipode = INTRINSIC. Two minimal arcs
    of equal length pi, sign of u_theta arbitrary. No coordinate choice removes
    it and the embedding does NOT remove it.
Bound explicitly preserved: atan2 returns (-pi, pi], so only the SIGN is
ambiguous, never the magnitude -> lem:bounded_velocity and the whole few-step
argument stand untouched.
Volunteered limitation (mathematical, not measured): antipodal pairs are
measure zero, but near them both branches +-pi enter the conditional
expectation that the squared error regresses, and partly cancel.

DELIBERATELY NOT merged with appendix prop:vanishing_gradient (zero gradient of
the chordal penalty at antipodal error). Tempting because both say "something
happens as dtheta -> pi", but they are different objects: that one is the LOSS
as a function of prediction error, this one is the TARGET as a function of
endpoints. Merging them would repeat the exact error being fixed.

Sweep for other continuity claims: clean. Remaining hits are unrelated
("continuous-time", "Continuous Normalizing Flows", the embedding's own title,
and the hypothesis of thm:angular_divergence).
MINOR, left alone: 04_method:25 says the log map is "resolving the minimal
circular arc displacement", which is unique EXCEPT at the antipode. Section 4.4
now states the exception four subsections later, so this is defensible as is.

### G6 + G7 CLOSED (2026-09-12), verified on origin/main
Both graded Qualify. Cryo AGREES with both, with one correction to G7's premise.

G6 (03_preliminaries:30) -- parallelizability
Half was already fixed in the G4 pass ("completely eliminating ... chart
transitions" had been narrowed). Added Oleksii's concrete part: the frame is now
written WITHOUT a global angular coordinate, which is the whole point given G2.
With a point as (A,q), q in S^1 subset C: E_A = (1,0), E_theta = (0, iq) is a
global orthonormal frame and parallel transport preserves components in it.
Explicitly: this removes neither the periodic wrap nor the cut locus.

G7 (03_preliminaries:30, 04_method:11,17-25) -- product geodesics, completeness
PREMISE CORRECTION: the audit says ((0,inf), dA^2) is incomplete. True of the
pre-G2 text. We now use the closure, and [0,inf) with dA^2 IS a complete metric
space. The residual problem is different and sharper: [0,inf) x S^1 is a
manifold WITH BOUNDARY, so it is not GEODESICALLY complete -- an inward geodesic
hits A=0 and cannot be extended. Chen & Lipman's RFM assumes complete manifolds
without boundary, so we are outside its assumptions. Now stated, citing
chen2024flow.
CODE CONFIRMS the concern: cylindrical_unet.forward returns a raw convolution
output, no softplus, so v_A is genuinely unconstrained; solver.py:100 clamps.
Rejected reparameterisation rho = log A and said why in the paper: it changes
the radial metric AND the probability path (A_t would become geometric
interpolation), which would invalidate every existing experiment.
Also added: the minimal arc is unique except at the antipode (method:25),
matching the cut-locus paragraph in 4.4.
Build clean 0/0/0, conclusion still p.9.

### MEASURABLE, NOT YET MEASURED -- needs Marcel's call
How often the clamp at solver.py:100 actually fires is still unknown, and it is
the difference between a footnote and a finding. It IS measurable now:
outputs/state/ holds checkpoints (28G, e.g. dim_cylindrical_16_s0/last.pt,
t1_cylindrical_ot_s0/last.pt). Needs a counter in CylindricalODESolver.step plus
one sampling pass. No paper text should quantify this until it is run.

### Clamp frequency MEASURED (2026-09-12) -- was the open item from G7
Probe: counting subclass of CylindricalODESolver.step, records how often raw
m_t + v_m*dt < 0 before the clamp at solver.py:100. Script kept at
scripts/clamp_probe.py. 64 fields per config, k in {1,2,4,8,100}.

  unsz_cylindrical_ot_16_s0       16x16   0% at every k
  unsz_cylindrical_ot_32_s0       32x32   0% at every k
  un_cylindrical_ot_scnull_s0     64x64   0% at every k
  abl_cylindrical_l2_unweighted_32_s0  32x32  max 0.0007% (k=8), zeros 0.0107%
  abl_cylindrical_l2_unweighted_32_s1  32x32  max 0.0032% (k=8), zeros 0.0275%

Conclusion: 4 of 6 checkpoints never trigger it. Worst case 0.003% of updates
and 0.03% of final pixels. INACTIVE at k <= 4 in every run but one (s1 at k=4,
0.0004%) -- i.e. inactive across the few-step regime the main claim lives in.
Not numerical dust though: worst raw amplitude -0.043 against a prior of
m ~ U[0,1], so about 4% of the amplitude scale. Real excursions, just rare.

Marcel's call (correct): same paper_unet protocol, same unweighted L2, same OT
coupling, same prior -> the local checkpoints are the same setting as the
cluster runs, so the number goes into the paper without provenance hedging.
One sentence added to the method paragraph, no table.
All abl_ arms are OT too (checked ablation_loss32.yaml), so the 0% vs >0% split
is between runs, not between couplings. Sample far too small to claim anything.

### G8 CLOSED (2026-09-12) -- the only item that required touching src/
Repo commit 6c0fd4a. The cut-locus half was already fixed proactively; what G8
added and I had NOT done was a deterministic tie rule.
FINDING, sharper than the audit states: the tie was not merely unspecified, it
was PRECISION-DEPENDENT.
    float32   diff = +pi  ->  u_theta = -pi
    float64   diff = +pi  ->  u_theta = +pi
float32's pi rounds ABOVE true pi, so sin(diff) < 0 and atan2 drops to the other
branch; float64's rounds below. Same maths, opposite target, decided by the last
bit of the mantissa. Repo trains in float32.
Second finding: THREE hand-rolled copies of the wrap existed, and flow/bridge.py
used a different convention entirely -- (d+pi) % 2pi - pi, range [-pi,pi), tie at
-pi -- while its own docstring claimed [-pi, pi].
Fix: utils/complex_ops.wrap_to_pi(d) = pi - remainder(pi - d, 2pi). Tie -> +pi in
every precision; matches atan2 to rounding over 20001 points; range (-pi, pi].
Wired into manifolds/cylindrical.py (log_map, geodesic_path) and flow/bridge.py.
Tests pin tie/agreement/range. pytest 565 passed, ruff clean, mypy clean.
Paper: one sentence, and it is TRUE of the code, which was the whole point of
refusing to write the rule before implementing it.
Marcel's call: do NOT unify the dead GeodesicFlowBridge path. Left alone.

### G10 + G11 CLOSED (2026-09-12), verified on origin/main
G10 (04_method:52-57,69) "partly verified, partly overstated". AGREE.
Half was already done in the cut-locus pass (the embedding removes the
coordinate discontinuity and leaves the antipodal one). Remaining half fixed:
"the three-channel embedding, which is forced by the representation, since the
raw angle cannot be passed to a network without a branch cut" -> a network CAN
take the raw angle, it just meets a branch discontinuity. Now a choice, not a
necessity.
G11 (03_preliminaries:34-42) "verified with assumptions". AGREE.
Strictness of |z(t)| <= (1-t)A_0 + t A_1 was asserted on theta_0 != theta_1
alone. Equality holds iff the two terms are non-negative multiples of one
another, so it also needs 0 < t < 1 and both amplitudes non-zero, and the phase
condition is mod 2pi. All three stated.
Cryo addition, same defect one sentence later: prop:attenuation divided by A
without saying A_0 = A_1 = A. Assumption lived only in the preceding paragraph.
Now carried by the proposition.
Build 0/0/0, conclusion still p.9.

### STILL MISSING
G9 arrived truncated at "The cylindrical angular target satisfies (|u_\theta|"
-- no verdict, no proposed fix. NOT adjudicated. Asked Marcel for the source
file so the rest of the table can be read directly instead of pasted piecemeal.
From the executive verdict, two items remain unaccounted for:
  - divergence asserted from a proof that only establishes an upper bound
    (thm:angular_divergence ends at |theta_dot| <= ||v||/A)
  - nonsignificance read as equivalence (abstract AND Contributions)

================================================================================
## DEFERRED BACKLOG -- Marcel parked these on 2026-09-12, pick up after G12/G13
================================================================================

### D1. Divergence asserted from a proof that only gives an UPPER bound
Location: 07_appendix.tex, Proposition thm:angular_divergence.
This is a LOGIC error, not wording, and it is the one remaining hard defect.

  Statement:  "The induced angular velocity theta_dot DIVERGES as O(A^-1)
               near the origin."
  Proof ends: "|theta_dot| <= A||v||/A^2 = ||v||/A = O(A^-1)."

An upper bound of O(A^-1) does not establish divergence. theta_dot == 0 also
satisfies that bound -- e.g. a purely radial flow, where the numerator
x*ydot - y*xdot vanishes identically. So the proposition as stated does not
follow from its own proof.

What is actually true and IS provable: for the LINEAR bridge specifically,
05_experiments already notes the numerator Im(conj(z_0) z_1) = A_0 A_1 sin(dtheta)
is CONSTANT along the path, so theta_dot = const / |z_t|^2 and the peak really
does scale as the inverse square of the closest approach to the origin. That is
a sharp statement about the bridge, not a generic statement about "a generative
flow in the plane".
=> Minimal fix: restate the proposition for the linear interpolation path with a
   non-zero transverse component, where the bound is attained, OR retitle it as
   an upper bound and move the divergence claim to the bridge, where the constant
   numerator makes it exact. Prefer the latter: the empirical section already
   measures the tail index ~1.0, consistent with the bridge version.
Note the appendix corollary added in G1 (circumference 2*pi*r vs 2*pi) already
gives the geometric reason, so the repair has somewhere to attach.

### D2. Nonsignificance read as equivalence
Three live sites, all saying the same thing, so ONE decision governs all three:
  00_abstract     "...and the two geometries are statistically indistinguishable
                   at convergence."
  01_introduction "Run to convergence, the two geometries are statistically
                   indistinguishable." (inside the Contributions bullet)
  05_experiments  "...and at $k=100$ no difference is significant."
Failing to reject the null is not evidence of equivalence, especially at n = 5
seeds where power is low. 05_experiments is already the most defensible of the
three ("no difference is significant" is a statement ABOUT the test).
=> Options: (a) align abstract and intro to the experiments phrasing, which is
   free and honest; (b) run a TOST / equivalence test against a pre-stated
   margin and then the word "indistinguishable" is earned. (b) needs the per-seed
   numbers, which are on the cluster.
CAREFUL: 01_introduction's wording is MINE, from the Contributions rewrite. I
changed the voice, not the strength of the claim, so the defect predates me but
I now own the sentence.

### G13 CLOSED (2026-09-12) -- "Incorrect as a universal statement". AGREE.
z(t) = 0 requires (1-t)z_0 = -t z_1: exactly antipodal phases AND the amplitude
ratio matching t, i.e. t = A_0/(A_0+A_1). Probability zero under any continuous
phase law, so "force mass through the origin" was false as written.
Three sites fixed: 03_preliminaries (now names the crossing time and says the
phase is undefined there), 01_introduction prose, teaser caption.
Second half was SELF-INFLICTED: "low-amplitude states that do not lie on the
physical signal manifold" contradicted our own G2 redefinition to [0,inf) x S^1,
thirty lines earlier in the same section. Dropped.
NOTE: main text crossed from p.9 to p.10 here, i.e. over the ICLR 9-page limit.
Marcel: irrelevant, this is a preprint. Revisit before any ICLR submission.

### G12 CLOSED (2026-09-12) -- three parts, all shipped
(a) equal amplitudes in the proposition: already done in the G11 pass.
(b) wrapped difference. The integral assumed uniformity on [-pi,pi), true of the
    WRAPPED difference, false of the raw one (triangular on [-2pi,2pi]).
    Verified on 2e7 samples: raw density 0.070..0.117..0.070, wrapped flat 0.1250.
    CRUCIAL: the RESULT was never at risk -- |cos(./2)| has period 2pi, so both
    laws give the same expectation (measured 0.636678 vs 2/pi = 0.636620). The
    text now says so, because a reader who spots the triangular density would
    otherwise conclude the number is wrong. delta defined once, used as the
    integration variable.
(c) Marcel asked to turn this to the cylinder's advantage instead of retracting.
    Done, and it is the better argument: the coupling dependence IS the
    asymmetry. A Cartesian cost can only be quoted for a given law of delta and
    must be bought down with a coupling; |u_theta| <= pi holds for every pair
    under every coupling with no distributional assumption. Section 5 already
    stated the cylindrical half ("independently of the coupling"); the
    preliminaries now set it up.
    Same defect found in two more places and fixed with it: the 46% figure is
    also an independent-coupling number, correctly qualified in the experiments
    but quoted bare in the ABSTRACT and the CONTRIBUTIONS bullet. Both now say
    "under independent coupling", and the bullet adds that the cylindrical bound
    holds "at any coupling".
Cost named to Marcel and accepted: 36.33% becomes a conditional figure rather
than a headline number.

### G17 CLOSED (2026-09-12) -- trivial, agreed
theta_dot = Im(conj(z) zdot)/|z|^2 divides by |z|^2. Method now says "wherever
z_t != 0"; the appendix proof notes atan2 is differentiable exactly where A > 0.

### G19 CLOSED (2026-09-12) -- diagnosis accepted, remedy rejected
Oleksii is right that "the phase component of the regression target is
unbounded" is false: the Cartesian model regresses z_1 - z_0, bounded whenever
the endpoints are, and has no phase component at all. We were comparing our
target against their diagnostic.
REJECTED his fix ("present bounded u_theta as a property of the chosen
parameterisation"), which removes the claim without replacing it. Same shape as
G1: right diagnosis, deflationary remedy.
What replaced it is an apples-to-apples comparison that does exist: the
cylindrical path has theta_t = theta_0 + t u_theta, so it turns at the CONSTANT
rate u_theta with |u_theta| <= pi, while the Cartesian path's turn rate is
unbounded. Both are dtheta/dt along a path. The mechanism that matters is that a
coarse integrator must resolve that rate -- which is exactly what the few-step
results measure. Abstract now says "a rate no cylindrical path ever exceeds".

### G18 / D1 CLOSED (2026-09-12) -- the big one
Old proposition claimed divergence, proved only |theta_dot| <= ||v||/A. A purely
radial flow has c = 0, theta_dot == 0, and satisfies that bound. Did not follow.
REPLACED with what is exactly provable, verified numerically BEFORE writing:
  - numerator constant along the bridge: Im(conj(z_t) zdot_t) = Im(conj(z_0) z_1) = c
  - sup_t |theta_dot| = |c| / d^2, d = min_t |z_t|, attained at closest approach
  - equal amplitudes: closest approach at t = 1/2, sup = 2|tan(delta/2)|
    (verified vs dense-grid search over 4e5 bridges: median rel. err 9e-8)
  - delta ~ U(-pi,pi]: P(sup > x) = 1 - (2/pi)arctan(x/2) = 4/(pi x) + O(x^-3),
    Pareto index exactly 1 (verified at x = 10, 100, 1000)
TWO MENTAL-MODEL CORRECTIONS now stated in the paper:
  - the bridge does NOT approach the origin arbitrarily closely for fixed
    endpoints: d >= |c| / |z_1 - z_0|, the distance from origin to the line.
  - the divergence is in delta -> pi, NOT in A -> 0. The whole "paths fall into
    the origin" picture, which G13 started correcting, is now gone.
SCOPE STATED, NOT OVERCLAIMED: Pareto(1) fixes the amplitudes. With random
amplitudes a second route to a small numerator opens (c vanishes as either
amplitude does) and the tail is HEAVIER. My sim on m ~ U[0,1] fits index 0.91,
and the ratio to 4/(pi x) grows 1.8 -> 3.4 over x = 10..1e4, so it is not
Pareto(1) there. The paper says the measured ~1.0 is consistent with the
analysis but not derived from it.
CAUTION recorded: my sim gave P(peak > pi) = 0.53 vs the paper's 46%, but my
sim draws BOTH endpoints with uniform phase while the paper bridges from the
prior to a copula target with correlated amplitude and phase. Different
distributions -- my run is a check of the algebra, not a reproduction of Table 1.
Dependent sites updated: circumference corollary, method lemma (now quotes the
exact peak), experiments cross-ref (now to cor:pareto_tail), abstract
("singularity" -> "heavy-tailed distribution"), and the contributions bullet.
Verified: ZERO occurrences of \mathcal{O}(A^{-1}) remain anywhere in the paper.

### F9 / S5 -- NO ACTION NEEDED (already fixed earlier today)
F9 (factorised OT equality condition) was the first fix of the session, from
Oleksii's Overleaf comment. Current text already states the exact iff condition
and the mu = nu counterexample, which is what F9 asks for.
S5's "topological necessity" went in the G1 pass (grep "topological" = 0 in the
experiments). S5's other half asks for an association claim rather than a causal
one; the text already reads "is consistent with the few-step results below".
PUSHED BACK on that half -- nothing to change.

### F12, S2, S7, S12, S13 CLOSED (2026-09-12), verified on origin/main
S7 (= deferred D2). Five sites, one more than I had found: the CONCLUSION said
CyFM "matches" the baselines, the strongest equivalence claim in the paper. All
four offending sites now say "we detect no significant difference"; the fifth
was already correct. Protocol paragraph states the power caveat and what an
equivalence claim would need (prespecified margin + equivalence test).

S12. Two halves, and the first is a FREE DEFENCE we were not using: the headline
claim is an INTERSECTION over step counts and resolutions, so it is an
intersection-union test -- every component must pass on its own and the
conjunction needs NO multiplicity correction. Now stated, pre-empting the
standard "you ran many tests" objection. Second half conceded: best-of-four
selects on the evaluation data, so "flatters both geometries equally" was an
assertion; dropped, comparison made descriptive, nominal p-values disclaimed.

S13. "Equally straight" from a range of values is not an equality test, and a
lower amplitude marginal does not establish mediation. Now: similar straightness
explicitly untested, and a decomposition (consistent amplitude gap, MIXED phase)
reported as such rather than as a cause.

S2. "Exact distribution" was wrong: bridges are exact, the distribution is Monte
Carlo over 1e6 sampled endpoint pairs. Now separated, with the seed, and pointed
at the closed-form peak from the new Proposition.

F12. "No architecture can undo it" is an unproved universal impossibility.
Oleksii's suggested narrowing ("no guarantee") is WEAKER than what is provable,
so I used the stronger form instead: a factorised plan reassembles endpoints
from different samples, so the population optimum of the flow-matching objective
transports to that reassembled law; capacity does not change which target is
being fitted.

================================================================================
## SESSION TOTAL (2026-09-12)
Closed: G1-G8, G10-G13, G17-G19, F9, F12, S2, S5, S7, S12, S13, plus the
cut-locus item and both deferred items D1/D2. Every one verified by a local
pdflatex+bibtex build (0 undefined refs, 0 undefined cites, 0 overfull) and by
`git show origin/main:` rather than through the MCP server.
Never received: G9 (truncated at "|u_theta|"), G14, G15, G16.
Code: one commit, 6c0fd4a, wrap_to_pi + tests. pytest 565 passed, ruff, mypy.
Paper main text grew from 9 to 10 pages; Marcel: irrelevant for the preprint,
revisit before an ICLR submission.
