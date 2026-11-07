Rule used to choose the two thresholds (fixed from the label inventory alone, before any model was trained):

1. **Candidate bands.** Lower threshold in [4.25, 4.50] m^2, upper threshold in [4.60, 5.00] m^2, on a 0.005 m^2 grid.
   The bands span the hours-tertile and natural-break candidates of `build/SIZE_INVENTORY.md` (4.30 / 4.45 and
   4.62 / 4.67) and the round proposal 4.35 / 4.85.
2. **Support constraint.** Every size class must hold, for BEV and for ICE separately, at least 100 h, 3 car models and
   5 identity groups with a drive table, so that the size task can be evaluated inside each powertrain and the
   powertrain task inside each size class. The unconstrained schemes fail this: hours tertiles leave MID x BEV with
   17 h and the natural breaks with 1 h, because the BEV fleet is either compact (Bolt, Niro, Born) or large (Rivian).
3. **Balance.** Among the admissible pairs, take those that maximise the smallest of the six size x {BEV, ICE} cells
   (309 h). This plateau is lower in [4.385, 4.50] and upper in [4.805, 5.00].
4. **Boundary placement inside the plateau.** The lower threshold sits at the midpoint of the widest gap between
   platform footprints (Kia Sportage 5th gen 4.421 | Acura TLX 1st gen 4.439), so no platform is within 0.008 m^2 of
   it. The upper threshold sits inside the gap that separates the two-row mid-size vehicles (Ioniq 5 facelift 4.875
   is the largest) from the first full-size sedan, three-row SUV and pickup (Hyundai Genesis 4.896, Honda Pilot 4.915,
   Ford Maverick 4.929), rounded to 0.01 m^2; it keeps both Ioniq 5 generations in one class. The widest gap of the upper
   plateau (4.973 to 5.06) was not used because it would put the three-row Palisade, Pilot, MDX 4th gen, GV80 and the
   Tesla Model S into MID.
5. **Reading.** SMALL = compact cars and compact crossovers (Civic, Corolla, Prius, Elantra, RAV4, CR-V, Tiguan,
   Escape, Sportage, Bolt, Niro, Born); MID = mid-size sedans and two-row mid-size SUVs (Camry, Accord, Sonata, Tucson,
   Santa Fe, RDX, MDX 3rd gen, Model 3, Model Y, EV6, GV60, Ioniq 5, Ioniq 6, Mach-E); LARGE = full-size sedans,
   three-row SUVs, minivans and pickups (Palisade, Pilot, Explorer, MDX 4th gen, Model S, Model X, Rivian R1, EV9,
   Sienna, F-150, RAM).
