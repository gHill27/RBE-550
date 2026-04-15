// secondary_shaft.scad
// Standalone file - includes library and calls secondary shaft assembly

include <transmission.scad>;

// Rotate to match the orientation used in main assembly
rotate([0, 90, 180])
secondary_shaft_assembly();