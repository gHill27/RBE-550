// primary_shaft.scad
// Standalone file - includes library and calls primary shaft assembly

include <transmission.scad>;

// Rotate to match the orientation used in main assembly
rotate([0, 90, 0])
primary_shaft_assembly();