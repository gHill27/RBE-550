// Idealized model of SM-465 transmission
// Worcester Polytechnic Institute
// RBE-550 Motion Planning
// Daniel Montrallo Flickinger, PhD
//
// Produce a projected view of the counter shaft,
// to generate dimensions, for the 'transmission' assignment



use <transmission.scad>

// produce a projection of the primary shaft:
projection() rotate([0, 90, 0]) secondary_shaft_assembly();
