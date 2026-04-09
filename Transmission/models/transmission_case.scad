// transmission_case.scad - 3D model of transmission case
// Standalone version for collision checking

// Parameters
bearing_radius = 40;
bearing_offset_height = 215;
cs_bearing_offset_height = 100;
case_thickness = 25;
case_length = 280;
case_width = 210;
case_height = 300;

module case() {
    eps = 0.01;
    union() {
        base();
        
        // ends (with bearing holes)
        translate([0.5*case_length + 0.5*case_thickness - eps, 0, 0.5*case_height])
            sidewall_bearing();
        
        translate([-0.5*case_length - 0.5*case_thickness + eps, 0, 0.5*case_height])
            sidewall_bearing();
        
        // sides
        translate([0, -0.5*case_width + 0.5*case_thickness, 0.5*case_height + 0.5*case_thickness - eps])
        sidewall();
    
        translate([0, 0.5*case_width - 0.5*case_thickness, 0.5*case_height + 0.5*case_thickness - eps])
        sidewall();
    }
}

module sidewall() {
    difference() {
        cube(size = [case_length, case_thickness, case_height], center = true);
        
        // PTO access hole
        translate([-0.25 * case_length, 0, -0.2 * case_height])
        cube(size = [0.35 * case_length, 1.2 * case_thickness, 0.45 * case_height], center = true);
    }
}

module sidewall_bearing() {
    difference() {
        cube(size = [case_thickness, case_width, case_height + case_thickness], center = true);
        
        // mainshaft bearing hole
        translate([0, 0, bearing_offset_height + case_thickness - 0.5*(case_height + case_thickness)])
        rotate([0, 90, 0])
        cylinder(h = case_thickness + 2, r = bearing_radius, center = true);
        
        // countershaft bearing hole
        translate([0, 0, cs_bearing_offset_height + case_thickness - 0.5*(case_height + case_thickness)])
        rotate([0, 90, 0])
        cylinder(h = case_thickness + 2, r = bearing_radius, center = true);
    }
}

module base() {
    cube(size = [case_length, case_width, case_thickness], center = true);
}

// Render the case
color([0.5, 0.5, 1.0, 0.7])
case();