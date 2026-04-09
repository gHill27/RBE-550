// secondary_shaft.scad - 3D model of secondary shaft assembly
// Standalone version for collision checking

// Parameters (copied from transmission.scad)
gear_thickness = 25;
collar_thickness = 30;
syncro_thickness = 8;

bearing_offset_height = 215;
cs_bearing_offset_height = 100;
gear1_radius = 65;
gear2_radius = 53;
gear3_radius = 45;
gear_rev_primary = 60;
case_thickness = 25;
case_length = 280;

prim_sec_shaft_offset = bearing_offset_height - cs_bearing_offset_height;

gear1_sec_radius = prim_sec_shaft_offset - gear1_radius;
gear2_sec_radius = prim_sec_shaft_offset - gear2_radius;
gear3_sec_radius = prim_sec_shaft_offset - gear3_radius;
gear_rev_secondary = prim_sec_shaft_offset - gear_rev_primary - 15;
countergear_radius = 70;

secondary_shaft_length = case_length + 2*case_thickness;
shaft_radius = 18;

// Modules
module shaft(length, radius) {
    color("gray")
    difference() {
        cylinder(h = length, r = radius, center=true);
        translate([0,0,length/2 + 0.01])
        cylinder(h = 0.1*length, r = 0.5*radius, center=true);
        translate([0,0,-length/2 - 0.01])
        cylinder(h = 0.1*length, r = 0.5*radius, center=true);
    }
}

module gear(radius) {
    color([1.0, 1.0, 1.0, 0.95])
    linear_extrude(height = gear_thickness, center=true, twist = 15)
    gear_core_straight(100, [0.9*radius, radius, radius, 0.9*radius]);
}

module straight_gear(radius) {
    color([1.0, 1.0, 1.0, 0.95])
    linear_extrude(height = gear_thickness, center=true)
    gear_core_straight(100, [0.9*radius, radius, radius, 0.9*radius]);
}

module gear_core_straight(num, radii) {
    polygon([for (i=[0:num-1], a=i*360/num, r=radii[i%len(radii)]) [ r*cos(a), r*sin(a) ]]);
}

module secondary_shaft_assembly() {
    // Center the shaft at origin
    total_length = secondary_shaft_length;
    center_offset = -total_length/2 + 38;
    
    shaft(secondary_shaft_length, shaft_radius);
    
    // first gear
    translate([0, 0, center_offset])
    gear(gear1_sec_radius);
    
    // reverse gear
    translate([0, 0, center_offset + syncro_thickness + collar_thickness + gear_thickness])
    straight_gear(gear_rev_secondary);
    
    // second gear
    translate([0, 0, center_offset + 2 * syncro_thickness + collar_thickness + 2 * gear_thickness])
    gear(gear2_sec_radius);
    
    // third gear
    translate([0, 0, center_offset + 2 * syncro_thickness + collar_thickness + 3 * gear_thickness])
    gear(gear3_sec_radius);
    
    // counter gear
    translate([0, 0, total_length/2 - case_thickness - 5])
    gear(countergear_radius);
}

// Render the assembly
secondary_shaft_assembly();