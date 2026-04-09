// primary_shaft.scad - 3D model of primary shaft assembly
// Standalone version for collision checking

// Parameters (copied from transmission.scad)
gear_thickness = 25;
collar_thickness = 30;
syncro_thickness = 8;

syncro12_radius = 48;
syncro34_radius = 45;

gear1_radius = 65;
gear2_radius = 53;
gear3_radius = 45;
gear_rev_primary = 60;

collar12_radius = 60;
collar34_radius = 60;

primary_shaft_length = 330;
shaft_radius = 18;

// Import the gear generation modules from transmission.scad
// Copy the necessary modules here (since we can't use <transmission.scad> easily)

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

module collar(radius) {
    color([0.6, 0.6, 0.8, 0.85])
    union() {
        cylinder(h = 0.35 * collar_thickness, r1 = 0.9 * radius, r2 = radius, center=true);
        translate([0, 0, 0.35 * collar_thickness/2 + 0.3 * collar_thickness/2])
            cylinder(h = 0.3 * collar_thickness, r = 0.8 * radius, center=true);
        translate([0, 0, 0.35 * collar_thickness + 0.3 * collar_thickness + 0.35 * collar_thickness/2])
            cylinder(h = 0.35 * collar_thickness, r1 = radius, r2 = 0.9 * radius, center=true);
    }
}

module syncro(radius) {
    color("#c4ba21")
    linear_extrude(height = syncro_thickness, center=true)
    gear_core_straight(200, [radius, 0.95*radius, 0.95*radius, radius]);
}

module primary_shaft_assembly() {
    // Center the shaft at origin for easier positioning
    total_length = primary_shaft_length;
    
    // Shaft (centered)
    shaft(primary_shaft_length, shaft_radius);
    
    // Calculate offsets from center
    shaft_offset = 20;
    center_offset = -total_length/2 + shaft_offset + 10;
    
    // 3-4 syncro
    translate([0, 0, center_offset + shaft_offset])
    syncro(syncro34_radius);
    
    // 3-4 collar
    translate([0, 0, center_offset + shaft_offset + syncro_thickness])
    collar(collar34_radius);
    
    // 3-4 syncro
    translate([0, 0, center_offset + shaft_offset + syncro_thickness + collar_thickness])
    syncro(syncro34_radius);
    
    // 3rd gear
    translate([0, 0, center_offset + shaft_offset + 2 * syncro_thickness + collar_thickness])
    gear(gear3_radius);
     
    // 2nd gear
    translate([0, 0, center_offset + shaft_offset + 2 * syncro_thickness + collar_thickness + gear_thickness])
    gear(gear2_radius);
    
    // 1-2 syncro
    translate([0, 0, center_offset + shaft_offset + 2 * syncro_thickness + collar_thickness + 2 * gear_thickness])
    syncro(syncro12_radius);
    
    // reverse driven gear
    translate([0, 0, center_offset + shaft_offset + 3 * syncro_thickness + collar_thickness + 2 * gear_thickness])
    straight_gear(gear_rev_primary);
    
    // 1-2 collar
    translate([0, 0, center_offset + shaft_offset + 3 * syncro_thickness + collar_thickness + 3 * gear_thickness])
    collar(collar12_radius);
    
    // 1-2 syncro
    translate([0, 0, center_offset + shaft_offset + 3 * syncro_thickness + 2 * collar_thickness + 3 * gear_thickness])
    syncro(syncro12_radius);
 
    // 1st gear
    translate([0, 0, center_offset + shaft_offset + 4 * syncro_thickness + 2 * collar_thickness + 3 * gear_thickness])
    gear(gear1_radius);
}

// Render the assembly
primary_shaft_assembly();