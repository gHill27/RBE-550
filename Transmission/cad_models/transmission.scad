// Idealized model of SM-465 transmission
// Worcester Polytechnic Institute
// RBE-550 Motion Planning
// Daniel Montrallo Flickinger, PhD
//
// Library for the 'transmission' assignment




// gear, collar, and syncro parameters:

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


// case parameters:

bearing_radius = 40;
bearing_offset_height = 215;
cs_bearing_offset_height = 100;

case_thickness = 25;
case_length = 280;
case_width = 210;
case_height = 300;


// calculate parameters for the secondary shaft

prim_sec_shaft_offset = bearing_offset_height - cs_bearing_offset_height;

gear1_sec_radius = prim_sec_shaft_offset - gear1_radius;
gear2_sec_radius = prim_sec_shaft_offset - gear2_radius;
gear3_sec_radius = prim_sec_shaft_offset - gear3_radius;

gear_rev_secondary = prim_sec_shaft_offset - gear_rev_primary - 15;
countergear_radius = 70;


// shaft parameters:

primary_shaft_length = 330;
secondary_shaft_length = case_length + 2*case_thickness;

shaft_radius = 18;




module transmission() {
    
    
    translate([-85,
               0,
               bearing_offset_height + 0.5*       case_thickness])
    rotate([0, 90, 0])
    primary_shaft_assembly();
    
    
    translate([0.5*case_length + case_thickness,
               0,
               cs_bearing_offset_height + 0.5*case_thickness])
    rotate([0, 90, 180])
    secondary_shaft_assembly();
    
    



    color([0.5, 0.5, 1.0, 0.45])
    case();
    
    
}





module transmission_apart() {
    
    
    translate([-100,
               0,
               120 + bearing_offset_height + 0.5*       case_thickness])
    rotate([0, 115, 0])
    primary_shaft_assembly();
    
    
    translate([0.5*case_length + case_thickness,
               0,
               cs_bearing_offset_height + 0.5*case_thickness])
    rotate([0, 90, 180])
    secondary_shaft_assembly();
    
    



    color([0.5, 0.5, 1.0, 0.45])
    case();
    
    
}


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
        
        translate([0,
                   -0.5*case_width + 0.5*case_thickness,
                    0.5*case_height + 0.5*case_thickness - eps])
        sidewall();
    
        translate([0,
                   0.5*case_width - 0.5*case_thickness,
                   0.5*case_height + 0.5*case_thickness - eps])
        sidewall();
        
    }
    
}


module sidewall() {
    difference() {
        cube(size = [case_length, case_thickness, case_height],
             center = true);
        
        // PTO access hole
        translate([-0.25 * case_length, 0, -0.2 * case_height])
        cube(size = [0.35 * case_length, 1.2 * case_thickness, 0.45 * case_height],
             center = true);
    }
    
}

module sidewall_bearing() {
    difference() {
        cube(size = [case_thickness, case_width, case_height + case_thickness],
             center = true);
        
        // mainshaft bearing
        translate([0, 0, bearing_offset_height +case_thickness - 0.5 *(case_height + case_thickness)])
        rotate([0, 90, 0])
        cylinder(h = case_thickness + 1, r = bearing_radius, center = true);
        
        
        // countershaft bearing
        translate([0, 0, cs_bearing_offset_height +case_thickness - 0.5 *(case_height + case_thickness)])
        rotate([0, 90, 0])
        cylinder(h = case_thickness + 1, r = bearing_radius, center = true);
        
        
    }
    
}


module base() {
    cube(size = [case_length, case_width, case_thickness],
         center = true);
}



module primary_shaft_assembly() {
     
 
    shaft(primary_shaft_length, shaft_radius); 
 
    shaft_offset = 20;
     
    // 3-4 syncro
    translate([0, 0, shaft_offset])
    syncro(syncro34_radius);
    
    // 3-4 collar
    translate([0, 0, shaft_offset + syncro_thickness])
    collar(collar34_radius);
    
    // 3-4 syncro
    translate([0, 0, shaft_offset + syncro_thickness + collar_thickness])
    syncro(syncro34_radius);
    
    
    // 3rd gear
    translate([0, 0, shaft_offset + 2 * syncro_thickness + collar_thickness])
    gear(gear3_radius);
     
    // 2nd gear
    translate([0, 0, shaft_offset + 2 * syncro_thickness + collar_thickness + gear_thickness])
    gear(gear2_radius);
    
    // 1-2 syncro
    translate([0, 0, shaft_offset + 2 * syncro_thickness + collar_thickness + 2 * gear_thickness])
    syncro(syncro12_radius);
    
    // reverse driven gear
    translate([0, 0, shaft_offset + 3 * syncro_thickness + collar_thickness + 2 * gear_thickness])
    straight_gear(gear_rev_primary);
    
    // 1-2 collar
    translate([0, 0, shaft_offset + 3 * syncro_thickness + collar_thickness + 3 * gear_thickness])
    collar(collar12_radius);
    
    // 1-2 syncro
    translate([0, 0, shaft_offset + 3 * syncro_thickness + 2 * collar_thickness + 3 * gear_thickness])
    syncro(syncro12_radius);
 
    // 1st gear
    translate([0, 0, shaft_offset + 4 * syncro_thickness + 2 * collar_thickness + 3 * gear_thickness])
    gear(gear1_radius);

}
 
 
 
module secondary_shaft_assembly() {
     
 
    shaft(secondary_shaft_length, shaft_radius); 
 
    shaft_offset = 38;
    

    // first gear
    translate([0, 0, shaft_offset])
    gear(gear1_sec_radius);
    
    // reverse gear
    translate([0, 0, shaft_offset + syncro_thickness + collar_thickness + gear_thickness])
    straight_gear(gear_rev_secondary);
    
    // second gear
    translate([0, 0, shaft_offset + 2 * syncro_thickness + collar_thickness + 2 * gear_thickness])
    gear(gear2_sec_radius);
    
    // third gear
    translate([0, 0, shaft_offset + 2 * syncro_thickness + collar_thickness + 3 * gear_thickness])
    gear(gear3_sec_radius);
    
    // counter gear
    translate([0, 0, secondary_shaft_length - 2 * case_thickness - 5])
    gear(countergear_radius);
    
}



module shaft(length, radius) {
    color("gray")
    difference() {
        cylinder(h = length, r = radius);
        
        translate([0,0,length+0.01])
        rotate([180,0,0])
        cylinder(h = 0.1*length, r = 0.5*radius);
        
        translate([0,0,-0.01])
        cylinder(h = 0.1*length, r = 0.5*radius);
        
    }
    // TODO: add splines for fun
}
 
module gear(radius) {
     color([1.0, 1.0, 1.0, 0.95])
        linear_extrude(height = gear_thickness,
                       twist = 15)
        gear_core_straight(100,
                [0.9*radius,
                 radius,
                 radius,
                 0.9*radius]);
}

module straight_gear(radius) {
     color([1.0, 1.0, 1.0, 0.95])
        linear_extrude(height = gear_thickness)
            gear_core_straight(100,
                [0.9*radius,
                 radius,
                 radius,
                 0.9*radius]);
} 

module gear_core_straight(num, radii) {
  function r(a) = (floor(a / 10) % 2) ? 10 : 8;
  polygon([for (i=[0:num-1], a=i*360/num, r=radii[i%len(radii)]) [ r*cos(a), r*sin(a) ]]);
}


module collar(radius) {
    color([0.6, 0.6, 0.8, 0.85])
    union() {
        cylinder(h = 0.35 * collar_thickness,
                 r1 = 0.9 * radius,
                 r2 = radius);
        translate([0, 0, 0.35 * collar_thickness])
            cylinder(h = 0.3 * collar_thickness,
                     r = 0.8 * radius);
        translate([0, 0, 0.65 * collar_thickness])
            cylinder(h = 0.35 * collar_thickness,
                     r1 = radius,
                     r2 = 0.9 * radius);
    }
}

module syncro(radius) {
    color("#c4ba21")
        linear_extrude(height = syncro_thickness)
            gear_core_straight(200,
                [radius,
                 0.95*radius,
                 0.95*radius,
                 radius]);
}
 
 
 