
# verify_fix.py
import numpy as np
from geometry import build_mainshaft, build_countershaft, build_case, get_countershaft_pose
from collision import is_collision_free
from config import Q_START

def verify_configuration():
    print("=" * 60)
    print("VERIFYING FIXED CONFIGURATION")
    print("=" * 60)
    
    mainshaft_cyls = build_mainshaft()
    countershaft_cyls = build_countershaft()
    case_walls = build_case()
    countershaft_pose = get_countershaft_pose()
    
    # Use the adjusted start position
    q_start = Q_START
    
    print(f"\nCountershaft position: y={countershaft_pose[1]}mm")
    print(f"Mainshaft start position: y={q_start[1]}mm")
    print(f"Vertical separation: {q_start[1] - countershaft_pose[1]:.1f}mm")
    
    # Check collision
    collision_free = is_collision_free(
        q_start, mainshaft_cyls, countershaft_cyls, 
        case_walls, countershaft_pose
    )
    
    print(f"\nStart configuration collision-free? {collision_free}")
    
    if collision_free:
        print("\n✓ SUCCESS! Start configuration is valid.")
        
        # Test some moves
        print("\nTesting simple moves:")
        test_moves = [
            ([50, 0, 0], "Move right 50mm"),
            ([0, 20, 0], "Move up 20mm"),
            ([0, 0, 50], "Move forward 50mm"),
            ([100, 0, -50], "Move right and down"),
        ]
        
        for delta, desc in test_moves:
            q_test = q_start.copy()
            q_test[:3] += delta
            free = is_collision_free(q_test, mainshaft_cyls, countershaft_cyls, 
                                     case_walls, countershaft_pose)
            print(f"  {desc}: {'✓ Free' if free else '✗ Collision'}")
    else:
        print("\n✗ Still in collision. Need further adjustment.")
        
        # Find minimum safe y-offset
        print("\nFinding minimum safe y-offset:")
        for y_offset in [30, 40, 50, 60, 70, 80]:
            q_test = np.array([0.0, y_offset, 0.0, 1.0, 0.0, 0.0, 0.0])
            free = is_collision_free(q_test, mainshaft_cyls, countershaft_cyls,
                                     case_walls, countershaft_pose)
            print(f"  y={y_offset}mm: {'✓ Free' if free else '✗ Collision'}")
            if free:
                print(f"\n  Recommended start y-offset: {y_offset}mm")
                break

if __name__ == "__main__":
    verify_configuration()