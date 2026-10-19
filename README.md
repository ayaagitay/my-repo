# Part A: Human Motion to Robot
    
## How did you determine what the columns in the CSV represent?
The CSV has no header and I used help of AI, which suggested me to start from the Unitree LAFAN1 format (root position, root quaternion, then joint angles) and I checked it against the data and the robot model. 
The file has 36 columns, and the 29-DoF G1 in MuJoCo has 36 position values: 7 for the floating base and 29 for the joints. 
`play_motion.py` asserts that the two match.

Then I checked each group. Column 3 is about 0.77 in the first row, a normal pelvis height in metres, so columns 1-3 are the root position. 
Columns 4-7 have a length of 1, so they are a quaternion. The last one is close to 1, so the order is x, y, z, w, and the script reorders it because MuJoCo wants w first. 
The last 29 columns are joint angles in radians, in the same order as the joints printed from the XML: legs, waist, left arm, right arm. All values are inside the joint limits.

Finally I rendered the motion. The robot spins in place and bows, and the right limbs move as expected, so the order is correct.