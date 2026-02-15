# all joints information
dic_joints = {
# Head
#0
"HS":{"name":"HeadYaw", "min":-2.0857, "max":2.0857, "default": 0.0184},      
#1
"HUD":{"name":"HeadPitch", "min":-0.7068, "max":0.6371, "default": 0},
# left                                              
#2
"LSUD":{"name":"LShoulderPitch", "min":-2.0857, "max": 2.0857, "default": 1.765},
#3
"LSS":{"name":"LShoulderRoll", "min":0.0087, "max":1.5620,"default":0.09}, 
#4
"LES":{"name":"LElbowYaw", "min":-2.0857, "max":2.0857, "default":-1.718},
#5
"LEUD":{"name":"LElbowRoll", "min":-1.5620, "max":-0.0087,"default":-0.119},
#6
"LWS":{"name":"LWristYaw", "min":-1.8239, "max":1.8239, "default":0.067},
#7
"LH":{"name":"LHand", "min":0.0, "max":1.0, "default":0.678},
# right
#8
"RSUD":{"name":"RShoulderPitch", "min":-2.0857, "max":2.0857,"default":1.749},
#9
"RSS":{"name":"RShoulderRoll", "min":-1.5620, "max":-0.0087,"default":-0.105},
#10
"RES":{"name":"RElbowYaw", "min":-2.0857, "max":2.0857, "default":1.687},
#11
"REUD":{"name":"RElbowRoll", "min":0.0087, "max":1.5620, "default":0.101},
#12
"RWS":{"name":"RWristYaw", "min":-1.8239, "max":1.8239, "default":-0.054},
#13
"RH":{"name":"RHand", "min":0.0, "max":1.0, "default":0.677},
# bottom
#14
"BS":{"name":"HipRoll", "min":-0.5149, "max":0.5149, "default":-0.0153},
#15
"BUD":{"name":"HipPitch", "min":-1.0385, "max":1.0385,"default":-0.026},
#16
"KUD":{"name":"KneePitch", "min":-0.5149, "max":0.5149, "default":0.0031}
}

# absolute joint order
joint_order = [
    "HS", "HUD", 
    "LSUD", "LSS", "LES", "LEUD", "LWS", "LH",
    "RSUD", "RSS", "RES", "REUD", "RWS", "RH",
    "BS", "BUD", "KUD"
]