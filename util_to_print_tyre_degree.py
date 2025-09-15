import math

n = 100
radius = 0.0875

for i in range(n):
    angle_deg = i * 360 / n
    angle_rad = math.radians(angle_deg)
    x = radius * math.cos(angle_rad)
    z = radius * math.sin(angle_rad)
    contype = 4 if i % 2 == 1 else 8
    conaffinity = 4 if i % 2 == 1 else 8
    print(f'''<body name="rearwheel_cylinder{i}" pos="{x:.6f} 0 {z:.6f}" euler="90 {-angle_deg:.6f} 0" >
    <joint type="hinge" name="rear_wheel_cylinder{i}_joint" pos="0 0 0" axis="0 0 1" range="-inf inf" damping="0.001"/>
    <geom name="rear_omniwheel_cylinder{i}" type="mesh" mesh="cylinder_of_tire" mass="0.0001" class="后轮_default" material="front_wheel_material" friction="0.8 0.01 0.01" contype="{contype}" conaffinity="{conaffinity}" condim="4" priority="10"/>
</body>''')