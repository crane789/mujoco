# -*- coding: utf-8 -*-
import sys
import os
import mujoco
import mujoco.viewer
import numpy as np
import time
import csv
import scipy.spatial.transform
import matplotlib.pyplot as plt
import matplotlib
import json
from datetime import datetime

MODEL_PATH = r"C:\mujoco-3.3.3-windows-x86_64\new_T4\T4_with_idleg.xml"
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b'carmainpart_body')

# 获取执行器ID和关节ID

front_leg_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, b'front_leg')
front_wheel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, b'front_wheel') 
rear_wheel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, b'rear_wheel')
idleg_slide_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, b'idleg_slide')  # 支撑腿伸缩执行器
    
# 获取关节ID用于读取实际位置
front_leg_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, b'front_leg')
front_wheel_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, b'front_wheel')  # 前轮关节
idleg_slide_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, b'idleg_slide')  # 支撑腿关节
    
# 获取关节在qpos中的正确地址
front_leg_qpos_addr = -1
front_wheel_qpos_addr = -1
idleg_slide_qpos_addr = -1
if front_leg_joint_id >= 0:
    front_leg_qpos_addr = model.jnt_qposadr[front_leg_joint_id]
if front_wheel_joint_id >= 0:
    front_wheel_qpos_addr = model.jnt_qposadr[front_wheel_joint_id]
if idleg_slide_joint_id >= 0:
    idleg_slide_qpos_addr = model.jnt_qposadr[idleg_slide_joint_id]
    
print("[OK] 成功获取执行器和关节ID")
print(f"  前车把关节ID: {front_leg_joint_id}, qpos地址: {front_leg_qpos_addr}")
print(f"  前轮关节ID: {front_wheel_joint_id}, qpos地址: {front_wheel_qpos_addr}")
print(f"  支撑腿关节ID: {idleg_slide_joint_id}, qpos地址: {idleg_slide_qpos_addr}")
print(f"  模型总关节数: {model.njnt}, 总执行器数: {model.nu}")
        

roll_list = []
roll_rate_list = []
time_list = []
front_leg_ctrl_list = []  # 记录前车把控制信号的列表
front_leg_actual_list = []  # 记录前车把实际位置的列表
front_wheel_ctrl_list = []  # 记录前轮控制信号的列表（目标转速）
front_wheel_actual_list = []  # 记录前轮实际转速的列表
idleg_ctrl_list = []  # 记录支撑腿控制信号的列表
idleg_actual_list = []  # 记录支撑腿实际位置的列表
robot_pos_x_list = []  # 记录机器人X位置的列表
robot_pos_y_list = []  # 记录机器人Y位置的列表

prev_roll = None
prev_front_wheel = None
dt = model.opt.timestep

# ===== 支撑腿控制参数 (基于T4ctrl/config.py) =====
# 机器人几何参数
Rw = 0.1  # 车轮半径 (m)
L = 0.45  # 轴距 (m)
H = 0.35  # 车架平面到轮子中心的垂直距离 (m)
lh = Rw + H  # 车架平面到地面的高度 (m)
la = 0.28  # 车架质心面到斜支撑腿的横向距离 (m)
theta0 = -np.radians(20)  # 支撑腿安装角度 (rad)
ls = lh / np.cos(theta0)  # 支撑腿的原长，定义为竖直时支撑腿的长度 (m)
h0 = 0  # 支撑腿原长的修正项 (m)

def deg_to_rad(deg):
    """角度转弧度"""
    return deg * np.pi / 180.0

def dps_to_rad_per_sec(dps):
    """度每秒转弧度每秒"""
    return dps * np.pi / 180.0

def interpolate_value(start_val, end_val, progress):
    """线性插值，progress从0到1"""
    return start_val + (end_val - start_val) * progress

def theta2h(theta_deg):
    """
    根据目标侧倾角计算支撑腿需要的长度
    基于T4ctrl/bcid.py的theta2h函数
    
    Args:
        theta_deg: 目标侧倾角 (度)
    
    Returns:
        lg: 地面横向距离 (m)
        h: 支撑腿伸缩长度 (m)，负数表示收缩
    """
    theta = np.radians(theta_deg)
    
    # 优化后的lg计算：原始lg减去侧倾角对应的轮子半径投影
    lg_original = lh*np.sin(theta) + la*np.cos(theta) - np.tan(theta+theta0)*(lh*np.cos(theta)-la*np.sin(theta))
    lg = lg_original - np.sin(theta) * Rw
    
    h = (lh*np.cos(theta) - la*np.sin(theta)) / np.cos(theta+theta0) - ls - h0
    
    return lg, h

def thetaDot2hDot(h, theta_deg, thetaDot_dps):
    """
    根据侧倾角速度计算支撑腿需要的运动速度
    基于T4ctrl/bcid.py的thetaDot2hDot函数
    
    Args:
        h: 当前支撑腿伸缩位置 (m)
        theta_deg: 当前侧倾角 (度)
        thetaDot_dps: 侧倾角速度 (度/秒)
    
    Returns:
        hDot: 支撑腿运动速度 (m/s)
        lgDot: 地面横向速度 (m/s)
    """
    theta = np.radians(theta_deg)
    thetaDot = np.radians(thetaDot_dps)
    
    hDot = thetaDot/np.cos(theta+theta0)*((h+ls+h0)*np.sin(theta+theta0)-la*np.cos(theta)-lh*np.sin(theta))
    lgDot = thetaDot/np.cos(theta+theta0)*(lh*np.cos(theta0)-(ls+h+h0)+la*np.sin(theta0))
    
    return hDot, lgDot

# 运动规划PLAN - 修改为专门测试前轮PID的版本
delta_1 = 20  #起始delta
delta_2 = 60 #目标delta，加速到该角度并保持
delta_end = -10 #结束状态delta，可以不是0
omega_st = 100 # 起始前轮转速
omega_mid = 429  # 加速到该转速并保持
omega_ed = 0  # 结束状态前轮转速
waitTime = 3
runTime = 8  # 延长运行时间以观察收敛效果
rollSpeed = 4
phi_rdot_st = 0
phi_rdot_ed = 0
theta = 0
l_leg = 0.01

# 前轮PID测试PLAN：专注于前轮转速控制
PLAN = [
    # 步骤1: 等待阶段 - 所有参数为0
    ["bcid", waitTime, delta_1, delta_1, omega_st, omega_st, 'pos', 0, 0, 0, 0],
    # 步骤2: 前轮加速测试 - 从0加速到目标转速
    ["bcid", waitTime, delta_1, delta_2, omega_st, omega_mid, 'pos', 0, 0, 0, 0],
    # 步骤3: 保持高速运行 - 测试稳态性能
    ["bcid", runTime, delta_2, delta_2, omega_mid, omega_mid, 'pos', 0, 0, 0, 0],
    # 步骤4: 减速测试 - 从高速减速到0
    ["bcid", waitTime, delta_2, delta_end, omega_mid, omega_ed, 'pos', 0, 0, 0, 0],
]

print("PLAN已配置为前轮PID测试模式，专注于转速控制性能")
if l_leg == 0:
    print("支撑腿设置为动态调节模式 (根据theta参数调节长度)")
else:
    print(f"支撑腿设置为固定长度模式: {l_leg:.3f}m")

def execute_plan_step(plan_step, current_time, step_start_time, previous_values=None, last_control_values=None):
    """
    执行PLAN中的一个步骤
    plan_step格式: ["bcid", duration, deltaF_st, deltaF_ed, omega_f_st, omega_f_ed, 'pos', theta, theta_dot, phi_rdot_st, phi_rdot_ed]
    根据全局变量l_leg控制支撑腿模式：
    - l_leg == 0: 动态模式，根据theta和theta_dot参数调节支撑腿长度
    - l_leg != 0: 固定模式，支撑腿始终保持l_leg长度
    返回True如果步骤完成，False如果仍在进行中
    """
    try:
        duration = plan_step[1]  # 运动时间
        deltaF_st = plan_step[2]  # 前车把起始角度(度)
        deltaF_ed = plan_step[3]  # 前车把终止角度(度)
        omega_f_st = plan_step[4]  # 前轮起始转速(dps)
        omega_f_ed = plan_step[5]  # 前轮终止转速(dps)
        target_theta = plan_step[7]  # 目标侧倾角(度)
        theta_dot = plan_step[8]  # 侧倾角速度(度/秒)
        phi_rdot_st = plan_step[9]   # 后轮起始转速(dps)
        phi_rdot_ed = plan_step[10]  # 后轮终止转速(dps)
        
        # 如果有前一步的值，使用前一步的结束值作为当前步的起始值，实现真正的连续性
        if previous_values is not None:
            deltaF_st = previous_values['deltaF']
            omega_f_st = previous_values['omega_f']
            phi_rdot_st = previous_values['phi_rdot']
        
        # 计算当前步骤的进度 (0到1)
        elapsed_time = current_time - step_start_time
        progress = min(elapsed_time / duration, 1.0) if duration > 0 else 1.0
        
        # 使用平滑的S曲线插值，减少突变
        smooth_progress = 3 * progress**2 - 2 * progress**3  # S曲线插值
        
        # 插值计算当前目标值
        current_deltaF = interpolate_value(deltaF_st, deltaF_ed, smooth_progress)  # 度
        current_omega_f = interpolate_value(omega_f_st, omega_f_ed, smooth_progress)  # dps
        current_phi_rdot = interpolate_value(phi_rdot_st, phi_rdot_ed, smooth_progress)  # dps
        
        # 单位转换：度→弧度，dps→rad/s
        front_leg_target = deg_to_rad(current_deltaF)
        front_wheel_target = dps_to_rad_per_sec(current_omega_f)
        rear_wheel_target = dps_to_rad_per_sec(current_phi_rdot)
        
        # ===== 支撑腿控制逻辑 (已注释掉) =====
        # 获取当前实际侧倾角
        # quat = data.xquat[body_id]  # [w, x, y, z]
        # euler = scipy.spatial.transform.Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_euler('xyz')
        # current_theta_rad = euler[0]  # 当前侧倾角，单位：弧度
        # current_theta_deg = np.degrees(current_theta_rad)  # 转换为度
        
        # 根据全局l_leg变量选择支撑腿控制模式
        # if abs(l_leg) < 1e-6:  # l_leg接近0，使用动态模式
        #     # 动态模式：基于theta和theta_dot调节支撑腿长度
        #     # 使用PLAN中的theta参数进行动态调节
        #     theta_error = target_theta - current_theta_deg
        #     idleg_slide_target = -0.05 + theta_error * 0.01  # 简单的比例控制
        #     control_mode = "动态调节模式"
        # else:
        #     # 固定模式：支撑腿保持l_leg长度，完全忽略PLAN中的theta参数
        #     idleg_slide_target = l_leg
        #     control_mode = f"固定长度模式"
        #     # 在固定模式下，直接跳过theta相关的计算
        
        # 获取当前支撑腿位置（用于调试显示）
        # current_h = 0.0
        # if idleg_slide_qpos_addr >= 0:
        #     current_h = data.qpos[idleg_slide_qpos_addr]  # 支撑腿当前位置
        
        # 限制支撑腿运动范围，扩展正向范围以支持l_leg=0.05等正值
        # 原范围(-0.12m到0m)扩展为(-0.12m到0.08m)
        # idleg_slide_target = np.clip(idleg_slide_target, -0.12, 0.08)
        
        # 应用控制信号变化率限制（避免急剧变化）
        max_change_rate = 0.1  # 弧度/步
        # max_idleg_change_rate = 0.01  # m/步，支撑腿变化率限制 (已注释)
        if last_control_values is not None:
            # 限制前车把的变化率
            if abs(front_leg_target - last_control_values['front_leg']) > max_change_rate:
                if front_leg_target > last_control_values['front_leg']:
                    front_leg_target = last_control_values['front_leg'] + max_change_rate
                else:
                    front_leg_target = last_control_values['front_leg'] - max_change_rate
            
            # 限制支撑腿的变化率 (已注释)
            # if 'idleg_slide' in last_control_values:
            #     if abs(idleg_slide_target - last_control_values['idleg_slide']) > max_idleg_change_rate:
            #         if idleg_slide_target > last_control_values['idleg_slide']:
            #             idleg_slide_target = last_control_values['idleg_slide'] + max_idleg_change_rate
            #         else:
            #             idleg_slide_target = last_control_values['idleg_slide'] - max_idleg_change_rate
        
        # 设置执行器控制信号
        if front_leg_id >= 0:
            data.ctrl[front_leg_id] = front_leg_target
        if front_wheel_id >= 0:
            data.ctrl[front_wheel_id] = front_wheel_target
        if rear_wheel_id >= 0:
            data.ctrl[rear_wheel_id] = rear_wheel_target
        # 支撑腿控制：保持固定位置l_leg
        if idleg_slide_id >= 0:
            data.ctrl[idleg_slide_id] = l_leg  # 始终保持初始位置
        
        # 定期输出进度信息（每1秒一次）
        if int(elapsed_time) != int(elapsed_time - 0.05):
            # 计算实际位置用于调试
            actual_deltaF = 0.0
            actual_idleg = 0.0  # 重新启用支撑腿显示用于调试
            actual_front_wheel_speed = 0.0
            if front_leg_qpos_addr >= 0:
                actual_deltaF = np.degrees(data.qpos[front_leg_qpos_addr])
            if idleg_slide_qpos_addr >= 0:  # 重新启用支撑腿显示用于调试
                actual_idleg = data.qpos[idleg_slide_qpos_addr]
            if front_wheel_qpos_addr >= 0 and prev_front_wheel is not None:
                current_angle = data.qpos[front_wheel_qpos_addr]
                actual_front_wheel_speed = np.degrees((current_angle - prev_front_wheel) / dt)
            
            print(f"    进度: {progress:.0%} | 前车把目标: {current_deltaF:.1f}° | 实际: {actual_deltaF:.1f}°")
            print(f"    支撑腿目标: {l_leg:.3f}m (固定模式) | 实际: {actual_idleg:.3f}m | 差值: {abs(actual_idleg - l_leg):.4f}m")  # 重新启用用于调试
            print(f"    前轮目标: {current_omega_f:.0f}dps | 实际: {actual_front_wheel_speed:.0f}dps | 控制信号: {front_wheel_target:.2f}rad/s")
        
        # 返回当前的控制值，用于下一步的连续性
        current_values = {
            'deltaF': current_deltaF,
            'omega_f': current_omega_f,
            'phi_rdot': current_phi_rdot
        }
        
        # 保存当前控制信号值用于变化率限制
        control_values = {
            'front_leg': front_leg_target,
            'front_wheel': front_wheel_target,
            'rear_wheel': rear_wheel_target,
            'idleg_slide': l_leg  # 支撑腿固定位置控制
        }
        
        return progress >= 1.0, current_values, control_values  # 返回是否完成当前步骤和当前控制值
        
    except Exception as e:
        print(f"执行PLAN步骤时出错: {e}")
        print(f"步骤数据: {plan_step}")
        return True, None, None

matplotlib.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 添加字体备选方案
try:
    # 尝试设置中文字体
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except:
    # 如果中文字体设置失败，使用英文
    plt.rcParams['font.family'] = 'DejaVu Sans'

def read_log_file(log_file_path):
    """
    读取log文件中的实物数据
    根据用户要求：
    - 第7n-2行包含theta、thetaDot、deltaF和omegaF（实物数据）
    - 第7n+2行包含filteredTorque数据
    """
    real_theta_list = []
    real_thetaDot_list = []
    real_deltaF_list = []
    real_omegaF_list = []  # 新增：实物前轮转速
    filtered_torque_list = []
    real_time_list = []
    
    try:
        # 优先 UTF-8（含 BOM），失败再回退 GBK（忽略非法字节）
        try:
            with open(log_file_path, 'r', encoding='utf-8-sig') as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            with open(log_file_path, 'r', encoding='gbk', errors='ignore') as f:
                lines = f.readlines()
        
        print(f"✓ 读取log文件: {log_file_path} (共{len(lines)}行)")
        
        # 按照用户指定的行号规则提取数据
        n = 1
        successful_reads = 0
        
        while successful_reads < 1000:  # 限制读取数量
            # 第7n-2行的索引（行号从1开始，索引从0开始）
            theta_line_idx = 7 * n - 2 - 1  # 第7n-2行的索引
            # 第7n+2行的索引
            torque_line_idx = 7 * n + 2 - 1  # 第7n+2行的索引
            
            if torque_line_idx >= len(lines):
                break
            
            try:
                # 读取第7n-2行的theta、thetaDot、deltaF和omegaF数据
                if theta_line_idx < len(lines):
                    theta_line = lines[theta_line_idx].strip()
                    if theta_line.startswith('{') and theta_line.endswith('}'):
                        theta_data = json.loads(theta_line)
                        if 'theta' in theta_data and 'thetaDot' in theta_data:
                            # 过滤掉全零数据，获取有意义的数据
                            if theta_data['theta'] != 0 or theta_data['thetaDot'] != 0:
                                real_theta_list.append(theta_data['theta'])
                                real_thetaDot_list.append(theta_data['thetaDot'])
                                # 读取deltaF数据，如果存在的话
                                real_deltaF_list.append(theta_data.get('deltaF', 0.0))
                                # 读取omegaF数据（前轮转速），如果存在的话
                                real_omegaF_list.append(theta_data.get('omegaF', 0.0))
                
                # 读取第7n+2行的filteredTorque数据
                if torque_line_idx < len(lines):
                    torque_line = lines[torque_line_idx].strip()
                    if torque_line.startswith('{') and torque_line.endswith('}'):
                        torque_data = json.loads(torque_line)
                        if 'filteredTorque' in torque_data and 'timestamp' in torque_data:
                            filtered_torque_list.append(torque_data['filteredTorque'])
                            real_time_list.append(torque_data['timestamp'])
                            
                            successful_reads += 1
                
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                pass
            
            n += 1
        
        print(f"✓ 成功读取log数据: theta({len(real_theta_list)}点), deltaF({len(real_deltaF_list)}点), omegaF({len(real_omegaF_list)}点), torque({len(filtered_torque_list)}点)")
        
        if len(real_theta_list) == 0:
            print("⚠ 未读取到有效数据，可能是文件格式问题")
        
        # 确保数据长度一致
        min_length = min(len(real_theta_list), len(real_deltaF_list), len(real_omegaF_list), len(filtered_torque_list), len(real_time_list))
        if min_length > 0:
            real_theta_list = real_theta_list[:min_length]
            real_thetaDot_list = real_thetaDot_list[:min_length]
            real_deltaF_list = real_deltaF_list[:min_length]
            real_omegaF_list = real_omegaF_list[:min_length]
            filtered_torque_list = filtered_torque_list[:min_length] 
            real_time_list = real_time_list[:min_length]
        
        return (np.array(real_time_list), np.array(real_theta_list), 
                np.array(real_thetaDot_list), np.array(real_deltaF_list), 
                np.array(real_omegaF_list), np.array(filtered_torque_list))
        
    except FileNotFoundError:
        print(f"✗ 错误: 找不到文件 {log_file_path}")
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    except Exception as e:
        print(f"✗ 读取log文件时出错: {e}")
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

# 读取实物数据
log_file_path = "C:\\mujoco-3.3.3-windows-x86_64\\new_T4\\log_06011349_G1_deltaF=0.txt"
real_time, real_theta, real_thetaDot, real_deltaF, real_omegaF, filtered_torque = read_log_file(log_file_path)

# 运行仿真
print("\n🚀 开始仿真...")
print(f"  运动规划: {len(PLAN)}个步骤, 总时长: {sum([step[1] for step in PLAN]):.1f}秒")
print("  PLAN配置为连续性模式，前车把将在步骤间保持连续")

with mujoco.viewer.launch_passive(model, data) as viewer:
    # 初始化支撑腿位置为l_leg设定值
    if idleg_slide_id >= 0:
        # 强制设置支撑腿初始位置
        data.ctrl[idleg_slide_id] = l_leg
        
        # 同时设置qpos初始位置（如果可能）
        if idleg_slide_qpos_addr >= 0:
            data.qpos[idleg_slide_qpos_addr] = l_leg
        
        print(f"🔧 支撑腿初始化位置设置为: {l_leg:.3f}m")
        print(f"💡 支撑腿控制模式: {'动态调节模式 (基于theta参数)' if abs(l_leg) < 1e-6 else f'固定长度模式 (保持{l_leg:.3f}m)'}")
                
        # 检查初始位置是否设置成功
        if idleg_slide_qpos_addr >= 0:
            actual_initial_pos = data.qpos[idleg_slide_qpos_addr]
            print(f"📏 支撑腿实际初始位置: {actual_initial_pos:.3f}m")
    
    start_time = time.time()
    current_plan_step = 0
    step_start_time = 0
    previous_values = None  # 存储前一步的控制值
    last_control_values = None  # 存储上一帧的控制信号值，用于变化率限制
    
    # 打印当前步骤信息
    if current_plan_step < len(PLAN):
        current_step = PLAN[current_plan_step]
        print(f"\n► 步骤 {current_plan_step + 1}/{len(PLAN)}: {current_step[1]}s")
    count=0
    while current_plan_step < len(PLAN):
        current_time = time.time() - start_time
        count+=1
        # 执行当前PLAN步骤，传递前一步的控制值以保持连续性
        step_completed, current_values, control_values = execute_plan_step(
            PLAN[current_plan_step], current_time, step_start_time, previous_values, last_control_values)
        
        # 更新上一帧控制值
        if control_values is not None:
            last_control_values = control_values
        
        # 如果当前步骤完成，进入下一步骤
        if step_completed:
            print(f"  ✓ 步骤 {current_plan_step + 1} 完成")
            # 保存当前步的最终控制值，作为下一步的起始值
            previous_values = current_values
            current_plan_step += 1
            step_start_time = current_time
            
            # 打印下一步骤信息
            if current_plan_step < len(PLAN):
                current_step = PLAN[current_plan_step]
                print(f"► 步骤 {current_plan_step + 1}/{len(PLAN)}: {current_step[1]}s")
        
        # 记录仿真数据
        quat = data.xquat[body_id]  # [w, x, y, z]
        euler = scipy.spatial.transform.Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_euler('xyz')
        roll = euler[0]  # 侧倾角，单位：弧度
        if prev_roll is not None:
            droll = np.unwrap([prev_roll, roll])
            roll_rate = (droll[1] - droll[0]) / dt
        else:
            roll_rate = 0.0
        prev_roll = roll

        # 记录机器人位置
        robot_pos_x = data.xpos[body_id][0]  # X位置
        robot_pos_y = data.xpos[body_id][1]  # Y位置
        robot_pos_x_list.append(robot_pos_x)
        robot_pos_y_list.append(robot_pos_y)

        roll_list.append(roll)
        roll_rate_list.append(roll_rate)
        time_list.append(current_time)
        
        # 记录前车把控制信号（转换为度数）
        if front_leg_id >= 0:
            front_leg_ctrl_list.append(np.degrees(data.ctrl[front_leg_id]))
        else:
            front_leg_ctrl_list.append(0.0)
            
        # 记录前车把实际位置（转换为度数）
        if front_leg_qpos_addr >= 0:
            # 使用正确的qpos地址获取关节实际位置
            actual_pos_rad = data.qpos[front_leg_qpos_addr]
            actual_pos_deg = np.degrees(actual_pos_rad)
            front_leg_actual_list.append(actual_pos_deg)
        else:
            front_leg_actual_list.append(0.0)
        
        # 记录前轮控制信号（转换为dps）
        if front_wheel_id >= 0:
            front_wheel_ctrl_list.append(np.degrees(data.ctrl[front_wheel_id]))  # rad/s转dps
        else:
            front_wheel_ctrl_list.append(0.0)
            
        # 记录前轮实际转速（计算角速度）
        if front_wheel_qpos_addr >= 0:
            current_front_wheel_angle = data.qpos[front_wheel_qpos_addr]  # 当前角度
            if prev_front_wheel is not None:
                # 计算角速度：(当前角度 - 前一角度) / 时间步长
                front_wheel_velocity = (current_front_wheel_angle - prev_front_wheel) / dt
                front_wheel_actual_list.append(np.degrees(front_wheel_velocity))  # rad/s转dps
            else:
                front_wheel_actual_list.append(0.0)
            prev_front_wheel = current_front_wheel_angle
        else:
            front_wheel_actual_list.append(0.0)
        
        # 记录支撑腿控制信号（单位：米）
        if idleg_slide_id >= 0:
            idleg_ctrl_list.append(data.ctrl[idleg_slide_id])
        else:
            idleg_ctrl_list.append(0.0)
            
        # 记录支撑腿实际位置（单位：米）
        if idleg_slide_qpos_addr >= 0:
            actual_idleg_pos = data.qpos[idleg_slide_qpos_addr]
            idleg_actual_list.append(actual_idleg_pos)
        else:
            idleg_actual_list.append(0.0)
        # 仿真步进
        mujoco.mj_step(model, data)
        if count%50==0:
            viewer.sync()
    
    print(f"\n🎯 仿真完成！总用时: {time_list[-1]:.2f}秒")
    
    # 计算机器人移动距离
    if len(robot_pos_x_list) > 1:
        start_pos_x = robot_pos_x_list[0]
        end_pos_x = robot_pos_x_list[-1]
        start_pos_y = robot_pos_y_list[0] 
        end_pos_y = robot_pos_y_list[-1]
        distance_moved = np.sqrt((end_pos_x - start_pos_x)**2 + (end_pos_y - start_pos_y)**2)
        print(f"📏 机器人实际移动距离: {distance_moved:.3f}m")
        print(f"📊 理论移动距离 (429dps × 0.1m × 8s): {0.749 * 8:.3f}m")
        print(f"⚙️  移动效率: {distance_moved / (0.749 * 8) * 100:.1f}%")
    else:
        print("⚠️  无法计算移动距离：位置数据不足")

print("\n📊 生成对比图表...")

plt.figure(figsize=(18, 12))

# 第一个子图：theta对比
plt.subplot(2, 3, 1)
plt.plot(np.array(time_list), np.degrees(roll_list), label='Simulation theta (deg)', color='blue', linewidth=2)
if len(real_theta) > 0:
    plt.plot(real_time, real_theta, label='Real theta (deg)', color='red', linestyle='--', linewidth=2)
    plt.title(f'Theta Comparison (Sim:{len(time_list)}, Real:{len(real_theta)} points)')
else:
    plt.title('Theta Comparison (Simulation Only)')
    plt.text(0.5, 0.5, 'No real theta data found\nCheck log file format', 
             ha='center', va='center', transform=plt.gca().transAxes, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))
plt.xlabel('Time (s)')
plt.ylabel('Theta (deg)')
plt.legend()
plt.grid(True)

# 第二个子图：thetaDot对比
plt.subplot(2, 3, 2)
plt.plot(np.array(time_list), np.degrees(roll_rate_list), label='Simulation thetaDot (deg/s)', color='blue', linewidth=2)
if len(real_thetaDot) > 0:
    plt.plot(real_time, real_thetaDot, label='Real thetaDot (deg/s)', color='red', linestyle='--', linewidth=2)
    plt.title(f'ThetaDot Comparison (Sim:{len(time_list)}, Real:{len(real_thetaDot)} points)')
else:
    plt.title('ThetaDot Comparison (Simulation Only)')
    plt.text(0.5, 0.5, 'No real thetaDot data found\nCheck log file format', 
             ha='center', va='center', transform=plt.gca().transAxes,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))
plt.xlabel('Time (s)')
plt.ylabel('ThetaDot (deg/s)')
plt.legend()
plt.grid(True)

# 第三个子图：前轮转速对比
plt.subplot(2, 3, 3)
plt.plot(np.array(time_list), front_wheel_ctrl_list, label='Simulation Control Target (dps)', color='blue', linewidth=2, linestyle='-')
plt.plot(np.array(time_list), front_wheel_actual_list, label='Simulation Actual Speed (dps)', color='cyan', linewidth=2, linestyle='-')
if len(real_omegaF) > 0:
    # omegaF在log文件中单位可能是rad/s，转换为dps
    plt.plot(real_time, np.degrees(real_omegaF), label='Real Front Wheel Speed (dps)', color='red', linestyle='--', linewidth=2)
    plt.title(f'Front Wheel Speed Comparison (Target: {omega_ed:.0f} dps, Sim:{len(time_list)}, Real:{len(real_omegaF)} points)')
else:
    plt.title(f'Front Wheel Speed Comparison (Target: {omega_ed:.0f} dps, Simulation Only)')
    plt.text(0.02, 0.02, 'No real omegaF data found\nCheck log file format', 
             transform=plt.gca().transAxes, fontsize=8, verticalalignment='bottom',
             bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.5))
plt.xlabel('Time (s)')
plt.ylabel('Front Wheel Speed (dps)')
plt.legend()
plt.grid(True)
# 添加目标转速参考线
plt.axhline(y=omega_ed, color='red', linestyle=':', alpha=0.5, label=f'Target: {omega_ed:.0f} dps')

# 第四个子图：支撑腿位置对比（仅仿真数据，无实物对比）
plt.subplot(2, 3, 4)
if len(idleg_ctrl_list) > 0 or len(idleg_actual_list) > 0:
    if len(idleg_ctrl_list) > 0:
        plt.plot(np.array(time_list), np.array(idleg_ctrl_list), label='Simulation Control Target (m)', color='orange', linewidth=2, linestyle='-')
    if len(idleg_actual_list) > 0:
        plt.plot(np.array(time_list), np.array(idleg_actual_list), label='Simulation Actual Position (m)', color='red', linewidth=2, linestyle='-')
    plt.xlabel('Time (s)')
    plt.ylabel('Support Leg Position (m)')
    plt.title('Support Leg Extension (Simulation Only - No Real Data Available)')
    plt.legend()
    plt.grid(True)
    # 添加参考线显示支撑腿范围
    plt.axhline(y=-0.12, color='gray', linestyle=':', alpha=0.5, label='Min Extension')
    plt.axhline(y=0, color='gray', linestyle=':', alpha=0.5, label='Max Extension')
    # 添加说明文字
    plt.text(0.02, 0.98, 'Note: Support leg data is simulation only\nNo real robot data available for comparison', 
             transform=plt.gca().transAxes, fontsize=8, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow", alpha=0.8))
else:
    plt.text(0.5, 0.5, 'No support leg data\nCheck theta parameters in PLAN', 
             ha='center', va='center', transform=plt.gca().transAxes,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.5))
    plt.title('Support Leg Data (Not Available)')

# 第五个子图：前车把转角对比
plt.subplot(2, 3, 5)
plt.plot(np.array(time_list), front_leg_ctrl_list, label='Simulation deltaF Control (deg)', color='blue', linewidth=2, linestyle='-')
plt.plot(np.array(time_list), front_leg_actual_list, label='Simulation deltaF Actual (deg)', color='cyan', linewidth=2, linestyle='-')
if len(real_deltaF) > 0:
    plt.plot(real_time, real_deltaF, label='Real deltaF (deg)', color='red', linestyle='--', linewidth=2)
    plt.title(f'Front Steering Angle Comparison (Sim:{len(time_list)}, Real:{len(real_deltaF)} points)')
else:
    plt.title('Front Steering Angle Comparison (Simulation Only)')
    plt.text(0.02, 0.02, 'No real deltaF data found\nCheck log file format', 
             transform=plt.gca().transAxes, fontsize=8, verticalalignment='bottom',
             bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.5))
plt.xlabel('Time (s)')
plt.ylabel('Front Steering Angle (deg)')
plt.legend()
plt.grid(True)

# 第六个子图：机器人移动轨迹和参数信息
plt.subplot(2, 3, 6)
if len(robot_pos_x_list) > 1:
    # 绘制移动轨迹
    plt.plot(robot_pos_x_list, robot_pos_y_list, 'b-', linewidth=3, alpha=0.8, label='Robot Trajectory')
    plt.plot(robot_pos_x_list[0], robot_pos_y_list[0], 'go', markersize=10, label='Start')
    plt.plot(robot_pos_x_list[-1], robot_pos_y_list[-1], 'ro', markersize=10, label='End')
    
    # 计算移动距离
    distance_moved = np.sqrt((robot_pos_x_list[-1] - robot_pos_x_list[0])**2 + 
                           (robot_pos_y_list[-1] - robot_pos_y_list[0])**2)
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title(f'Robot Movement Trajectory\nDistance: {distance_moved:.3f}m')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    # 添加当前参数和性能信息
    info_text = f"Current Parameters:\n"
    # 尝试读取XML中的实际参数
    try:
        with open(MODEL_PATH, 'r', encoding='utf-8') as f:
            xml_content = f.read()
        import re
        kv_match = re.search(r'<motor name="front_wheel".*?kv="([^"]+)"', xml_content)
        damping_match = re.search(r'<joint name="front_wheel".*?damping="([^"]+)"', xml_content)
        forcerange_match = re.search(r'<motor name="front_wheel".*?forcerange="([^"]+)"', xml_content)
        
        info_text += f"Front Wheel kv: {kv_match.group(1) if kv_match else 'unknown'}\n"
        info_text += f"Front Wheel damping: {damping_match.group(1) if damping_match else '0'}\n"
        info_text += f"Forcerange: {forcerange_match.group(1) if forcerange_match else 'unknown'}\n\n"
    except:
        info_text += "kv=15, damping=0.5, forcerange=±4000\n\n"
    
    info_text += f"Performance:\n"
    info_text += f"Target Speed: {omega_ed:.0f} dps\n"
    info_text += f"Theoretical Distance (8s): {0.749 * 8:.3f}m\n"
    info_text += f"Actual Distance: {distance_moved:.3f}m\n"
    info_text += f"Efficiency: {distance_moved / (0.749 * 8) * 100:.1f}%\n\n"
    
    # 简单的收敛性分析
    if len(front_wheel_actual_list) > 100:
        # 取后80%的数据分析稳态性能
        stable_start = int(len(front_wheel_actual_list) * 0.2)
        stable_data = np.array(front_wheel_actual_list[stable_start:])
        stable_mean = np.mean(stable_data)
        stable_std = np.std(stable_data)
        info_text += f"Convergence Analysis:\n"
        info_text += f"Stable Mean: {stable_mean:.0f} dps\n"
        info_text += f"Oscillation (±std): {stable_std:.1f} dps"
    
    plt.text(1.05, 0.5, info_text, transform=plt.gca().transAxes, fontsize=9,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
else:
    plt.text(0.5, 0.5, 'No position data available\nCheck simulation setup', 
             ha='center', va='center', transform=plt.gca().transAxes,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.5))
    plt.title('Robot Movement Trajectory (No Data)')

plt.tight_layout()

# 保存图表到文件，文件名包含时间戳和参数信息
# from datetime import datetime
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# # 从XML文件读取当前前轮参数用于文件名
# try:
#     with open(MODEL_PATH, 'r', encoding='utf-8') as f:
#         xml_content = f.read()
#     
#     # 提取kv和damping参数
#     import re
#     kv_match = re.search(r'<motor name="front_wheel".*?kv="([^"]+)"', xml_content)
#     damping_match = re.search(r'<joint name="front_wheel".*?damping="([^"]+)"', xml_content)
#     forcerange_match = re.search(r'<motor name="front_wheel".*?forcerange="([^"]+)"', xml_content)
#     
#     kv_value = kv_match.group(1) if kv_match else "unknown"
#     damping_value = damping_match.group(1) if damping_match else "0"
#     forcerange_value = forcerange_match.group(1) if forcerange_match else "unknown"
#     
#     filename = f"sim_{timestamp}_kv{kv_value}_damp{damping_value}_force{forcerange_value.replace(' ', 'to').replace('-', 'n')}.png"
# except:
#     filename = f"simulation_results_{timestamp}_kv15_damping0.5.png"

# plt.savefig(filename, dpi=300, bbox_inches='tight')
# print(f"📊 图表已保存为: {filename}")

plt.show()

print("✓ 分析完成！")