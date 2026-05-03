@echo off
echo ===================================================
echo 🏃‍♂️ 实验 1/4: Baseline (无边界注意力，无动态门控)
echo ===================================================
python train_joint.py --model_type Cascade --use_dynamic_gate False --use_boundary_attn False --use_fgm False

echo ===================================================
echo 🏃‍♂️ 实验 2/4: + Boundary Attn (仅开启边界注意力)
echo ===================================================
python train_joint.py --model_type Cascade --use_dynamic_gate False --use_boundary_attn True --use_fgm False

echo ===================================================
echo 🏃‍♂️ 实验 3/4: + Dynamic Gate (仅开启动态门控)
echo ===================================================
python train_joint.py --model_type Cascade --use_dynamic_gate True --use_boundary_attn False --use_fgm False

echo ===================================================
echo 🏃‍♂️ 实验 4/4: Full Model (完整级联模型，双机制开启)
echo ===================================================
python train_joint.py --model_type Cascade --use_dynamic_gate True --use_boundary_attn True --use_fgm False

echo.
echo 🎉 恭喜！所有 4 个消融实验已全部运行完毕！
