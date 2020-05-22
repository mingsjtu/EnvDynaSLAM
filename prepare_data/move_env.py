import os
import numpy as np 


move_dir="/media/gm/Data/SLAM/self_video/5.16morn/xiaomi_20200516_081150"
# move_dir="/home/gm/SLAM/self_preprocess/my_video/move_2_191036"
env_dir="/media/gm/Data/SLAM/self_video/5.16morn/huawei_20200516_081324"
# env_dir="/home/gm/SLAM/self_preprocess/my_video/env_2_191129"

move_txt=open(move_dir+"/rgb.txt",'r')
env_txt=open(env_dir+"/rgb.txt",'r')
out_txt=open(move_dir+"/join_rgb.txt",'w')

move_file=os.listdir(move_dir)
env_file=os.listdir(env_dir)

move_list=[]
env_list=[]
for move_line in move_txt.readlines():
    move_line=move_line.strip().replace('\n', '').replace('\r', '')
    move_list.append([float(move_line.split('    ')[0]),move_line.split('    ')[1]])

for env_line in env_txt.readlines():
    env_line=env_line.strip().replace('\n', '').replace('\r', '')
    env_list.append([float(env_line.split('    ')[0]),env_line.split('    ')[1]])
    # print(move_line)
# for i in move_dir:
    # for j 
# print(move_list)

thr=0.03
start_env=0
move_env_list=[]
dis=0
num=0

for move_line in move_list:
    if move_line[0]<env_list[0][0] or move_line[0]>env_list[len(env_list)-1][0]:
        print(move_line)
        continue
    for i in range(start_env,len(env_list)):
        env_line=env_list[i]
        if move_line[0]-env_line[0]<thr:
            # print("move",move_line)
            # print("env",env_line)
            dis+=move_line[0]-env_line[0]
            num+=1

            move_line=move_line+env_line
            move_env_list.append(move_line)
            start_env=i
            break

print("avg dis ",dis/num)
# print("num_out,num_in ",dis/num)

for i in move_env_list:
    for s in i:
        out_txt.write(str(s)+' ')
    out_txt.write('\r\n')
# out_txt.writelines(move_env_list)