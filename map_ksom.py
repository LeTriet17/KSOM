from CSom import *
import pickle
import os
from utils import *


model=load_model('../ksoms.ckpt')
processed_data=load_data('../data_preprocess')
PNodes_arr=create_pnode(model, processed_data)
sum_quan_err=0
for i in range(len(PNodes_arr)):
    SuitNode, _, _ = model.FindBestMatchingNode(PNodes_arr[i])
    sum_quan_err += model.CalculateDistance_PNode2CNode(PNodes_arr[i], SuitNode)
    # print(PNodes_arr[i])
    # print(SuitNode)
    SuitNode.addPNode(PNodes_arr, i)

# for iy, ix in np.ndindex(model.m_Som.shape):
#     for i in range(len(model.m_Som[iy,ix].PNodes)):
#         print(iy," ",ix," ",model.m_Som[iy,ix].PNodes[i].corpus)

print(f"Quantization Error {sum_quan_err/len(PNodes_arr)}")

with open("node_ksom.txt", "w+", encoding="utf-8") as f:
    for iy, ix in np.ndindex(model.m_Som.shape):
        cNode = model.m_Som[iy, ix]
        if len(cNode.PNodes) > 0:
            f.write(f"Node: ({iy}, {ix})--------------------\n")
            for i in cNode.PNodes.keys():
                label = processed_data[i][-1]
                writing_style = processed_data[i][1]
                original_text = processed_data[i][2]
                text = " ".join(processed_data[i][0])
                f.write(f"POST {i}: {original_text}\n ENCODE: {writing_style}, {text}\n LABEL: {label}\n-------------------------------------------------------------------------------\n")
            f.write("\n\n")
