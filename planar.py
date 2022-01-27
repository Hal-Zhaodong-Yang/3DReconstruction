import numpy as np
import matplotlib.pyplot as plt

landmarks = [[1.2, 1.0],
             [2.4, 3.5],
             [-1.0, 0.5],
             [3.4, -1.5],
             [5.1, 0.6],
             [-0.1, -0.7],
             [3.1, 1.9]]

landmarks = np.array(landmarks)

robots = [[0, 0, 0],
          [-1.9, 4, - np.pi / 8],
          [2.1, -2.1, np.pi / 3]]

robots = np.array(robots)


bearings = []
for landmark in landmarks:
    bearings.append([])
    for robot in robots:
        angle = np.arctan2(landmark[1] - robot[1], landmark[0] - robot[0]) - robot[2] #+ np.random.normal(0, 0.01)
        if angle < -np.pi:
            angle += 2 * np.pi
        if angle > np.pi:
            angle -= 2 * np.pi
        bearings[-1].append(angle)

bearings = np.array(bearings)
print("bearings")
print(bearings)

A = []
for bearing in bearings:
    measurement = [[np.cos(bearing[0]), np.sin(bearing[0])],
                   [np.cos(bearing[1]), np.sin(bearing[1])],
                   [np.cos(bearing[2]), np.sin(bearing[2])]]
    A.append([])
    for i in range(2):
        for j in range(2):
            for k in range(2):
                A[-1].append(measurement[0][i] * measurement[1][j] * measurement[2][k])

A = np.array(A)
#print(A)

u, s, vh = np.linalg.svd(A)

T = vh[-1,:].reshape(2,2,2)

print(T)

KCB = [[-T[0][0][1], -T[0][1][1]],
       [T[0][0][0], T[0][1][0]]]
LCB = [[-T[1][0][1], -T[1][1][1]],
       [T[1][0][0], T[1][1][0]]]
KCB = np.array(KCB)
LCB = np.array(LCB)
M = np.matmul(np.linalg.inv(LCB), KCB)
print(M)
w, v = np.linalg.eig(M)
print("v")
print(v)
                
KAB = [[-T[1][0][0], -T[1][1][0]],
       [T[0][0][0], T[0][1][0]]]

eB1 = v[:,1]
eB3 = v[:,0]
print("eB1, eB3")
print(eB1, eB3)


eA2 = np.matmul(KAB,eB1)
eC2 = np.matmul(KCB,eB3)
eA3 = np.matmul(KAB,eB3)
eC1 = np.matmul(KCB,eB1)
print("eA2, eC2")
print(eA2, eC2)

thetaBA = np.arctan2(eB1[1],eB1[0]) - np.arctan2(eA2[1],eA2[0])
thetaBC = np.arctan2(eB3[1],eB3[0]) - np.arctan2(eC2[1],eC2[0])

while thetaBA > np.pi / 2:
    thetaBA -= np.pi
while thetaBA < -np.pi / 2:
    thetaBA += np.pi
while thetaBC > np.pi / 2:
    thetaBC -= np.pi
while thetaBC < -np.pi / 2:
    thetaBC += np.pi

AB_real = np.sqrt((robots[0][0] - robots[1][0])**2 + (robots[0][1] - robots[1][1])**2)
print("thetaBA, thetaBC")
print(thetaBA, thetaBC)

##################
'''
eA3 = np.matmul(KAB,eB1)
eC1 = np.matmul(KCB,eB3)

print("eA2, eC2")
print(eA3, eC1)

thetaBA = np.arctan2(eB1[1],eB1[0]) - np.arctan2(eA3[1],eA3[0])
thetaBC = np.arctan2(eB3[1],eB3[0]) - np.arctan2(eC1[1],eC1[0])

while thetaBA > np.pi / 2:
    thetaBA -= np.pi
while thetaBA < -np.pi / 2:
    thetaBA += np.pi
while thetaBC > np.pi / 2:
    thetaBC -= np.pi
while thetaBC < -np.pi / 2:
    thetaBC += np.pi

t_real = np.sqrt((robots[0][0] - robots[1][0])**2 + (robots[0][1] - robots[1][1])**2)
print("thetaBA, thetaBC")
print(thetaBA, thetaBC)
'''

angABC = np.arctan2(eB3[1],eB3[0]) - np.pi - np.arctan2(eB1[1],eB1[0])
angBAC = np.arctan2(eA2[1],eA2[0]) - np.arctan2(eA3[1],eA3[0])
angACB = np.pi - angABC - angBAC
print("angABC, angBAC, angACB")
print(angABC, angBAC, angACB)
AC = np.sin(angABC) * AB_real / np.sin(angACB)
print("AC")
print(AC)

R = [[np.cos(thetaBA),-np.sin(thetaBA)],
     [np.sin(thetaBA),np.cos(thetaBA)]]
t = [np.cos(np.arctan2(eB1[1],eB1[0])) * AB_real, np.sin(np.arctan2(eB1[1],eB1[0])) * AB_real]
print("R,t")
print(R,t)

Q = [[np.cos(thetaBA - thetaBC),-np.sin(thetaBA - thetaBC)],
     [np.sin(thetaBA - thetaBC),np.cos(thetaBA - thetaBC)]]
s = [np.cos(np.arctan2(eC1[1],eC1[0])) * AC, np.sin(np.arctan2(eC1[1],eC1[0])) * AC]
print("Q,s")
print(Q,s)


plt.figure(1)
plt.xlim(-4, 6)
plt.ylim(-3, 5)
#_, ax = plt.subplots(figsize=(7, 7))
plt.scatter(landmarks[:,0], landmarks[:,1], c="red", marker="o", label="Actual points")
for robot in robots:
    plt.arrow(robot[0], robot[1], np.cos(robot[2]),np.sin(robot[2]),head_width=0.2)
#plt.arrow(t[0],t[1],np.cos(-thetaBA),np.sin(-thetaBA), head_width=0.2, color ="green")

print("estimate the location of landmarks")
estimate_landmarks = []
for bearing in bearings:
    D = [[-np.sin(bearing[0]), np.cos(bearing[0]), 0],
          [np.cos(bearing[1]) * R[1][0] - np.sin(bearing[1]) * R[0][0], 
           np.cos(bearing[1]) * R[1][1] - np.sin(bearing[1]) * R[0][1], 
           np.cos(bearing[1]) * t[1] - np.sin(bearing[1]) * t[0]],
           [np.cos(bearing[2]) * Q[1][0] - np.sin(bearing[2]) * Q[0][0], 
           np.cos(bearing[2]) * Q[1][1] - np.sin(bearing[2]) * Q[0][1], 
           np.cos(bearing[2]) * s[1] - np.sin(bearing[2]) * s[0]]]
    
    U_lo,S_lo,vh_loc = np.linalg.svd(D)
    landmark_loc = vh_loc[-1,:]
    landmark_loc = np.array(landmark_loc)
    print(landmark_loc)
    landmark_loc = landmark_loc / landmark_loc[-1]
    estimate_landmarks.append(landmark_loc[0:-1])

estimate_landmarks = np.array(estimate_landmarks)
print("estimate_landmarks")
print(estimate_landmarks)

plt.scatter(estimate_landmarks[:,0],estimate_landmarks[:,1], c="blue", marker="+", label="Estimate points")
plt.legend(loc = 'upper right')
plt.savefig("planar.png")
Rt = list(R)
Rt[0].append(t[0])
Rt[1].append(t[1])
Rt = np.array(Rt)
print("Rt")
print(Rt)
vb = np.matmul(Rt, [landmarks[0][0], landmarks[0][1], 1])
print(vb)
print(np.arctan2(-1.58,3.992))
    
print(np.random.normal(0,0.05,100))
    

