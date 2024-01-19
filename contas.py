import numpy as np
largura = 9.6e-3
distancia = 10e-2
Omega = 2*np.pi*(1-np.cos(np.arctan(largura/(2*distancia))))
print(f"Solid Angle: {round(Omega,4)} Sr")

# Hight of the LiAlO2 and LiF 
H = np.array([54, 144])
QH=2e-6 #Coulombs
# Area of the Implanted Sample  (in 50 channels) in "detector 0"
A = 116
QA = 20e-6 #Coulombs
e = 1.602e-19 # Coulombs

sigma = 1.39e-27 # cm^2/sr
NTau = H/(Omega*sigma*QH/e)
print(f"NTau LiAlO2: {'{:.2e}'.format(NTau[0])} atoms per cm^2")
print(f"NTau LiF: {'{:.2e}'.format(NTau[1])} atoms per cm^2")
NX = A/(Omega*sigma*QA/e)
print(f"NX: {'{:.2e}'.format(NX)} atoms per cm^2")
print("For comparison:")
print(f"NX LiAlO2: {'{:.2e}'.format(NTau[0]*50)} atoms per cm^2")
print(f"NX LiF: {'{:.2e}'.format(NTau[1]*50)} atoms per cm^2")