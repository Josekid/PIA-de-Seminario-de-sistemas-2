"""
Entrenamiento Perceptron
"""

import xlrd
import pandas as pd

def entrenar(theta,fac_ap,w1,w2,epochs,x1,x2,d,n_muestras):
    errores = True
    while errores:
        errores = False
        for i in range(n_muestras):
            z = ((x1[i] * w1)+(x2[i] * w2)) - theta # calculamos z
            
            if z >= 0:
                z = 1
            else:
                z = 0
            
            if z != d[i]:
                errores = True
                # calcular errores
                error = (d[i] - z)
                # ajustar theta
                theta = theta + (-(fac_ap * error))
                # ajustar pesos
                w1 = w1 + (x1[i] * error * fac_ap)
                w2 = w2 + (x2[i] * error * fac_ap)
                epochs +=1
                
    return w1, w2, epochs, theta
                

# ciclo principal
if __name__ == "__main__":
    # Leer archivo excel CAMBIAR RUTA A DONDE SE HAYA GUARDADO EXCEL
    archivo_excel = pd.read_excel("C:\py/perceptron_data.xlsx")
    theta = 0.2
    fac_ap = 0.2
    w1 = 0.5
    w2 = 0.5
    epochs = 0
    x1 = archivo_excel["x1"]
    x2 = archivo_excel["x2"]
    d = archivo_excel["d"]
    n_muestras = len(d)
    w1, w2, epochs, theta = entrenar(theta,fac_ap,w1,w2,epochs,x1,x2,d,n_muestras)
    print ("w1 = ", w1)
    print ("w2 = ", w2)
    print ("Theta =", theta)
    print ("epochs =", epochs)
    
