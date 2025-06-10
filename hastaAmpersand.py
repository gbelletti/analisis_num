import numpy as np
from scipy.fftpack import fft2, ifft2
from numpy.fft import fftshift, ifftshift
from PIL import Image
import matplotlib.pyplot as plt

def modificar_coeficientes(fft_img, mensaje, delta):
    fft_mod = fft_img.copy()
    fila, cols = fft_mod.shape
    index = 0
    total_bits = len(mensaje)
    
    for i in range(fila):
        for j in range(cols):
            if index >= total_bits:
                break
            
            # trabajo solo en el primer cuadrante y después hago el conjugado
            if (i < fila//2 and j < cols//2):
                # aislo parte real
                a = fft_mod[i, j].real
                
                # guardo el signo de a en signo_a
                if (a >= 0): signo_a = 1 
                else: signo_a= -1
                    
                # fórmula para q=abs(round(a/δ))
                q_a = np.abs(np.round(a / delta))
                
                if (index < total_bits):
                    bit = mensaje[index]
                    # si tengo que ocultar 1 y q par, q++
                    # si tengo que ocultar 0 y q impar, q++
                    if (bit == 1 and q_a % 2 == 0) or (bit == 0 and q_a % 2 == 1):
                        q_a += 1
                        
                    # a'=signo.q_a.delta    
                    a_nuevo = signo_a * q_a * delta
                    # como ya oculté un bit de mensaje, itero
                    index += 1
                else:
                    # si ya terminé de ocultar el mensaje, no modifico el dato orig.
                    a_nuevo = a
                
                
# tengo que volver a verificar el index, porque puede que al terminar de asignar el mensaje en la parte real me salga de rango
                
                if (index < total_bits):
                    # aislo la parte imaginaria
                    b = fft_mod[i, j].imag
                    
                    # guardo el signo de b en signo_b
                    if (b >= 0): signo_b = 1 
                    else: signo_b= -1
                        
                    # fórmula para q=abs(round(b/δ))
                    q_b = np.abs(np.round(b / delta))
                    
                    bit = mensaje[index]
                    
                    # si tengo que ocultar 1 y q par, q++
                    # si tengo que ocultar 0 y q impar, q++
                    if (bit == 1 and q_b % 2 == 0) or (bit == 0 and q_b % 2 == 1):
                        q_b += 1
                        
                    # b'=signo.q_b.delta  
                    b_nuevo = signo_b * q_b * delta
                    index += 1
                else:
                    b_nuevo = fft_mod[i, j].imag
                
                # si terminé de modificar, guardo el bit como a'+b'j
                fft_mod[i, j] = a_nuevo + 1j * b_nuevo
    
    
    return fft_mod

# FUNCIONES PARA DECODIFICAR PREGUNTAS DE LA CONSULTA

# No uso texto_a_binario porque devuelve un str de 110101011101 no me sirve porque dps uso funciones tipo packbits
# Esta nueva devuelve un np.array de tipo uint8 seria [1,1,0,1]
def textoaBinario(texto):
    lista_bytes = []
    for c in texto:
        ascii = ord(c)
        lista_bytes.append(ascii)
        
    bytes_array = np.array(lista_bytes,dtype=np.uint8)
    bits_array = np.unpackbits(bytes_array)
    
    return bits_array

def bits_a_texto(bits_array):
    bits_validos = bits_array[:len(bits_array) // 8 * 8]
    mensaje_bytes = np.packbits(bits_validos)
    # decodifica los bytes a texto con utf-8
    # tobytes: secuencia de bytes
    # con replace reemplazo por � si no reconoce algun caracter
    return mensaje_bytes.tobytes().decode('utf-8', errors='replace')

# Funcion para leer mensaje hasta longitud_mensaje
def extraer_hasta_long(matriz_fft, longitud_mensaje, delta=1):
    bits = []
    filas, cols = matriz_fft.shape
    
    for i in range(filas//2):
        for j in range(cols//2):
            
            if(len(bits) >= longitud_mensaje):
                break
            
            real = matriz_fft[i, j].real
            imag = matriz_fft[i, j].imag
                      
            q_real = round(abs(real) / delta)
            bits.append(q_real % 2)
                
            if(len(bits) >= longitud_mensaje):
                break
            
            q_imag = round(abs(imag) / delta)
            bits.append(q_imag % 2)
           # :hastaEstaLongitud
    Bits = np.array(bits[:longitud_mensaje],dtype=np.uint8)
    return Bits 

# Funcion para leer mensaje hasta &
def extraer_hasta_ampersand(matriz_fft, delta):
    bits = []
    filas, cols = matriz_fft.shape
    
    for i in range(filas//2):
        for j in range(cols//2):
            real = matriz_fft[i, j].real
            imag = matriz_fft[i, j].imag
            
            q_real = round(abs(real) / delta)
            bits.append(q_real % 2)
            
            q_imag = round(abs(imag) / delta)
            bits.append(q_imag % 2)
            
            # Paso a texto asi veo si es &
            mensaje_actual = bits_a_texto(np.array(bits, dtype=np.uint8))
            
            # Encontramos & --> mostramos hasta esa pos
            if '&' in mensaje_actual:
                ampersand = mensaje_actual.find('&')
                return mensaje_actual[:ampersand]  
            
    # Esto seria en caso de que la pregunta no termine con & (o sea mostraria la pregunta y después basura)
    return mensaje_actual


# OCULTAMOS
imhost = np.array(Image.open("elrocholinus.png").convert("L"))
txt = "Que metodo usaron para codificar y decodificar el ejercicio 2? Que pasa con delta si se modifica en el ejercicio 3?&"
bits_mensaje = textoaBinario(txt)
long_pregunta = len(bits_mensaje)
delta = 1

host_tft = fftshift(fft2(imhost))



# estego_txt es una matriz con componentes imaginarias que tiene el mensaje oculto
estego_txt = modificar_coeficientes(host_tft,bits_mensaje,delta)
estego_plot = (ifft2(ifftshift(estego_txt))).real

#np.save("preguntas.npy", estego_txt)

# DECODIFICAMOS

# Este es el caso anterior: leiamos hasta len(bits_mensaje)
# bits_recuperados = extraer_hasta_long(estego_txt,long_pregunta, delta)
# mensaje_recuperado = bits_a_texto(bits_recuperados)
im_p = np.load("imagen_estego_preguntas.npy")
mensaje_rec = extraer_hasta_ampersand(im_p, 1)


#mensaje_recuperado = extraer_hasta_ampersand(estego_txt, delta)
msj_inv = (ifft2(ifftshift(im_p))).real

plt.figure(figsize=(8, 8))
plt.imshow(msj_inv, cmap='gray')
plt.title("Imagen estego con preguntas")
plt.axis('off')

plt.tight_layout()
plt.show()

#print("Mensaje recuperado desde memoria: ", mensaje_recuperado)
print("Mensaje recuperado desde disco: ", mensaje_rec)









