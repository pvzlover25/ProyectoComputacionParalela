# Proyecto Final
**Nombre:** Marco Antonio Aguayo Solis

**Ramo:** Introducción a la Computación Paralela

---

Los códigos de este proyecto están presentes en las carpetas "Archivos_CU" y "Archivos_IPYNB", y se ejecutan de la siguiente manera:

**Archivos_CU:**
- Bajar el archivo en cuestión y colocarlo en una carpeta
- Abrir la consola
- Para compilar el archivo, ejecutar la siguiente línea de comando en la consola
  
  `nvcc nombre_algoritmo.cu -o nombre_algoritmo`
  - "nombre_algoritmo" corresponde al nombre del archivo con el que se quiere trabajar
- Para ejecutar el archivo ya compilado, ejecutar la siguiente línea de comando

  `./nombre_algoritmo n p`
  - "n" corresponde al tamaño del arreglo que se quiere ordenar
  - "p" corresponde al numero de buckets a utilizar en el algoritmo. Este dato es especifico para el algoritmo Sample Sort (archivo de nombre sample_sort.cu), y se omite al ejecutar los demás algoritmos.

**Archivos_IPYNB**
- Bajar el archivo en cuestión
- Subir el archivo en Google Colab
- Ir a la pestaña "Entorno de ejecución" y seleccionar "Cambiar tipo de entorno de ejecución"
- En "Acelerador por hardware", seleccionar "GPU T4" y hacer clic en guardar.
- Ejecutar todas las celdas correspondientes
  - Para que el código funcione correctamente, solo es necesario ejecutar las últimas tres celdas. Las primeras dos celdas pueden ejecutarse, pero no es necesario.
  - En la última celda, se pueden cambiar los valores númericos para ejecutar el algoritmo con un arreglo de distinto tamaño y, en el caso del Sample Sort, una cantidad distinta de buckets
