# Proyecto-Desarrollo-de-IA
_Para este proyecto se usan técnicas de visión computacional tradicional con OPENCV para localizar la placa dentro de la imagen mediante identificación de contornos, posteriormente identificar el contorno de los caracteres y segmentarlos. Luego a través de una CNN entrenada, realizar la clasificación de cada uno de esos caracteres y así identificar la placa del predio.Por último, se crea una interfaz gráfica e integra el modelo a la API REST desarrollada._

 

## Requerimientos 🚀
Antes de comenzar, debe tener lo siguiente:

En su sistema operativo Linux o Windows debe tener ANACONDA instalado.
En caso que no lo tenga, puede descargarlo  aquí https://www.anaconda.com/products/individual
* [Python 3.8] -  Ya viene instalado en Anaconda.


## Desarrollo y prueba local🔧

Se puede implementar la API Rest en su ambiente virtual de Anaconda prompt(Esta es la consola que maneja Anaconda)
Siga los pasos correctamente para obtener una copia del proyecto en funcionamiento en tu máquina local:

1. Abrir Anaconda prompt en tu computador y escribir

```
git clone https://github.com/maria-ramirez/Proyecto-Desarrollo-de-IA.git
```
Puede observar el modelo usado en el directorio  ```models``` , también, encuentra las imagenes que se cargará en el modelo para la predicción  en el directorio ```uploads```  y los directorios  ```templates```  y  ```static```  se encuentra el diseño de la página web del API. 

Todas las dependencias necesarias se encuentran instaladas en el archivo  **[app_funciona.py](https://github.com/maria-ramirez/Proyecto-Desarrollo-de-IA/blob/main/app_funciona.py)**

2. Active el ambiente Python 

```
python app_funciona.py
```
3. Por último en el browser de su navegador colocar la url que te arroga para probar su funcionamiento  

```
http://127.0.0.1:5000/
```

Ahora ya tienes la API en tu máquina y puedes probar el modelo creado por medio de la url. 


## Autores ✒️


* **Alba Ramirez** - *Documentación*
* **Milton Guarin** - *Documentación* 







