# Redes Neurais Convolucionais - LIBRAS :hand: :raised_hand: :fist: :point_up:
---
<b>Deep Learning & Visão computacional: REDES NEURAIS CONVOLUCIONAIS https://link.medium.com/Jjnt43P0K3</b>  


<p align="center">
  <img src="demo/demo_cnn_libras.gif">
</p>

---

Esse projeto tem como objetivo gerar um classificador com Redes Neurais Convolucionais para reconhecimento de gestos do alfabeto em LIBRAS. 

#Deep Learning #LIBRAS #InteligenciaArtificial #CNN #Python3 <br> <br>

```
INPUT => CONV => POOL => CONV => POOL => CONV => POOL => FC => FC => OUTPUT 
```

#### Requisitos

> conda env create -f environment.yml 

#### Execução
> conda activate cnn_libras  
> python app_64x64x1.py 

#### Estrutura dos arquivos

> Dataset/ - Contem o dataset e scripts para gerar novas imagens <br>
> Main/ <br>
>> cnn/ <br>
>>> __ init __.py  - Estrutura das camadas da CNN <br>
>> train.py  - Execução de treinamento importando a estrutura da cnn. <br>
>> app.py - Teste do modelo para reconhecimento em real-time com OpenCV e o modelo de CNN treinado.<br>
<br>

> Models/ - Contem graficos, imagens de modelos e modelos treinados <br> 

> logs/  - Logs de execução com informações de epocas, validação e sumario <br> 

<br>


#### Referências

CNN: http://cs231n.github.io/convolutional-networks/ 

Documentação Keras: https://keras.io/

--- 
@Author: [Lucas Lacerda](https://www.linkedin.com/in/lucaaslb/)  :beer: :pizza:

