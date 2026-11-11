from utils_data import read_dataset, visualize_k_means
from Kmeans import *
from KNN import KNN

IDX_IMAGE = 10

if __name__ == '__main__':

    gt_json_path = r'C:\Users\niltr\OneDrive\Escritorio\UNI\2n de carrera\2n semestre\IA\Codi Proj2 2025\Practica2\images\gt.json'
    root_folder_path = r'C:\Users\niltr\OneDrive\Escritorio\UNI\2n de carrera\2n semestre\IA\Codi Proj2 2025\Practica2\images'


    # import images with color
    train_imgs, _, train_color_labels, test_imgs, _, test_color_labels = read_dataset(
        root_folder=root_folder_path, gt_json=gt_json_path, with_color=True
    )

    # import images in grayscale
    train_imgs_grayscale, train_class_labels, _, test_imgs_grayscale, test_class_labels, _ = read_dataset(
        root_folder=root_folder_path, gt_json=gt_json_path, with_color=False
    )

    # BELOW HERE YOU CAN CALL ANY FUNCTION THAT YOU HAVE PROGRAMED TO ANSWER THE QUESTIONS OF THE EXAM #
    # this code is just for you, you won't have to upload it after the exam #

    # this is an example of how to call some functions that you have programed
    km = KMeans(test_imgs[IDX_IMAGE], K=4)
    km.fit()
    # print(km.centroids)
    #visualize_k_means(km, (80, 60, 3))
    
    # Test question 1
    # Pregunta 1: Donada la primera imatge del test set quins centroides obtindrem en aplicar el kmeans si escollim per paràmetres el mètode d’inicialització ’first’ i K=3, en l’espai RGB? Nota: l’ordre que apareixen els centroides no és rellevant.
    IDX = 0
    km = KMeans(test_imgs[IDX], K=3, options={'km_init' : 'first'})
    km.fit()
    #print(km.centroids)
    # ANS: d
    
    # Test question 2
    # Pregunta 2: A quins tipus de peça de roba pertanyen els tres veïns més propers de la primera imatge del test set segons el KNN si aquest s’inicialitza amb totes les imatges del train set? Nota: l’ordre que apareixen els tipus de peça de roba no és rellevant.
    knn = KNN(train_imgs_grayscale, train_class_labels)
    knn.get_k_neighbours( np.array([test_imgs_grayscale[IDX]]) , 3)
    #print(knn.neighbors[0])
    # ANS : c
    
    # E1 --> Quina és la matriu de distàncies euclidianes entre cada punt i cadascun dels dos centroides?
    points = np.array([[120, 24, 89], [282, 7, 75], [71, 151, 83], [3, 132, 167], [212, 10, 5], [40, 139, 171]])
    centroids = np.array([[192, 77, 12], [77, 261, 223]])
    
    #print(distance(points, centroids)) #matriu de distancies
    #print(distance(np.array([[71, 151, 83]]), centroids)) #distancia del punt als dos centroides
    
    # E2 --> ¿cuáles son las etiquetas asignadas a cada punto tras ejecutar la asignación de clusters (get_labels()) en el algoritmo KMeans? K=2
    km2 = KMeans(points, K=2)
    km2.centroids = centroids
    km2.get_labels()
    #print(km2.labels)
    
    # E3
    
    # E4 Dada la imagen número 100 del conjunto de test, se extrae el valor RGB del píxel situado en la coordenada (52, 32). 
    # Utilizando la función get_color_prob() del módulo utils, que devuelve las probabilidades de pertenencia a cada color, y el vector utils.colors 
    # que contiene los nombres de los colores, ¿cuál es el color con mayor probabilidad para ese píxel?
    test_img4 = test_imgs[100]
    point_RGB4 = test_img4[52, 32].reshape((-1, 3))
    
    #print(point_RGB4) #valor RGB del pixel (vermell, verd i blau)
    #print(utils.colors[np.argmax(utils.get_color_prob(point_RGB4), axis=1)]) #color amb més prop
    
    # E5 Dada la primera imagen del conjunto de test (test_imgs[0]), ¿cuáles son los centroides obtenidos al aplicar el algoritmo KMeans
    #  con K=3 y método de inicialización 'first' en el espacio RGB?
    test_img5 = test_imgs[0]
    km5 = KMeans(test_imgs[0], K=3, options={'km_init' : 'first'})
    km5.fit()
    #print(km5.centroids)  #representen els 3 colors de la imatge agrupats per kmeans
    
    # E6 #Donada la imatge número 40 del conjunt de test, utilitzant KMeans per trobar el nombre òptim de clusters 
    # (fins a un màxim de 10), quins colors representen els centroides corresponents?
    test_img6 = test_imgs[40]
    km6 = KMeans(test_img6)
    km6.find_bestK(10)
    #print(get_colors(km6.centroids)) #colors dels centroides
    
    # E8
    #Quina proporció, en percentatge, d’imatges de "Flip Flops" hi ha entre les primeres 50 imatges del conjunt 
    # de test quan es fa la predicció amb KNN (inicialitzat amb totes les imatges del conjunt de train) utilitzant els 5 veïns més propers?
    # What proportion of "Flip Flops" is theree between the first 50 of the test set if KNN is initialized with all the train set and predicts using the 5 nearest neighbors?
    knn7 = KNN(train_imgs_grayscale, train_class_labels)
    prediction7 = knn7.predict(test_imgs_grayscale[:50], 5)
    #print(prediction7)
    #print((np.sum('Flip Flops' == prediction7) / 50) * 100) #proporcio de flip flops
    
    # E9 Donada la primera imatge del conjunt de test (en escala de grisos), quins són els tipus de peça de roba corresponents als 3 veïns 
    # més propers segons el KNN, quan aquest es inicialitza amb totes les imatges del conjunt d’entrenament?
    knn9 = KNN(train_imgs_grayscale, train_class_labels)
    knn9.get_k_neighbours( np.array([test_imgs_grayscale[0]]) , 3)
    #print(knn9.neighbors[0]) #veins més propers
    
