# Vectorized version of st_k_means for mortaltiy clustering.

import numpy as np
from itertools import product

class vectorized_st_kmeans_():
    """
    Contient les fonctions qui servent à déployer l'algorithme de FCMeans Spatio-Temporel adapté aux données de mortalité.
    """

    def __init__(self, n_clusters=3, m=2, lambda_=0.4, n_init=1, max_iter=100, epsi=1e-8, **kwargs) -> None:
        """
        Initialise l'objet vectorized_st_kmeans_.
        """
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.m = m
        self.lambda_ = lambda_
        self.epsi = epsi


    def euclidienne_distance(self, p: np.ndarray , q: np.ndarray, power: int = 2) -> float:
        """Calcule la distance euclidienne entre deux vecteurs."""
        return np.sqrt(np.sum((p - q) ** power))


    def hellinger_distance(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        Calcule la distance de Hellinger entre deux distributions de probabilité.
        
        Args:
            p (np.ndarray): Distribution de probabilité 1 (somme(p) doit être égale à 1).
            q (np.ndarray): Distribution de probabilité 2 (somme(q) doit être égale à 1).
            
        Returns:
            float: Distance de Hellinger entre p et q.
        """
        # Vérification que les distributions sont valides
        #if not np.isclose(np.sum(p), 1) or not np.isclose(np.sum(q), 1):
        #    raise ValueError("Les distributions doivent être des distributions de probabilité (somme = 1).")
        if len(p) != len(q):
            raise ValueError("Les distributions doivent avoir la même taille.")
        return np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))
       


    def initialization(
        self, 
        x: tuple[np.ndarray, np.ndarray],
        **kwargs
        ) -> list[np.ndarray]:
        """
        Initialise les prototypes (centroids) pour les données d'entrée.
        Args:
            x (tuple[np.ndarray, np.ndarray]) : Données de shape (n, T, p) et (n, T, q)
            k (int): Nombre de clusters
        Returns:
            list[np.ndarray]: Liste de deux tableaux numpy contenant les prototypes pour chaque modalité.
        """
        n, _, _ = x[0].shape
        indices = np.random.choice(n, self.n_clusters, replace=False)
        return [x[0][indices], x[1][indices]]
        

    def update_centroid_euclidienne(self,
        x: tuple[np.ndarray, np.ndarray], 
        w: np.ndarray, 
        c: tuple[np.ndarray, np.ndarray], 
        **kwargs
    ) -> np.ndarray:
        """
        Update centroids using Euclidean distance for c_{j,t}^{(2), k+1}.

        Args:
            x [(np.ndarray),(np.ndarray)]: Input data, shape [(n, T, p), (n, T, q)]
            w (np.ndarray): Assignment weights, shape (n, C, T, L)
            c [(np.ndarray), (np.ndarray)]: Current centroids, shape [(C, T, p), (C, T, q)]
            lambda_ (float): Regularization parameter
            m (float): Fuzziness parameter

        Returns:
            np.ndarray: Updated centroids of shape (C, T, p).
        """
        x_euclidean = x[1] # [I, T, V]
        c_euclidean = c[1] # [K, T, V]
        u_euclidean = w[:, :, :, 1]  # [I, K, T] Assuming the second dimension corresponds to Euclidean distance


        I, T, V = x_euclidean.shape  
        K, _, _ = c_euclidean.shape
        updated_c = np.zeros_like(c_euclidean)  # Initialize updated centroids array
        
        # Reshape for broadcasting
        # [I, T, V] -> [I, 1, T, V], [K, T, V] -> [1, K, T, V], [I, K, T] -> [I, K, T, 1]
        x_broadcasted = x_euclidean[:, np.newaxis, :]  # [I, 1, T, V]
        u_broadcasted = u_euclidean[:, :, :, np.newaxis]  # [I, K, T, 1]
        

        u_prod_x = np.sum((u_broadcasted ** self.m) * x_broadcasted, axis=0)  # [K, T, V]
        u_sum = np.sum(u_broadcasted ** self.m, axis=0)  # [K, T, 1]

        # Calculate lambda term
        lambda_term = I * self.lambda_ ** (T - np.arange(T))[np.newaxis, :, np.newaxis]  # [1, T, 1]
        lambda_term[:, -1] = 0  # Shift lambda term to match c_t_plus_1

        # Calculate c_t_plus_1
        c_t_plus_1 = np.zeros_like(c_euclidean)  # [K, T, V]
        c_t_plus_1[:, :-1] = c_euclidean[:, 1:, :]  # [K, T-1, V]
        
        # Update centroids
        updated_c = (u_prod_x + lambda_term * c_t_plus_1) / (u_sum + lambda_term + self.epsi) # [K, T, q]

        return updated_c

    def update_centroid_hellinger(self, 
        x: tuple[np.ndarray, np.ndarray], 
        w: np.ndarray, 
        c: list[np.ndarray], 
        **kwargs
    ) -> np.ndarray:
        """ Met à jour les centroides en utilisant la distance de Hellinger de c_{j,t}^{(1), k+1}.
            Args:
                x (tuple[np.ndarray, np.ndarray]): Données d'entrée, de forme [(n, T, p), (n, T, q)]
                w (np.ndarray): Matrice des poids d'affectation, de forme (n, C, T, L)
                c (list[np.ndarray]): Liste des centroides actuels, de forme [C, T, p]
                lambda_ (float): Paramètre de régularisation
                m (float): Paramètre de fuzziness
            Returns:
                np.ndarray: Centroides mis à jour de forme (C, T, p).
        """
        x_hellinger = x[0] # [I, T, A]
        c_hellinger = c[0] # [K, T, A]
        u_hellinger = w[:, :, :, 0]  # [I, K, T] Assuming the second dimension corresponds to Euclidean distance

        # Reshape for broadcasting and square root
        root_x_broadcasted = np.sqrt(x_hellinger[:, np.newaxis, :])  # [I, 1, T, A]
        root_c = np.sqrt(c_hellinger)  # [K, T, A]
        u_broadcasted = u_hellinger[:, :, :, np.newaxis]  # [I, K, T, 1]

        u_prod_x = np.sum((u_broadcasted ** self.m) * root_x_broadcasted, axis=0)  # [K, T, A]
       
        # Calculate lambda term
        I, T, A = x_hellinger.shape
        lambda_term = I * self.lambda_ ** (T - np.arange(T))[np.newaxis, :, np.newaxis]  # [1, T, 1]
        
        # Calculate c_t_plus_1
        c_t_plus_1 = np.zeros_like(root_c)  # [K, T, A]
        c_t_plus_1[: , :-1] = root_c[:, 1:, :]  # [K, T-1, A] En

        # Update centroids
        updated_c = (u_prod_x + lambda_term * c_t_plus_1) ** 2
        updated_c /= updated_c.sum(axis=-1, keepdims=True) + self.epsi  # Normalize and avoid division by zero

        
        return updated_c


    def update_centroid(
        self, 
        x: tuple[np.ndarray, np.ndarray], 
        w: np.ndarray, 
        c: list[np.ndarray], 
        **kwargs
    ) -> list[np.ndarray]:
        """
        Met à jour les centroides.

        Args:
            x (tuple[np.ndarray,np.ndarray]): Données d'entrée, de forme [(n, T, p), (n, T, q)]
            w (np.ndarray): Matrice de poids d'affectation, de forme (n, C, T, L)
            c (list[np.ndarray]): Liste des centroides actuels, de forme [(C, T, p), (C, T, q)]
            lambda_ (float): Paramètre de régularisation
            m (float): Paramètre de fuzziness

        Returns:
            list[np.ndarray]: Centroides mis à jour de forme [(C, T, p), (C, T, q)]
        """
        return [self.update_centroid_hellinger(x, w, c),
                self.update_centroid_euclidienne(x, w, c)]



    def update_weights(
        self, 
        x: tuple[np.ndarray, np.ndarray], 
        c: list[np.ndarray], 
        **kwargs 
    ) -> np.ndarray:
        """
        Met à jour les poids d'affectation des clusters.

        Args:
            x (tuple[np.ndarray,np.ndarray]): Données d'entrée, de forme [(n, T, p), (n, T, q)].
            c (list[np.ndarray]): Liste des centroides actuels, de forme [(C, T, p), (C, T, q)].
            m (float): Paramètre de fuzziness.

        Returns:
            np.ndarray: Matrice des poids mis à jour de forme (n, k, T, L).
        """

        n, T, _ = x[1].shape
        k, _, _ = c[1].shape
        L = 2
        updated_w = np.zeros((n, k, T, L))
        dist = np.zeros((n, k, T, L))

        for i, t, j in product(range(n), range(T), range(k)):
            dist[i,j,t,0] = self.hellinger_distance(x[0][i, t, :], c[0][j, t, :]) ** 2 #* np.sqrt(x[0].shape[2]) 
            dist[i,j,t,1] = np.linalg.norm(x[1][i, t, :] - c[1][j, t, :]) ** 2 #/ np.sqrt(x[1].shape[2])

        inv_dist =  (1 / (dist + self.epsi)) ** (1 / (self.m - 1))
        updated_w = inv_dist / np.sum(inv_dist, axis=(1,3), keepdims=True)

        return updated_w



    def loss_hellinger(self,
        x: tuple[np.ndarray, np.ndarray],
        w: np.ndarray,
        c: list[np.ndarray], 
        **kwargs
    ) -> float:
        """
        Calcule la perte.

        Args:
            x (tuple[np.ndarray,np.ndarray]): Données d'entrée, de forme [(n, T, p), (n, T, q)]
            w (np.ndarray): Matrice de poids d'affectation, de forme (n, C, T, L)
            c (list[np.ndarray]): Liste des centroides actuels, de forme [(C, T, p), (C, T, q)]
            m (float): Paramètre de fuzziness.
            lambda_ (float): Paramètre de régularisation

        Returns:
            float: Perte
        """

        u_hellinger = w[:, :, :, 0]  # [I, K, T]
        x_hellinger = x[0]  # [I, T, A]
        c_hellinger = c[0]  # [K, T, A]

        # Reshape for broadcasting 
        x_broadcasted = x_hellinger[:, np.newaxis, :]  # [I, 1, T, A]
        c_broadcasted = c_hellinger[np.newaxis, :, :, :]  # [1, K, T, A]

        # C_t_plus_1
        c_t_plus_1 = c_hellinger[:, 1:, :]  # [K, T-1, A]
        c_t_plus_1_broadcasted = c_t_plus_1[np.newaxis, :, :, :]  # [1, K, T-1, A]

        # lambda term
        I, T, A = x_hellinger.shape
        lambda_term = self.lambda_ ** (T - np.arange(T-1))[np.newaxis, np.newaxis, :]  # [1, 1, T-1]


        # Calculate hellinger distance
        sqrt_x_broadcasted = np.sqrt(x_broadcasted)  # [I, 1, T, A]
        sqrt_c_broadcasted = np.sqrt(c_broadcasted)  # [1, K, T, A]
        sqrt_c_t_plus_1_broadcasted = np.sqrt(c_t_plus_1_broadcasted)  # [1, K, T, A]
        d1 = u_hellinger ** self.m * (0.5 * np.sum((sqrt_x_broadcasted - sqrt_c_broadcasted) ** 2, axis=-1))   # [I, K, T]
        d2 = np.zeros_like(d1)  # Initialize d2 with the same shape as d1
        d2[:, :, :-1] = lambda_term * (0.5 * np.sum((sqrt_c_broadcasted[:, :, :-1] - sqrt_c_t_plus_1_broadcasted) ** 2, axis=-1))  # [1, K, T-1]

        return np.sum(d1 + d2)  # Total loss

    
    def loss_euclidean(
        self,
        x: tuple[np.ndarray, np.ndarray],
        w: np.ndarray,
        c: list[np.ndarray], 
        **kwargs
    ) -> float:
        """
        Calcule la distance euclidienne entre deux vecteurs.

        Args:
            x (np.ndarray): Premier vecteur.
            y (np.ndarray): Deuxième vecteur.

        Returns:
            float: Distance euclidienne.
        """

        u_euclidean = w[:, :, :, 1]  # [I, K, T]
        x_euclidean = x[1]  # [I, T, V]
        c_euclidean = c[1]  # [K, T, V]
        
        # Reshape for broadcasting 
        x_broadcasted = x_euclidean[:, np.newaxis, :]  # [I, 1, T, V]
        c_broadcasted = c_euclidean[np.newaxis, :, :, :]  # [1, K, T, V]

        # C_t_plus_1
        c_t_plus_1 = c_euclidean[:, 1:, :]  # [K, T-1, A]
        c_t_plus_1_broadcasted = c_t_plus_1[np.newaxis, :, :, :]  # [1, K, T-1, A]

        # lambda term
        I, T, V = x_euclidean.shape
        lambda_term = self.lambda_ ** (T - np.arange(T-1))[np.newaxis, np.newaxis, :]  # [1, 1, T]


        # Calculate hellinger distance
        d1 = u_euclidean ** self.m * np.sum((x_broadcasted - c_broadcasted) ** 2, axis=-1)    # [I, K, T]
        d2 = np.zeros_like(d1)  # Initialize d2 with the same shape as d1
        d2[:, :, :-1] = lambda_term * np.sum((c_broadcasted[:,:,:-1] - c_t_plus_1_broadcasted) ** 2, axis=-1)  # [1, K, T-1]

        return np.sum(d1 + d2)  # Total loss

    
    def loss_function(
        self,
        x: tuple[np.ndarray, np.ndarray],
        w: np.ndarray,
        c: list[np.ndarray], 
        **kwargs
    ) -> float:
        """
        Calcule la perte.

        Args:
            x (tuple[np.ndarray,np.ndarray]): Données d'entrée, de forme [(n, T, p), (n, T, q)]
            w (np.ndarray): Matrice de poids d'affectation, de forme (n, C, T, L)
            c (list[np.ndarray]): Liste des centroides actuels, de forme [(C, T, p), (C, T, q)]
            m (float): Paramètre de fuzziness.
            lambda_ (float): Paramètre de régularisation

        Returns:
            float: Perte
        """
        #print(f"Hellinger : {self.loss_hellinger(x, w, c, m, lambda_)}, Euclidienne : {self.loss_euclidean(x, w, c, m, lambda_)}")
        return self.loss_hellinger(x, w, c) + self.loss_euclidean(x, w, c)


    def clustering(
        self, 
        x: tuple[np.ndarray, np.ndarray],
        **kwargs
    ) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, float]:
        """
        Algorithme de clustering avec mise à jour des prototypes et des poids.
        Args:
            x [(np.array), (np.array)] : données de shape [(n, T, p), (n, T, q)]
            k (int): nombre de clusters
            lambda_ (float): paramètre de régularisation
            m (float): paramètre de fuzziness
            epsilon (float): critère d'arrêt
            max_iter (int): nombre maximal d'itérations

        Returns:    
            similarity_matrix (np.array): matrice de similarité
            c (np.array): prototypes des clusters
            w (np.array): degrés d'appartenance
            loss (float): valeur de la fonction de perte
        """
        c = self.initialization(x)
        n, T, _ = x[0].shape
        iteration = 0
        loss_prev = float('inf')

        while iteration < self.max_iter:
            w = self.update_weights(x, c)
            c = self.update_centroid(x, w, c)
            
            loss = self.loss_function(x, w, c)
            print(f"Iteration {iteration}, Loss: {loss}")

            if np.abs(loss - loss_prev) < self.epsi:
                print(f"L'algorithme a convergé après {iteration} itérations.")
                break
            loss_prev = loss
            iteration += 1

        if iteration == self.max_iter:
            print("L'algorithme n'a pas convergé.")

        # Calcul de la matrice de similarité
        w_combined_features = np.sum(w, axis=3)
        a = np.argmax(w_combined_features, axis=1)
        similarity_matrix = np.zeros((n, n))
        for i, j in product(range(n), range(n)):
            similarity_matrix[i, j] = 1 - (np.sum(a[i] != a[j]) / T)

        return similarity_matrix, c, w, loss

    def fit(
        self,
        x: tuple[np.ndarray, np.ndarray],
        **kwargs
    ) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, float]:
        """
        Exécute l'algorithme de clustering et retourne les résultats.
        Args:
            x [(np.array), (np.array)] : données de shape [(n, T, p), (n, T, q)]
        Returns:
            similarity_matrix (np.array): matrice de similarité
            c (list[np.ndarray]): prototypes des clusters
            w (np.ndarray): degrés d'appartenance
            loss (float): valeur de la fonction de perte
        """
        loss = float('inf')
        algorithm = vectorized_st_kmeans_(x= x, k=self.n_clusters, n_init=1, m=self.m, lambda_=self.lambda_, epsilon=self.epsi, max_iter=self.max_iter)
        for _ in range(self.n_init):
            print(f"Initialization {_+1}")
            sm_current, clus_current, w_current, loss_current = algorithm.clustering(x)
            if loss_current < loss:
                loss = loss_current
                clus = clus_current
                w = w_current
                similarity_matrix = sm_current

        return similarity_matrix, clus, w, loss

