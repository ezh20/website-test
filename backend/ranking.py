from __future__ import print_function
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from numpy import linalg as LA


class ranking:
    # Load the CSV file into a Pandas dataframe
    def __init__(self):
        self.df = pd.read_csv('../data/output.csv')

        # Rename the "sypnopsis" column to "synopsis"
        self.df = self.df.rename(columns={'sypnopsis': 'synopsis'})

        # Drop Duplicate names
        self.df.drop_duplicates(subset='Name', inplace=True)

        self.anime_id_to_index = {anime_id:index for index, anime_id in enumerate(self.df['MAL_ID'])}
        self.anime_name_to_id = {name:mid for name, mid in zip(self.df['Name'], self.df['MAL_ID'])}
        self.anime_id_to_name = {v:k for k,v in self.anime_name_to_id.items()}
        self.anime_name_to_index = {name:self.anime_id_to_index[self.anime_name_to_id[name]] for name in self.df['Name']}
        self.anime_index_to_name = {v:k for k,v in self.anime_name_to_index.items()}
        n_feats = 5000
        tfidf_vec = self.build_vectorizer(max_features=n_feats, stop_words="english")
        doc_by_vocab = np.empty([len(self.df), n_feats])
        doc_by_vocab = tfidf_vec.fit_transform(self.df['synopsis'].values.astype('U'))
        doc_by_vocab = doc_by_vocab.toarray()
        index_to_vocab = {i:v for i, v in enumerate(tfidf_vec.get_feature_names())}

        self.movie_sims_cos = self.build_movie_sims_cos(1000, self.anime_index_to_name, doc_by_vocab, self.anime_name_to_index, self.get_sim)

        movie_sims_jac = self.build_movie_sims_jac(1000,self.df['Genres'])


    def build_vectorizer(self, max_features, stop_words, max_df=0.8, min_df=10, norm='l2'):
        """Returns a TfidfVectorizer object with the above preprocessing properties.
        
        Note: This function may log a deprecation warning. This is normal, and you
        can simply ignore it.
        
        Parameters
        ----------
        max_features : int
            Corresponds to 'max_features' parameter of the sklearn TfidfVectorizer 
            constructer.
        stop_words : str
            Corresponds to 'stop_words' parameter of the sklearn TfidfVectorizer constructer. 
        max_df : float
            Corresponds to 'max_df' parameter of the sklearn TfidfVectorizer constructer. 
        min_df : float
            Corresponds to 'min_df' parameter of the sklearn TfidfVectorizer constructer. 
        norm : str
            Corresponds to 'norm' parameter of the sklearn TfidfVectorizer constructer. 

        Returns
        -------
        TfidfVectorizer
            A TfidfVectorizer object with the given parameters as its preprocessing properties.
        """
        # YOUR CODE HERE
        vectorizer = TfidfVectorizer(max_features = max_features, stop_words=stop_words, max_df=max_df, min_df=min_df, norm=norm)
        return vectorizer



    def get_sim(self, mov1, mov2, input_doc_mat, input_movie_name_to_index):
        """Returns a float giving the cosine similarity of 
        the two movie transcripts.
        
        Params: {mov1 (str): Name of the first movie.
                mov2 (str): Name of the second movie.
                input_doc_mat (numpy.ndarray): Term-document matrix of movie transcripts, where 
                        each row represents a document (movie transcript) and each column represents a term.
                movie_name_to_index (dict): Dictionary that maps movie names to the corresponding row index 
                        in the term-document matrix.}
        Returns: Float (Cosine similarity of the two movie transcripts.)
        """
        # YOUR CODE HERE
        index1 = input_movie_name_to_index[mov1]
        index2 = input_movie_name_to_index[mov2]
        arr1 = input_doc_mat[index1]
        arr2 = input_doc_mat[index2]
        numerator = np.dot(arr1,arr2)
        denomenator = LA.norm(arr1)*LA.norm(arr2)
        
        return numerator/denomenator

    def build_movie_sims_cos(self, n_mov, movie_index_to_name, input_doc_mat, movie_name_to_index, input_get_sim_method):
        """Returns a movie_sims matrix of size (num_movies,num_movies) where for (i,j):
            [i,j] should be the cosine similarity between the movie with index i and the movie with index j
            
        Note: You should set values on the diagonal to 1
        to indicate that all movies are trivially perfectly similar to themselves.
        
        Params: {n_mov: Integer, the number of movies
                movie_index_to_name: Dictionary, a dictionary that maps movie index to name
                input_doc_mat: Numpy Array, a numpy array that represents the document-term matrix
                movie_name_to_index: Dictionary, a dictionary that maps movie names to index
                input_get_sim_method: Function, a function to compute cosine similarity}
        Returns: Numpy Array 
        """
        # YOUR CODE HERE
        movie_sims_matrix = np.zeros((n_mov, n_mov))
        
        for i in range(0, n_mov):
            for j in range(0, n_mov):
                sim_score = input_get_sim_method(movie_index_to_name[i], movie_index_to_name[j], input_doc_mat, movie_name_to_index)
                movie_sims_matrix[i][j] = sim_score
        
        return movie_sims_matrix


    def build_movie_sims_jac(self, n_mov, input_data):
        """Returns a movie_sims_jac matrix of size (num_movies,num_movies) where for (i,j) :
            [i,j] should be the jaccard similarity between the category sets for movies i and j
            such that movie_sims_jac[i,j] = movie_sims_jac[j,i]. 
            
        Note: 
            Movies sometimes contain *duplicate* categories! You should only count a category once
            
            A movie should have a jaccard similarity of 1.0 with itself.
        
        Params: {n_mov: Integer, the number of movies,
                input_data: List<Dictionary>, a list of dictionaries where each dictionary 
                        represents the movie_script_data including the script and the metadata of each movie script}
        Returns: Numpy Array 
        """
        genre_sims = np.zeros((n_mov, n_mov))
        
        # YOUR CODE HERE
        for i in range (0, n_mov):
            for j in range (0, n_mov):
                Al = input_data[i].split(',')
                Al = [s.strip() for s in Al]
                A = set(Al)
                
                Bl = input_data[j].split(',')
                Bl = [s.strip() for s in Bl]
                B = set(Bl)
                jac_sim = 0
                if(len(A.union(B)) > 0):
                    jac_sim = len(A.intersection(B))/len(A.union(B))
                else:
                    jac_sim = 0
                genre_sims[i][j] = jac_sim
                genre_sims[j][i] = jac_sim
                
        
        return genre_sims




    def get_ranked_movies(self, mov, matrix, anime_name_to_index, anime_index_to_name):
        """
        Return sorted rankings (most to least similar) of movies as 
        a list of two-element tuples, where the first element is the 
        movie name and the second element is the similarity score
        
        Params: {mov: String,
                matrix: np.ndarray}
        Returns: List<Tuple>
        """
        
        # Get movie index from movie name
        mov_idx = anime_name_to_index[mov]
        
        # Get list of similarity scores for movie
        score_lst = matrix[mov_idx]
        mov_score_lst = [(anime_index_to_name[i], s) for i,s in enumerate(score_lst)]
        
        # Do not account for movie itself in ranking
        mov_score_lst = mov_score_lst[:mov_idx] + mov_score_lst[mov_idx+1:]
        
        # Sort rankings by score
        mov_score_lst = sorted(mov_score_lst, key=lambda x: -x[1])
        
        return mov_score_lst

    def multiply_jac_sim(self, anime, genres, arr, anime_name_to_index, df):
        # Get movie index from movie name
        anime_idx = anime_name_to_index[anime]
        score_lst = []
        
        for i,tup in enumerate(arr):
            
            A = set(genres)
            l = df['Genres'][anime_name_to_index[tup[0]]].split(',')
            l = [s.strip() for s in l]
            B = set(l)
            jac_sim = 0
            if(len(A.union(B)) > 0):
                jac_sim = len(A.intersection(B))/len(A.union(B))        
            arr[i] = (tup[0], tup[1]*jac_sim)
            
        arr = sorted(arr, key=lambda x: -x[1])

        return arr
        
        
    def multiply_ratings(self, arr, df):
        for i, tup in enumerate(arr):
            score = df['Score'][i]
            score = 'hi'
            try:
                score = float(score)
            except:
                score = 5
            arr[i] = (tup[0], tup[1]*score)
            
        arr = sorted(arr, key=lambda x: -x[1])
        return arr



    def get_ranking(self, anime, genres):
        initial_ranking = self.get_ranked_movies(anime, self.movie_sims_cos, self.anime_name_to_index, self.anime_index_to_name)
        ranking_jac = self.multiply_jac_sim(anime, genres, initial_ranking, self.anime_name_to_index, self.df)
        ranking_score = self.multiply_ratings(ranking_jac, self.df) 
        return ranking_score[:10]  