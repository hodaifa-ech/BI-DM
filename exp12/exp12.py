import numpy as np
from numpy . linalg import norm
A = np . array ([2 , 1 , 2 , 3 , 2 , 9])
B = np . array ([3 , 4 , 2 , 4 , 5 , 5])
cosine_sim = np . dot (A , B ) / ( norm ( A ) * norm ( B ) )
print ( f " Cosine Similarity : { cosine_sim :.4 f } " )
A = {1 , 2 , 3 , 5 , 7}
B = {1 , 2 , 4 , 8 , 9}

def jaccard_similarity (A , B ) :
    intersection = len ( A . intersection ( B ) )
    union = len ( A . union ( B ) )
    return intersection / union

 def jaccard_distance (A , B ) :
  return 1 - jaccard_similarity (A , B )

print ( f " Jaccard Similarity : { jaccard_similarity (A , B ) :.2 f } " )
print ( f " Jaccard Distance : { jaccard_distance (A , B ) :.2 f } " )

import numpy as np
point1 = np . array ([4 , 4 , 2])
point2 = np . array ([1 , 2 , 1])
euclidean_dist = np . linalg . norm ( point1 - point2 )
print ( f " Euclidean Distance : { euclidean_dist :.4 f } " )

def manhattan_distance (a , b ) :
    return sum ( abs ( x - y ) for x , y in zip (a , b ) )
A = [2 , 4 , 4 , 6]
B = [5 , 5 , 7 , 8]
print ( f " Manhattan Distance : { manhattan_distance (A , B ) } " )


import numpy as np
import matplotlib . pyplot as plt
def estimate_coef (x , y ) :
    n = np . size ( x )
    m_x , m_y = np . mean ( x ) , np . mean ( y )
    SS_xy = np . sum ( y * x ) - n * m_y * m_x
    SS_xx = np . sum ( x * x ) - n * m_x * m_x
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1 * m_x
    return ( b_0 , b_1 )