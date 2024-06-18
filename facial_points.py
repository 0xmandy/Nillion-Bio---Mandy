from nada_dsl import *

def nada_main():
    party1 = Party(name="Party1")
    party2 = Party(name="Party2")
    
    points1 = [SecretInteger(Input(name=f"point1_{i}", party=party1)) for i in range(68)]
    points2 = [SecretInteger(Input(name=f"point2_{i}", party=party2)) for i in range(68)]
    
    def euclidean_distance(points1, points2):
        return sum([(p1 - p2) ** 2 for p1, p2 in zip(points1, points2)]) ** 0.5
    
    distance = euclidean_distance(points1, points2)
    
    threshold = Integer(30)  
    is_same_person = distance < threshold
    
    return [Output(is_same_person, "is_same_person", party1), Output(distance, "distance", party1)]
