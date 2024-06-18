import numpy as np
from tinybio import node, preprocess, token, request, reveal

def compare_facial_points(points1, points2):
    nodes = [node(), node()]

    length = len(points1)
    preprocess(nodes, length=length)

    points1 = np.array(points1).flatten()
    points2 = np.array(points2).flatten()

    reg_masks_1 = nodes[0].masks(request.registration(points1))
    reg_masks_2 = nodes[1].masks(request.registration(points2))

    reg_token_1 = token.registration(reg_masks_1, points1)
    reg_token_2 = token.registration(reg_masks_2, points2)

    auth_masks_1 = nodes[0].masks(request.authentication(points2))
    auth_masks_2 = nodes[1].masks(request.authentication(points1))

    auth_token_1 = token.authentication(auth_masks_1, points2)
    auth_token_2 = token.authentication(auth_masks_2, points1)

    shares_1 = nodes[0].authenticate(reg_token_1, auth_token_1)
    shares_2 = nodes[1].authenticate(reg_token_2, auth_token_2)

    result_1 = reveal(shares_1)
    result_2 = reveal(shares_2)

    return result_1, result_2

if __name__ == "__main__":
    points1 = [(30, 50), (40, 60), (50, 70)]  
    points2 = [(32, 52), (42, 62), (52, 72)]  

    result = compare_facial_points(points1, points2)
    print("result", result)
