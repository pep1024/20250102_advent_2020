import pandas as pd
import numpy as np
import re

def read_input(file = 'input.txt'):
    x = pd.read_csv(file, header=None, names=['information'])
    return(x)

def get_tile_number(n, df):
    
    text = df["information"].iloc[11 * (n - 1)]
    match = re.search(r"Tile (\d+):", text)
    return(int(match.group(1)))

def get_image(n, df):
    
    text = df["information"].iloc[(11 * (n - 1) + 1):(11 * (n - 1) + 11)]
    
    return(text)

def image_to_matrix(img):

    z=[]
    for y in img:
        z.append([int(char=='.') for char in y])
    z_np = np.array(z)
    return(z_np)

def matrix_to_image(m):
    z=[]
    for y in m:
        z.append("".join(['.' if x == 1 else '#' for x in y]))
    z_df = pd.Series(z)
    z_df.name = "information"
    return(z_df)

def transform(m):
    # Generate the 8 transformations of the matrix
    rot90_1 = np.rot90(m, k=-1)  # 90 degrees clockwise
    rot90_2 = np.rot90(m, k=-2)  # 180 degrees clockwise
    rot90_3 = np.rot90(m, k=-3)  # 270 degrees clockwise
    flip_horiz = np.fliplr(m)    # Horizontal flip
    flip_vert = np.flipud(m)     # Vertical flip
    # main_diag_flip = m.T         # Flip along the main diagonal (transpose)
    # anti_diag_flip = np.transpose(np.fliplr(m))  # Flip along the anti-diagonal
    d1 = np.rot90(flip_horiz, k=-1)  # Horizontal flip + 90 degrees rotation
    d2 = np.rot90(flip_vert, k=-1) # Vertical flip + 90 degrees rotation
    # Store all matrices in a dictionary for display
    ma_transformed = {
        "Original": m,
        "90° Rotation": rot90_1,
        "180° Rotation": rot90_2,
        "270° Rotation": rot90_3,
        "Horizontal Flip": flip_horiz,
        "Vertical Flip": flip_vert,
#        "Main Diagonal Flip": main_diag_flip,
#        "Anti-Diagonal Flip": anti_diag_flip
        "d1": d1,
        "d2": d2
    }
    return(pd.Series(ma_transformed))

def boundary_to_decimal(matrix):
    boundary_1 = ''.join(map(str, matrix[0]))  # Row 0
    boundary_2 = ''.join(map(str, matrix[:, -1]))  # Column 9
    boundary_3 = ''.join(map(str, matrix[-1]))  # Row 9
    boundary_4 = ''.join(map(str, matrix[:, 0]))  # Column 0
    
    # Convert to decimal
    boundary_1_decimal = int(boundary_1, 2)
    boundary_2_decimal = int(boundary_2, 2)
    boundary_3_decimal = int(boundary_3, 2)
    boundary_4_decimal = int(boundary_4, 2)
    
    return ([boundary_1_decimal, boundary_2_decimal, boundary_3_decimal, boundary_4_decimal])

def get_all_boundaries(df):
    z = []
    # Each tile defined in 11 rows
    for i in range(len(df)//11):
        tile = get_tile_number(i+1, df)
        img = get_image(i+1, df)
        m = image_to_matrix(img)
        m_all = transform(m)
        for j in range(8):
            boundary = boundary_to_decimal(m_all.iloc[j])
            for k in range(len(boundary)):
                z.append([tile, j+1, k+1, boundary[k]])
    resp = pd.DataFrame(z, columns=['id', 'transformation', 'boundary', 'value'])
    return(resp)

def get_all_transforms(df):
    z = []
    for i in range(len(df)//11):
        tile = get_tile_number(i+1, df)
        img = get_image(i+1, df)
        m = image_to_matrix(img)
        m_all = transform(m)
        for j in range(8):
            boundary = boundary_to_decimal(m_all.iloc[j])
            z.append([tile, j+1, boundary])
    resp = pd.DataFrame(z, columns=['id', 'transformation', 'boundaries'])
    return(resp)

def n_coincidences(b, id, boundaries):
    new_df = boundaries[boundaries['id']!=id] 
    matches = new_df[new_df['value'] == b]['id'].nunique()
    return(matches)

def coincidence_piece(b, id, position,  boundaries):
    new_df = boundaries[boundaries['id']!=id] 
    matches = new_df[new_df['value'] == b][['id', 'transformation']]
    return(matches)