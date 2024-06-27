import os
import glob
import numpy as np

def parse_ply(file_path):
    with open(file_path, 'rb') as f:
        if f.read(3).decode('ascii') != "ply":
            raise ValueError("Not a PLY file")

        vertex_count = face_count = 0
        face_props = []

        # Parse header
        while True:
            line = f.readline().strip().decode('ascii')
            if line == "end_header":
                break
            elif line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
            elif line.startswith("element face"):
                face_count = int(line.split()[-1])
            elif line.startswith("property list") and face_count > 0:
                face_props = line.split()[-2:]

        print(f"Vertex count: {vertex_count}")
        print(f"Face count: {face_count}")
        print(f"Face properties: {face_props}")

        # Read vertices
        vertices = np.fromfile(f, dtype='<f4', count=vertex_count*3).reshape(-1, 3)

        # Read faces
        if face_props[-2] == 'uchar':
            face_dtype = np.dtype([('count', '<u1'), ('indices', '<i4', (3,))])
        elif face_props[-2] == 'uint':
            face_dtype = np.dtype([('count', '<u4'), ('indices', '<i4', (3,))])
        elif face_props[-2] == 'int':
            face_dtype = np.dtype([('count', '<i4'), ('indices', '<i4', (3,))])
        else:
            raise ValueError(f"Unsupported face property type: {face_props[-2]}")
        
        faces = np.fromfile(f, dtype=face_dtype, count=face_count)['indices']

        print(f"Parsed {len(vertices)} vertices and {len(faces)} faces")

        return vertices, faces

def write_obj(filepath, vertices, faces, chunk_size=1000000):
    with open(filepath, 'w') as f:
        # Write vertices
        for i in range(0, len(vertices), chunk_size):
            chunk = vertices[i:i+chunk_size]
            for v in chunk:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")

        # Write faces
        for i in range(0, len(faces), chunk_size):
            chunk = faces[i:i+chunk_size]
            for face in chunk:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

def decimate_mesh(vertices, faces, target_faces):
    # Simple decimation: keep every nth face
    n = max(len(faces) // target_faces, 1)
    decimated_faces = faces[::n]
    return vertices, decimated_faces

# Set these paths according to your setup
input_dir = "/home/john/Desktop/3D-Pose/MultiPly/code/outputs/Hi4D/taichi01/test_mesh/0/"
output_dir = "/home/john/Desktop/3D-Pose/MultiPly/code/outputs/Hi4D/taichi01/reduce_obj/0/"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Get all PLY files in the input directory
ply_files = glob.glob(os.path.join(input_dir, "*.ply"))
print(f"Found {len(ply_files)} PLY files")

for ply_file in ply_files:
    print(f"\nProcessing: {ply_file}")
    try:
        # Parse PLY file
        vertices, faces = parse_ply(ply_file)

        # Decimate mesh
        target_faces = max(int(len(faces) * 0.1), 4)  # 1% of original faces, minimum 4
        decimated_vertices, decimated_faces = decimate_mesh(vertices, faces, target_faces)

        print(f"Original faces: {len(faces)}, Decimated faces: {len(decimated_faces)}")

        # Prepare output filename
        output_file = os.path.join(output_dir, os.path.splitext(os.path.basename(ply_file))[0] + ".obj")

        # Write OBJ file
        write_obj(output_file, decimated_vertices, decimated_faces)
        print(f"Exported to: {output_file}")

    except Exception as e:
        print(f"Error processing {ply_file}: {str(e)}")

print("Conversion and decimation complete!")
