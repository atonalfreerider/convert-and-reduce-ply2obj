import os
import glob
import numpy as np
import struct
from collections import defaultdict

def parse_ply(file_path):
    with open(file_path, 'rb') as f:
        content = f.read()

    if content[:3].decode('ascii') != "ply":
        raise ValueError("Not a PLY file")

    # Find the start of vertex data
    header_end = content.index(b'end_header\n') + len(b'end_header\n')
    header = content[:header_end].decode('ascii')

    # Parse header
    lines = header.split('\n')
    vertex_count = face_count = 0
    vertex_props = []
    face_props = None
    for line in lines:
        if line.startswith("element vertex"):
            vertex_count = int(line.split()[-1])
        elif line.startswith("element face"):
            face_count = int(line.split()[-1])
        elif line.startswith("property") and vertex_count > 0 and face_count == 0:
            vertex_props.append(line.split()[-1])
        elif line.startswith("property list") and "vertex_indices" in line:
            face_props = line.split()

    print(f"Vertex count: {vertex_count}")
    print(f"Face count: {face_count}")
    print(f"Vertex properties: {vertex_props}")
    print(f"Face properties: {face_props}")

    # Read vertices
    vertex_size = 4 * len(vertex_props)
    vertex_data = content[header_end:header_end + vertex_count * vertex_size]
    vertices = np.frombuffer(vertex_data, dtype='<f4').reshape(-1, len(vertex_props))
    vertices = vertices[:, :3]  # Only keep x, y, z coordinates

    # Read faces
    face_data_start = header_end + vertex_count * vertex_size
    face_data = content[face_data_start:]
    
    faces = []
    offset = 0
    for i in range(face_count):
        if offset + 4 > len(face_data):
            print(f"Warning: Reached end of face data after {i} faces")
            break
        
        # Read the number of vertices for this face
        if face_props[2] == 'uchar':
            num_vertices = struct.unpack('<B', face_data[offset:offset+1])[0]
            offset += 1
        elif face_props[2] in ['uint', 'int']:
            num_vertices = struct.unpack('<I', face_data[offset:offset+4])[0]
            offset += 4
        else:
            raise ValueError(f"Unsupported face property type: {face_props[2]}")
        
        if offset + 4 * num_vertices > len(face_data):
            print(f"Warning: Not enough data for face {i}. Expected {num_vertices} vertices.")
            break
        
        face = struct.unpack(f'<{num_vertices}I', face_data[offset:offset+4*num_vertices])
        offset += 4 * num_vertices
        faces.append(face)
        
        if i < 5 or i >= face_count - 5:
            print(f"Face {i}: {face}")

    faces = np.array(faces)
    print(f"Parsed {len(vertices)} vertices and {len(faces)} faces")
    print(f"First few faces: {faces[:5]}")
    print(f"Last few faces: {faces[-5:]}")

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

def compute_quadric(v, faces, vertices):
    q = np.zeros((4, 4))
    for face in faces:
        if v in face:
            tri = vertices[face]
            normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
            normal /= np.linalg.norm(normal)
            a, b, c = normal
            d = -np.dot(normal, tri[0])
            p = np.array([a, b, c, d])
            q += np.outer(p, p)
    return q

def compute_error(q, v):
    v = np.append(v, 1)
    return np.dot(v, np.dot(q, v))

def decimate_mesh(vertices, faces, target_faces):
    print("Starting decimation...")
    vertices = np.array(vertices, dtype=np.float32, copy=True)
    
    # Compute initial quadrics
    quadrics = [compute_quadric(i, faces, vertices) for i in range(len(vertices))]
    
    # Create edge heap
    edge_heap = []
    for face in faces:
        for i in range(3):
            v1, v2 = face[i], face[(i+1)%3]
            if v1 < v2:
                q = quadrics[v1] + quadrics[v2]
                cost = compute_error(q, (vertices[v1] + vertices[v2]) / 2)
                heapq.heappush(edge_heap, (cost, (v1, v2)))

    # Decimate
    while len(faces) > target_faces and edge_heap:
        _, (v1, v2) = heapq.heappop(edge_heap)
        
        # Check if this edge is still valid
        if v2 not in faces[faces == v1].flatten():
            continue
        
        # Compute optimal position
        q = quadrics[v1] + quadrics[v2]
        q_3x3 = q[:3, :3]
        b = -q[:3, 3]
        try:
            new_pos = np.linalg.solve(q_3x3, b)
        except np.linalg.LinAlgError:
            new_pos = (vertices[v1] + vertices[v2]) / 2

        # Update mesh
        vertices[v1] = new_pos
        quadrics[v1] = q
        faces = np.array([f for f in faces if v2 not in f])
        faces[faces == v2] = v1

        # Update edges
        affected_vertices = np.unique(faces[np.any(faces == v1, axis=1)].flatten())
        for v in affected_vertices:
            if v != v1:
                q = quadrics[v1] + quadrics[v]
                cost = compute_error(q, (vertices[v1] + vertices[v]) / 2)
                heapq.heappush(edge_heap, (cost, (min(v1, v), max(v1, v))))

        if len(faces) % 1000 == 0:
            print(f"Faces remaining: {len(faces)}")

    print("Decimation completed.")
    return vertices, faces


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
        target_faces = max(int(len(faces) * 0.01), 4)  # 1% of original faces, minimum 4
        decimated_vertices, decimated_faces = decimate_mesh(vertices, faces, target_faces)

        print(f"Original faces: {len(faces)}, Decimated faces: {len(decimated_faces)}")

        # Prepare output filename
        output_file = os.path.join(output_dir, os.path.splitext(os.path.basename(ply_file))[0] + ".obj")

        # Write OBJ file
        write_obj(output_file, decimated_vertices, decimated_faces)
        print(f"Exported to: {output_file}")

    except Exception as e:
        print(f"Error processing {ply_file}: {str(e)}")
        import traceback
        traceback.print_exc()

print("Conversion and decimation complete!")
