import os
import glob
import numpy as np
import struct
import argparse

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

def fast_decimate_mesh(vertices, faces, target_faces):
    print("Starting fast decimation...")
    
    # Determine the bounding box of the mesh
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)
    
    # Calculate the number of cells in each dimension
    cell_count = int((len(faces) / target_faces) ** (1/3))
    cell_size = (max_coords - min_coords) / cell_count
    
    # Create a dictionary to store the clusters
    clusters = {}
    
    # Cluster vertices
    for i, vertex in enumerate(vertices):
        cell_coords = tuple((vertex - min_coords) // cell_size)
        if cell_coords not in clusters:
            clusters[cell_coords] = []
        clusters[cell_coords].append(i)
    
    # Create new vertices (cluster centers)
    new_vertices = []
    old_to_new = {}
    for cluster in clusters.values():
        new_index = len(new_vertices)
        for old_index in cluster:
            old_to_new[old_index] = new_index
        new_vertices.append(np.mean(vertices[cluster], axis=0))
    
    # Create new faces
    new_faces = []
    for face in faces:
        new_face = [old_to_new[v] for v in face]
        if len(set(new_face)) == 3:  # Only keep faces with 3 unique vertices
            new_faces.append(new_face)
    
    print(f"Fast decimation completed. New vertex count: {len(new_vertices)}, New face count: {len(new_faces)}")
    return np.array(new_vertices), np.array(new_faces)

def process_ply_file(ply_file, output_dir, decimate_factor):
    try:
        # Parse PLY file
        vertices, faces = parse_ply(ply_file)

        # Decimate mesh
        target_faces = max(int(len(faces) * decimate_factor), 4)  # Minimum 4 faces
        decimated_vertices, decimated_faces = fast_decimate_mesh(vertices, faces, target_faces)

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

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Convert PLY files to decimated OBJ files.")
    parser.add_argument("input_dir", help="Directory containing input PLY files")
    parser.add_argument("output_dir", help="Directory for output OBJ files")
    parser.add_argument("--decimate", type=float, default=0.01, help="Decimation factor (0.01 = 1% of original faces)")
    
    # Parse arguments
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Get all PLY files in the input directory
    ply_files = glob.glob(os.path.join(args.input_dir, "*.ply"))
    print(f"Found {len(ply_files)} PLY files")

    for ply_file in ply_files:
        print(f"\nProcessing: {ply_file}")
        process_ply_file(ply_file, args.output_dir, args.decimate)

    print("Conversion and decimation complete!")

if __name__ == "__main__":
    main()
