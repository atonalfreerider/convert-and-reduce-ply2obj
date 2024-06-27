import os
import glob
import argparse
import pymeshlab

def process_ply_file(input_file, output_file, target_percentage):
    print(f"Processing: {input_file}")
    
    # Create a new MeshSet
    ms = pymeshlab.MeshSet()
    
    # Load the PLY file
    ms.load_new_mesh(input_file)
    
    # Get the current face count
    original_face_count = ms.current_mesh().face_number()
    
    print(f"Original face count: {original_face_count}")
    
    # Calculate target face number
    target_face_number = int(original_face_count * target_percentage)
    
    # Simplify the mesh
    print("Starting PyMeshLab simplification...")
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=target_face_number, preservenormal=True)
    
    # Get the new face count
    new_face_count = ms.current_mesh().face_number()
    
    print(f"Simplified face count: {new_face_count}")
    
    # Save the result as OBJ
    ms.save_current_mesh(output_file)
    
    print(f"Exported to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Convert PLY files to decimated OBJ files using PyMeshLab.")
    parser.add_argument("input_dir", help="Directory containing input PLY files")
    parser.add_argument("output_dir", help="Directory for output OBJ files")
    parser.add_argument("--decimate", type=float, default=0.1, help="Decimation factor (0.1 = 10% of original faces)")
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    ply_files = glob.glob(os.path.join(args.input_dir, "*.ply"))
    print(f"Found {len(ply_files)} PLY files")

    for ply_file in ply_files:
        try:
            output_file = os.path.join(args.output_dir, os.path.splitext(os.path.basename(ply_file))[0] + ".obj")
            process_ply_file(ply_file, output_file, args.decimate)
        except Exception as e:
            print(f"Error processing {ply_file}: {str(e)}")
            import traceback
            traceback.print_exc()

    print("Conversion and decimation complete!")

if __name__ == "__main__":
    main()
