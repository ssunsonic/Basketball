import os

def get_available_filename(output_dir, base_name, extension):
    counter = 1
    output_path = os.path.join(output_dir, f"{base_name}.{extension}")
    
    while os.path.exists(output_path):
        output_path = os.path.join(output_dir, f"{base_name}{counter}.{extension}")
        counter += 1
    
    return output_path