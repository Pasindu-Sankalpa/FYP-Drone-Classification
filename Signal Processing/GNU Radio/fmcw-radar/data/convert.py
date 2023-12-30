import struct

# Path to the binary file
binary_file_path = "data.bin"

# Path to the CSV file where converted data will be stored
csv_file_path = "converted_data.csv"

# Open the binary file in binary read mode
with open(binary_file_path, "rb") as binary_file:
    # Read the binary data
    binary_data = binary_file.read()

# Convert binary data to decimal numbers using struct.unpack
floats = struct.unpack(f"{len(binary_data)//4}f", binary_data)

# Open the CSV file in write mode
with open(csv_file_path, "w") as csv_file:
    # Write decimal numbers to the CSV file in comma-separated format
    csv_file.write(",".join(map(str, floats)))

print("Conversion complete.")
